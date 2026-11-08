"""Opt-in Langfuse tracing for Sidekick (Langfuse Python SDK v4).

One trace per chat question, with spans for rewrite, retrieve, rerank
and generate, so a slow question can be debugged step by step. Langfuse
records span timing natively; that is the point of this module.

Tracing is OFF unless all three env vars are set:

    LANGFUSE_BASE_URL   (or LANGFUSE_HOST) - the Langfuse server URL
    LANGFUSE_PUBLIC_KEY - the project's public key
    LANGFUSE_SECRET_KEY - the project's secret key

With them unset, every helper here is a no-op: no errors, no startup
failure, no slowdown. Tracing is also fail-safe: if the SDK is missing
or the server is down, calls fall back to the untraced path instead of
raising. Tracing is async and non-blocking: the SDK batches events
off the request path, so a tracing call never holds up a token.

Structure follows Langfuse's tracing best practices
(https://langfuse.com/docs/observability/best-practices), because the
shape of a trace is what evaluators, dashboards and dataset runs read:

- One trace per turn, one session per conversation. The API passes the
  conversation id as ``session_id``, so the Sessions view replays the
  whole chat instead of showing unrelated single questions.
- The root observation carries both input (the question) and output
  (the answer). Without the output the tracing table and every
  LLM-as-a-judge evaluator see an empty result.
- Observations are typed. LLM calls arrive as ``generation`` runs from
  the LangChain callback handler (which also supplies the model), and
  retrieval steps are typed ``retriever`` rather than a generic span.
- ``request_id``, ``endpoint`` and ``model`` ride in metadata, not in
  the name, so names stay low-cardinality and a trace can still be
  matched to its structured log line.

Notes on the v4 SDK (LangChain callback handler, OTEL-native):
- Handler import:  from langfuse.langchain import CallbackHandler
- Client:           from langfuse import get_client
- Manual spans:     client.start_as_current_observation(as_type="span", ...)
- Trace attributes: from langfuse import propagate_attributes
- v4 uses LANGFUSE_BASE_URL; LANGFUSE_HOST is accepted here as a
  fallback and translated before the client is created.

Two threading notes. First, Langfuse's context propagation does not
follow ThreadPoolExecutor worker threads, so the query rewrite (which
runs in a worker thread) shows as its own trace instead of nesting
under the question trace. The timing is still there. Second, spans
opened inside a generator (stream_ask) close when the generator is
exhausted or the client disconnects; a disconnect closes the span,
it does not leak it.
"""

import contextlib
import functools
import os

ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def tracing_enabled():
    """True only when a server URL and both keys are set."""
    base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST", "")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    return bool(base_url and public_key and secret_key)


def _ensure_base_url():
    """v4 reads LANGFUSE_BASE_URL; accept LANGFUSE_HOST as a fallback."""
    if not os.environ.get("LANGFUSE_BASE_URL") and os.environ.get("LANGFUSE_HOST"):
        os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]


@functools.lru_cache(maxsize=8)
def _get_handler(base_url, public_key, secret_key):
    """Build the LangChain callback handler once per env triple.

    Returns None when the SDK is missing or the client cannot be
    created. Never raises: a tracing failure must never break a
    question. Keyed on the env values so tests with different envs
    never share a stale handler.
    """
    try:
        _ensure_base_url()
        from langfuse import get_client

        get_client()  # initialise the singleton from the env vars
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        return None


def get_callbacks():
    """LangChain callbacks for the current call, or [] when off."""
    if not tracing_enabled():
        return []
    handler = _get_handler(
        os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST", ""),
        os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        os.environ.get("LANGFUSE_SECRET_KEY", ""),
    )
    return [handler] if handler is not None else []


def run_config(metadata=None, run_name=None):
    """RunnableConfig for a LangChain invoke/stream call.

    Returns None when tracing is off, so call sites pass the result
    straight through: chain.invoke(x, config=tracing.run_config()).
    LangChain runnables accept config=None.
    """
    callbacks = get_callbacks()
    if not callbacks:
        return None
    config = {"callbacks": callbacks}
    if metadata:
        config["metadata"] = metadata
    if run_name:
        config["run_name"] = run_name
    return config


def invoke_llm(llm, prompt, run_name=None):
    """Invoke an LLM with the tracing callback attached when tracing
    is on. Plain llm.invoke(prompt) otherwise, so hand-rolled fakes
    with an invoke(prompt) signature keep working (see tests).
    """
    config = run_config(run_name=run_name)
    if config is None:
        return llm.invoke(prompt)
    return llm.invoke(prompt, config=config)


def _propagate(session_id=None, user_id=None, tags=None):
    """propagate_attributes(...) for the current observation tree.

    session_id groups every trace of one conversation, user_id enables
    per-user cost views, and tags carry stable business dimensions
    (they are immutable, so they must be set at creation time). Any
    subset may be passed; when nothing applies — or when the installed
    SDK predates propagate_attributes — this is an empty context.
    """
    attrs = {}
    if session_id:
        attrs["session_id"] = str(session_id)
    if user_id:
        attrs["user_id"] = str(user_id)
    if tags:
        attrs["tags"] = list(tags)
    if not attrs:
        return contextlib.nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(**attrs)
    except Exception:
        return contextlib.nullcontext()


@contextlib.contextmanager
def _observation(
    name,
    input=None,
    metadata=None,
    as_type="span",
    session_id=None,
    user_id=None,
    tags=None,
):
    """A manual span that yields None when tracing is off or fails.

    The span is created with the minimal constructor and metadata is
    attached in a guarded update(), so a future SDK signature change
    can lose the metadata but never the span itself.

    Exceptions from the wrapped block are re-raised untouched and mark
    the span as an error; a tracing problem never swallows, replaces
    or duplicates an application error.
    """
    if not tracing_enabled():
        yield None
        return
    stack = contextlib.ExitStack()
    try:
        _ensure_base_url()
        from langfuse import get_client

        client = get_client()
        obs = stack.enter_context(
            client.start_as_current_observation(as_type=as_type, name=name)
        )
    except Exception:
        stack.close()
        yield None
        return
    try:
        if input is not None or metadata:
            try:
                obs.update(input=input, metadata=metadata or {})
            except Exception:
                pass
        stack.enter_context(_propagate(session_id=session_id, user_id=user_id, tags=tags))
        yield obs
    except BaseException as exc:
        try:
            obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        try:
            stack.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise
    else:
        stack.close()


def question_span(question, metadata=None, session_id=None, user_id=None, tags=None):
    """One trace per chat question, opened around ask()/stream_ask().

    The question text is the span input (so the trace input is the user
    message, not a raw argument blob); session_id pulls the turn into
    its conversation, and tags mark which feature produced it. The
    LangChain callback handler (attached via run_config) nests the
    generation run under this span.
    """
    return _observation(
        "sidekick-question",
        input={"question": question},
        metadata=metadata,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    )


def span(name, metadata=None, as_type="span", input=None, session_id=None, user_id=None, tags=None):
    """A named span for one step, or a root span for a non-chat task.

    as_type picks the most specific observation type (for example
    "retriever" for a lookup step), which is what Langfuse uses to
    drive the agent graph and per-type analytics. A root span (e.g. a
    document summary) can carry tags and a session too.
    """
    return _observation(
        name,
        input=input,
        metadata=metadata,
        as_type=as_type,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
    )


def set_output(obs, text):
    """Set the output of an observation (the root's becomes the trace output).

    A no-op without an observation or text, so call sites never need to
    ask whether tracing is on.
    """
    if obs is None or text is None:
        return
    try:
        obs.update(output=text)
    except Exception:
        pass


def flush():
    """Send buffered events now. For short-lived scripts, before exit.

    Long-running servers do not need this: the SDK flushes on its own
    background thread and registers an atexit hook.
    """
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


def shutdown():
    """Flush and stop the background exporter, swallowing SDK failures."""
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client

        get_client().shutdown()
    except Exception:
        pass
