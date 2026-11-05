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

Notes on the v4 SDK (LangChain callback handler, OTEL-native):
- Handler import:  from langfuse.langchain import CallbackHandler
- Client:           from langfuse import get_client
- Manual spans:     client.start_as_current_observation(as_type="span", ...)
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


@contextlib.contextmanager
def _observation(name, input=None, metadata=None):
    """A manual span that yields None when tracing is off or fails.

    The span is created with the minimal constructor and metadata is
    attached in a guarded update(), so a future SDK signature change
    can lose the metadata but never the span itself.
    """
    if not tracing_enabled():
        yield None
        return
    try:
        _ensure_base_url()
        from langfuse import get_client

        client = get_client()
        with client.start_as_current_observation(as_type="span", name=name) as obs:
            if input is not None or metadata:
                try:
                    obs.update(input=input, metadata=metadata or {})
                except Exception:
                    pass
            yield obs
    except Exception:
        yield None


def question_span(question, metadata=None):
    """One trace per chat question, opened around ask()/stream_ask().

    The question text is the span input; model and conversation id
    ride in metadata. The LangChain callback handler (attached via
    run_config) nests the generation run under this span.
    """
    return _observation("sidekick-question", input={"question": question}, metadata=metadata)


def span(name, metadata=None):
    """A named span for one step inside a question trace."""
    return _observation(name, metadata=metadata)
