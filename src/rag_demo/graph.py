"""LangGraph wrap of the RAG ask chain (Phase 2).

The graph is the same linear chain (episodic -> rewrite -> retrieve ->
generate), orchestrated as a LangGraph StateGraph instead of an LCEL
chain. Same steps, same primitives, same behavior: the nodes call the
same helpers the chain uses (rewrite_query, retriever.search, the same
prompt template, _invoke_chain, ThinkBlockFilter), so the answer for a
question is the same whichever path runs it. No new capabilities here:
no conditional retrieval, no self-correcting loops, no tool calls, no
new persistent state. Those are design decisions for a later review.

Routing: ask()/stream_ask() in chain.py dispatch here when
Settings.use_langgraph is set (RAG_DEMO_USE_LANGGRAPH=1). The API, CLI,
and evals need no changes; they keep calling chain.ask/stream_ask.

Streaming: the generate node writes each LLM chunk with LangGraph's
stream writer (get_stream_writer) and the stream wrapper drives
graph.stream(stream_mode=["custom", "values"]). "messages" mode does
NOT surface tokens for OllamaLLM (a plain BaseLLM, not a chat model);
"events" is not a valid .stream() mode in langgraph 1.2. The custom
mode is the working path (verified in the sandbox).

Disconnect: closing a graph.stream() generator does NOT interrupt a
running node (it runs to completion), so the generate node checks a
threading.Event before each pull and the stream wrapper sets it in its
finally before closing the iterator. That is what abandons generation
on client disconnect, matching the chain's close() behavior.
"""

from __future__ import annotations

import threading
import time
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from . import caching, tracing
from .audit import log_question
from .caching import get_cached_episodic_memory, get_cached_retriever
from .chain import (
    Answer,
    ThinkBlockFilter,
    _generation_hits,
    _invoke_chain,
    _model_name,
    _weak_evidence,
    build_prompt,
    format_episodic,
    format_hits,
    general_knowledge_answer,
    is_abstention,
    is_fallback_answer,
    is_small_talk,
    small_talk_answer,
    strip_think,
)
from .config import Settings, get_llm, get_settings
from .observability import (
    log_error_event,
    log_question_event,
    new_request_id,
    record_cache_hit,
    record_cache_reload,
    record_llm_error,
    record_question,
)
from .ollama_guard import OllamaTimeoutError, iter_with_idle_timeout
from .retriever import Hit, Retriever
from .rewrite import rewrite_query

# Same head-buffer length as chain._stream_ask_impl: tokens are held
# until 200 visible chars so an abstention can be caught before display.
HEAD_CHECK_LEN = 200


class _CancellableStream:
    """Wraps the graph stream so closing it first signals the generate
    node to stop (via the cancel event) and only then closes the graph
    iterator.

    The ordering matters: the idle-timeout wrapper closes its source
    iterator in its own finally, which runs before the stream wrapper's
    finally (the GeneratorExit unwinds the for-loop iterator first).
    Closing the graph stream alone waits for the running node to
    finish, so without the event set first the node would run to
    completion on disconnect. Setting the event here makes the node
    stop after its in-flight pull, and the close then returns promptly.
    The event is set before the inner close is attempted, so the node
    stops even when that close raises ValueError (stream mid-pull in
    the idle-timeout worker)."""

    def __init__(self, stream_iter, cancel_event: threading.Event):
        self._it = stream_iter
        self._cancel_event = cancel_event

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        self._cancel_event.set()
        close = getattr(self._it, "close", None)
        if close is not None:
            close()


class QuestionState(TypedDict, total=False):
    """Per-question data flowing through the graph. Plain data only;
    runtime deps (settings, llm, retriever) are closed over by the
    factory, not carried in state."""

    question: str
    history: list[tuple[str, str]]
    document_id: str | None
    user_groups: list[str] | None
    standalone: str
    episodic_context: str
    hits: list[Hit]
    context: str
    text: str
    small_talk: bool


def build_question_graph(
    settings: Settings | None = None,
    *,
    llm=None,
    retriever: Retriever | None = None,
    streaming: bool = False,
    cancel_event: threading.Event | None = None,
    timings: dict | None = None,
    capture: dict | None = None,
    ctx: dict | None = None,
):
    """Build the question graph. One graph per question: cheap to
    compile, and each call gets its own timings/capture/cancel state.

    settings: resolved Settings (defaults via get_settings()).
    llm: the model to call (defaults via get_llm(settings)).
    retriever: explicit retriever (e.g. an in-memory upload index);
        None means the app-lifetime cached retriever, as in build_chain.
    streaming: when True the generate node streams the LLM and writes
        each chunk with the stream writer; when False it uses the
        guarded one-shot _invoke_chain (timeout + retry, as in ask()).
    cancel_event: set by the stream wrapper on disconnect; the
        generate node stops pulling when it is set.
    timings/capture: the call's working dicts, updated by the nodes
        (retrieve_ms, retrieve_calls, captured hits for citations).
    ctx: node-to-caller outputs (resolved retriever, groups,
        source_filter) for the answer assembly after the run.
    """
    settings = settings or get_settings()
    llm = llm if llm is not None else get_llm(settings)
    timings = timings if timings is not None else {}
    capture = capture if capture is not None else {}
    ctx = ctx if ctx is not None else {}

    def _episodic_node(state: QuestionState) -> dict:
        # Same block as chain._ask_impl/_stream_ask_impl: episodic
        # memory is supplementary context only, never merged into
        # history, and a failure here never breaks the question.
        with tracing.span("episodic"):
            _t0 = time.perf_counter()
            context = ""
            try:
                episodes = get_cached_episodic_memory(settings).search(
                    state["question"], k=3
                )
                if episodes:
                    context = format_episodic(episodes)
            except Exception:
                pass
            finally:
                timings["episodic_ms"] = (time.perf_counter() - _t0) * 1000
            return {"episodic_context": context}

    def _rewrite_node(state: QuestionState) -> dict:
        with tracing.span("rewrite"):
            history = state.get("history") or []
            if settings.rewrite_enabled and history:
                _t0 = time.perf_counter()
                standalone = rewrite_query(state["question"], history, llm)
                timings["rewrite_ms"] = (time.perf_counter() - _t0) * 1000
            else:
                standalone = state["question"]
            return {"standalone": standalone}

    def _retrieve_node(state: QuestionState) -> dict:
        # The one retrieval per question. Mirrors build_chain's cached-
        # retriever logic and its _retrieve timing/capture: the hits
        # land in capture so the answer reuses them for citations
        # instead of searching a second time (see #58).
        with tracing.span("retrieve"):
            resolved = retriever
            if resolved is None:
                _key = caching._retriever_key(settings)
                _was_cached = _key in caching._RETRIEVERS
                resolved = get_cached_retriever(settings)
                if _was_cached:
                    record_cache_hit()
                else:
                    record_cache_reload()
            groups = (
                state.get("user_groups")
                if state.get("user_groups") is not None
                else settings.user_groups
            )
            document_id = state.get("document_id")
            source_filter = f"upload:{document_id}" if document_id else None
            ctx["retriever"] = resolved
            ctx["groups"] = groups
            ctx["source_filter"] = source_filter
            _t0 = time.perf_counter()
            try:
                hits = resolved.search(
                    state["standalone"],
                    user_groups=groups,
                    source_filter=source_filter,
                )
                capture["hits"] = hits
            finally:
                timings["retrieve_ms"] = (timings.get("retrieve_ms") or 0.0) + (
                    time.perf_counter() - _t0
                ) * 1000
                timings["retrieve_calls"] = timings.get("retrieve_calls", 0) + 1
            parts = []
            if state.get("episodic_context"):
                parts.append(state["episodic_context"])
            parts.append(format_hits(hits))
            context = "\n\n".join(p for p in parts if p)
            return {"hits": hits, "context": context}

    def _generate_node(state: QuestionState) -> dict:
        with tracing.span("generate"):
            prompt = build_prompt(settings)
            if streaming:
                # Stream the LLM directly (not via the LCEL fragment's
                # .stream()): abandoning a fragment stream does not stop
                # the underlying LLM stream — its transform cleanup
                # drains the remaining chunks instead — so a disconnect
                # would run generation to completion. llm.stream()
                # stops pulling as soon as the node stops iterating,
                # which is what the cancel event relies on. The prompt
                # text is the same template the fragment formats.
                writer = get_stream_writer()
                prompt_text = prompt.format(
                    context=state.get("context", ""),
                    question=state["standalone"],
                )
                parts: list[str] = []
                it = iter(
                    llm.stream(
                        prompt_text,
                        config=tracing.run_config(run_name="generate"),
                    )
                )
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    try:
                        chunk = next(it)
                    except StopIteration:
                        break
                    # OllamaLLM yields str; chat models (and the fake
                    # LLM) yield message chunks — normalize to text.
                    text = (
                        chunk.content
                        if hasattr(chunk, "content")
                        else str(chunk)
                    )
                    parts.append(text)
                    writer({"token": text})
                return {"text": "".join(parts)}
            # One-shot generation with the shared timeout and retry
            # policy, exactly as chain._invoke_chain provides it. The
            # fragment takes the {context, question} dict, not the
            # bare question string. Note the retry difference from
            # the chain: a transient failure retries only this LLM
            # call (retrieval already ran, in its own node), where
            # the chain retries retrieve+LLM together. Same outcome.
            fragment = prompt | llm | StrOutputParser()
            text = _invoke_chain(
                fragment,
                {
                    "context": state.get("context", ""),
                    "question": state["standalone"],
                },
                settings,
                config=tracing.run_config(run_name="generate"),
            )
            return {"text": text}

    def _small_talk_node(state: QuestionState) -> dict:
        # small_talk_answer already wraps itself in the "small-talk"
        # span and uses the guarded invoke_llm.
        text = small_talk_answer(state["question"], llm, settings)
        return {"text": text, "small_talk": True}

    def _route(state: QuestionState) -> str:
        return "small_talk" if is_small_talk(state["question"]) else "episodic"

    graph = StateGraph(QuestionState)
    graph.add_node("episodic", _episodic_node)
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("small_talk", _small_talk_node)
    graph.add_conditional_edges(
        START, _route, {"small_talk": "small_talk", "episodic": "episodic"}
    )
    graph.add_edge("episodic", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("small_talk", END)
    return graph.compile()


def _run_graph_question(
    question: str,
    settings: Settings,
    llm,
    history: list[tuple[str, str]] | None,
    user_groups: list[str] | None,
    retriever: Retriever | None,
    document_id: str | None,
    timings: dict,
    capture: dict,
    ctx: dict,
) -> dict:
    """Run the non-streaming graph for one question. Returns the final
    state. No logging here: the caller assembles the answer and logs,
    so the streaming one-shot fallback can reuse this without
    double-logging the question."""
    graph = build_question_graph(
        settings,
        llm=llm,
        retriever=retriever,
        streaming=False,
        timings=timings,
        capture=capture,
        ctx=ctx,
    )
    return graph.invoke(
        {
            "question": question,
            "history": history or [],
            "document_id": document_id,
            "user_groups": user_groups,
        }
    )


def ask_via_graph(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
    session_id: str | None = None,
) -> Answer:
    """Non-streaming ask through the graph. Same signature, same
    Answer, same logging as chain.ask(). Opens one Langfuse trace per
    question (a no-op when tracing is off)."""
    settings = settings or get_settings()
    metadata = {"model": settings.ollama_model}
    if document_id:
        metadata["document_id"] = document_id
    if request_id:
        metadata["request_id"] = request_id
    if endpoint:
        metadata["endpoint"] = endpoint
    with tracing.question_span(
        question, metadata=metadata, session_id=session_id, tags=("feature:chat",)
    ) as obs:
        answer = _ask_via_graph_impl(
            question,
            settings=settings,
            history=history,
            user_groups=user_groups,
            retriever=retriever,
            document_id=document_id,
            request_id=request_id,
            endpoint=endpoint,
        )
        tracing.set_output(obs, answer.text)
        return answer


def _ask_via_graph_impl(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
) -> Answer:
    settings = settings or get_settings()
    history = history or []
    request_id = request_id or new_request_id()
    model = _model_name(settings)
    llm = get_llm(settings)
    timings: dict = {
        "rewrite_ms": None,
        "episodic_ms": None,
        "retrieve_ms": None,
        "retrieve_calls": 0,
        "generate_ms": None,
        "total_ms": None,
    }
    capture: dict = {}
    ctx: dict = {}
    # Whether this question built a fresh retriever (cache reload) or
    # reused the cached one (hit). Snapshot before the graph runs, as
    # in chain._ask_impl; an explicitly passed retriever is neither.
    cache_reload = retriever is None and (
        caching._retriever_key(settings) not in caching._RETRIEVERS
    )
    started = time.perf_counter()
    try:
        final = _run_graph_question(
            question,
            settings,
            llm,
            history,
            user_groups,
            retriever,
            document_id,
            timings,
            capture,
            ctx,
        )
        text = strip_think(final.get("text") or "")
        # Server-side hybrid fallback, as in chain._ask_impl: if the
        # model abstained but hybrid is on, answer from general
        # knowledge instead of showing "No relevant answer".
        if settings.hybrid_fallback and is_abstention(text):
            text = general_knowledge_answer(
                final.get("standalone") or question, llm, settings
            )
    except Exception as e:
        # No behavior change: the error still propagates to the caller
        # (the API returns 500 as before). It is only counted and logged.
        record_llm_error(e)
        log_error_event(
            request_id=request_id,
            endpoint=endpoint,
            where="generate",
            error_type=type(e).__name__,
            error=str(e),
            model=model,
        )
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    if final.get("small_talk"):
        # Small talk bypasses retrieval entirely, as in chain._ask_impl.
        # generate_ms here is the graph run time; the chain times only
        # the small-talk call, which dominates the run either way.
        small_talk_timings: dict = {
            "rewrite_ms": None,
            "episodic_ms": None,
            "retrieve_ms": None,
            "retrieve_calls": 0,
            "generate_ms": latency_ms,
            "total_ms": latency_ms,
        }
        record_question(small_talk_timings, streamed=False)
        log_question_event(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            question=question,
            timings=small_talk_timings,
            hits=0,
            sources=[],
            grounded=True,
            weak_evidence=False,
            standalone=False,
            streamed=False,
            cache_reload=False,
        )
        return Answer(
            text=text,
            sources=[],
            hits=[],
            latency_ms=0.0,
            grounded=True,
            standalone_question=None,
        )
    # generate_ms is graph time minus retrieval, so it includes prompt
    # formatting and retriever construction: an approximation, not a
    # pure LLM timer (see observability.log_question_event).
    timings["generate_ms"] = max(
        latency_ms - (timings.get("retrieve_ms") or 0.0), 0.0
    )
    timings["total_ms"] = latency_ms
    standalone = final.get("standalone") or question
    # Citations reuse the hits generation already retrieved: no second
    # search. Traced as its own span so the reuse is visible in the
    # trace, as in chain._ask_impl.
    with tracing.span("retrieve-hits"):
        hits = _generation_hits(
            capture,
            ctx.get("retriever") or retriever,
            standalone,
            ctx.get("groups"),
            ctx.get("source_filter"),
        )
    sources = sorted({h.source for h in hits})
    answer = Answer(
        text=text,
        sources=sources,
        hits=hits,
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(text),
        standalone_question=standalone if standalone != question else None,
        weak_evidence=_weak_evidence(ctx.get("retriever") or retriever),
    )
    log_question(settings, question, answer, latency_ms)
    record_question(timings, streamed=False)
    log_question_event(
        request_id=request_id,
        endpoint=endpoint,
        model=model,
        question=question,
        timings=timings,
        hits=len(hits),
        sources=sources,
        grounded=answer.grounded,
        weak_evidence=answer.weak_evidence,
        standalone=answer.standalone_question is not None,
        streamed=False,
        cache_reload=cache_reload,
    )
    # Save to episodic memory for long-term recall (skip if it was a
    # fallback to avoid polluting memory with "I don't know"
    # responses). Same cached instance as the search in the node.
    if answer.grounded:
        try:
            get_cached_episodic_memory(settings).add(question, text)
        except Exception:
            pass
    return answer


def stream_ask_via_graph(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
    session_id: str | None = None,
):
    """Streaming ask through the graph. Same yielded events as
    chain.stream_ask(): {"token": str} chunks as the LLM generates,
    then a final {"answer": Answer}.

    Opens one Langfuse trace per question (a no-op when tracing is
    off). The trace stays open until the stream is exhausted or the
    client disconnects."""
    settings = settings or get_settings()
    metadata = {"model": settings.ollama_model}
    if document_id:
        metadata["document_id"] = document_id
    if request_id:
        metadata["request_id"] = request_id
    if endpoint:
        metadata["endpoint"] = endpoint
    with tracing.question_span(
        question, metadata=metadata, session_id=session_id, tags=("feature:chat",)
    ) as obs:
        for event in _stream_ask_via_graph_impl(
            question,
            settings=settings,
            history=history,
            user_groups=user_groups,
            retriever=retriever,
            document_id=document_id,
            request_id=request_id,
            endpoint=endpoint,
        ):
            if "answer" in event:
                tracing.set_output(obs, event["answer"].text)
            yield event


def _stream_ask_via_graph_impl(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
):
    settings = settings or get_settings()
    history = history or []
    request_id = request_id or new_request_id()
    model = _model_name(settings)
    llm = get_llm(settings)
    timings: dict = {
        "rewrite_ms": None,
        "episodic_ms": None,
        "retrieve_ms": None,
        "retrieve_calls": 0,
        "generate_ms": None,
        "total_ms": None,
    }
    capture: dict = {}
    ctx: dict = {}
    cache_reload = retriever is None and (
        caching._retriever_key(settings) not in caching._RETRIEVERS
    )
    started = time.perf_counter()
    cancel_event = threading.Event()
    graph = build_question_graph(
        settings,
        llm=llm,
        retriever=retriever,
        streaming=True,
        cancel_event=cancel_event,
        timings=timings,
        capture=capture,
        ctx=ctx,
    )
    parts: list[str] = []
    think_filter = ThinkBlockFilter()
    # Buffer the head of the stream to detect abstentions before displaying.
    # If the model declines to answer, we suppress the abstention and stream
    # a general-knowledge fallback instead (when hybrid is on).
    head_buf: list[str] = []
    head_text = ""
    abstention_detected = False
    final_state: dict = {}
    stream_iter = None
    try:
        stream_iter = _CancellableStream(
            graph.stream(
                {
                    "question": question,
                    "history": history,
                    "document_id": document_id,
                    "user_groups": user_groups,
                },
                stream_mode=["custom", "values"],
            ),
            cancel_event,
        )
        for mode, payload in iter_with_idle_timeout(
            stream_iter,
            settings.stream_idle_timeout_seconds,
            label="chat-stream",
        ):
            if mode == "values":
                # The full state after a node finished; the last one
                # is the final state for the answer assembly below.
                final_state = payload
                continue
            chunk = payload.get("token", "")
            parts.append(chunk)
            visible = think_filter.feed(chunk)
            if not visible:
                continue
            if not abstention_detected and len(head_text) < HEAD_CHECK_LEN:
                head_buf.append(visible)
                head_text += visible
                if len(head_text) >= HEAD_CHECK_LEN or "</think>" in "".join(parts):
                    # Enough to judge — check for abstention.
                    if settings.hybrid_fallback and is_abstention(head_text):
                        abstention_detected = True
                        # Don't yield the abstention; fall through to fallback below.
                        head_buf = []
                        head_text = ""
                        break
                    else:
                        # Not an abstention — flush the buffer and continue streaming.
                        for b in head_buf:
                            yield {"token": b}
                        head_buf = []
            else:
                yield {"token": visible}
    except OllamaTimeoutError:
        # A stalled stream must fail fast, not burn a second full
        # generation on the one-shot fallback.
        raise
    except Exception as e:
        # Streaming not supported by this LLM, or the graph run failed
        # — fall back to one shot. Counted once here; if the one-shot
        # retry also fails, it is counted again and the error
        # propagates (same as before).
        record_llm_error(e)
        log_error_event(
            request_id=request_id,
            endpoint=endpoint,
            where="stream",
            error_type=type(e).__name__,
            error=str(e),
            model=model,
        )
        try:
            fallback = _run_graph_question(
                question,
                settings,
                llm,
                history,
                user_groups,
                retriever,
                document_id,
                timings,
                capture,
                ctx,
            )
            text = fallback.get("text") or ""
            final_state = fallback
        except Exception as e2:
            record_llm_error(e2)
            log_error_event(
                request_id=request_id,
                endpoint=endpoint,
                where="generate",
                error_type=type(e2).__name__,
                error=str(e2),
                model=model,
            )
            raise
        visible = strip_think(text)
        yield {"token": visible}
        parts = [text]
    finally:
        # Release the model call as soon as the stream ends, errors, or is
        # abandoned. Abandonment is the disconnect path: the API closes
        # this generator, GeneratorExit lands at the yield above, and
        # closing the stream sets the cancel event (via
        # _CancellableStream) before closing the graph iterator, so the
        # generate node stops pulling further chunks instead of running
        # to completion. The set() here is belt-and-braces for paths
        # that never went through the stream close.
        cancel_event.set()
        if stream_iter is not None:
            close = getattr(stream_iter, "close", None)
            if close is not None:
                try:
                    close()
                except ValueError:
                    # The graph stream is mid-pull in the idle-timeout
                    # worker; that pull was the last one — nothing is
                    # ever pulled again — so the worker is simply
                    # abandoned with the executor (same as chain).
                    pass
    if final_state.get("small_talk"):
        # Small talk bypasses retrieval entirely, as in
        # chain._stream_ask_impl: the whole text goes out as one token.
        text = final_state.get("text") or ""
        _ms = (time.perf_counter() - started) * 1000
        small_talk_timings: dict = {
            "rewrite_ms": None,
            "episodic_ms": None,
            "retrieve_ms": None,
            "retrieve_calls": 0,
            "generate_ms": _ms,
            "total_ms": _ms,
        }
        record_question(small_talk_timings, streamed=True)
        log_question_event(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            question=question,
            timings=small_talk_timings,
            hits=0,
            sources=[],
            grounded=True,
            weak_evidence=False,
            standalone=False,
            streamed=True,
            cache_reload=False,
        )
        yield {"token": text}
        yield {
            "answer": Answer(
                text=text,
                sources=[],
                hits=[],
                latency_ms=0.0,
                grounded=True,
                standalone_question=None,
            )
        }
        return
    text = strip_think("".join(parts))
    text = strip_think(text)  # belt-and-braces: no reasoning in stored answers
    standalone = final_state.get("standalone") or question
    # Server-side hybrid fallback: if abstention was detected mid-stream (or
    # in the one-shot path), stream a general-knowledge answer instead.
    if settings.hybrid_fallback and (abstention_detected or is_abstention(text)):
        fb_text = general_knowledge_answer(standalone, llm, settings)
        for i in range(0, len(fb_text), 50):
            yield {"token": fb_text[i : i + 50]}
        text = fb_text
    else:
        # Yield buffered head first (in order), then the filter's tail,
        # so tokens == answer.text (test invariant). The tail is already
        # included in `text` via parts, so don't append it again.
        if head_buf:
            for b in head_buf:
                yield {"token": b}
        flushed = think_filter.flush()
        if flushed:
            yield {"token": flushed}
    latency_ms = (time.perf_counter() - started) * 1000
    timings["generate_ms"] = max(
        latency_ms - (timings.get("retrieve_ms") or 0.0), 0.0
    )
    timings["total_ms"] = latency_ms
    # Citations reuse the hits generation already retrieved (see
    # _generation_hits): no second search. Traced as its own span so the
    # reuse is visible in the trace.
    with tracing.span("retrieve-hits"):
        hits = _generation_hits(
            capture,
            ctx.get("retriever") or retriever,
            standalone,
            ctx.get("groups"),
            ctx.get("source_filter"),
        )
    answer = Answer(
        text=text,
        sources=sorted({h.source for h in hits}),
        hits=hits,
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(text),
        standalone_question=standalone if standalone != question else None,
        weak_evidence=_weak_evidence(ctx.get("retriever") or retriever),
    )
    log_question(settings, question, answer, latency_ms)
    record_question(timings, streamed=True)
    log_question_event(
        request_id=request_id,
        endpoint=endpoint,
        model=model,
        question=question,
        timings=timings,
        hits=len(hits),
        sources=answer.sources,
        grounded=answer.grounded,
        weak_evidence=answer.weak_evidence,
        standalone=answer.standalone_question is not None,
        streamed=True,
        cache_reload=cache_reload,
    )
    # Save to episodic memory (skip ungrounded fallbacks). Same cached
    # instance as the search in the node.
    if answer.grounded:
        try:
            get_cached_episodic_memory(settings).add(question, text)
        except Exception:
            pass
    yield {"answer": answer}
