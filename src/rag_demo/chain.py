"""RAG chain: retrieve -> prompt -> LLM, with source citations."""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from . import caching
from . import tracing
from .audit import log_question
from .caching import get_cached_episodic_memory, get_cached_retriever
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
from .retriever import Hit, Retriever, format_citation
from .rewrite import rewrite_query

SYSTEM = """You answer questions using ONLY the context below.
For greetings or remarks that aren't questions (e.g. "hi", "thanks", "that's
wonderful"), respond briefly and naturally without citations.
For factual questions: if the context doesn't contain the answer, say you
don't know. Cite the source for each fact you use, like [api-reference.md]."""

SYSTEM_HYBRID = """You answer questions using the context below when it contains the answer.
Cite the source for each fact you use, like [api-reference.md].
For greetings or remarks that aren't questions (e.g. "hi", "thanks", "that's
wonderful"), respond briefly and naturally without citations.
If the context doesn't contain the answer to a factual question, answer from
your general knowledge — but start your response with exactly this line:
🌐 General knowledge (not from your corpus)."""

# Markers the hybrid prompt instructs the LLM to emit; detected in code so
# every surface (UI, CLI, MCP) can label fallback answers reliably.
FALLBACK_MARKERS = ("🌐", "general knowledge")


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (qwen3 et al.).

    Handles unclosed blocks (model cut off mid-reasoning) by dropping
    everything from <think> onward. Also handles orphaned </think>
    (closing tag without opening tag) by dropping everything before it.
    """
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed think block — drop the trailing reasoning.
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # Orphaned closing tag (no opening tag before it) — drop everything
    # up to and including it. Handles malformed model output.
    if "</think>" in text and "<think>" not in text.split("</think>")[0]:
        text = text.split("</think>", 1)[-1]
    # Clean any remaining stray tags.
    text = text.replace("</think>", "").replace("<think>", "")
    return text.strip()


class ThinkBlockFilter:
    """Stateful filter stripping <think>...</think> from a token stream.

    Tags may be split across chunks; suppressed content is never yielded.
    """

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        """Feed one token; returns the visible text to yield (may be '')."""
        self._buf += token
        out = []
        while True:
            if not self._in_think:
                start = self._buf.find("<think>")
                # Handle orphaned </think> (no opening tag): drop everything
                # up to and including it. This handles malformed model output
                # where reasoning leaks without the opening tag.
                orphan_end = self._buf.find("</think>")
                if start == -1 and orphan_end != -1:
                    self._buf = self._buf[orphan_end + 8 :]
                    continue
                if start == -1:
                    # Keep a tail in case the tag is split across chunks.
                    if len(self._buf) > 7:
                        out.append(self._buf[:-7])
                        self._buf = self._buf[-7:]
                    break
                if start > 0:
                    out.append(self._buf[:start])
                self._buf = self._buf[start + 7 :]
                self._in_think = True
            else:
                end = self._buf.find("</think>")
                if end == -1:
                    if len(self._buf) > 8:
                        self._buf = self._buf[-8:]
                    break
                self._buf = self._buf[end + 8 :]
                self._in_think = False
        return "".join(out)

    def flush(self) -> str:
        """Return any buffered visible text (call at stream end)."""
        if self._in_think:
            return ""
        buf, self._buf = self._buf, ""
        return buf


def is_fallback_answer(text: str) -> bool:
    head = strip_think(text).lstrip()[:120].lower()
    return head.startswith("🌐") or "general knowledge" in head


ABSTENTION_PHRASES = (
    "no relevant answer",
    "don't know",
    "do not know",
    "not in the context",
    "not in context",
    "cannot answer",
    "can't answer",
    "no information",
)


def is_abstention(text: str) -> bool:
    """Detect when the model declined to answer from context."""
    head = strip_think(text).lstrip()[:200].lower()
    return any(p in head for p in ABSTENTION_PHRASES)


SMALL_TALK_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "how are you",
)


def is_small_talk(question: str) -> bool:
    """Detect greetings/remarks that don't need retrieval."""
    q = question.strip().lower().rstrip("!.,?")
    return q in SMALL_TALK_PATTERNS


def small_talk_answer(question: str, llm) -> str:
    """Brief friendly response without retrieval or citations."""
    prompt = (
        "The user said: \"" + question + "\"\n"
        "Respond briefly and warmly (1-2 sentences). You are Sidekick, "
        "a helpful RAG assistant. Do not mention sources or citations."
    )
    out = tracing.invoke_llm(llm, prompt, run_name="small-talk")
    text = out.content if hasattr(out, "content") else str(out)
    return strip_think(text).strip()


def general_knowledge_answer(question: str, llm) -> str:
    """Direct general-knowledge answer, bypassing retrieval."""
    prompt = (
        "Answer the following question from your general knowledge. "
        "Be concise and accurate.\n\nQuestion: " + question
    )
    out = tracing.invoke_llm(llm, prompt, run_name="general-knowledge")
    text = out.content if hasattr(out, "content") else str(out)
    text = strip_think(text)
    return "🌐 General knowledge (not from your corpus).\n" + text.strip()


@dataclass
class Answer:
    text: str
    sources: list[str]
    hits: list[Hit]
    latency_ms: float | None = None
    grounded: bool = True  # False when answered from general knowledge (hybrid)
    standalone_question: str | None = None  # rewritten follow-up, if any
    weak_evidence: bool = False  # True when the relevance floor dropped every candidate


def _weak_evidence(retriever) -> bool:
    """True when the relevance floor dropped every retrieved candidate —
    the answer was produced with no surviving evidence."""
    candidates = getattr(retriever, "last_candidates", 0) or 0
    dropped = getattr(retriever, "last_floor_dropped", 0) or 0
    return candidates > 0 and dropped >= candidates


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{format_citation(h)}]\n{h.text}" for h in hits)


def format_episodic(episodes: list[dict]) -> str:
    """Format episodic memory hits as supplementary context, clearly labeled
    as prior conversations (not current-turn history)."""
    if not episodes:
        return ""
    lines = ["Prior conversations (for background only — the current question takes precedence):"]
    for e in episodes:
        lines.append(f"Q: {e['question']}\nA: {e['answer']}")
    return "\n\n".join(lines)


def _model_name(settings: Settings) -> str:
    """Display model for logs (same rule as audit.log_question)."""
    return (
        settings.ollama_model
        if settings.llm_provider == "ollama"
        else settings.llm_provider
    )

def _generation_hits(
    capture: dict,
    retriever: Retriever,
    standalone: str,
    groups: list[str] | None,
    source_filter: str | None,
) -> list[Hit]:
    """Hits for the answer: the ones generation already retrieved.

    The chain retrieves once to build the prompt; those hits are the
    citations. Only if generation somehow ran without retrieving (a
    future chain variant, an error path) do we run one search here.
    Same query, groups and filter as generation, so the result is
    identical to the old second search.
    """
    hits = capture.get("hits")
    if hits is None:
        hits = retriever.search(
            standalone, user_groups=groups, source_filter=source_filter
        )
    return hits


def build_chain(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    user_groups: list[str] | None = None,
    episodic_context: str = "",
    document_id: str | None = None,
    timings: dict | None = None,
    capture: dict | None = None,
):
    settings = settings or get_settings()
    # Default path reuses the app-lifetime cached retriever (one FAISS
    # load, one BM25 build, one shared embeddings object, one lazy
    # reranker) instead of rebuilding per question. An explicitly passed
    # retriever (e.g. an in-memory upload index) still wins and is never
    # cached. Loading stays lazy: nothing is built at import time, the
    # first question pays the load. See caching.py.
    if retriever is None:
        # Hit-or-miss check around the cached call (PR #57 merge
        # recipe): snapshot the key first; a hit counts as a hit, a
        # miss builds and counts as a reload. An explicitly passed
        # retriever is neither and is not counted.
        _key = caching._retriever_key(settings)
        _was_cached = _key in caching._RETRIEVERS
        retriever = get_cached_retriever(settings)
        if _was_cached:
            record_cache_hit()
        else:
            record_cache_reload()
    # Resolve groups once: the SAME filtered hits feed the prompt AND the
    # returned answer. Never let the LLM see what the user may not.
    groups = user_groups if user_groups is not None else settings.user_groups
    # Document-scoped chat: only retrieve chunks from this document's source.
    source_filter = f"upload:{document_id}" if document_id else None
    system = SYSTEM_HYBRID if settings.hybrid_fallback else SYSTEM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with citations.",
            ),
        ]
    )
    def _retrieve(q: str) -> list[Hit]:
        # Time the retrieval inside the chain so ask()/stream_ask()
        # can report retrieve_ms separately from generate_ms. Rerank
        # time stays inside retrieve_ms (see observability.py).
        # The one retrieval per question: store the hits in `capture` so
        # the answer reuses them for citations instead of searching again.
        _t0 = time.perf_counter()
        try:
            hits = retriever.search(
                q, user_groups=groups, source_filter=source_filter
            )
            if capture is not None:
                capture["hits"] = hits
            return hits
        finally:
            if timings is not None:
                timings["retrieve_ms"] = (
                    timings.get("retrieve_ms") or 0.0
                ) + (time.perf_counter() - _t0) * 1000
                timings["retrieve_calls"] = timings.get("retrieve_calls", 0) + 1

    # Named so the step shows as "retrieve" in a Langfuse trace.
    retrieve = RunnableLambda(_retrieve).with_config({"run_name": "retrieve"})

    def _format_context(hits: list[Hit]) -> str:
        parts = []
        if episodic_context:
            parts.append(episodic_context)
        parts.append(format_hits(hits))
        return "\n\n".join(p for p in parts if p)

    chain = (
        {
            "context": retrieve
            | RunnableLambda(_format_context).with_config(
                {"run_name": "format-context"}
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm(settings)
        | StrOutputParser()
    )
    return chain, retriever, groups


def ask(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
) -> Answer:
    """history: list of (question, answer) tuples for multi-turn context.
    user_groups: asker's groups for permission-aware retrieval (defaults to
    settings.user_groups). retriever: inject a pre-built retriever (e.g.
    an in-memory upload index). document_id: scope retrieval to a single
    uploaded document's chunks. request_id/endpoint: carried into the
    structured log line (api.py supplies them; direct callers get a
    generated id and endpoint "direct").

    Opens one Langfuse trace per question (a no-op when tracing is off).
    """
    settings = settings or get_settings()
    metadata = {"model": settings.ollama_model}
    if document_id:
        metadata["document_id"] = document_id
    with tracing.question_span(question, metadata=metadata):
        return _ask_impl(
            question,
            settings=settings,
            history=history,
            user_groups=user_groups,
            retriever=retriever,
            document_id=document_id,
            request_id=request_id,
            endpoint=endpoint,
        )


def _ask_impl(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
) -> Answer:
    import time
    settings = settings or get_settings()
    history = history or []
    request_id = request_id or new_request_id()
    model = _model_name(settings)
    llm = get_llm(settings)
    # Small talk bypasses retrieval entirely — no need to search the corpus
    # for "hi".
    if is_small_talk(question):
        _t0 = time.perf_counter()
        try:
            text = small_talk_answer(question, llm)
        except Exception as e:
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
        _ms = (time.perf_counter() - _t0) * 1000
        timings: dict = {
            "rewrite_ms": None,
            "episodic_ms": None,
            "retrieve_ms": None,
            "retrieve_calls": 0,
            "generate_ms": _ms,
            "total_ms": _ms,
        }
        record_question(timings, streamed=False)
        log_question_event(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            question=question,
            timings=timings,
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
    # Episodic memory: retrieve relevant past conversations as SUPPLEMENTARY
    # context for the answer. It is deliberately NOT merged into `history`:
    # history feeds query rewriting, and prior-session Q&As are not the
    # current conversation — letting them drive the rewrite lets stale
    # context override the user's actual question. The cached instance is
    # shared with the save below, so episodic memory is built once per
    # question, not twice.
    timings = {
        "rewrite_ms": None,
        "episodic_ms": None,
        "retrieve_ms": None,
        "retrieve_calls": 0,
        "generate_ms": None,
        "total_ms": None,
    }
    episodic_context = ""
    _t0 = time.perf_counter()
    try:
        _episodic = get_cached_episodic_memory(settings)
        _episodes = _episodic.search(question, k=3)
        if _episodes:
            episodic_context = format_episodic(_episodes)
    except Exception:
        pass
    finally:
        timings["episodic_ms"] = (time.perf_counter() - _t0) * 1000
    _t0 = time.perf_counter()
    standalone = (
        rewrite_query(question, history, llm)
        if settings.rewrite_enabled and history
        else question
    )
    if settings.rewrite_enabled and history:
        timings["rewrite_ms"] = (time.perf_counter() - _t0) * 1000
    started = time.perf_counter()
    source_filter = f"upload:{document_id}" if document_id else None
    # Whether this question built a fresh retriever (cache reload) or
    # reused the cached one (hit). Snapshot the key before build_chain
    # rebinds the name; an explicitly passed retriever is neither.
    cache_reload = retriever is None and (
        caching._retriever_key(settings) not in caching._RETRIEVERS
    )
    # `capture` collects the hits generation retrieves, so the answer
    # reuses them for citations instead of searching a second time.
    capture: dict = {}
    chain, retriever, groups = build_chain(
        settings,
        retriever=retriever,
        user_groups=user_groups,
        episodic_context=episodic_context,
        document_id=document_id,
        timings=timings,
        capture=capture,
    )
    try:
        text = chain.invoke(standalone, config=tracing.run_config(run_name="generate"))
        text = strip_think(text)
        # Server-side hybrid fallback: if the model abstained but hybrid is on,
        # answer from general knowledge instead of showing "No relevant answer".
        if settings.hybrid_fallback and is_abstention(text):
            text = general_knowledge_answer(standalone, llm)
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
    # generate_ms is chain time minus retrieval, so it includes prompt
    # formatting and retriever construction: an approximation, not a
    # pure LLM timer (see observability.log_question_event).
    timings["generate_ms"] = max(
        latency_ms - (timings.get("retrieve_ms") or 0.0), 0.0
    )
    timings["total_ms"] = latency_ms
    # Citations reuse the hits generation already retrieved (see
    # _generation_hits): no second search. Traced as its own span so the
    # reuse is visible in the trace.
    with tracing.span("retrieve-hits"):
        hits = _generation_hits(capture, retriever, standalone, groups, source_filter)
    sources = sorted({h.source for h in hits})
    answer = Answer(
        text=text,
        sources=sources,
        hits=hits,
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(text),
        standalone_question=standalone if standalone != question else None,
        weak_evidence=_weak_evidence(retriever),
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
    # Save to episodic memory for long-term recall (skip if it was a fallback
    # to avoid polluting memory with "I don't know" responses). Same
    # cached instance as the search above.
    if answer.grounded:
        try:
            get_cached_episodic_memory(settings).add(question, text)
        except Exception:
            pass
    return answer


def stream_ask(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
):
    """Streaming variant of ask(). Yields {"token": str} chunks as the LLM
    generates, then a final {"answer": Answer}. Powers st.write_stream.
    request_id/endpoint are carried into the structured log line (see ask).

    Opens one Langfuse trace per question (a no-op when tracing is off).
    The trace stays open until the stream is exhausted or the client
    disconnects.
    """
    settings = settings or get_settings()
    metadata = {"model": settings.ollama_model}
    if document_id:
        metadata["document_id"] = document_id
    with tracing.question_span(question, metadata=metadata):
        yield from _stream_ask_impl(
            question,
            settings=settings,
            history=history,
            user_groups=user_groups,
            retriever=retriever,
            document_id=document_id,
            request_id=request_id,
            endpoint=endpoint,
        )


def _stream_ask_impl(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
    document_id: str | None = None,
    request_id: str | None = None,
    endpoint: str | None = None,
):
    import time
    settings = settings or get_settings()
    history = history or []
    request_id = request_id or new_request_id()
    model = _model_name(settings)
    llm = get_llm(settings)
    # Small talk bypasses retrieval entirely.
    if is_small_talk(question):
        _t0 = time.perf_counter()
        try:
            text = small_talk_answer(question, llm)
        except Exception as e:
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
        _ms = (time.perf_counter() - _t0) * 1000
        timings: dict = {
            "rewrite_ms": None,
            "episodic_ms": None,
            "retrieve_ms": None,
            "retrieve_calls": 0,
            "generate_ms": _ms,
            "total_ms": _ms,
        }
        record_question(timings, streamed=True)
        log_question_event(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            question=question,
            timings=timings,
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
    # Episodic memory: supplementary context only — never merged into
    # history, so it can't hijack the query rewrite (see ask()).
    # Cached instance, shared with the save below: built once.
    timings = {
        "rewrite_ms": None,
        "episodic_ms": None,
        "retrieve_ms": None,
        "retrieve_calls": 0,
        "generate_ms": None,
        "total_ms": None,
    }
    episodic_context = ""
    _t0 = time.perf_counter()
    try:
        _episodic = get_cached_episodic_memory(settings)
        _episodes = _episodic.search(question, k=3)
        if _episodes:
            episodic_context = format_episodic(_episodes)
    except Exception:
        pass
    finally:
        timings["episodic_ms"] = (time.perf_counter() - _t0) * 1000
    _t0 = time.perf_counter()
    standalone = (
        rewrite_query(question, history, llm)
        if settings.rewrite_enabled and history
        else question
    )
    if settings.rewrite_enabled and history:
        timings["rewrite_ms"] = (time.perf_counter() - _t0) * 1000
    started = time.perf_counter()
    source_filter = f"upload:{document_id}" if document_id else None
    # Whether this question built a fresh retriever (cache reload) or
    # reused the cached one (hit). Snapshot the key before build_chain
    # rebinds the name; an explicitly passed retriever is neither.
    cache_reload = retriever is None and (
        caching._retriever_key(settings) not in caching._RETRIEVERS
    )
    # `capture` collects the hits generation retrieves, so the answer
    # reuses them for citations instead of searching a second time.
    capture: dict = {}
    chain, retriever, groups = build_chain(
        settings,
        retriever=retriever,
        user_groups=user_groups,
        episodic_context=episodic_context,
        document_id=document_id,
        timings=timings,
        capture=capture,
    )
    parts: list[str] = []
    think_filter = ThinkBlockFilter()
    # Buffer the head of the stream to detect abstentions before displaying.
    # If the model declines to answer, we suppress the abstention and stream
    # a general-knowledge fallback instead (when hybrid is on).
    head_buf: list[str] = []
    head_text = ""
    abstention_detected = False
    HEAD_CHECK_LEN = 200
    try:
        for chunk in chain.stream(standalone, config=tracing.run_config(run_name="generate")):
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
    except Exception as e:
        # Streaming not supported by this LLM — fall back to one shot.
        # Counted once here; if the one-shot retry also fails, it is
        # counted again and the error propagates (same as before).
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
            text = chain.invoke(standalone, config=tracing.run_config(run_name="generate"))
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
    text = strip_think("".join(parts))
    text = strip_think(text)  # belt-and-braces: no reasoning in stored answers
    # Server-side hybrid fallback: if abstention was detected mid-stream (or
    # in the one-shot path), stream a general-knowledge answer instead.
    if settings.hybrid_fallback and (abstention_detected or is_abstention(text)):
        fb_text = general_knowledge_answer(standalone, llm)
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
        hits = _generation_hits(capture, retriever, standalone, groups, source_filter)
    answer = Answer(
        text=text,
        sources=sorted({h.source for h in hits}),
        hits=hits,
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(text),
        standalone_question=standalone if standalone != question else None,
        weak_evidence=_weak_evidence(retriever),
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
    # instance as the search above.
    if answer.grounded:
        try:
            get_cached_episodic_memory(settings).add(question, text)
        except Exception:
            pass
    yield {"answer": answer}
