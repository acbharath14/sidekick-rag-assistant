"""RAG chain: retrieve -> prompt -> LLM, with source citations."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import Settings, get_llm, get_settings
from .retriever import Hit, Retriever
from .rewrite import rewrite_query
from .audit import log_question

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
    everything from <think> onward.
    """
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed think block — drop the trailing reasoning.
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
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


@dataclass
class Answer:
    text: str
    sources: list[str]
    hits: list[Hit]
    latency_ms: float | None = None
    grounded: bool = True  # False when answered from general knowledge (hybrid)
    standalone_question: str | None = None  # rewritten follow-up, if any


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.source}]\n{h.text}" for h in hits)


def build_chain(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    user_groups: list[str] | None = None,
):
    settings = settings or get_settings()
    retriever = retriever or Retriever(settings)
    # Resolve groups once: the SAME filtered hits feed the prompt AND the
    # returned answer. Never let the LLM see what the user may not.
    groups = user_groups if user_groups is not None else settings.user_groups
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
    retrieve = RunnableLambda(lambda q: retriever.search(q, user_groups=groups))
    chain = (
        {
            "context": retrieve | RunnableLambda(format_hits),
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
) -> Answer:
    """history: list of (question, answer) tuples for multi-turn context.
    user_groups: asker's groups for permission-aware retrieval (defaults to
    settings.user_groups). retriever: inject a pre-built retriever (e.g.
    an in-memory upload index)."""
    import time

    settings = settings or get_settings()
    history = history or []
    llm = get_llm(settings)
    standalone = (
        rewrite_query(question, history, llm)
        if settings.rewrite_enabled and history
        else question
    )
    started = time.perf_counter()
    chain, retriever, groups = build_chain(
        settings, retriever=retriever, user_groups=user_groups
    )
    text = chain.invoke(standalone)
    text = strip_think(text)
    latency_ms = (time.perf_counter() - started) * 1000
    hits = retriever.search(standalone, user_groups=groups)
    sources = sorted({h.source for h in hits})
    answer = Answer(
        text=text,
        sources=sources,
        hits=hits,
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(text),
        standalone_question=standalone if standalone != question else None,
    )
    log_question(settings, question, answer, latency_ms)
    return answer


def stream_ask(
    question: str,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_groups: list[str] | None = None,
    retriever: Retriever | None = None,
):
    """Streaming variant of ask(). Yields {"token": str} chunks as the LLM
    generates, then a final {"answer": Answer}. Powers st.write_stream."""
    import time

    settings = settings or get_settings()
    history = history or []
    llm = get_llm(settings)
    standalone = (
        rewrite_query(question, history, llm)
        if settings.rewrite_enabled and history
        else question
    )
    started = time.perf_counter()
    chain, retriever, groups = build_chain(
        settings, retriever=retriever, user_groups=user_groups
    )
    parts: list[str] = []
    think_filter = ThinkBlockFilter()
    try:
        for chunk in chain.stream(standalone):
            parts.append(chunk)
            visible = think_filter.feed(chunk)
            if visible:
                yield {"token": visible}
    except Exception:
        # Streaming not supported by this LLM — fall back to one shot.
        text = chain.invoke(standalone)
        visible = strip_think(text)
        yield {"token": visible}
        parts = [text]
    text = strip_think("".join(parts)) + think_filter.flush()
    text = strip_think(text)  # belt-and-braces: no reasoning in stored answers
    latency_ms = (time.perf_counter() - started) * 1000
    hits = retriever.search(standalone, user_groups=groups)
    answer = Answer(
        text=text,
        sources=sorted({h.source for h in hits}),
        hits=hits,
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(text),
        standalone_question=standalone if standalone != question else None,
    )
    log_question(settings, question, answer, latency_ms)
    yield {"answer": answer}
