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
    out = llm.invoke(prompt)
    text = out.content if hasattr(out, "content") else str(out)
    return strip_think(text).strip()


def general_knowledge_answer(question: str, llm) -> str:
    """Direct general-knowledge answer, bypassing retrieval."""
    prompt = (
        "Answer the following question from your general knowledge. "
        "Be concise and accurate.\n\nQuestion: " + question
    )
    out = llm.invoke(prompt)
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
    # Small talk bypasses retrieval entirely — no need to search the corpus
    # for "hi".
    if is_small_talk(question):
        text = small_talk_answer(question, llm)
        return Answer(
            text=text,
            sources=[],
            hits=[],
            latency_ms=0.0,
            grounded=True,
            standalone_question=None,
        )
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
    # Server-side hybrid fallback: if the model abstained but hybrid is on,
    # answer from general knowledge instead of showing "No relevant answer".
    if settings.hybrid_fallback and is_abstention(text):
        text = general_knowledge_answer(standalone, llm)
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
    # Small talk bypasses retrieval entirely.
    if is_small_talk(question):
        text = small_talk_answer(question, llm)
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
    # Buffer the head of the stream to detect abstentions before displaying.
    # If the model declines to answer, we suppress the abstention and stream
    # a general-knowledge fallback instead (when hybrid is on).
    head_buf: list[str] = []
    head_text = ""
    abstention_detected = False
    HEAD_CHECK_LEN = 200
    try:
        for chunk in chain.stream(standalone):
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
    except Exception:
        # Streaming not supported by this LLM — fall back to one shot.
        text = chain.invoke(standalone)
        visible = strip_think(text)
        yield {"token": visible}
        parts = [text]
    text = strip_think("".join(parts)) + think_filter.flush()
    text = strip_think(text)  # belt-and-braces: no reasoning in stored answers
    # Server-side hybrid fallback: if abstention was detected mid-stream (or
    # in the one-shot path), stream a general-knowledge answer instead.
    if settings.hybrid_fallback and (abstention_detected or is_abstention(text)):
        fb_text = general_knowledge_answer(standalone, llm)
        for i in range(0, len(fb_text), 50):
            yield {"token": fb_text[i : i + 50]}
        text = fb_text
    elif head_buf:
        # Flush any buffered head that wasn't yielded (non-abstention case
        # where stream ended before reaching HEAD_CHECK_LEN).
        for b in head_buf:
            yield {"token": b}
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
