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


@dataclass
class Answer:
    text: str
    sources: list[str]
    hits: list[Hit]


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.source}]\n{h.text}" for h in hits)


def build_chain(settings: Settings | None = None, retriever: Retriever | None = None):
    settings = settings or get_settings()
    retriever = retriever or Retriever(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with citations.",
            ),
        ]
    )
    retrieve = RunnableLambda(lambda q: retriever.search(q))
    chain = (
        {
            "context": retrieve | RunnableLambda(format_hits),
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm(settings)
        | StrOutputParser()
    )
    return chain, retriever


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
    chain, retriever = build_chain(settings, retriever=retriever)
    text = chain.invoke(standalone)
    hits = retriever.search(standalone, user_groups=user_groups)
    sources = sorted({h.source for h in hits})
    answer = Answer(text=text, sources=sources, hits=hits)
    log_question(settings, question, answer, (time.perf_counter() - started) * 1000)
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
    chain, retriever = build_chain(settings, retriever=retriever)
    parts: list[str] = []
    try:
        for chunk in chain.stream(standalone):
            parts.append(chunk)
            yield {"token": chunk}
    except Exception:
        # Streaming not supported by this LLM — fall back to one shot.
        text = chain.invoke(standalone)
        yield {"token": text}
        parts = [text]
    text = "".join(parts)
    hits = retriever.search(standalone, user_groups=user_groups)
    answer = Answer(text=text, sources=sorted({h.source for h in hits}), hits=hits)
    log_question(settings, question, answer, (time.perf_counter() - started) * 1000)
    yield {"answer": answer}
