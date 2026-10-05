"""RAG chain: retrieve -> prompt -> LLM, with source citations."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import Settings, get_llm, get_settings
from .retriever import Hit, Retriever

SYSTEM = """You answer questions using ONLY the context below.
If the context doesn't contain the answer, say you don't know. Cite the source
for each fact you use, like [api-reference.md]."""


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


def ask(question: str, settings: Settings | None = None) -> Answer:
    settings = settings or get_settings()
    chain, retriever = build_chain(settings)
    text = chain.invoke(question)
    hits = retriever.search(question)
    sources = sorted({h.source for h in hits})
    return Answer(text=text, sources=sources, hits=hits)
