"""FAISS-backed retrieval over the indexed corpus."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings, get_embeddings, get_settings


@dataclass
class Hit:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.index_dir.exists():
            raise RuntimeError(
                f"index not found at {self.settings.index_dir} — run `python -m rag_demo.ingest` first"
            )
        self.store = FAISS.load_local(
            str(self.settings.index_dir),
            get_embeddings(self.settings),
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.settings.top_k
        results: list[tuple[Document, float]] = self.store.similarity_search_with_score(query, k=k)
        return [
            Hit(text=d.page_content, source=d.metadata.get("source", "?"), score=float(s))
            for d, s in results
        ]

    def as_langchain_retriever(self, k: int | None = None):
        return self.store.as_retriever(search_kwargs={"k": k or self.settings.top_k})
