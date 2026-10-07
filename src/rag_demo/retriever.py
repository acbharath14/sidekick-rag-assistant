"""FAISS-backed retrieval over the indexed corpus.

Dense mode (default): pure vector similarity.
Hybrid mode (RAG_DEMO_RETRIEVAL=hybrid): BM25 + dense with reciprocal rank
fusion over the top candidates of each side. Reranking (RAG_DEMO_RERANK=1)
applies a cross-encoder to the fused candidates — opt-in, downloads a small
model on first use (never on CI).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings, get_embeddings, get_settings

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - rank-bm25 is a hard dependency
    BM25Okapi = None

def _tokenize(text: str) -> list[str]:
    """Word tokens: strips punctuation so `weight_kg` matches weight_kg."""
    return re.findall(r"\w+", text.lower())


RRF_K = 60
FUSION_CANDIDATES = 20


@dataclass
class Hit:
    text: str
    source: str
    score: float


def rrf_fuse(rankings: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Reciprocal rank fusion over ranked hit lists.

    Hits are matched by (text, source); fused score = sum over rankings of
    1/(rrf_k + rank). Only rank order matters, so dense L2 distances and BM25
    scores are never mixed directly.
    """
    fused: dict[tuple[str, str], list] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.text, hit.source)
            entry = fused.get(key)
            if entry is None:
                entry = [hit, 0.0]
                fused[key] = entry
            entry[1] += 1.0 / (rrf_k + rank)
    scored = sorted(fused.values(), key=lambda e: e[1], reverse=True)
    return [Hit(text=h.text, source=h.source, score=s) for h, s in scored[:k]]


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
        self.chunks = self._load_chunks()
        self.bm25 = None
        if self.settings.retrieval == "hybrid":
            if BM25Okapi is None:
                raise RuntimeError(
                    "hybrid retrieval needs the rank-bm25 package — pip install -r requirements.txt"
                )
            if not self.chunks:
                raise RuntimeError(
                    f"hybrid retrieval needs {self.settings.index_dir}/chunks.json — "
                    "rebuild the index with the current ingest"
                )
            self.bm25 = BM25Okapi([_tokenize(c['text']) for c in self.chunks])
        self._reranker = None

    def _load_chunks(self) -> list[dict]:
        p = self.settings.index_dir / "chunks.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def _dense_search(self, query: str, k: int) -> list[Hit]:
        results: list[tuple[Document, float]] = self.store.similarity_search_with_score(query, k=k)
        return [
            Hit(text=d.page_content, source=d.metadata.get("source", "?"), score=float(s))
            for d, s in results
        ]

    def _bm25_search(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(
                text=self.chunks[i]["text"],
                source=self.chunks[i]["source"],
                score=float(scores[i]),
            )
            for i in top
        ]

    def _maybe_rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if self._reranker is None:
            # Lazy: sentence_transformers (and torch) are never imported on CI.
            from .rerank import Reranker

            self._reranker = Reranker()
        return self._reranker.rerank(query, hits, k)

    def search(
        self, query: str, k: int | None = None, rerank: bool | None = None
    ) -> list[Hit]:
        k = k or self.settings.top_k
        rerank = self.settings.rerank_enabled if rerank is None else rerank
        wide = rerank or self.settings.retrieval == "hybrid"
        want = FUSION_CANDIDATES if wide else k
        if self.settings.retrieval == "hybrid" and self.bm25 is not None:
            dense = self._dense_search(query, k=FUSION_CANDIDATES)
            bm25 = self._bm25_search(query, k=FUSION_CANDIDATES)
            hits = rrf_fuse([dense, bm25], k=want)
        else:
            hits = self._dense_search(query, k=want)
        if rerank:
            hits = self._maybe_rerank(query, hits, k)
        return hits[:k]

    def as_langchain_retriever(self, k: int | None = None):
        return self.store.as_retriever(search_kwargs={"k": k or self.settings.top_k})
