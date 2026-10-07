"""Cross-encoder reranking (opt-in via RAG_DEMO_RERANK=1).

The sentence_transformers import is lazy and the model downloads on first
use — this module is never touched under RAG_DEMO_FAKE=1, so CI stays fast
and torch-free.
"""

from __future__ import annotations

from .retriever import Hit


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(scores, hits), key=lambda pair: pair[0], reverse=True)
        return [
            Hit(text=h.text, source=h.source, score=float(s))
            for s, h in ranked[:top_k]
        ]
