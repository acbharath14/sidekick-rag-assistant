"""FAISS-backed retrieval over the indexed corpus.

Dense mode (RAG_DEMO_RETRIEVAL=dense): pure vector similarity.
Hybrid mode (default): BM25 + dense with reciprocal rank fusion over the
top candidates of each side. Reranking (on by default;
RAG_DEMO_RERANK=0 disables it) applies a cross-encoder to the fused
candidates — downloads a small model on first use (never on CI).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import INDEX_FORMAT_VERSION, Settings, get_embeddings, get_settings

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
    allowed_groups: tuple = ("*",)
    # Phase 5 (A2+A3) grounding fields. `source` stays the bare filename
    # (evals compare on it); `section` + `chunk_id` are the passage-level
    # identity. `dense_score` is the FAISS L2 distance when the hit came
    # from dense search (None for BM25-only hits). `confidence` is set by
    # apply_floor: a 0-1 absolute relevance signal, or None when no
    # absolute signal exists for the hit.
    chunk_id: str = ""
    section: str = ""
    dense_score: float | None = None
    confidence: float | None = None


def format_citation(hit: Hit) -> str:
    """Passage-level citation: source + section, not filename alone."""
    return f"{hit.source} › {hit.section}" if hit.section else hit.source


def citation_source(citation: str) -> str:
    """Filename part of a citation (before the section separator)."""
    return citation.split(" › ", 1)[0]


def confidence_band(confidence: float | None) -> str:
    """Qualitative band for UI display; raw scores are debug-only."""
    if confidence is None:
        return "—"
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.5:
        return "Medium"
    return "Low"


def confidence_for(hit: Hit, score_kind: str) -> float | None:
    """Map a hit's final score to a 0-1 absolute confidence.

    score_kind is the stage that produced the final ranking:
    - "rerank": cross-encoder logit; sigmoid maps logit 0 -> 0.5.
    - "dense": FAISS L2 distance on (near-)normalized embeddings, where
      distance 0 -> 1.0 and distance 2 -> 0.0.
    - "rrf": deliberately unsupported. RRF scores are purely relative
      ranks within one query's candidate pool — a garbage query still
      produces ranked hits — so they carry no absolute relevance signal
      and must never feed a floor. Hybrid hits without reranking fall
      back to their dense distance when they have one (see apply_floor).
    """
    import math

    if score_kind == "rerank":
        return 1.0 / (1.0 + math.exp(-hit.score))
    if score_kind == "dense":
        if hit.dense_score is None:
            return None
        return max(0.0, min(1.0, 1.0 - hit.dense_score / 2.0))
    return None


def apply_floor(
    hits: list[Hit], floor: float, score_kind: str
) -> tuple[list[Hit], int]:
    """Drop hits below the relevance floor; return (kept, dropped_count).

    The floor needs an absolute signal, in priority order: the rerank
    logit when reranked, else the dense L2 distance when the hit has
    one. A hit with no absolute signal (e.g. a BM25-only hit when
    reranking is off) cannot be judged and is kept — dropping it on a
    meaningless number would be worse than keeping it. floor <= 0
    disables filtering but still stamps confidence for display.
    """
    kept: list[Hit] = []
    dropped = 0
    for hit in hits:
        if score_kind == "rerank":
            conf = confidence_for(hit, "rerank")
        elif hit.dense_score is not None:
            conf = confidence_for(hit, "dense")
        else:
            conf = None
        hit.confidence = conf
        if floor > 0 and conf is not None and conf < floor:
            dropped += 1
        else:
            kept.append(hit)
    return kept, dropped


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
            # Keep the dense distance for the floor: a fused hit that
            # came from dense search carries an absolute signal even
            # though its fused RRF score is purely relative. Replace,
            # don't mutate — the input rankings belong to the caller.
            if hit.dense_score is not None and entry[0].dense_score is None:
                from dataclasses import replace

                entry[0] = replace(entry[0], dense_score=hit.dense_score)
    scored = sorted(fused.values(), key=lambda e: e[1], reverse=True)
    return [
        Hit(
            text=h.text,
            source=h.source,
            score=s,
            allowed_groups=h.allowed_groups,
            chunk_id=h.chunk_id,
            section=h.section,
            dense_score=h.dense_score,
        )
        for h, s in scored[:k]
    ]


def _index_format_current(index_dir) -> bool:
    """True when the on-disk index carries Phase 5 chunk metadata."""
    from pathlib import Path

    marker = Path(index_dir) / "index_format.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("version") == INDEX_FORMAT_VERSION
    except (OSError, ValueError):
        return False


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        store=None,
        chunks: list[dict] | None = None,
        embeddings=None,
    ):
        """store/chunks: inject a pre-built index (e.g. an in-memory upload
        index) instead of loading from settings.index_dir.
        embeddings: inject a shared embeddings object (see caching.py);
        when omitted, one is built via get_embeddings (old behavior)."""
        self.settings = settings or get_settings()
        if store is not None:
            self.store = store
            self.chunks = chunks or []
        else:
            if not self.settings.index_dir.exists():
                raise RuntimeError(
                    f"index not found at {self.settings.index_dir} — run `python -m rag_demo.ingest` first"
                )
            self.store = FAISS.load_local(
                str(self.settings.index_dir),
                embeddings if embeddings is not None else get_embeddings(self.settings),
                allow_dangerous_deserialization=True,
            )
            self.chunks = self._load_chunks()
        # Phase 5 (A2+A3): an index built before chunk-level citations
        # lacks chunk_id/section metadata. It still answers, but citations
        # degrade to filename-only — so flag it instead of silently
        # dropping the new fields. Injected stores/chunks (uploads,
        # tests) carry no on-disk format and are never stale.
        self.index_stale = False
        if store is None:
            self.index_stale = not _index_format_current(self.settings.index_dir)
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
        # Phase 5 floor bookkeeping, updated by every search() call.
        self.last_candidates = 0
        self.last_floor_dropped = 0

    def _load_chunks(self) -> list[dict]:
        p = self.settings.index_dir / "chunks.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def _dense_search(self, query: str, k: int) -> list[Hit]:
        results: list[tuple[Document, float]] = self.store.similarity_search_with_score(query, k=k)
        return [
            Hit(
                text=d.page_content,
                source=d.metadata.get("source", "?"),
                score=float(s),
                allowed_groups=tuple(d.metadata.get("allowed_groups", ["*"])),
                chunk_id=d.metadata.get("chunk_id", ""),
                section=d.metadata.get("section", ""),
                dense_score=float(s),
            )
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
                allowed_groups=tuple(self.chunks[i].get("allowed_groups", ["*"])),
                chunk_id=self.chunks[i].get("chunk_id", ""),
                section=self.chunks[i].get("section", ""),
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
        self,
        query: str,
        k: int | None = None,
        rerank: bool | None = None,
        user_groups: list[str] | None = None,
    ) -> list[Hit]:
        """user_groups: groups the asker belongs to. ACL is default-deny:
        public ("*") hits are visible to everyone; restricted hits require an
        explicit group match. Filtering happens BEFORE the prompt — never rely
        on the LLM to withhold what it has seen."""
        k = k or self.settings.top_k
        rerank = self.settings.rerank_enabled if rerank is None else rerank
        if user_groups is None:
            user_groups = self.settings.user_groups
        groups = set(user_groups or [])
        # ACL filtering always runs (default-deny), so always pull a wide
        # candidate pool — trimming to k happens after filtering/reranking.
        want = FUSION_CANDIDATES
        if self.settings.retrieval == "hybrid" and self.bm25 is not None:
            dense = self._dense_search(query, k=FUSION_CANDIDATES)
            bm25 = self._bm25_search(query, k=FUSION_CANDIDATES)
            hits = rrf_fuse([dense, bm25], k=want)
        else:
            hits = self._dense_search(query, k=want)
        hits = [
            h
            for h in hits
            if "*" in h.allowed_groups or groups & set(h.allowed_groups)
        ]
        # Phase 5 (A2+A3) relevance floor. The floor runs on the full
        # candidate pool BEFORE trimming to k, so a passing hit beyond
        # the first k is not lost to a failing one. The score kind is
        # the stage that produced the final ranking: rerank logits when
        # reranked, else the dense distance carried on each hit (RRF
        # scores themselves are never floored — see confidence_for).
        # last_floor_dropped / last_candidates let callers (chain) tell
        # "nothing passed the floor" apart from "nothing was retrieved".
        score_kind = "dense"
        if rerank and hits:
            # Rank the whole pool, then floor, then trim — rerank() takes
            # top_k, so pass the pool size to keep every candidate.
            # Skipped on empty pools so no model is loaded for nothing.
            hits = self._maybe_rerank(query, hits, len(hits))
            score_kind = "rerank"
        pre_floor = len(hits)
        hits, dropped = apply_floor(hits, self.settings.relevance_floor, score_kind)
        self.last_candidates = pre_floor
        self.last_floor_dropped = dropped
        return hits[:k]

    def as_langchain_retriever(self, k: int | None = None):
        return self.store.as_retriever(search_kwargs={"k": k or self.settings.top_k})
