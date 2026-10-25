"""Pipeline walkthrough: chunk → embed → retrieve, on a specimen or a
pasted document.

Streamlit-free: app.py renders the stages; this module computes them using
the REAL pipeline code paths — ingest.chunk_documents for chunking,
config.get_embeddings for embedding, and retriever.Retriever (with an
in-memory FAISS store injected) for retrieval. Nothing here is a mockup of
the pipeline; it is the pipeline, run on a small document.

Deliberately NOT included: generation. The walkthrough stops at retrieval —
it shows what evidence a question would retrieve, not an answer. There is
also no "summarize this document" step: summarization is a separate
capability, not a stage of the Q&A pipeline, and adding one here would
smuggle a new feature into a visibility view.

Under RAG_DEMO_FAKE=1 the embedding stage uses FakeEmbeddings (384-dim
deterministic vectors): chunking and retrieval mechanics still run, but the
scores are not meaningful — callers must say so (see WalkthroughResult.fake).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings, get_embeddings
from .ingest import chunk_documents
from .retriever import Retriever, confidence_band, format_citation

# Labeled specimen: describes the real pipeline, so questions about
# chunking / embedding / retrieval retrieve the matching passage. This is
# demonstration text, not corpus content, and the UI must label it as such.
SPECIMEN_TITLE = "Specimen: how Sidekick answers a question"
SPECIMEN_TEXT = """# How Sidekick answers a question

## Chunking

Long documents are split into short passages of 500 characters with a
50-character overlap, so each passage is small enough to embed and cite on
its own.

## Embedding

Each passage is turned into a vector: a list of numbers that captures what
the passage means. Passages with similar meanings get vectors that sit
close together.

## Retrieval

When you ask a question, the question is embedded the same way, and the
passages whose vectors sit closest to the question's vector are retrieved
as evidence for the answer.
"""

#: The walkthrough's stages, in order. Guarded by test: a summarize or
#: generation step must never creep in here unnoticed.
STAGES = ("chunking", "embedding", "retrieval")


@dataclass
class ChunkRow:
    chunk_id: str
    section: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class EmbeddingInfo:
    provider: str
    model: str
    dim: int
    vector_count: int
    preview: list[float]  # first components of the first chunk's vector
    fake: bool


@dataclass
class RetrievalRow:
    rank: int
    chunk_id: str
    citation: str
    confidence: str
    score: float
    text: str


@dataclass
class WalkthroughResult:
    doc_name: str
    query: str
    chunks: list[ChunkRow] = field(default_factory=list)
    embedding: EmbeddingInfo | None = None
    retrieval_stages: list[str] = field(default_factory=list)
    hits: list[RetrievalRow] = field(default_factory=list)
    candidates: int = 0
    floor_dropped: int = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def chunk_stage(text: str, doc_name: str, settings: Settings) -> list[ChunkRow]:
    """Stage 1 — the real splitter, with chunk ids + sections stamped."""
    docs = [
        Document(
            page_content=text,
            metadata={"source": doc_name, "doc_id": doc_name},
        )
    ]
    return [
        ChunkRow(
            chunk_id=c.metadata.get("chunk_id", ""),
            section=c.metadata.get("section", ""),
            text=c.page_content,
        )
        for c in chunk_documents(docs, settings)
    ]


def retrieval_stages_for(settings: Settings) -> list[str]:
    """Which retrieval stages run under these settings, in order."""
    stages = ["BM25", "dense", "RRF fusion"] if settings.retrieval == "hybrid" else ["dense"]
    if settings.rerank_enabled:
        stages.append("rerank")
    if settings.relevance_floor > 0:
        stages.append("relevance floor")
    return stages


def run_walkthrough(
    text: str, doc_name: str, query: str, settings: Settings
) -> WalkthroughResult:
    """Run chunk → embed → retrieve on one document + one query."""
    result = WalkthroughResult(doc_name=doc_name, query=query)
    result.chunks = chunk_stage(text, doc_name, settings)
    if not result.chunks:
        return result

    # Stage 2 — the real embeddings object (fake under RAG_DEMO_FAKE=1).
    embeddings = get_embeddings(settings)
    vectors = embeddings.embed_documents([c.text for c in result.chunks])
    dim = len(vectors[0]) if vectors else 0
    result.embedding = EmbeddingInfo(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dim=dim,
        vector_count=len(vectors),
        preview=[round(float(v), 4) for v in vectors[0][:5]] if vectors else [],
        fake=settings.fake,
    )

    # Stage 3 — the real retriever over an in-memory index built from
    # exactly the chunks above (same store + chunks injection the upload
    # path uses, so hybrid/RRF/rerank/floor all behave as in chat).
    chunk_docs = [
        Document(
            page_content=c.text,
            metadata={
                "source": doc_name,
                "doc_id": doc_name,
                "chunk_id": c.chunk_id,
                "section": c.section,
                "allowed_groups": ["*"],
            },
        )
        for c in result.chunks
    ]
    store = FAISS.from_documents(chunk_docs, embeddings)
    chunk_dicts = [
        {
            "text": c.text,
            "source": doc_name,
            "doc_id": doc_name,
            "chunk_id": c.chunk_id,
            "section": c.section,
            "allowed_groups": ["*"],
        }
        for c in result.chunks
    ]
    retriever = Retriever(settings, store=store, chunks=chunk_dicts)
    hits = retriever.search(query, k=settings.top_k)
    result.retrieval_stages = retrieval_stages_for(settings)
    result.candidates = retriever.last_candidates
    result.floor_dropped = retriever.last_floor_dropped
    result.hits = [
        RetrievalRow(
            rank=i + 1,
            chunk_id=h.chunk_id,
            citation=format_citation(h),
            confidence=confidence_band(h.confidence),
            score=h.score,
            text=h.text,
        )
        for i, h in enumerate(hits)
    ]
    return result
