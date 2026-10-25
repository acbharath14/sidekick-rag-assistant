"""Pipeline-walkthrough tests (fake tier): stage outputs, specimen chunking,
retrieval display data. No Streamlit involved — the walkthrough logic is
Streamlit-free by design."""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import get_settings
from rag_demo.walkthrough import (
    SPECIMEN_TEXT,
    STAGES,
    chunk_stage,
    retrieval_stages_for,
    run_walkthrough,
)


def _settings():
    return get_settings()


# ---------------------------------------------------------------- stages ---


def test_stages_are_exactly_chunk_embed_retrieve():
    # Guards against a summarize or generation step creeping into the
    # walkthrough unnoticed: visibility only, no new capability.
    assert STAGES == ("chunking", "embedding", "retrieval")
    assert "summarize" not in STAGES and "generate" not in STAGES


def test_specimen_chunks_carry_ids_and_sections():
    chunks = chunk_stage(SPECIMEN_TEXT, "specimen", _settings())
    assert len(chunks) >= 2
    ids = [c.chunk_id for c in chunks]
    assert all(ids) and len(set(ids)) == len(ids)
    # Every chunk sits under one of the specimen's headings, and the
    # section follows the chunk's start (chunk 0 starts at the H1 title,
    # so its section is the title, not the first H2).
    assert chunks[0].section == "How Sidekick answers a question"
    assert {c.section.split(" > ")[-1] for c in chunks} <= {
        "How Sidekick answers a question",
        "Chunking",
        "Embedding",
        "Retrieval",
    }
    assert len({c.section for c in chunks}) >= 2  # attribution varies


def test_specimen_chunks_respect_chunk_size():
    settings = _settings()
    for c in chunk_stage(SPECIMEN_TEXT, "specimen", settings):
        assert c.chars <= settings.chunk_size


def test_run_walkthrough_fake_embedding_shape():
    result = run_walkthrough(
        SPECIMEN_TEXT, "specimen", "How does chunking work?", _settings()
    )
    assert result.chunk_count == len(result.chunks) >= 2
    assert result.embedding is not None
    assert result.embedding.fake is True
    assert result.embedding.dim == 384
    assert result.embedding.vector_count == result.chunk_count
    assert len(result.embedding.preview) == 5


def test_run_walkthrough_retrieval_rows_reference_chunks():
    result = run_walkthrough(
        SPECIMEN_TEXT, "specimen", "How does chunking work?", _settings()
    )
    assert result.hits
    chunk_ids = {c.chunk_id for c in result.chunks}
    for i, hit in enumerate(result.hits, start=1):
        assert hit.rank == i
        assert hit.chunk_id in chunk_ids
        assert hit.citation.startswith("specimen")
        assert hit.text
    assert result.candidates >= len(result.hits)


def test_retrieval_stages_reflect_settings():
    settings = _settings()
    assert settings.retrieval == "hybrid"
    assert retrieval_stages_for(settings) == ["BM25", "dense", "RRF fusion"]
    settings.retrieval = "dense"
    assert retrieval_stages_for(settings) == ["dense"]


def test_run_walkthrough_pasted_text():
    text = "# Notes\n\nThe deployment runbook lives in the docs folder."
    result = run_walkthrough(text, "pasted", "Where is the runbook?", _settings())
    assert result.chunk_count >= 1
    assert result.hits
    assert all(h.citation.startswith("pasted") for h in result.hits)


def test_run_walkthrough_empty_text():
    result = run_walkthrough("", "empty", "anything?", _settings())
    assert result.chunk_count == 0
    assert result.embedding is None
    assert result.hits == []
