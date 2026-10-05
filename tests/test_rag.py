"""Tests run with RAG_DEMO_FAKE=1: deterministic fakes, no model downloads."""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

import pytest

from rag_demo import ingest
from rag_demo.chain import ask, build_chain, format_hits
from rag_demo.config import get_settings
from rag_demo.retriever import Retriever


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    s = get_settings()
    s.index_dir = tmp_path_factory.mktemp("index")
    return s


@pytest.fixture(scope="module")
def _indexed(settings):
    ingest.build_index(settings)
    return settings


def test_ingest_indexes_all_docs(settings):
    store = ingest.build_index(settings)
    assert store.index.ntotal > 0


def test_chunks_carry_source_metadata(settings):
    docs = ingest.load_documents(settings)
    chunks = ingest.chunk_documents(docs, settings)
    assert chunks, "expected at least one chunk"
    assert all(c.metadata.get("source", "").endswith(".md") for c in chunks)


def test_retriever_returns_hits_with_sources(_indexed):
    r = Retriever(_indexed)
    hits = r.search("API token expiry", k=3)
    assert len(hits) == 3
    assert all(h.source.endswith(".md") and h.text for h in hits)


def test_retriever_returns_k_hits(_indexed):
    # With FakeEmbeddings the vector space is random, so this asserts
    # structure, not which doc wins. Real-embedding quality is measured
    # by evals/eval_retrieval.py.
    r = Retriever(_indexed)
    hits = r.search("anything at all", k=4)
    assert len(hits) == 4
    assert all(h.source.endswith(".md") and h.text for h in hits)


def test_chain_answers_with_citations(_indexed):
    answer = ask("What is the max parcel weight?", _indexed)
    assert answer.text
    assert answer.sources, "expected source citations"


def test_format_hits_marks_sources():
    from rag_demo.retriever import Hit

    out = format_hits([Hit(text="hello", source="a.md", score=0.1)])
    assert "[a.md]" in out and "hello" in out


def test_golden_eval_runs_end_to_end(_indexed, monkeypatch):
    # The fake embedding space is random; this asserts the eval harness runs
    # end-to-end and reports a rate between 0 and 1.
    import evals.eval_retrieval as ev

    monkeypatch.setattr(ev, "THRESHOLD", 0.0)
    assert ev.main(settings=_indexed) == 0


def test_golden_file_is_well_formed():
    golden = json.loads(
        Path(__file__).resolve().parent.parent.joinpath("evals/golden.json").read_text()
    )
    assert len(golden) >= 10
    for item in golden:
        assert item["question"] and item["expected_sources"]


def test_markdown_source_loads_docs_with_citation_names():
    from rag_demo.sources import MarkdownDirectorySource

    docs = MarkdownDirectorySource(
        Path(__file__).resolve().parent.parent / "docs"
    ).load()
    assert len(docs) == 5
    assert all(d.metadata["source"].endswith(".md") for d in docs)
    assert {d.metadata["source"] for d in docs} == {
        "api-reference.md",
        "deployment-runbook.md",
        "onboarding.md",
        "security-policy.md",
        "troubleshooting.md",
    }


def test_ingest_accepts_explicit_source(settings):
    from rag_demo.sources import MarkdownDirectorySource

    source = MarkdownDirectorySource(
        Path(__file__).resolve().parent.parent / "docs"
    )
    store = ingest.build_index(settings, source)
    assert store.index.ntotal > 0


def test_playwright_source_is_registered():
    from rag_demo.sources import SOURCES, PlaywrightDocsSource

    assert SOURCES["playwright-docs"] is PlaywrightDocsSource
    assert SOURCES["meridian"].__name__ == "MarkdownDirectorySource"
