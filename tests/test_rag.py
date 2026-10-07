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


def test_rrf_fuse_orders_by_rank():
    from rag_demo.retriever import Hit, rrf_fuse

    a = Hit(text="doc-a", source="a.md", score=0.0)
    b = Hit(text="doc-b", source="b.md", score=0.0)
    fused = rrf_fuse([[a, b], [a]], k=2)
    assert [h.source for h in fused] == ["a.md", "b.md"]
    assert fused[0].score > fused[1].score


def test_rrf_fuse_dedupes_by_text_and_source():
    from rag_demo.retriever import Hit, rrf_fuse

    a1 = Hit(text="same", source="a.md", score=0.0)
    a2 = Hit(text="same", source="a.md", score=0.0)
    fused = rrf_fuse([[a1], [a2]], k=2)
    assert len(fused) == 1
    # ranked 1st in both lists: 2 * 1/(60+1)
    assert fused[0].score == pytest.approx(2 / 61)


def test_chunks_json_written_on_ingest(_indexed):
    p = _indexed.index_dir / "chunks.json"
    assert p.exists(), "ingest should persist chunks.json for hybrid retrieval"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data and all({"text", "source", "doc_id"} <= set(c) for c in data)


def test_hybrid_retrieval_finds_keyword_doc(_indexed, monkeypatch):
    monkeypatch.setenv("RAG_DEMO_RETRIEVAL", "hybrid")
    s = get_settings()
    s.index_dir = _indexed.index_dir
    r = Retriever(s)
    hits = r.search("weight_kg", k=3)
    assert hits
    assert any(h.source == "api-reference.md" for h in hits)


def test_rewrite_no_history_returns_question_unchanged():
    from rag_demo.rewrite import rewrite_query

    assert rewrite_query("hello?", [], llm=None) == "hello?"


def test_rewrite_uses_history_with_fake_llm():
    from langchain_core.language_models.fake import FakeListLLM

    from rag_demo.rewrite import rewrite_query

    llm = FakeListLLM(responses=["What is the max parcel weight for express?"])
    out = rewrite_query(
        "what about express?",
        [("What is the max parcel weight?", "1200 kg per parcel")],
        llm,
    )
    assert out == "What is the max parcel weight for express?"


def test_reranker_import_is_lazy():
    # Importing the module must not pull in sentence_transformers/torch (CI).
    import sys

    assert "sentence_transformers" not in sys.modules
    import rag_demo.rerank  # noqa: F401

    assert "sentence_transformers" not in sys.modules


def test_ask_accepts_history(_indexed):
    answer = ask(
        "what about express?",
        _indexed,
        history=[("What is the max parcel weight?", "1200 kg per parcel")],
    )
    assert answer.text


def test_confluence_mock_loads_pages_with_acls():
    from rag_demo.sources import MockConfluenceSource

    docs = MockConfluenceSource(
        Path(__file__).resolve().parent.parent / "fixtures" / "confluence"
    ).load()
    assert len(docs) == 4
    by_source = {d.metadata["source"]: d for d in docs}
    assert "confluence:ENG/Salary bands FY27" in by_source
    restricted = by_source["confluence:ENG/Salary bands FY27"]
    assert restricted.metadata["allowed_groups"] == ["eng-leads"]
    public = by_source["confluence:ENG/Deploy checklist"]
    assert public.metadata["allowed_groups"] == ["*"]
    assert "Deploy checklist" in public.page_content  # HTML was stripped


def test_html_to_text_strips_tags():
    from rag_demo.sources import html_to_text

    assert html_to_text("<h1>Title</h1><p>Some <b>bold</b> text.</p>") == "Title\n\nSome bold text."


def test_github_source_parses_tree_and_blobs(monkeypatch):
    import base64

    from rag_demo.sources import GitHubDocsSource

    tree = {"tree": [
        {"type": "blob", "path": "docs/guide.md", "sha": "abc123"},
        {"type": "blob", "path": "src/main.py", "sha": "def456"},
    ], "truncated": False}
    blob = {"content": base64.b64encode(b"# Guide\n\nHello docs.").decode()}

    def fake_api(self, path):
        return blob if "/git/blobs/" in path else tree

    monkeypatch.setattr(GitHubDocsSource, "_api", fake_api)
    docs = GitHubDocsSource(repo="owner/repo").load()
    assert len(docs) == 1  # only the .md file
    assert docs[0].metadata["source"] == "github:owner/repo/docs/guide.md"
    assert "Hello docs" in docs[0].page_content
    assert docs[0].metadata["allowed_groups"] == ["*"]


def test_github_source_truncated_tree_raises(monkeypatch):
    from rag_demo.sources import GitHubDocsSource

    monkeypatch.setattr(
        GitHubDocsSource, "_api", lambda self, path: {"tree": [], "truncated": True}
    )
    with pytest.raises(RuntimeError, match="truncated"):
        GitHubDocsSource(repo="owner/repo").load()


def test_acl_filtering_hides_restricted_hits(_indexed_confluence):
    settings, _ = _indexed_confluence
    r = Retriever(settings)
    # eng-leads sees the restricted page; eng-all does not.
    leads_hits = r.search("salary bands", k=10, user_groups=["eng-leads"])
    assert any("Salary bands" in h.source for h in leads_hits)
    all_hits = r.search("salary bands", k=10, user_groups=["eng-all"])
    assert not any("Salary bands" in h.source for h in all_hits)
    # No groups given -> no filtering (backwards compatible).
    unfiltered = r.search("salary bands", k=10)
    assert any("Salary bands" in h.source for h in unfiltered)


def test_audit_log_writes_one_line_per_ask(_indexed, tmp_path, monkeypatch):
    from rag_demo.audit import log_question
    from rag_demo.chain import Answer

    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("RAG_DEMO_AUDIT_LOG", str(log_path))
    s = get_settings()
    answer = Answer(text="hi", sources=["a.md"], hits=[])
    log_question(s, "hello?", answer, 12.5)
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["question"] == "hello?"
    assert entry["sources"] == ["a.md"]
    assert entry["latency_ms"] == 12.5
    assert "answer" not in entry  # answers not logged by default


def test_audit_log_disabled_by_default(_indexed, tmp_path, monkeypatch):
    from rag_demo.audit import log_question
    from rag_demo.chain import Answer

    monkeypatch.delenv("RAG_DEMO_AUDIT_LOG", raising=False)
    s = get_settings()
    log_question(s, "hello?", Answer(text="hi", sources=[], hits=[]), 1.0)
    # No file created anywhere we can see; just assert no crash.
    assert True


@pytest.fixture(scope="module")
def _indexed_confluence(tmp_path_factory):
    from rag_demo.sources import MockConfluenceSource

    s = get_settings()
    s.index_dir = tmp_path_factory.mktemp("index-confluence")
    ingest.build_index(
        s,
        MockConfluenceSource(
            Path(__file__).resolve().parent.parent / "fixtures" / "confluence"
        ),
    )
    return s, True
