"""Phase 5 grounding tests (fake tier): relevance floor, passage-level
citations, structural agentic sources, stale-index handling."""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from langchain_core.documents import Document

from rag_demo import ingest
from rag_demo.agent import (
    StepResult,
    format_docs_evidence,
    sources_from_step,
    web_sources,
)
from rag_demo.chain import ask, format_hits
from rag_demo.config import INDEX_FORMAT_VERSION, get_settings
from rag_demo.preflight import index_format_current, run_preflight
from rag_demo.retriever import (
    Hit,
    Retriever,
    apply_floor,
    citation_source,
    confidence_band,
    confidence_for,
    format_citation,
    rrf_fuse,
)


# ------------------------------------------------------------ citations ---


def test_format_citation_with_and_without_section():
    assert format_citation(Hit(text="t", source="a.md", score=0.0, section="Auth")) == (
        "a.md › Auth"
    )
    assert format_citation(Hit(text="t", source="a.md", score=0.0)) == "a.md"


def test_citation_source_strips_section():
    assert citation_source("a.md › Auth") == "a.md"
    assert citation_source("a.md") == "a.md"


def test_format_hits_cites_section():
    out = format_hits([Hit(text="hello", source="a.md", score=0.1, section="Auth")])
    assert "[a.md › Auth]" in out and "hello" in out
    out_plain = format_hits([Hit(text="hello", source="a.md", score=0.1)])
    assert "[a.md]" in out_plain


# -------------------------------------------------- confidence + floor ---


def test_confidence_band_thresholds():
    assert confidence_band(None) == "—"
    assert confidence_band(0.9) == "High"
    assert confidence_band(0.75) == "High"
    assert confidence_band(0.5) == "Medium"
    assert confidence_band(0.49) == "Low"


def test_confidence_for_rerank_is_sigmoid():
    assert confidence_for(Hit(text="t", source="a.md", score=0.0), "rerank") == pytest.approx(0.5)
    assert confidence_for(Hit(text="t", source="a.md", score=5.0), "rerank") > 0.99
    assert confidence_for(Hit(text="t", source="a.md", score=-5.0), "rerank") < 0.01


def test_confidence_for_dense_maps_distance():
    assert confidence_for(Hit(text="t", source="a.md", score=0.0, dense_score=0.0), "dense") == pytest.approx(1.0)
    assert confidence_for(Hit(text="t", source="a.md", score=0.0, dense_score=1.0), "dense") == pytest.approx(0.5)
    assert confidence_for(Hit(text="t", source="a.md", score=0.0, dense_score=2.0), "dense") == pytest.approx(0.0)
    assert confidence_for(Hit(text="t", source="a.md", score=0.0), "dense") is None
    # RRF scores are relative ranks, never an absolute signal.
    assert confidence_for(Hit(text="t", source="a.md", score=0.03), "rrf") is None


def test_apply_floor_drops_below_floor_and_counts():
    hits = [
        Hit(text="good", source="a.md", score=2.0),
        Hit(text="bad", source="b.md", score=-2.0),
    ]
    kept, dropped = apply_floor(hits, 0.5, "rerank")
    assert [h.text for h in kept] == ["good"]
    assert dropped == 1
    assert kept[0].confidence == pytest.approx(0.88, abs=0.01)


def test_apply_floor_keeps_hits_without_absolute_signal():
    # BM25-only hit under a dense floor: no distance to judge on, so it
    # is kept rather than dropped on a meaningless number.
    hits = [Hit(text="kw", source="a.md", score=12.0)]
    kept, dropped = apply_floor(hits, 0.5, "dense")
    assert kept == hits and dropped == 0
    assert kept[0].confidence is None


def test_apply_floor_disabled_still_stamps_confidence():
    hits = [Hit(text="t", source="a.md", score=0.0, dense_score=0.4)]
    kept, dropped = apply_floor(hits, 0.0, "dense")
    assert kept == hits and dropped == 0
    assert kept[0].confidence == pytest.approx(0.8)


def test_rrf_fuse_preserves_chunk_identity():
    dense_hit = Hit(text="same", source="a.md", score=0.1, chunk_id="a.md#s:0",
                    section="S", dense_score=0.3)
    bm25_hit = Hit(text="same", source="a.md", score=9.0)
    fused = rrf_fuse([[dense_hit], [bm25_hit]], k=1)
    assert len(fused) == 1
    assert fused[0].chunk_id == "a.md#s:0"
    assert fused[0].section == "S"
    assert fused[0].dense_score == pytest.approx(0.3)
    # Fusion must not mutate the caller's hits.
    assert bm25_hit.dense_score is None


# ------------------------------------------------------- chunk identity ---


def test_chunk_documents_assigns_ids_and_sections():
    body = (
        "# Guide\n\n" + "intro word " * 120
        + "\n\n## Details\n\n" + "detail word " * 120
    )
    doc = Document(page_content=body, metadata={"source": "guide.md"})
    chunks = ingest.chunk_documents([doc], get_settings())
    assert chunks, "expected chunks"
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    assert all(c.metadata["chunk_id"].startswith("guide.md") for c in chunks)
    sections = {c.metadata["section"] for c in chunks}
    assert "Guide" in sections
    assert "Guide > Details" in sections
    # Ordinal is the trailing :N of the chunk id.
    assert chunks[0].metadata["chunk_id"].endswith(":0")


def test_chunk_documents_without_headings_has_empty_section():
    doc = Document(page_content="plain text " * 120, metadata={"source": "plain.md"})
    chunks = ingest.chunk_documents([doc], get_settings())
    assert chunks
    assert all(c.metadata["section"] == "" for c in chunks)
    assert chunks[0].metadata["chunk_id"] == "plain.md:0"


# ------------------------------------------------------------- index fmt ---


@pytest.fixture
def tmp_settings(tmp_path):
    s = get_settings()
    s.index_dir = tmp_path / "index"
    return s


def test_build_index_writes_format_marker_and_chunk_fields(tmp_settings):
    ingest.build_index(tmp_settings)
    marker = json.loads((tmp_settings.index_dir / "index_format.json").read_text())
    assert marker["version"] == INDEX_FORMAT_VERSION
    data = json.loads((tmp_settings.index_dir / "chunks.json").read_text())
    assert data and all("chunk_id" in c and "section" in c for c in data)
    assert all(c["chunk_id"] for c in data)


def test_incremental_rebuilds_when_format_stale(tmp_settings):
    ingest.build_index(tmp_settings)
    (tmp_settings.index_dir / "index_format.json").write_text(
        json.dumps({"version": 1, "features": []})
    )
    store = ingest.build_index(tmp_settings, incremental=True)
    assert store is not None, "stale format must force a rebuild, not a skip"
    marker = json.loads((tmp_settings.index_dir / "index_format.json").read_text())
    assert marker["version"] == INDEX_FORMAT_VERSION


def test_retriever_flags_stale_index(tmp_settings):
    ingest.build_index(tmp_settings)
    s = get_settings()
    s.index_dir = tmp_settings.index_dir
    s.retrieval = "dense"
    assert Retriever(s).index_stale is False
    (tmp_settings.index_dir / "index_format.json").unlink()
    assert Retriever(s).index_stale is True


def test_preflight_reports_stale_but_not_failed(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"")
    (index_dir / "index_format.json").write_text(json.dumps({"version": 1}))

    class FakeSettings:
        fake = True
        ollama_base_url = "http://localhost:11434"
        ollama_model = "qwen3:8b"

    result = run_preflight(FakeSettings(), "meridian", index_dir)
    assert result.index_ok is True
    assert result.index_stale is True
    assert result.problems == []
    assert result.all_ok is True  # stale warns; it does not disable chat
    assert index_format_current(index_dir) is False
    (index_dir / "index_format.json").write_text(
        json.dumps({"version": INDEX_FORMAT_VERSION})
    )
    assert index_format_current(index_dir) is True
    assert run_preflight(FakeSettings(), "meridian", index_dir).index_stale is False


# ------------------------------------------------------- floor in search ---


class _StubStore:
    """Duck-typed FAISS stand-in returning fixed (Document, L2) pairs."""

    def __init__(self, pairs):
        self.pairs = pairs

    def similarity_search_with_score(self, query, k):
        return self.pairs[:k]


def test_search_applies_floor_and_tracks_drops():
    s = get_settings()
    s.retrieval = "dense"
    s.relevance_floor = 0.5
    pairs = [
        (Document(page_content="close", metadata={"source": "a.md", "chunk_id": "a.md:0", "section": "S"}), 0.2),
        (Document(page_content="far", metadata={"source": "b.md", "chunk_id": "b.md:0", "section": "S"}), 1.6),
    ]
    r = Retriever(s, store=_StubStore(pairs), chunks=[])
    hits = r.search("q", k=2)
    assert [h.text for h in hits] == ["close"]
    assert r.last_candidates == 2
    assert r.last_floor_dropped == 1
    assert hits[0].confidence == pytest.approx(0.9)


def test_ask_marks_weak_evidence_when_floor_drops_everything():
    class FloorStubRetriever:
        last_floor_dropped = 2
        last_candidates = 2

        def search(self, query, user_groups=None):
            return []

    answer = ask("anything at all", get_settings(), retriever=FloorStubRetriever())
    assert answer.hits == []
    assert answer.weak_evidence is True
    assert answer.sources == []


# ------------------------------------------------- structural agentic src ---


def test_search_docs_tool_cites_section():
    from rag_demo.agent import _search_docs_tool

    class StubRetriever:
        def search(self, query, k=None):
            return [Hit(text="passage", source="a.md", score=0.1, section="Auth")]

    out = _search_docs_tool("anything", retriever=StubRetriever())
    assert "[a.md › Auth]" in out and "passage" in out


def test_sources_from_docs_hits_are_structural():
    hits = [Hit(text="t", source="a.md", score=0.1, chunk_id="a.md#s:0", section="Auth")]
    assert sources_from_step({"tool": "search_docs"}, StepResult(text="...", hits=hits)) == [
        "a.md › Auth"
    ]


def test_web_sources_exclude_markers():
    text = (
        "[TOOL_ERROR] Tavily search timed out\n\n"
        "[CACHED 5s ago]\n\n"
        "[Real Title](https://example.com/page)"
    )
    assert web_sources(text) == ["Real Title"]
    assert sources_from_step({"tool": "search_web"}, StepResult(text=text, hits=None)) == [
        "Real Title"
    ]
    assert web_sources("[NO_RESULTS] nothing here.") == []


def test_format_docs_evidence_empty():
    assert format_docs_evidence([]) == "No matching passages found."
