"""Tests for the per-source regression suites (evals/per_source.py).

Run with RAG_DEMO_FAKE=1: deterministic fakes, no model downloads.
Proves the runner logic on stub hits — real-index numbers come from
a run against real indexes on real hardware, not from this file.
"""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import evals.per_source as ps
from rag_demo.config import get_settings
from rag_demo.retriever import Hit

EVALS = Path(__file__).resolve().parent.parent / "evals"


def _hit(source, section="", text="passage"):
    return Hit(text=text, source=source, score=0.1, section=section)


def _prompt(**over):
    p = {"id": "t", "question": "q?", "expected_sources": [],
         "expected_sections": [], "expect_keywords": [],
         "expect_keywords_any": [], "expect_absent_sources": [],
         "user_groups": None}
    p.update(over)
    return p


# --- prompt files: shape + counts -------------------------------------------

def test_prompt_files_parse_and_count():
    sets = ps.load_prompt_sets()
    assert len(sets["meridian"]) == 44  # 20 golden + 9 regression + 15 gaps
    assert len(sets["playwright-docs"]) == 12
    assert len(sets["github-docs"]) == 8
    assert len(sets["confluence-mock"]) == 7
    for source in ps.ENTERPRISE_SOURCES:
        assert sets[source] == []
    for source, prompts in sets.items():
        for p in prompts:
            assert p["id"] and p["question"]
            assert isinstance(p["expected_sources"], list)
            assert isinstance(p["expected_sections"], list)


def test_meridian_gap_sections_are_real_doc_sections():
    """Every gap prompt's expected section must be a real H1 > H2 path
    in the Meridian corpus — no invented sections."""
    import re

    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    paths = set()
    for doc in docs_dir.glob("*.md"):
        stack: list[str] = []
        for line in doc.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not m:
                continue
            level = len(m.group(1))
            while len(stack) >= level:
                stack.pop()
            stack.append(m.group(2).strip())
            paths.add(" > ".join(stack))
    gaps = json.loads((EVALS / "per_source" / "meridian.json").read_text())
    for p in gaps:
        for sec in p["expected_sections"]:
            assert sec in paths, f"{p['id']}: {sec!r} not a real doc section"


def test_github_sections_are_real_readme_or_architecture_headings():
    """Same grounding rule for github-docs prompts."""
    import re

    repo = Path(__file__).resolve().parent.parent
    headings = set()
    for name in ("README.md", "ARCHITECTURE.md"):
        for line in (repo / name).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if m:
                headings.add(m.group(2).strip())
    prompts = json.loads((EVALS / "per_source" / "github-docs.json").read_text())
    for p in prompts:
        for sec in p["expected_sections"]:
            assert any(sec in h or h in sec for h in headings), (
                f"{p['id']}: {sec!r} matches no README/ARCHITECTURE heading")


def test_playwright_sources_are_doc_filenames():
    prompts = json.loads((EVALS / "per_source" / "playwright-docs.json").read_text())
    for p in prompts:
        for s in p["expected_sources"]:
            assert s.startswith("playwright-docs/") and s.endswith(".md"), s


# --- score_prompt ------------------------------------------------------------

def test_score_retrieval_hit_miss_and_na():
    hits = [_hit("api-reference.md")]
    assert ps.score_prompt(_prompt(expected_sources=["api-reference.md"]), hits)["retrieval"] is True
    assert ps.score_prompt(_prompt(expected_sources=["other.md"]), hits)["retrieval"] is False
    assert ps.score_prompt(_prompt(), hits)["retrieval"] is None


def test_score_absent_sources():
    hits = [_hit("confluence:OPS/TMS deploy checklist")]
    p = _prompt(expect_absent_sources=["confluence:OPS/Driver pay bands FY27"])
    assert ps.score_prompt(p, hits)["retrieval_absent"] is True
    hits2 = [_hit("confluence:OPS/Driver pay bands FY27")]
    assert ps.score_prompt(p, hits2)["retrieval_absent"] is False
    assert ps.score_prompt(_prompt(), hits)["retrieval_absent"] is None


def test_score_precision_section_substring_and_source_only():
    full = "Meridian Logistics — Shipment Tracking API Reference > Track a shipment"
    p = _prompt(expected_sources=["api-reference.md"],
                expected_sections=["Meridian Logistics — Shipment Tracking API Reference > Track a shipment"])
    assert ps.score_prompt(p, [_hit("api-reference.md", section=full)])["precision"] is True
    # H2-only expectation still matches the full path (substring rule).
    p2 = _prompt(expected_sources=["api-reference.md"],
                 expected_sections=["Track a shipment"])
    assert ps.score_prompt(p2, [_hit("api-reference.md", section=full)])["precision"] is True
    assert ps.score_prompt(p2, [_hit("api-reference.md", section="Other")])["precision"] is False
    # No sections => source-level only (confluence-mock pages).
    p3 = _prompt(expected_sources=["confluence:OPS/TMS deploy checklist"])
    assert ps.score_prompt(p3, [_hit("confluence:OPS/TMS deploy checklist")])["precision"] is True
    assert ps.score_prompt(p3, [_hit("confluence:OPS/EDI retry policy")])["precision"] is False


def test_score_rerank_improved_worse_absent_and_na():
    p = _prompt(expected_sources=["b.md"])
    before = [_hit("a.md"), _hit("b.md"), _hit("c.md")]
    after = [_hit("b.md"), _hit("a.md")]
    assert ps.score_prompt(p, after, before)["rerank"] == {"before": 2, "after": 1, "improved": True}
    assert ps.score_prompt(p, before, after)["rerank"]["improved"] is False
    assert ps.score_prompt(p, [_hit("a.md")], [_hit("a.md")])["rerank"]["before"] is None
    assert ps.score_prompt(p, after, None)["rerank"] is None


def test_score_accuracy_keywords_and_any():
    p = _prompt(expect_keywords=["90 days", "token_expired"])
    assert ps.score_prompt(p, [], answer_text="Tokens last 90 days; 401 token_expired")["accuracy"] is True
    assert ps.score_prompt(p, [], answer_text="Tokens last 90 days")["accuracy"] is False
    assert ps.score_prompt(p, [], answer_text=None)["accuracy"] is None
    assert ps.score_prompt(_prompt(), [], answer_text="anything")["accuracy"] is None
    p2 = _prompt(expect_keywords_any=["getByRole", "get_by_role"])
    assert ps.score_prompt(p2, [], answer_text="use getByRole")["accuracy"] is True
    assert ps.score_prompt(p2, [], answer_text="use locator")["accuracy"] is False


def test_aggregation_counts_applicable_only():
    rows = [
        {"retrieval": True, "precision": True, "accuracy": None,
         "rerank": {"before": 2, "after": 1, "improved": True}},
        {"retrieval": False, "precision": False, "accuracy": None,
         "rerank": {"before": None, "after": 1, "improved": None}},
        {"retrieval": None, "precision": None, "accuracy": None, "rerank": None},
    ]
    assert ps._rate(rows, "retrieval") == (1, 2)
    assert ps._rate(rows, "accuracy") == (0, 0)
    assert ps._rerank_rate(rows) == (1, 1, 2, 2)


# --- run_source skip + stub behavior -----------------------------------------

def test_run_source_skips_missing_index():
    settings = get_settings()
    out = ps.run_source("playwright-docs", settings)
    assert out["status"] == "skipped"
    assert "no index" in out["reason"]


def test_run_source_enterprise_skip_reason():
    settings = get_settings()
    for source in ps.ENTERPRISE_SOURCES:
        out = ps.run_source(source, settings)
        assert out["status"] == "skipped"
        assert "not connected" in out["reason"]


def test_run_source_github_hint_mentions_repo():
    settings = get_settings()
    out = ps.run_source("github-docs", settings)
    assert out["status"] == "skipped"
    assert "--repo" in out["reason"]


class _StubRetriever:
    """Canned hits; raises on rerank=False to simulate an unavailable
    cross-encoder (the sandbox has no sentence_transformers)."""

    def __init__(self, settings):
        self.index_stale = False

    def search(self, question, k=None, rerank=None, user_groups=None):
        if rerank is False:
            raise ImportError("no cross-encoder here")
        return [_hit("api-reference.md",
                     section="Meridian Logistics — Shipment Tracking API Reference > Authentication")]


def test_run_source_with_stub_retriever_scores_and_marks_rerank_na(monkeypatch):
    import evals.per_source as mod

    monkeypatch.setattr(mod, "INDEX_DIRS", {"meridian": "x"})
    monkeypatch.setattr(mod, "index_present", lambda d: True)
    settings = get_settings()
    out = mod.run_source("meridian", settings, retriever_factory=_StubRetriever)
    assert out["status"] == "ran"
    assert len(out["results"]) == 44
    token_row = next(r for r in out["results"]
                     if r["prompt"]["id"] == "token-expiry")
    assert token_row["retrieval"] is True
    assert token_row["rerank"] is None  # reranker unavailable => n/a, not crash


def test_main_skips_everything_without_indexes_and_exits_zero(monkeypatch, capsys):
    # Deterministic regardless of committed indexes in the checkout:
    # with no index anywhere, every source skips and the run exits 0
    # even under --fail-on-miss (skips are not misses).
    monkeypatch.setattr(ps, "index_present", lambda d: False)
    settings = get_settings()
    assert ps.main(settings=settings) == 0
    assert ps.main(settings=settings, fail_on_miss=True) == 0
    assert ps.main(settings=settings, only_source="bogus") == 2


def test_main_runs_in_sandbox_checkout(capsys):
    # Real checkout state: confluence-mock has a committed (stale)
    # index, everything else skips. Retrieval-only exit code is 0
    # whether that index loads or search errors per prompt.
    assert ps.main(settings=get_settings()) == 0


# --- confluence-mock fixture expansion ---------------------------------------

def test_mock_fixtures_include_new_pages():
    from rag_demo.sources import MockConfluenceSource

    docs = MockConfluenceSource(
        Path(__file__).resolve().parent.parent / "fixtures" / "confluence"
    ).load()
    titles = {d.metadata["source"] for d in docs}
    assert len(docs) == 6
    assert "confluence:OPS/API key rotation schedule" in titles
    assert "confluence:OPS/Incident severity levels" in titles
    assert "confluence:OPS/Driver pay bands FY27" in titles
