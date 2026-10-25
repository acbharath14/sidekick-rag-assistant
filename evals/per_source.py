"""Per-source regression suites: one prompt set per corpus, four metrics.

Extends the existing eval machinery (golden.json, regression_cases.json,
regression.py's check shapes) instead of building a parallel runner:

- Meridian keeps the existing 20 golden items + 9 regression cases, plus
  gap prompts in per_source/meridian.json for corpus topics the golden
  set never asked about (track response, environments, feature flags,
  on-call, week 2/3, buddy system, classification, storage, access,
  third parties, 401-with-fresh-token, 5xx, search-no-results, rate
  limits). Four golden/regression questions appear in both suites and
  are counted once per form, as the existing suites treat them.
- Playwright docs, GitHub repo docs, and Confluence mock each get a
  per_source/<source>.json prompt file grounded in that source's real
  content (Playwright v1.63.0 docs structure, this repo's README +
  ARCHITECTURE.md, the mock fixtures).
- Enterprise sources (confluence, jira, outlook, aws-s3) are SKIPPED
  when unconnected — no mock data, no fake prompts. A skipped source
  is reported, never silently dropped, and a connected-but-promptless
  enterprise source reports "no prompt set" rather than fake coverage.

Metrics per prompt (a metric is None, not failed, when its inputs
don't exist — e.g. no expected sections means precision is source-level
only, and no keywords means accuracy doesn't apply):
  retrieval  — an expected source appears in the retrieved hits.
  reranking — rank of the first expected hit before vs after rerank.
  precision  — a hit cites the expected source AND an expected section
               (section match is a case-insensitive substring of the
               hit's section path, so H1 prefixes and heading-extraction
               quirks don't fail an otherwise exact citation).
  accuracy   — (answers mode only) the answer contains the expected
               key facts (all of expect_keywords, and any of
               expect_keywords_any when that list is present).

Two tiers, mirroring regression.py:
  PYTHONPATH=src python -m evals.per_source
      Retrieval-only: no LLM needed. Under RAG_DEMO_FAKE=1 the
      embedding space is random, so rates are structural only — real
      numbers come from a run against real indexes on real hardware.
  PYTHONPATH=src python -m evals.per_source --answers
      Also runs ask() per prompt and checks answer content. Refuses
      under RAG_DEMO_FAKE=1 (fake answers can't contain real facts),
      exactly like regression.py --real-llm.

Exit code 0 on a completed run (skips never fail it); --fail-on-miss
exits 1 if any applicable retrieval check misses, for later CI gating
once real-index baselines exist.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import get_settings  # noqa: E402
from rag_demo.ingest import INDEX_DIRS  # noqa: E402
from rag_demo.preflight import index_present  # noqa: E402
from rag_demo.retriever import Retriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed run order: real-content sources first, enterprise last (skipped).
SOURCES = (
    "meridian",
    "playwright-docs",
    "github-docs",
    "confluence-mock",
    "confluence",
    "jira",
    "outlook",
    "aws-s3",
)
ENTERPRISE_SOURCES = ("confluence", "jira", "outlook", "aws-s3")


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(prompt: dict, source: str, index: int) -> dict:
    """One prompt shape for every suite, following golden.json +
    regression_cases.json conventions (question, expected_sources,
    expect_keywords) plus Phase 5 fields (expected_sections) and the
    per-source additions (user_groups, expect_absent_sources,
    expect_keywords_any for cross-language syntax prompts)."""
    return {
        "id": prompt.get("id") or f"{source}-{index}",
        "question": prompt["question"],
        "expected_sources": list(prompt.get("expected_sources", [])),
        "expected_sections": list(prompt.get("expected_sections", [])),
        "expect_keywords": list(prompt.get("expect_keywords", [])),
        "expect_keywords_any": list(prompt.get("expect_keywords_any", [])),
        "expect_absent_sources": list(prompt.get("expect_absent_sources", [])),
        "user_groups": prompt.get("user_groups"),
    }


def load_prompt_sets() -> dict[str, list[dict]]:
    """source -> normalized prompts. Meridian merges the existing suites
    with the gap file; enterprise sources intentionally have no file."""
    here = Path(__file__).resolve().parent
    sets: dict[str, list[dict]] = {}
    meridian: list[dict] = []
    for name in ("golden.json", "regression_cases.json"):
        meridian.extend(_load_json(here / name))
    meridian.extend(_load_json(here / "per_source" / "meridian.json"))
    sets["meridian"] = [_normalize(p, "meridian", i) for i, p in enumerate(meridian)]
    for source in ("playwright-docs", "github-docs", "confluence-mock"):
        raw = _load_json(here / "per_source" / f"{source}.json")
        sets[source] = [_normalize(p, source, i) for i, p in enumerate(raw)]
    for source in ENTERPRISE_SOURCES:
        sets[source] = []
    return sets


def _first_expected_rank(hits, expected_sources: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.source in expected_sources:
            return rank
    return None


def score_prompt(prompt: dict, hits, hits_norerank=None, answer_text: str | None = None) -> dict:
    """Pure metric computation over one prompt's hits (and, in answers
    mode, its answer text). No Retriever, no LLM — unit-testable on
    stub hits. Each metric is True, False, or None (not applicable)."""
    expected = prompt["expected_sources"]
    sections = prompt["expected_sections"]
    result: dict = {}

    if expected:
        result["retrieval"] = any(h.source in expected for h in hits)
    else:
        result["retrieval"] = None

    if prompt["expect_absent_sources"]:
        result["retrieval_absent"] = all(
            h.source not in prompt["expect_absent_sources"] for h in hits
        )
    else:
        result["retrieval_absent"] = None

    if expected:
        if not sections:
            # Source-level only (e.g. confluence-mock pages carry no
            # markdown headings, so their chunks have no section).
            result["precision"] = any(h.source in expected for h in hits)
            result["precision_level"] = "source-only"
        else:
            ok = False
            for h in hits:
                if h.source not in expected:
                    continue
                hay = (h.section or "").lower()
                if any(sec.lower() in hay for sec in sections):
                    ok = True
                    break
            result["precision"] = ok
            result["precision_level"] = "source+section"
    else:
        result["precision"] = None
        result["precision_level"] = "n/a"

    if hits_norerank is None or not expected:
        result["rerank"] = None
    else:
        before = _first_expected_rank(hits_norerank, expected)
        after = _first_expected_rank(hits, expected)
        if before is None or after is None:
            result["rerank"] = {
                "before": before, "after": after, "improved": None,
            }
        else:
            result["rerank"] = {
                "before": before, "after": after, "improved": after < before,
            }

    if answer_text is None:
        result["accuracy"] = None
    elif not prompt["expect_keywords"] and not prompt["expect_keywords_any"]:
        result["accuracy"] = None
    else:
        text = answer_text.lower()
        all_ok = all(k.lower() in text for k in prompt["expect_keywords"])
        any_ok = (
            any(k.lower() in text for k in prompt["expect_keywords_any"])
            if prompt["expect_keywords_any"]
            else True
        )
        result["accuracy"] = all_ok and any_ok

    return result


def _rate(results: list[dict], key: str) -> tuple[int, int]:
    """(passed, applicable) for a boolean metric key."""
    applicable = [r for r in results if r.get(key) is not None]
    passed = sum(1 for r in applicable if r[key] is True)
    return passed, len(applicable)


def _rerank_rate(results: list[dict]) -> tuple[int, int, int, int]:
    """(improved, both_present, present_after, computed) for rerank."""
    comparable = [r["rerank"] for r in results if r.get("rerank") is not None]
    both = [x for x in comparable if x["before"] is not None and x["after"] is not None]
    improved = sum(1 for x in both if x["improved"])
    present_after = sum(1 for x in comparable if x["after"] is not None)
    return improved, len(both), present_after, len(comparable)


def run_source(source: str, settings, answers: bool = False,
               retriever_factory=None, ask_fn=None) -> dict:
    """Run one source's suite. Never raises for a missing index, a
    failed retriever build, or an unavailable reranker — those become
    skip reasons or n/a metrics, so one broken source can't kill the
    whole run."""
    prompts = load_prompt_sets()[source]
    index_dir = REPO_ROOT / INDEX_DIRS[source]
    if source in ENTERPRISE_SOURCES and not prompts:
        if not index_present(index_dir):
            return {"source": source, "status": "skipped",
                    "reason": "skipped (not connected)"}
        return {"source": source, "status": "skipped",
                "reason": "connected but no prompt set (prompts are authored "
                          "from a real corpus at connection time)"}
    if not index_present(index_dir):
        hint = f"PYTHONPATH=src python -m rag_demo.ingest --source {source}"
        if source == "github-docs":
            hint += " --repo owner/name"
        return {"source": source, "status": "skipped",
                "reason": f"no index — build with: {hint}"}
    try:
        retriever = (retriever_factory or Retriever)(
            dataclasses.replace(settings, index_dir=index_dir)
        )
    except Exception as e:  # noqa: BLE001 — a broken source is a skip
        return {"source": source, "status": "skipped",
                "reason": f"retriever failed: {e}"}
    warnings = []
    if getattr(retriever, "index_stale", False):
        warnings.append("stale index — citations degrade to filename-only until rebuilt")

    results = []
    for prompt in prompts:
        groups = prompt["user_groups"] if prompt["user_groups"] is not None else settings.user_groups
        try:
            hits = retriever.search(prompt["question"], user_groups=groups)
        except Exception as e:  # noqa: BLE001 — a crashed prompt is a miss
            results.append({"prompt": prompt, "error": str(e),
                            **{k: None for k in ("retrieval", "retrieval_absent", "precision", "rerank", "accuracy")}})
            continue
        try:
            hits_norerank = retriever.search(
                prompt["question"], rerank=False, user_groups=groups
            )
        except Exception:  # noqa: BLE001 — reranker unavailable => n/a
            hits_norerank = None
        answer_text = None
        if answers:
            answer = (ask_fn or _default_ask)(
                prompt["question"], settings, retriever
            )
            answer_text = answer.text
        results.append({"prompt": prompt,
                        **score_prompt(prompt, hits, hits_norerank, answer_text)})
    return {"source": source, "status": "ran", "results": results,
            "warnings": warnings}


def _default_ask(question, settings, retriever):
    from rag_demo.chain import ask

    return ask(question, settings, retriever=retriever)


def summarize(outcome: dict) -> dict:
    """Per-source metric rates from a run_source outcome."""
    if outcome["status"] != "ran":
        return outcome
    results = outcome["results"]
    ret_p, ret_n = _rate(results, "retrieval")
    abs_p, abs_n = _rate(results, "retrieval_absent")
    pre_p, pre_n = _rate(results, "precision")
    acc_p, acc_n = _rate(results, "accuracy")
    imp, both, present_after, computed = _rerank_rate(results)
    return {**outcome,
            "retrieval": (ret_p, ret_n),
            "retrieval_absent": (abs_p, abs_n),
            "precision": (pre_p, pre_n),
            "accuracy": (acc_p, acc_n),
            "rerank": (imp, both, present_after, computed)}


def main(settings=None, answers: bool = False, fail_on_miss: bool = False,
         only_source: str | None = None) -> int:
    settings = settings or get_settings()
    if answers and settings.fake:
        print("error: --answers needs a real LLM (unset RAG_DEMO_FAKE), "
              "like regression.py --real-llm", file=sys.stderr)
        return 2
    sets = load_prompt_sets()
    sources = [only_source] if only_source else list(SOURCES)
    if only_source and only_source not in SOURCES:
        print(f"error: unknown source {only_source!r} (one of {', '.join(SOURCES)})",
              file=sys.stderr)
        return 2
    mode = "retrieval + answers" if answers else "retrieval-only"
    print(f"per-source regression [{mode}]\n")

    totals = {"retrieval": [0, 0], "precision": [0, 0], "accuracy": [0, 0],
              "rerank_imp": [0, 0], "rerank_present": [0, 0]}
    any_miss = False
    for source in sources:
        outcome = summarize(run_source(source, settings, answers=answers))
        if outcome["status"] == "skipped":
            print(f"[{source}] skipped: {outcome['reason']}\n")
            continue
        n = len(outcome["results"])
        print(f"[{source}] {n} prompts")
        for w in outcome.get("warnings", []):
            print(f"  ! {w}")
        for row in outcome["results"]:
            p = row["prompt"]
            marks = []
            if row["retrieval"] is not None:
                marks.append(f"retrieval={'✓' if row['retrieval'] else '✗'}")
                any_miss = any_miss or not row["retrieval"]
            if row["retrieval_absent"] is not None:
                marks.append(f"absent={'✓' if row['retrieval_absent'] else '✗'}")
            if row["precision"] is not None:
                marks.append(f"precision={'✓' if row['precision'] else '✗'}")
            if row["rerank"] is not None:
                r = row["rerank"]
                if r["before"] is not None and r["after"] is not None:
                    marks.append(f"rerank={r['before']}→{r['after']}")
                elif r["after"] is not None:
                    marks.append(f"rerank=?→{r['after']}")
                else:
                    marks.append("rerank=n/a")
            if row["accuracy"] is not None:
                marks.append(f"accuracy={'✓' if row['accuracy'] else '✗'}")
            if row.get("error"):
                marks.append(f"error: {row['error']}")
            failed = row.get("error") or any(
                row.get(k) is False
                for k in ("retrieval", "retrieval_absent", "precision", "accuracy")
            )
            print(f"  {'MISS' if failed else 'PASS'} "
                  f"{p['id']}: {p['question'][:60]}  [{'; '.join(marks)}]")
        rp, rn = outcome["retrieval"]
        pp, pn = outcome["precision"]
        ap, an = outcome["accuracy"]
        imp, both, present_after, computed = outcome["rerank"]
        totals["retrieval"][0] += rp
        totals["retrieval"][1] += rn
        totals["precision"][0] += pp
        totals["precision"][1] += pn
        totals["accuracy"][0] += ap
        totals["accuracy"][1] += an
        totals["rerank_imp"][0] += imp
        totals["rerank_imp"][1] += both
        totals["rerank_present"][0] += present_after
        totals["rerank_present"][1] += computed
        print(f"  → retrieval {rp}/{rn} · precision {pp}/{pn} · "
              f"rerank improved {imp}/{both} (present after rerank {present_after}/{computed})"
              + (f" · accuracy {ap}/{an}" if an else "") + "\n")

    print("overall (run sources only, skipped sources excluded):")
    print(f"  retrieval {totals['retrieval'][0]}/{totals['retrieval'][1]} · "
          f"precision {totals['precision'][0]}/{totals['precision'][1]} · "
          f"rerank improved {totals['rerank_imp'][0]}/{totals['rerank_imp'][1]} "
          f"(present after rerank {totals['rerank_present'][0]}/{totals['rerank_present'][1]})"
          + (f" · accuracy {totals['accuracy'][0]}/{totals['accuracy'][1]}"
             if totals["accuracy"][1] else ""))
    if not answers:
        print("tip: re-run with --answers (Ollama running) for answer-content checks")
    if fail_on_miss and any_miss:
        print("FAIL: --fail-on-miss set and at least one retrieval check missed")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true",
                        help="also run ask() per prompt and check answer content (needs Ollama)")
    parser.add_argument("--fail-on-miss", action="store_true",
                        help="exit 1 if any applicable retrieval check misses")
    parser.add_argument("--source", default=None,
                        help="run one source only (e.g. --source meridian)")
    args = parser.parse_args()
    raise SystemExit(main(answers=args.answers, fail_on_miss=args.fail_on_miss,
                          only_source=args.source))
