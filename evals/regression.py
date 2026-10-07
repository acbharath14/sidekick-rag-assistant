"""End-to-end regression runner: standard prompts through the full ask() pipeline.

Demonstrates what the assistant can do — factual Q&A with citations,
abstention on unanswerable questions, small-talk handling, and multi-turn
follow-ups — and fails loudly if any of it regresses.

Two tiers:
  RAG_DEMO_FAKE=1 python -m evals.regression   # CI-safe: fake LLM.
      Checks answer structure (non-empty, expected sources cited).
  python -m evals.regression --real-llm        # needs Ollama running.
      Also checks answer content (keywords, abstention, natural small-talk).

Exit code 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.chain import ask  # noqa: E402
from rag_demo.config import get_settings  # noqa: E402

ABSTAIN_PHRASES = ("don't know", "do not know", "not contain", "no information")


def check_case(case: dict, answer, real_llm: bool) -> list[tuple[str, bool]]:
    """Returns [(check_name, passed)] for one case.

    Fake tier (CI-safe): proves the pipeline runs end-to-end — retrieval
    returns hits, the chain produces a cited answer. Which sources win is
    random with FakeEmbeddings, so specific-source assertions are real-tier
    only, where embeddings are deterministic.
    """
    checks: list[tuple[str, bool]] = []
    checks.append(("answer non-empty", bool(answer.text and answer.text.strip())))
    if case.get("expected_sources"):
        checks.append(("retrieval returned hits", bool(answer.sources)))
    if real_llm:
        if case.get("expected_sources"):
            got, exp = set(answer.sources), set(case["expected_sources"])
            checks.append(("expected sources cited", bool(got & exp)))
        text = answer.text.lower()
        if case.get("expect_keywords"):
            missing = [k for k in case["expect_keywords"] if k.lower() not in text]
            checks.append(("keywords present", not missing))
        if case.get("expect_abstain"):
            checks.append(
                ("abstains gracefully", any(p in text for p in ABSTAIN_PHRASES))
            )
        if case.get("expect_small_talk"):
            checks.append(
                ("natural reply (no abstention)",
                 not any(p in text for p in ABSTAIN_PHRASES)),
            )
        if case.get("expect_fallback"):
            checks.append(
                ("falls back to general knowledge (marked)",
                 not answer.grounded),
            )
    return checks


def main(real_llm: bool = False, settings=None) -> int:
    here = Path(__file__)
    cases = json.loads(here.with_name("regression_cases.json").read_text())
    settings = settings or get_settings()
    if real_llm and settings.fake:
        print("error: --real-llm needs a real LLM (unset RAG_DEMO_FAKE)", file=sys.stderr)
        return 2
    mode = "real LLM" if real_llm else "fake LLM (structure only)"
    print(f"regression: {len(cases)} cases [{mode}]\n")

    passed, failed = 0, 0
    for case in cases:
        history = [tuple(h) for h in case.get("history", [])]
        try:
            answer = ask(case["question"], settings, history=history)
        except Exception as e:  # noqa: BLE001 — a crash is a failed case
            print(f"[FAIL] {case['id']}: {case['question'][:60]}")
            print(f"       crashed: {e}\n")
            failed += 1
            continue
        checks = check_case(case, answer, real_llm)
        ok = all(p for _, p in checks)
        passed, failed = passed + ok, failed + (not ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['id']}: {case['question'][:60]}")
        for name, p in checks:
            print(f"       {'✓' if p else '✗'} {name}")
        print(f"       sources: {answer.sources}")
        preview = " ".join(answer.text.split())[:160]
        print(f"       → {preview}\n")

    print(f"{passed}/{len(cases)} passed [{mode}]")
    if not real_llm:
        print("tip: re-run with --real-llm (Ollama running) for answer-content checks")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="check answer content too (needs Ollama running)",
    )
    args = parser.parse_args()
    raise SystemExit(main(real_llm=args.real_llm))
