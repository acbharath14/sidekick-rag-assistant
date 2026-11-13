"""Jev routing eval: measure the router before it is wired into ask().

Runs every case in jev_routing_cases.json through the routing decision
(src/rag_demo/routing.py) and reports accuracy overall and per class,
plus decision latency. The router is measured, not the answer pipeline:
no retrieval, no generation, no FAISS index needed.

Needs Ollama running with the decision model pulled:

    ollama pull tev1:0.8b
    python -m evals.jev_routing_eval

Options: --model, --base-url, --threshold, --limit, --out (JSON report),
--min-accuracy (exit 1 below the bar; default 0, report-only).

Exit code 0 when the run completes, 1 on infrastructure failure
(client missing/unreachable), 2 on usage error.

Notes on the labelled set:
- Corpus cases are drawn from the repo's own eval files (golden.json,
  per_source/*.json, regression_cases.json) so they are grounded in the
  real indexed corpus. The regression "abstention" case is labelled
  corpus: the corpus was searched and the answer was not there, so the
  question needed the corpus.
- The regression "follow-up" case is EXCLUDED: "what about express?"
  is only answerable with its history, and this trial's router is
  context-free by design. History-aware routing is a follow-up question.
- The regression "small-talk" case is EXCLUDED: small talk never
  reaches the router (is_small_talk() runs first).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.routing import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    ROUTE_CORPUS,
    ROUTE_GENERAL,
    RoutingError,
    decide_route,
    load_client,
)


def score_cases(cases: list[dict], decider) -> dict:
    """Score labelled cases with a decider(question) -> RouteDecision.

    Pure function (no Ollama): unit tests drive it with a fake decider.
    """
    rows: list[dict] = []
    for case in cases:
        decision = decider(case["question"])
        ok = decision.route == case["expected"]
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "route": decision.route,
                "confidence": round(decision.confidence, 3),
                "elapsed_ms": round(decision.elapsed_ms, 1),
                "fallback": decision.fallback,
                "error": decision.error,
                "ok": ok,
            }
        )
    total = len(rows)
    correct = sum(r["ok"] for r in rows)
    corpus = [r for r in rows if r["expected"] == ROUTE_CORPUS]
    general = [r for r in rows if r["expected"] == ROUTE_GENERAL]
    latencies = [r["elapsed_ms"] for r in rows]
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "corpus_total": len(corpus),
        "corpus_correct": sum(r["ok"] for r in corpus),
        "general_total": len(general),
        "general_correct": sum(r["ok"] for r in general),
        "mean_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "median_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "max_ms": round(max(latencies), 1) if latencies else 0.0,
        "fallbacks": sum(1 for r in rows if r["fallback"]),
        "wrong": [r for r in rows if not r["ok"]],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the Jev routing decision (needs Ollama)."
    )
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("jev_routing_cases.json")),
        help="labelled cases file (default: evals/jev_routing_cases.json)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    parser.add_argument("--out", default="", help="write the JSON report here")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="exit 1 when accuracy is below this bar (default: report-only)",
    )
    args = parser.parse_args(argv)

    try:
        cases = json.loads(Path(args.cases).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read cases file {args.cases}: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("error: no cases to run", file=sys.stderr)
        return 2

    try:
        client = load_client(
            base_url=args.base_url, model=args.model, timeout=args.timeout
        )
    except RoutingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def decider(question: str):
        return decide_route(
            question,
            client=client,
            model=args.model,
            threshold=args.threshold,
            base_url=args.base_url,
            timeout=args.timeout,
        )

    print(
        f"jev routing eval: {len(cases)} cases "
        f"[model={args.model} threshold={args.threshold}]\n"
    )
    report = score_cases(cases, decider)
    for row in report["rows"]:
        mark = "PASS" if row["ok"] else "FAIL"
        fb = " (fallback)" if row["fallback"] else ""
        print(
            f"[{mark}] {row['id']}: {row['question'][:60]}"
            f"\n       expected {row['expected']}, got {row['route']}"
            f" (conf {row['confidence']}, {row['elapsed_ms']}ms){fb}"
        )
    print(
        f"\n{report['correct']}/{report['total']} correct "
        f"({report['accuracy']:.1%})"
    )
    print(
        f"corpus: {report['corpus_correct']}/{report['corpus_total']} | "
        f"general knowledge: {report['general_correct']}/{report['general_total']}"
    )
    print(
        f"latency: mean {report['mean_ms']}ms, median {report['median_ms']}ms, "
        f"max {report['max_ms']}ms; fallbacks: {report['fallbacks']}"
    )
    if report["wrong"]:
        print("\nwrong routes:")
        for row in report["wrong"]:
            print(f"  - {row['id']}: expected {row['expected']}, got {row['route']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.out}")
    if args.min_accuracy and report["accuracy"] < args.min_accuracy:
        print(
            f"\nbelow --min-accuracy {args.min_accuracy:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
