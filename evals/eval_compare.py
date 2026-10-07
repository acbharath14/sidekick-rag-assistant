"""Compare retrieval modes on the golden set.

Runs dense vs hybrid (and hybrid+rerank when real embeddings are available),
prints a comparison table, exits 1 if any mode drops below its baseline in
baselines.json. The rerank mode is skipped under RAG_DEMO_FAKE=1 because it
needs the cross-encoder model download.

Usage:
    RAG_DEMO_FAKE=1 python -m evals.eval_compare   # dense vs hybrid only
    python -m evals.eval_compare                    # all three modes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import get_settings  # noqa: E402
from rag_demo.retriever import Retriever  # noqa: E402


def hit_rate(settings, golden) -> float:
    retriever = Retriever(settings)
    hits = 0
    for item in golden:
        got = {h.source for h in retriever.search(item["question"], k=settings.top_k)}
        if got & set(item["expected_sources"]):
            hits += 1
    return hits / len(golden)


def main() -> int:
    here = Path(__file__)
    golden = json.loads(here.with_name("golden.json").read_text())
    baselines = json.loads(here.with_name("baselines.json").read_text())
    fake = get_settings().fake
    modes = [("dense", False, False), ("hybrid", True, False)]
    if not fake:
        modes.append(("hybrid+rerank", True, True))
    elif "hybrid+rerank" in baselines:
        print("(skipping hybrid+rerank: needs real embeddings + model download)")
    ok = True
    # Baselines are calibrated for real embeddings; under fakes the dense
    # vectors are random, so print the table without enforcing.
    enforce = not fake
    if not enforce:
        print("(baselines not enforced under RAG_DEMO_FAKE=1)")
    print(f"{'mode':<15}{'hit-rate':<10}{'baseline':<10}")
    for name, hybrid, rerank in modes:
        settings = get_settings()
        if hybrid:
            settings.retrieval = "hybrid"
        settings.rerank_enabled = rerank
        rate = hit_rate(settings, golden)
        baseline = baselines.get(name, 0.0)
        good = rate >= baseline if enforce else True
        ok = ok and good
        print(f"{name:<15}{rate:<10.2f}{baseline:<10.2f}{'OK' if good else 'REGRESSION'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
