"""Retrieval quality eval: hit-rate@k against the golden Q&A set.

Usage:
    RAG_DEMO_FAKE=1 python -m evals.eval_retrieval            # deterministic, no models
    python -m evals.eval_retrieval                            # real embeddings (downloads model once)

Exit code 0 when hit-rate@k meets the threshold, 1 otherwise — CI-friendly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import get_settings  # noqa: E402
from rag_demo.retriever import Retriever  # noqa: E402

THRESHOLD = float(os.environ.get("RAG_DEMO_EVAL_THRESHOLD", "0.8"))


def main(settings=None) -> int:
    golden = json.loads(Path(__file__).with_name("golden.json").read_text())
    settings = settings or get_settings()
    retriever = Retriever(settings)
    k = settings.top_k
    hits = 0
    for item in golden:
        got = {h.source for h in retriever.search(item["question"], k=k)}
        expected = set(item["expected_sources"])
        ok = bool(got & expected)
        hits += ok
        mark = "OK " if ok else "MISS"
        print(f"[{mark}] {item['question'][:70]} -> {sorted(got)}")
    rate = hits / len(golden)
    print(f"\nhit-rate@{k}: {rate:.2f} ({hits}/{len(golden)}), threshold {THRESHOLD:.2f}")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
