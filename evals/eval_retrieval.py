"""Retrieval quality eval: hit-rate@k against the golden Q&A set.

Usage:
    RAG_DEMO_FAKE=1 python -m evals.eval_retrieval            # deterministic, no models
    python -m evals.eval_retrieval                            # real embeddings (downloads model once)
    python -m evals.eval_retrieval --retrieval hybrid         # hybrid BM25+dense
    python -m evals.eval_retrieval --retrieval hybrid --rerank # + cross-encoder (real only)

Exit code 0 when hit-rate@k meets the threshold, 1 otherwise — CI-friendly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.config import get_settings  # noqa: E402
from rag_demo.retriever import Retriever  # noqa: E402

THRESHOLD = float(os.environ.get("RAG_DEMO_EVAL_THRESHOLD", "0.8"))


def main(settings=None, retrieval: str | None = None, rerank: bool = False) -> int:
    golden = json.loads(Path(__file__).with_name("golden.json").read_text())
    settings = settings or get_settings()
    if retrieval:
        settings.retrieval = retrieval
    if rerank:
        settings.rerank_enabled = True
    retriever = Retriever(settings)
    k = settings.top_k
    hits = 0
    mode = settings.retrieval + ("+rerank" if settings.rerank_enabled else "")
    for item in golden:
        got = {h.source for h in retriever.search(item["question"], k=k)}
        expected = set(item["expected_sources"])
        ok = bool(got & expected)
        hits += ok
        mark = "OK " if ok else "MISS"
        print(f"[{mark}] {item['question'][:70]} -> {sorted(got)}")
    rate = hits / len(golden)
    print(f"\nhit-rate@{k} [{mode}]: {rate:.2f} ({hits}/{len(golden)}), threshold {THRESHOLD:.2f}")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default=None)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(retrieval=args.retrieval, rerank=args.rerank))
