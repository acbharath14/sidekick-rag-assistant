"""CLI for agentic RAG mode.

Usage:
    PYTHONPATH=src python -m rag_demo.agentic_cli "Compare token expiry across services"
    PYTHONPATH=src python -m rag_demo.agentic_cli --verbose "Your question"
"""

from __future__ import annotations

import argparse
import sys

from .agent import agentic_ask
from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG: plan-act-verify-refine")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--verbose", action="store_true", help="Show plan/verify steps")
    parser.add_argument("--max-iterations", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    if args.question:
        questions = [args.question]
    else:
        print("Agentic RAG mode. Type questions (or /quit):")
        questions = []
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("/quit", "/exit", "quit"):
                break
            if q:
                questions.append(q)

    for q in questions:
        print(f"\n🤖 Agentic RAG: {q}\n")
        answer = agentic_ask(
            q, settings,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
        )
        print(answer.text)
        if answer.sources:
            print(f"\nSources: {', '.join(answer.sources)}")
        print(f"⚡ {answer.latency_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
