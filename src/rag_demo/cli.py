"""Conversational CLI over the RAG index."""

from __future__ import annotations

from .chain import ask
from .config import get_settings


def main() -> None:
    settings = get_settings()
    print("Meridian Logistics RAG assistant (type /quit to exit)")
    print(f"LLM: {settings.llm_provider} | embeddings: {settings.embedding_provider}")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/q"}:
            break
        answer = ask(question, settings)
        print(f"\n{answer.text}")
        if answer.sources:
            print(f"\nSources: {', '.join(answer.sources)}")


if __name__ == "__main__":
    main()
