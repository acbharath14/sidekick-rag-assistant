"""Conversational CLI over the RAG index."""

from __future__ import annotations

from .chain import ask
from .config import get_settings


def main() -> None:
    settings = get_settings()
    print("Meridian Logistics RAG assistant (type /quit to exit)")
    print(f"LLM: {settings.llm_provider} | embeddings: {settings.embedding_provider}")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/q"}:
            break
        answer = ask(question, settings, history=history)
        print(f"\n{answer.text}")
        if answer.sources:
            print(f"\nSources: {', '.join(answer.sources)}")
        history.append((question, answer.text))
        history = history[-6:]


if __name__ == "__main__":
    main()
