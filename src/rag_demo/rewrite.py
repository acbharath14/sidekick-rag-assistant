"""Query rewriting for multi-turn chat.

A follow-up like "what about express?" is unanswerable on its own; rewriting
it against the recent history ("What is the max parcel weight?") produces a
standalone question the retriever can handle. No-op when there is no history,
so single-turn latency is unchanged.
"""

from __future__ import annotations

REWRITE_PROMPT = """Rewrite the follow-up question as a standalone question, using the chat history for context. Reply with ONLY the rewritten question, no preamble.

Chat history:
{history}

Follow-up: {question}

Standalone question:"""


def format_history(history: list[tuple[str, str]], turns: int = 3) -> str:
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-turns:])


def rewrite_query(question: str, history: list[tuple[str, str]], llm) -> str:
    if not history:
        return question
    prompt = REWRITE_PROMPT.format(history=format_history(history), question=question)
    out = llm.invoke(prompt)
    text = out.content if hasattr(out, "content") else str(out)
    # Strip reasoning blocks (qwen3 et al. emit <think>…</think>); the
    # retriever needs just the standalone question.
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Take the last non-empty line — models sometimes echo the prompt.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = lines[-1] if lines else text
    return text.strip().strip('"').removeprefix("Standalone question:").strip()
