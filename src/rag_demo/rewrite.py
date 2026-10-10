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
    # Run the rewrite in a thread with a timeout — a stalled/slow LLM must not
    # hang the follow-up forever. Falls back to the raw question on timeout.
    import concurrent.futures

    def _do_rewrite() -> str:
        prompt = REWRITE_PROMPT.format(
            history=format_history(history), question=question
        )
        try:
            out = llm.invoke(prompt)
        except Exception:
            # Rewrite failed (e.g. LLM unreachable) — fall back to the raw
            # question rather than breaking the follow-up entirely.
            return question
        text = out.content if hasattr(out, "content") else str(out)
        # Strip reasoning blocks (qwen3 et al. emit <think>…</think>); the
        # retriever needs just the standalone question.
        import re

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
        # Take the last non-empty line — models sometimes echo the prompt.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = lines[-1] if lines else text
        text = text.strip().strip('"').removeprefix("Standalone question:").strip()
        # Guard against empty/garbage rewrites — a bad standalone question is
        # worse than the original follow-up.
        if not text or len(text) < 3:
            return question
        return text

    # NOTE: not using a context manager — on timeout we abandon the stuck
    # thread via shutdown(wait=False) instead of blocking until Ollama
    # finishes.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_do_rewrite)
    try:
        return future.result(timeout=90)
    except concurrent.futures.TimeoutError:
        # Rewrite took too long — answer the follow-up as-is.
        return question
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
