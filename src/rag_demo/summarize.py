"""LLM summarization for uploaded documents.

Short docs: one call. Long docs: map-reduce — summarize each chunk, then
summarize the summaries. Works with any langchain LLM (fake LLM in tests).

Robustness, in order of preference:
1. Transient LLM errors (connection blips, HTTP 5xx) are retried with backoff.
2. Chunks that still fail — or time out — are passed through RAW (truncated)
   so the reduce step still sees their content. Nothing is silently dropped.
3. The reduce step recurses when its combined input exceeds
   REDUCE_LIMIT_CHARS, so very long documents can't blow up one giant call.

Timeouts are deliberately NOT retried: a chunk that burns the full timeout
once is unlikely to succeed on an immediate retry, and each attempt costs up
to `timeout` seconds. It goes straight to the raw fallback instead.
"""

from __future__ import annotations

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_llm, get_settings

SINGLE_CALL_CHARS = 6000
CHUNK_CHARS = 4000
REDUCE_LIMIT_CHARS = 12000
RAW_FALLBACK_CHARS = 2000
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (5, 15)

SUMMARY_PROMPT = """Summarize the document below. Lead with the key points, then list any dates, deadlines, owners, or numbers named. Be concise.

Document:
{text}

Summary:"""

MAP_PROMPT = """Summarize this section of a longer document in 3-5 bullet points. Reply with ONLY the bullets.

Section:
{chunk}"""

REDUCE_PROMPT = """Combine these section summaries into one coherent summary. Lead with the key points, then any dates, deadlines, owners, or numbers. Be concise.

Section summaries:
{summaries}

Summary:"""

RAW_SECTION_TEMPLATE = (
    "[Unprocessed section — the summarizer could not process it in time; "
    "raw text follows]\n{chunk}"
)


def _invoke_once(llm, prompt: str, timeout: float | None = None) -> tuple[str, str]:
    """Single LLM attempt.

    Returns (text, status) where status is "ok", "timeout", or "error".
    """
    import concurrent.futures
    import os

    from .chain import strip_think

    if timeout is None:
        timeout = float(os.environ.get("RAG_DEMO_SUMMARY_TIMEOUT", "600"))

    def _do():
        out = llm.invoke(prompt)
        text = out.content if hasattr(out, "content") else str(out)
        return strip_think(text.strip())

    # NOTE: not using a context manager — on timeout we abandon the stuck
    # thread via shutdown(wait=False) instead of blocking until Ollama
    # finishes (which is what `with` does and why the UI hung forever).
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_do)
    try:
        return future.result(timeout=timeout), "ok"
    except concurrent.futures.TimeoutError:
        return "", "timeout"
    except Exception:
        return "", "error"
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _invoke(llm, prompt: str, timeout: float | None = None) -> str:
    """Invoke the LLM with a timeout; returns '' on timeout/failure."""
    text, _ = _invoke_once(llm, prompt, timeout)
    return text


def _invoke_with_retry(
    llm,
    prompt: str,
    timeout: float | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
) -> tuple[str, str]:
    """Invoke with retries for transient errors.

    Only "error" outcomes are retried — timeouts go straight to the caller
    (see module docstring for why). Returns (text, status).
    """
    text, status = _invoke_once(llm, prompt, timeout)
    for wait in backoff[: max(0, attempts - 1)]:
        if status != "error":
            break
        time.sleep(wait)
        text, status = _invoke_once(llm, prompt, timeout)
    return text, status


def _reduce_summaries(partials: list[str], llm, timeout: float | None = None) -> str:
    """Combine chunk summaries into one; recurse when the combined input is
    too large for a single comfortable reduce call."""
    combined = "\n\n".join(partials)
    if len(combined) <= REDUCE_LIMIT_CHARS or len(partials) <= 1:
        result, _ = _invoke_with_retry(llm, REDUCE_PROMPT.format(summaries=combined), timeout)
        return result
    mid = len(partials) // 2
    left = _reduce_summaries(partials[:mid], llm, timeout)
    right = _reduce_summaries(partials[mid:], llm, timeout)
    merged = [s for s in (left, right) if s]
    if not merged:
        return ""
    return _reduce_summaries(merged, llm, timeout)


def summarize(text: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    llm = get_llm(settings)
    if len(text) <= SINGLE_CALL_CHARS:
        result, _ = _invoke_with_retry(llm, SUMMARY_PROMPT.format(text=text))
        return result or "Summary didn't complete — the model may be overloaded. Try again."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    partials = []
    raw_fallbacks = 0
    for c in chunks:
        p, _ = _invoke_with_retry(llm, MAP_PROMPT.format(chunk=c))
        if p:
            partials.append(p)
        else:
            # Never silently drop content: the reduce step sees the raw
            # section instead of nothing.
            partials.append(RAW_SECTION_TEMPLATE.format(chunk=c[:RAW_FALLBACK_CHARS]))
            raw_fallbacks += 1
    summary = _reduce_summaries(partials, llm)
    if not summary:
        return "Summary didn't complete — the model may be overloaded. Try again."
    if raw_fallbacks:
        summary += (
            f"\n\n_Note: {raw_fallbacks} of {len(chunks)} sections couldn't be "
            "summarized in time and were included unprocessed._"
        )
    return summary
