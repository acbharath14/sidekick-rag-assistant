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

Streaming: `summarize_events` yields progress events for SSE consumers
(start / progress / done / error). `summarize` is a thin wrapper that runs
the same generator to completion, so both paths share one code path.

Parallel map (opt-in, RAG_DEMO_SUMMARIZE_PARALLEL_MAP=1): the map calls
run concurrently instead of one at a time. Same prompts, same model,
same reduce input order, so the summary is unchanged; only the wall
time can change. Default off: whether it helps depends on the Ollama
server's parallel-sequence capacity on the target hardware.

Timeouts and retries go through rag_demo.ollama_guard and Settings:
the per-call deadline is settings.summarize_timeout_seconds (600s,
unchanged), attempts and backoff are settings.llm_retry_attempts and
settings.llm_retry_backoff_seconds, and every call is recorded in the
shared Ollama tracker that /api/health exposes.
"""

from __future__ import annotations

import logging
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import tracing
from .config import Settings, get_llm, get_settings
from .observability import record_ollama_error, record_ollama_timeout
from .ollama_guard import TRACKER, track_call

logger = logging.getLogger(__name__)


def _record_llm_outcome(status: str) -> None:
    """Count an LLM call that did not return text. Only the outcome is
    counted here; the API layer counts the summarize itself, so one
    summarize is never counted twice."""
    if status == "timeout":
        record_ollama_timeout()
    elif status == "error":
        record_ollama_error()

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


def _invoke_once(
    llm, prompt: str, timeout: float | None = None, run_name: str | None = None
) -> tuple[str, str, BaseException | None]:
    """Single LLM attempt.

    Returns (text, status, exc) where status is "ok", "timeout", or
    "error" and exc is the failure when status is "error". The call is
    recorded in the shared Ollama tracker, like every other LLM call.
    run_name names the Langfuse run when tracing is on (e.g.
    "summarize-single"); None leaves the call untraced.
    """
    import concurrent.futures
    import os

    from .chain import strip_think

    if timeout is None:
        timeout = float(os.environ.get("RAG_DEMO_SUMMARY_TIMEOUT", "600"))

    def _do():
        # Runs in a worker thread, so its Langfuse run shows as its own
        # trace instead of nesting under the question trace. The timing
        # is still there.
        out = tracing.invoke_llm(llm, prompt, run_name=run_name)
        text = out.content if hasattr(out, "content") else str(out)
        return strip_think(text.strip())

    # NOTE: not using a context manager for the executor — on timeout we
    # abandon the stuck thread via shutdown(wait=False) instead of
    # blocking until Ollama finishes (which is what `with` does and why
    # the UI hung forever). The submit happens inside track_call so a
    # configured concurrency cap actually gates the call.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with track_call("summarize", acquire_timeout=timeout) as record:
            future = ex.submit(_do)
            try:
                return future.result(timeout=timeout), "ok", None
            except concurrent.futures.TimeoutError:
                record.timed_out = True
                return "", "timeout", None
            except Exception as e:
                record.errored = True
                return "", "error", e
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _invoke(llm, prompt: str, timeout: float | None = None, run_name: str | None = None) -> str:
    """Invoke the LLM with a timeout; returns '' on timeout/failure."""
    text, _, _ = _invoke_once(llm, prompt, timeout, run_name=run_name)
    return text


def _invoke_with_retry(
    llm,
    prompt: str,
    timeout: float | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
    run_name: str | None = None,
) -> tuple[str, str]:
    """Invoke with retries for transient errors.

    Only "error" outcomes are retried — timeouts go straight to the caller
    (see module docstring for why). Each retry is logged. Returns
    (text, status).
    """
    text, status, exc = _invoke_once(llm, prompt, timeout, run_name=run_name)
    attempt = 1
    for wait in backoff[: max(0, attempts - 1)]:
        if status != "error":
            break
        if exc is not None:
            TRACKER.record_retry("summarize", attempt, attempts, exc)
        else:
            logger.warning(
                "Summarize LLM call failed, retry %d/%d", attempt, attempts
            )
        attempt += 1
        time.sleep(wait)
        text, status, exc = _invoke_once(llm, prompt, timeout, run_name=run_name)
    return text, status


def _reduce_summaries(
    partials: list[str],
    llm,
    timeout: float | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
) -> str:
    """Combine chunk summaries into one; recurse when the combined input is
    too large for a single comfortable reduce call."""
    combined = "\n\n".join(partials)
    if len(combined) <= REDUCE_LIMIT_CHARS or len(partials) <= 1:
        result, status = _invoke_with_retry(
            llm,
            REDUCE_PROMPT.format(summaries=combined),
            timeout,
            attempts,
            backoff,
            run_name="summarize-reduce",
        )
        _record_llm_outcome(status)
        return result
    mid = len(partials) // 2
    left = _reduce_summaries(partials[:mid], llm, timeout, attempts, backoff)
    right = _reduce_summaries(partials[mid:], llm, timeout, attempts, backoff)
    merged = [s for s in (left, right) if s]
    if not merged:
        return ""
    return _reduce_summaries(merged, llm, timeout, attempts, backoff)


def _summarize_one_chunk(
    llm,
    chunk: str,
    timeout: float | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
) -> tuple[str, bool]:
    """Summarize one map chunk. Returns (partial, was_raw_fallback).

    Shared by the sequential and parallel map paths, so both run the
    same retries and the same raw fallback: a chunk that still fails is
    passed through raw (truncated) instead of being silently dropped.
    """
    p, status = _invoke_with_retry(
        llm,
        MAP_PROMPT.format(chunk=chunk),
        timeout,
        attempts,
        backoff,
        run_name="summarize-map",
    )
    _record_llm_outcome(status)
    if p:
        return p, False
    return RAW_SECTION_TEMPLATE.format(chunk=chunk[:RAW_FALLBACK_CHARS]), True


def _run_map_parallel(
    llm,
    chunks: list[str],
    timeout: float | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
):
    """Run the map calls concurrently. Yields progress events; returns
    (partials in chunk order, raw_fallback_count) via StopIteration.

    The map calls are independent (same prompt shape, same model), so
    running them together changes only the wall time, not the summary:
    each result is placed back at its chunk index, and the reduce step
    sees the same ordered list the sequential loop would have built.
    Workers are capped at 4: beyond that a CPU-only Ollama server just
    queues the extra sequences. A failure inside a worker propagates
    like a sequential failure (the outer generator turns it into an
    error event).
    """
    import concurrent.futures

    partials: list[str | None] = [None] * len(chunks)
    raw_fallbacks = 0
    done = 0
    workers = min(len(chunks), 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_summarize_one_chunk, llm, c, timeout, attempts, backoff): i
            for i, c in enumerate(chunks)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            partial, was_raw = fut.result()
            partials[i] = partial
            raw_fallbacks += 1 if was_raw else 0
            done += 1
            yield {"type": "progress", "done": done, "total": len(chunks), "stage": "map"}
    return [p if p is not None else "" for p in partials], raw_fallbacks


def summarize_events(text: str, settings: Settings | None = None):
    """Yield progress events for SSE streaming.

    Events (dicts):
      {"type": "start", "mode": "single" | "map_reduce", "total": n}
      {"type": "progress", "done": i, "total": n, "stage": "map"}
      {"type": "progress", "stage": "reduce"}
      {"type": "done", "summary": "..."}
      {"type": "error", "error": "..."}
    """
    try:
        settings = settings or get_settings()
        # Tests pass a bare object() as settings, so read through getattr
        # with the Settings defaults; a real Settings always has these.
        timeout = getattr(settings, "summarize_timeout_seconds", 600.0)
        attempts = getattr(settings, "llm_retry_attempts", RETRY_ATTEMPTS)
        backoff = getattr(settings, "llm_retry_backoff_seconds", RETRY_BACKOFF_SECONDS)
        llm = get_llm(settings)
        if len(text) <= SINGLE_CALL_CHARS:
            yield {"type": "start", "mode": "single", "total": 1}
            result, status = _invoke_with_retry(
                llm,
                SUMMARY_PROMPT.format(text=text),
                timeout,
                attempts,
                backoff,
                run_name="summarize-single",
            )
            _record_llm_outcome(status)
            summary = result or "Summary didn't complete — the model may be overloaded. Try again."
            yield {"type": "done", "summary": summary}
            return
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CHARS,
            chunk_overlap=200,
        )
        chunks = splitter.split_text(text)
        yield {"type": "start", "mode": "map_reduce", "total": len(chunks)}
        partials = []
        raw_fallbacks = 0
        parallel = getattr(settings, "summarize_parallel_map", False)
        if parallel and len(chunks) > 1:
            partials, raw_fallbacks = yield from _run_map_parallel(
                llm, chunks, timeout, attempts, backoff
            )
        else:
            for i, c in enumerate(chunks, 1):
                p, was_raw = _summarize_one_chunk(llm, c, timeout, attempts, backoff)
                partials.append(p)
                raw_fallbacks += 1 if was_raw else 0
                yield {"type": "progress", "done": i, "total": len(chunks), "stage": "map"}
        yield {"type": "progress", "stage": "reduce"}
        summary = _reduce_summaries(
            partials,
            llm,
            timeout,
            attempts,
            backoff,
        )
        if not summary:
            summary = "Summary didn't complete — the model may be overloaded. Try again."
        elif raw_fallbacks:
            summary += (
                f"\n\n_Note: {raw_fallbacks} of {len(chunks)} sections couldn't be "
                "summarized in time and were included unprocessed._"
            )
        yield {"type": "done", "summary": summary}
    except Exception as e:
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}


def summarize(text: str, settings: Settings | None = None) -> str:
    """Blocking wrapper around summarize_events — one shared code path."""
    summary: str | None = None
    for event in summarize_events(text, settings):
        if event["type"] == "done":
            summary = event["summary"]
        elif event["type"] == "error":
            raise RuntimeError(event["error"])
    return summary or "Summary didn't complete — the model may be overloaded. Try again."
