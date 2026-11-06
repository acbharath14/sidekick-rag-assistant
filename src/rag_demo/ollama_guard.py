"""Timeouts, bounded retries, and queue visibility for Ollama calls.

One choke point for every LLM call in the backend (chat, streaming,
summarization, the agent loop). Stdlib only, no new dependencies.

Policy (same as summarize.py, now shared everywhere):

1. Every Ollama call runs in a worker thread with a deadline. On timeout
   the stuck thread is abandoned, never joined, and the caller gets an
   OllamaTimeoutError with a plain message. A timed-out request fails
   fast instead of hanging the HTTP request forever.
2. Transient errors (connection errors, HTTP 5xx) are retried with
   backoff, bounded by the attempt count. Each retry is logged.
3. Timeouts are NEVER retried. A call that burned the full timeout once
   is unlikely to succeed on an immediate retry, and each attempt costs
   up to `timeout` seconds. Client-side timeouts and server-side 5xx
   are different things: a 504 status from Ollama is a 5xx and is
   retried, while our own deadline expiring is not.
4. Queue visibility: a process-wide tracker counts in-flight calls,
   calls that started while another was already running, total waits,
   timeouts, retries, and errors. /api/health exposes the snapshot,
   so a backed-up Ollama is observable from outside the process.

Concurrency cap (opt-in): set RAG_DEMO_OLLAMA_MAX_CONCURRENT to a
positive integer to serialize generations through a semaphore (useful
on the CPU-only iMac, where concurrent generations contend). Default 0
means no cap and no behavior change. When capped, a call waits for a
slot up to its own call timeout, then fails with OllamaTimeoutError;
the wait is recorded in the tracker snapshot.
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import re
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DEFAULT_LLM_TIMEOUT_SECONDS = 300.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = (5.0, 15.0)

_TRANSIENT_TYPE_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "RemoteDisconnected",
        "URLError",
        "ConnectTimeout",  # httpx connect timeout: the server never answered
    }
)
_TRANSIENT_MESSAGE_FRAGMENTS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
    "name resolution",
    "network is unreachable",
    "broken pipe",
    "remote end closed",
    "service unavailable",
    "bad gateway",
)
_SERVER_ERROR_RE = re.compile(r"\b(500|502|503|504)\b")
_SENTINEL = object()


class OllamaTimeoutError(TimeoutError):
    """An Ollama call (or the wait for a call slot) exceeded its deadline.

    Subclasses TimeoutError, so existing `except TimeoutError` handlers
    still catch it. Never retried: see the module docstring.
    """


def is_timeout_error(exc: BaseException) -> bool:
    """True when the error is a client-side timeout (our deadline)."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__
    if name in {
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }:
        return True
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg or "deadline exceeded" in msg


def is_transient_error(exc: BaseException) -> bool:
    """True when the error is worth one bounded retry.

    Connection errors and HTTP 5xx only. Timeouts are excluded here on
    purpose: is_timeout_error wins, so a timeout is never retried.
    """
    if is_timeout_error(exc):
        return False
    if isinstance(exc, (ConnectionError, BrokenPipeError)):
        return True
    if type(exc).__name__ in _TRANSIENT_TYPE_NAMES:
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    msg = str(exc).lower()
    if _SERVER_ERROR_RE.search(msg):
        return True
    return any(frag in msg for frag in _TRANSIENT_MESSAGE_FRAGMENTS)


class OllamaTracker:
    """Thread-safe counters for in-flight Ollama calls and queue waits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._in_flight = 0
            self._max_in_flight = 0
            self._calls_total = 0
            self._timeouts_total = 0
            self._retries_total = 0
            self._errors_total = 0
            self._queued_behind_total = 0
            self._waits_total = 0
            self._total_wait_seconds = 0.0
            self._max_wait_seconds = 0.0
            self._total_call_seconds = 0.0
            self._max_call_seconds = 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "in_flight": self._in_flight,
                "max_in_flight": self._max_in_flight,
                "calls_total": self._calls_total,
                "timeouts_total": self._timeouts_total,
                "retries_total": self._retries_total,
                "errors_total": self._errors_total,
                "queued_behind_total": self._queued_behind_total,
                "waits_total": self._waits_total,
                "total_wait_seconds": round(self._total_wait_seconds, 3),
                "max_wait_seconds": round(self._max_wait_seconds, 3),
                "max_call_seconds": round(self._max_call_seconds, 3),
                "total_call_seconds": round(self._total_call_seconds, 3),
            }

    def record_retry(self, label: str, attempt: int, attempts: int, exc: BaseException) -> None:
        with self._lock:
            self._retries_total += 1
        logger.warning(
            "Ollama call '%s' failed (%s: %s), retry %d/%d",
            label,
            type(exc).__name__,
            exc,
            attempt,
            attempts,
        )


class _CallRecord:
    """Outcome flags for one tracked call (see track_call)."""

    def __init__(self) -> None:
        self.timed_out = False
        self.errored = False


TRACKER = OllamaTracker()

# Optional concurrency cap, sized from the environment on first use.
# Recreated if the env value changes (in practice set once per process).
_sem_lock = threading.Lock()
_semaphore: threading.Semaphore | None = None
_semaphore_limit = 0


def _concurrency_limit() -> int:
    import os

    try:
        return max(0, int(os.environ.get("RAG_DEMO_OLLAMA_MAX_CONCURRENT", "0")))
    except ValueError:
        return 0


def _get_semaphore(limit: int) -> threading.Semaphore | None:
    global _semaphore, _semaphore_limit
    if limit <= 0:
        return None
    with _sem_lock:
        if _semaphore is None or _semaphore_limit != limit:
            _semaphore = threading.Semaphore(limit)
            _semaphore_limit = limit
        return _semaphore


@contextmanager
def track_call(label: str, acquire_timeout: float | None = None):
    """Record one Ollama call: in-flight count, queue wait, outcome.

    Yields a small record object. Callers that handle failures by
    returning a status instead of raising (summarize.py) mark the
    outcome on it: record.timed_out = True or record.errored = True.
    Raised OllamaTimeoutError and other exceptions are recorded
    automatically.

    When the concurrency cap is set, waits for a slot up to
    acquire_timeout seconds (callers pass the call timeout) and raises
    OllamaTimeoutError instead of queueing forever. Logs a line when
    the call starts behind others, so a backed-up Ollama is visible
    in the logs as well as in /api/health.
    """
    record = _CallRecord()
    sem = _get_semaphore(_concurrency_limit())
    wait_seconds = 0.0
    if sem is not None:
        started_wait = time.monotonic()
        acquired = sem.acquire(timeout=acquire_timeout)
        wait_seconds = time.monotonic() - started_wait
        if not acquired:
            with TRACKER._lock:
                TRACKER._timeouts_total += 1
            raise OllamaTimeoutError(
                f"Ollama call '{label}' timed out waiting for a free "
                f"slot after {acquire_timeout:g} seconds: Ollama is busy. "
                "Try again in a moment."
            )
    with TRACKER._lock:
        TRACKER._in_flight += 1
        TRACKER._max_in_flight = max(TRACKER._max_in_flight, TRACKER._in_flight)
        queued_behind = TRACKER._in_flight > 1
        if queued_behind:
            TRACKER._queued_behind_total += 1
    if queued_behind:
        logger.warning(
            "Ollama busy: '%s' started with %d call(s) already in flight",
            label,
            TRACKER._in_flight - 1,
        )
    if wait_seconds >= 0.01:
        # Sub-10ms semaphore handoffs are noise, not a queue: only
        # record waits a human would notice.
        with TRACKER._lock:
            TRACKER._waits_total += 1
            TRACKER._total_wait_seconds += wait_seconds
            TRACKER._max_wait_seconds = max(TRACKER._max_wait_seconds, wait_seconds)
        logger.warning(
            "Ollama queue: '%s' waited %.1f seconds for a slot",
            label,
            wait_seconds,
        )
    started = time.monotonic()
    try:
        yield record
    except OllamaTimeoutError:
        record.timed_out = True
        raise
    except BaseException:
        record.errored = True
        raise
    finally:
        duration = time.monotonic() - started
        with TRACKER._lock:
            TRACKER._in_flight -= 1
            TRACKER._calls_total += 1
            TRACKER._total_call_seconds += duration
            TRACKER._max_call_seconds = max(TRACKER._max_call_seconds, duration)
            if record.timed_out:
                TRACKER._timeouts_total += 1
            if record.errored:
                TRACKER._errors_total += 1
        if sem is not None:
            sem.release()


def ollama_snapshot() -> dict:
    """Current tracker counters, for /api/health."""
    return TRACKER.snapshot()


def reset_ollama_tracker() -> None:
    """Reset all tracker counters. Tests only."""
    TRACKER.reset()


def _llm_policy(settings) -> tuple[float, int, tuple[float, ...]]:
    """Timeout, attempts, backoff for one LLM call from Settings."""
    if settings is None:
        return (
            DEFAULT_LLM_TIMEOUT_SECONDS,
            DEFAULT_RETRY_ATTEMPTS,
            DEFAULT_RETRY_BACKOFF_SECONDS,
        )
    timeout = float(getattr(settings, "llm_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS))
    attempts = int(getattr(settings, "llm_retry_attempts", DEFAULT_RETRY_ATTEMPTS))
    backoff = getattr(
        settings, "llm_retry_backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS
    )
    return timeout, max(1, attempts), tuple(backoff) if backoff else (0.0,)


def call_with_timeout_and_retry(
    fn,
    *,
    timeout: float,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    backoff: tuple[float, ...] = DEFAULT_RETRY_BACKOFF_SECONDS,
    label: str = "ollama",
):
    """Run fn() with a deadline, retrying transient errors with backoff.

    Timeouts raise OllamaTimeoutError and are never retried. Other
    errors are retried only when is_transient_error says so, at most
    `attempts` total tries. The stuck thread is abandoned on timeout,
    never joined, so a hung Ollama cannot hang the caller.
    """
    attempts = max(1, int(attempts))
    backoff = tuple(backoff) if backoff else (0.0,)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            with track_call(label, acquire_timeout=timeout):
                # NOTE: not using a context manager for the executor. On
                # timeout we abandon the stuck thread via
                # shutdown(wait=False) instead of blocking until Ollama
                # finishes (same pattern as summarize._invoke_once).
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = ex.submit(fn)
                try:
                    return future.result(timeout=timeout)
                except OllamaTimeoutError:
                    raise
                except TimeoutError:
                    raise OllamaTimeoutError(
                        f"Ollama call '{label}' timed out after "
                        f"{timeout:g} seconds: the model took too long. "
                        "Try again in a moment."
                    )
                finally:
                    ex.shutdown(wait=False, cancel_futures=True)
        except OllamaTimeoutError:
            raise
        except Exception as e:
            last_exc = e
            if is_transient_error(e) and attempt < attempts:
                TRACKER.record_retry(label, attempt, attempts, e)
                wait = backoff[min(attempt - 1, len(backoff) - 1)]
                if wait > 0:
                    time.sleep(wait)
                continue
            raise
    if last_exc is not None:  # pragma: no cover: loop always raises or returns
        raise last_exc
    raise RuntimeError(f"Ollama call '{label}' failed without an error")


def invoke_llm(llm, prompt: str, settings=None, *, label: str = "llm") -> str:
    """One LLM call with the shared timeout and retry policy.

    Returns the reply text (.content when present, else str(out)).
    Raises OllamaTimeoutError on timeout, or the last transient error
    once retries are exhausted.
    """
    timeout, attempts, backoff = _llm_policy(settings)

    def _do() -> str:
        out = llm.invoke(prompt)
        return out.content if hasattr(out, "content") else str(out)

    return call_with_timeout_and_retry(
        _do, timeout=timeout, attempts=attempts, backoff=backoff, label=label
    )


def run_with_timeout(fn, timeout: float, *, label: str = "request"):
    """Run fn() with a request-level deadline.

    A timeout of 0 or less disables the deadline and calls fn()
    directly. On timeout raises OllamaTimeoutError; the stuck thread
    is abandoned, never joined.
    """
    if not timeout or timeout <= 0:
        return fn()
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(fn)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        raise OllamaTimeoutError(
            f"Request '{label}' timed out after {timeout:g} seconds: "
            "the model took too long. Try again in a moment."
        )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def iter_with_idle_timeout(iterator, idle_timeout: float, *, label: str = "stream"):
    """Yield from iterator, failing fast when it stalls.

    If no item arrives within idle_timeout seconds, raises
    OllamaTimeoutError and abandons the pump thread (daemon), so a
    stalled stream cannot hang the SSE response forever.
    """
    q: queue.Queue = queue.Queue()

    def _pump() -> None:
        try:
            for item in iterator:
                q.put(item)
            q.put(_SENTINEL)
        except BaseException as e:  # noqa: BLE001: re-raised to the consumer
            q.put(e)

    def _generate():
        thread = threading.Thread(target=_pump, daemon=True)
        with track_call(label, acquire_timeout=idle_timeout):
            thread.start()
            while True:
                try:
                    item = q.get(timeout=idle_timeout)
                except queue.Empty:
                    raise OllamaTimeoutError(
                        f"Ollama stream '{label}' stalled: no output for "
                        f"{idle_timeout:g} seconds. Try again in a moment."
                    )
                if item is _SENTINEL:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item

    return _generate()
