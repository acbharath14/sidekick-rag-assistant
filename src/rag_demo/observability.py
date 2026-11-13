"""Structured logging and in-memory metrics for the FastAPI backend.

Local surface only: one JSON log line per event on stdout, plus
GET /api/metrics. No external service, no new dependency (stdlib only).
This complements opt-in Langfuse tracing (tracing.py): tracing shows
traces in a cloud UI, this shows latency and health here, in the
process, with no keys set. The Nov 3 latency review had to be done by
reading code; these logs are what makes the next one readable.

Privacy (read before changing):
- NEVER log secrets, API keys, or full document text.
- Question text is logged TRUNCATED to the first QUESTION_PREVIEW_CHARS
  characters (whitespace collapsed), never the full question, and common
  secret shapes inside the preview are redacted (see _scrub_secrets).
  A truncated preview is enough to recognise the question in the log.
  The full question already lives in the opt-in audit log (audit.py)
  when the user enables it, and nowhere else.
- Answer text is never logged. Retrieved hits are logged as source
  names plus a count, never passage text. Filenames are logged as
  basenames, truncated. Exception messages are truncated and scrubbed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone

QUESTION_PREVIEW_CHARS = 120
ERROR_PREVIEW_CHARS = 200
FILENAME_PREVIEW_CHARS = 80
LOG_LEVEL = os.environ.get("RAG_DEMO_LOG_LEVEL", "INFO").upper()
MAX_SAMPLES = 1000  # latency samples kept per step (p95 is computed over these)

# Latency steps reported by /api/metrics. "rerank" is deliberately
# absent: reranking runs inside retriever.search and cannot be timed
# without touching retriever.py, so its cost stays inside retrieve_ms.
LATENCY_STEPS = (
    "rewrite",
    "episodic",
    "retrieve",
    "generate",
    "total",
    "summarize",
    "upload",
)

_SECRET_PATTERNS = (
    # key=value style: api_key=..., token: ..., password=...
    # Keeps the key name, redacts the value: api_key=[redacted].
    (
        re.compile(
            r"([A-Za-z0-9_.\-]*?(?:api[_-]?key|secret|token|password))"
            r"(\s*[:=]\s*)([^\s,;]+)",
            re.IGNORECASE,
        ),
        r"\1\2[redacted]",
    ),
    # Long sk- style bearer tokens: the whole token goes.
    (re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,})\b"), "[redacted]"),
)


def _scrub_secrets(text: str) -> str:
    """Redact common secret shapes. Best effort, not a guarantee: the
    truncation rules above are the real guard, this only catches the
    obvious case of a key pasted at the start of a question."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def truncate_preview(text: str, limit: int = QUESTION_PREVIEW_CHARS) -> str:
    """Collapse whitespace, scrub secret shapes, truncate to `limit`."""
    collapsed = " ".join((text or "").split())
    return _scrub_secrets(collapsed)[:limit]


def new_request_id() -> str:
    """Short id tying one HTTP request to its chain log line."""
    return uuid.uuid4().hex[:12]


class _DynamicStdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout at emit time, not at
    construction time, so pytest's capsys (which swaps sys.stdout)
    captures the lines, and uvicorn still gets them on real stdout."""

    def emit(self, record):
        self.stream = sys.stdout
        super().emit(record)


def get_logger() -> logging.Logger:
    """One logger for all structured events. Configured once: JSON lines
    on stdout, no propagation (uvicorn's handlers must not double-log)."""
    logger = logging.getLogger("rag_demo")
    if not logger.handlers:
        handler = _DynamicStdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        logger.propagate = False
    return logger


def log_event(event: str, level: str = "INFO", **fields) -> None:
    """Emit one JSON log line. `fields` must already be truncated and
    secret-free; values are coerced with str() as a safety net."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
    }
    record.update(fields)
    get_logger().info(json.dumps(record, ensure_ascii=False, default=str))


def log_question_event(
    *,
    request_id: str,
    endpoint: str | None,
    model: str,
    question: str,
    timings: dict,
    hits: int,
    sources: list[str],
    grounded: bool,
    weak_evidence: bool,
    standalone: bool,
    streamed: bool,
    cache_reload: bool,
    route: str | None = None,
    route_confidence: float | None = None,
    route_ms: float | None = None,
) -> None:
    """One structured line per answered question (streaming included).
    Per-step timings are ms floats, or None when the step did not run
    (no history means no rewrite; small talk means no retrieval).
    generate_ms is chain time minus retrieve_ms, so it includes prompt
    formatting and (on main) retriever construction; it is an
    approximation, not a pure LLM timer."""
    log_event(
        "question",
        request_id=request_id,
        endpoint=endpoint or "direct",
        model=model,
        mode="stream" if streamed else "chat",
        question_preview=truncate_preview(question),
        rewrite_ms=timings.get("rewrite_ms"),
        episodic_ms=timings.get("episodic_ms"),
        retrieve_ms=timings.get("retrieve_ms"),
        retrieve_calls=timings.get("retrieve_calls", 0),
        rerank_ms=None,  # inside retrieve_ms; see LATENCY_STEPS note
        generate_ms=timings.get("generate_ms"),
        total_ms=timings.get("total_ms"),
        hits=hits,
        sources=sources,
        grounded=grounded,
        weak_evidence=weak_evidence,
        standalone_question=standalone,
        cache_reload=cache_reload,
        route=route,
        route_confidence=route_confidence,
        route_ms=route_ms,
    )


def log_upload_event(
    *,
    request_id: str,
    doc_id: str,
    filename: str,
    chunks: int,
    chars: int,
    size_bytes: int,
    extractor: str,
    latency_ms: float,
) -> None:
    """One line per successful upload. Filename only, never content."""
    import os as _os

    log_event(
        "upload",
        request_id=request_id,
        doc_id=doc_id,
        filename=truncate_preview(_os.path.basename(filename or "upload"),
                                   FILENAME_PREVIEW_CHARS),
        chunks=chunks,
        chars=chars,
        size_bytes=size_bytes,
        extractor=extractor,
        latency_ms=round(latency_ms, 1),
    )


def log_summarize_event(
    *,
    request_id: str,
    doc_id: str,
    mode: str,
    chars: int,
    latency_ms: float,
) -> None:
    log_event(
        "summarize",
        request_id=request_id,
        doc_id=doc_id,
        mode=mode,
        chars=chars,
        latency_ms=round(latency_ms, 1),
    )


def log_error_event(
    *,
    request_id: str | None,
    endpoint: str | None,
    where: str,
    error_type: str,
    error: str,
    model: str | None = None,
    **extra,
) -> None:
    """One line per error. The message is truncated and scrubbed; it
    must never carry document text or secrets."""
    fields: dict = {
        "request_id": request_id,
        "endpoint": endpoint or "direct",
        "where": where,
        "error_type": error_type,
        "error": truncate_preview(_scrub_secrets(str(error)),
                                  ERROR_PREVIEW_CHARS),
    }
    if model:
        fields["model"] = model
    fields.update(extra)
    log_event("error", level="ERROR", **fields)


def classify_error(exc: BaseException) -> str:
    """Classify an LLM-call failure for the metrics counters."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return "ollama_timeout"
    if "ollama" in text or "connection" in text or "connect" in text:
        return "ollama_error"
    return "error"


class Metrics:
    """In-memory counters and latency summaries. Stdlib only, thread
    safe. Latency samples are bounded (last MAX_SAMPLES per step);
    mean/p95 are computed on read from those samples."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "questions": 0,
            "questions_streamed": 0,
            "uploads": 0,
            "upload_errors": 0,
            "summarizes": 0,
            "summarize_errors": 0,
            "ollama_errors": 0,
            "ollama_timeouts": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_reloads": 0,
        }
        self._samples: dict[str, deque] = {
            step: deque(maxlen=MAX_SAMPLES) for step in LATENCY_STEPS
        }

    def _record_latency(self, step: str, value: float | None) -> None:
        if value is None:
            return
        self._samples[step].append(float(value))

    def record_question(self, timings: dict, *, streamed: bool) -> None:
        """Count one answered question and sample its per-step timings.
        Steps that did not run (None) are not sampled."""
        with self._lock:
            self._counters["questions"] += 1
            if streamed:
                self._counters["questions_streamed"] += 1
            for step in ("rewrite", "episodic", "retrieve", "generate",
                         "total"):
                self._record_latency(step, timings.get(f"{step}_ms"))

    def record_upload(self, latency_ms: float, *, ok: bool) -> None:
        with self._lock:
            self._counters["uploads"] += 1
            if not ok:
                self._counters["upload_errors"] += 1
            self._record_latency("upload", latency_ms)

    def record_summarize(self, latency_ms: float, *, ok: bool) -> None:
        with self._lock:
            self._counters["summarizes"] += 1
            if not ok:
                self._counters["summarize_errors"] += 1
            self._record_latency("summarize", latency_ms)

    def record_llm_error(self, exc: BaseException) -> str:
        """Count one LLM-call failure; returns its kind."""
        kind = classify_error(exc)
        with self._lock:
            if kind == "ollama_timeout":
                self._counters["ollama_timeouts"] += 1
            elif kind == "ollama_error":
                self._counters["ollama_errors"] += 1
            else:
                self._counters["errors"] += 1
        return kind

    def record_ollama_error(self) -> None:
        with self._lock:
            self._counters["ollama_errors"] += 1

    def record_ollama_timeout(self) -> None:
        with self._lock:
            self._counters["ollama_timeouts"] += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._counters["cache_hits"] += 1

    def record_cache_reload(self) -> None:
        with self._lock:
            self._counters["cache_reloads"] += 1

    def reset(self) -> None:
        """Test helper only: zero counters and drop samples. There is
        deliberately no API endpoint for this."""
        with self._lock:
            for key in self._counters:
                self._counters[key] = 0
            for samples in self._samples.values():
                samples.clear()

    @staticmethod
    def _summary(samples: deque) -> dict:
        if not samples:
            return {"count": 0, "mean_ms": None, "p95_ms": None}
        ordered = sorted(samples)
        mean = sum(ordered) / len(ordered)
        # Nearest-rank p95: the smallest value at or above the 95th
        # percentile. Cheap and exact over the kept samples.
        rank = max(1, math.ceil(0.95 * len(ordered)))
        p95 = ordered[rank - 1]
        return {
            "count": len(ordered),
            "mean_ms": round(mean, 1),
            "p95_ms": round(p95, 1),
        }

    def snapshot(self) -> dict:
        """JSON shape served by GET /api/metrics."""
        with self._lock:
            counters = dict(self._counters)
            latency = {
                step: self._summary(self._samples[step])
                for step in LATENCY_STEPS
            }
        return {**counters, "latency_ms": latency, "cache": cache_snapshot(self)}


def cache_snapshot(metrics: "Metrics | None" = None) -> dict:
    """Cache section of /api/metrics.

    If the caching module ever exposes cache_stats() (it does not yet,
    on main or in the pending Wave 1 PR), prefer its numbers. Otherwise
    report this module's own hit/reload counters, which the chain's
    default-retriever path feeds: on main every default-path question
    builds a fresh Retriever, so it counts as a reload. An explicitly
    passed retriever (upload indexes, tests) is neither a hit nor a
    reload and is not counted."""
    metrics = metrics or METRICS
    try:
        from .caching import cache_stats  # type: ignore[attr-defined]

        stats = cache_stats()
        if isinstance(stats, dict):
            return {"available": True, **stats}
    except Exception:
        pass
    with metrics._lock:
        hits = metrics._counters["cache_hits"]
        reloads = metrics._counters["cache_reloads"]
    return {
        "available": False,
        "reason": "caching module exposes no cache_stats(); counts below "
                  "are recorded by the chain call sites",
        "hits": hits,
        "reloads": reloads,
    }


METRICS = Metrics()


def record_question(timings: dict, *, streamed: bool) -> None:
    METRICS.record_question(timings, streamed=streamed)


def record_upload(latency_ms: float, *, ok: bool) -> None:
    METRICS.record_upload(latency_ms, ok=ok)


def record_summarize(latency_ms: float, *, ok: bool) -> None:
    METRICS.record_summarize(latency_ms, ok=ok)


def record_llm_error(exc: BaseException) -> str:
    return METRICS.record_llm_error(exc)


def record_ollama_error() -> None:
    METRICS.record_ollama_error()


def record_ollama_timeout() -> None:
    METRICS.record_ollama_timeout()


def record_cache_hit() -> None:
    METRICS.record_cache_hit()


def record_cache_reload() -> None:
    METRICS.record_cache_reload()


def get_metrics_snapshot() -> dict:
    return METRICS.snapshot()


def reset_metrics() -> None:
    """Test helper only (see Metrics.reset)."""
    METRICS.reset()
