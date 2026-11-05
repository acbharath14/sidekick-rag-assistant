"""Per-client rate limiting for the FastAPI backend (Phase 1, Wave 3).

Stdlib only, in-memory sliding window, no new dependency. One bucket per
client per endpoint group (chat, upload, summarize), keyed by the
authenticated API key when auth is on, else by client IP.

Fail-safe by design: the FastAPI dependency in this module catches every
limiter error and lets the request through (fail open) instead of 500ing
the API. A broken limiter is logged, never fatal.

State lives in the process (a module-level singleton). That is the honest
scope for a single-user demo server: limits do not survive a restart and
are not shared across workers. uvicorn runs one worker here, so one
process is the whole deployment.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# Fixed window for every bucket: N requests per 60 seconds.
RATE_LIMIT_WINDOW_SECONDS = 60

# Upper bound on tracked client buckets, so a key-spraying client cannot
# grow the dict without bound. Oldest idle buckets are dropped first.
_MAX_TRACKED_CLIENTS = 10000

# Buckets the API exposes. Keep in sync with the Settings fields
# (rate_limit_<bucket>_per_minute) and the api.py dependency lists.
RATE_LIMIT_BUCKETS = ("chat", "upload", "summarize")


class SlidingWindowLimiter:
    """Thread-safe in-memory sliding-window limiter.

    Each key maps to a deque of monotonic timestamps (one per allowed
    request inside the current window). A request is allowed when fewer
    than ``limit`` timestamps fall inside the window; otherwise it is
    denied with the seconds until the oldest hit leaves the window
    (the Retry-After value). ``limit <= 0`` means unlimited.
    """

    def __init__(
        self,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        max_clients: int = _MAX_TRACKED_CLIENTS,
        clock=None,
    ) -> None:
        self._window = window_seconds
        self._max_clients = max(1, max_clients)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """Check one request. Returns (allowed, retry_after_seconds).

        retry_after_seconds is 0 when allowed, otherwise at least 1.
        """
        if limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            window = self._hits.get(key)
            if window is None:
                window = self._hits[key] = deque()
            cutoff = now - self._window
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= limit:
                retry_after = max(1, math.ceil(self._window - (now - window[0])))
                return False, retry_after
            window.append(now)
            if len(self._hits) > self._max_clients:
                self._prune_expired(now, keep=key)
            return True, 0

    def _prune_expired(self, now: float, keep: str) -> None:
        """Drop idle buckets. Caller must hold the lock."""
        for bucket_key in list(self._hits):
            window = self._hits[bucket_key]
            cutoff = now - self._window
            while window and window[0] <= cutoff:
                window.popleft()
            if not window:
                del self._hits[bucket_key]
        while len(self._hits) > self._max_clients:
            oldest = next(iter(self._hits))
            if oldest == keep:
                # The bucket in use is the oldest; drop the next one.
                keys = list(self._hits)
                if len(keys) < 2:
                    break
                oldest = keys[1]
            del self._hits[oldest]

    def reset(self) -> None:
        """Drop all tracked buckets. Used by tests to isolate cases."""
        with self._lock:
            self._hits.clear()


# Process-wide limiter instance. Dependencies share it so buckets persist
# across requests; tests reset it between cases.
_limiter = SlidingWindowLimiter()


def reset_rate_limiter() -> None:
    """Drop all tracked buckets (test isolation)."""
    _limiter.reset()


def client_key(api_key: str | None, client_host: str | None) -> str:
    """Bucket identity for one client.

    The API key is hashed (sha256) before use, so the raw key never sits
    in the limiter dict. Without a key, fall back to the client IP.
    """
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return f"key:{digest}"
    return f"ip:{client_host or 'unknown'}"


def _limit_for(settings, bucket: str) -> int:
    """Per-minute limit for one bucket, from Settings."""
    return {
        "chat": settings.rate_limit_chat_per_minute,
        "upload": settings.rate_limit_upload_per_minute,
        "summarize": settings.rate_limit_summarize_per_minute,
    }[bucket]


def _trusted_api_key(x_api_key: str | None) -> str | None:
    """Return the header key only when it is the configured key.

    When auth is off (no RAG_DEMO_API_KEY), any header value is
    untrusted: a client could rotate bogus keys to dodge the IP bucket.
    Auth runs before this dependency, so with auth on a key reaching
    here already passed require_api_key; this re-check only matters
    for the auth-off case.
    """
    expected = os.environ.get("RAG_DEMO_API_KEY", "").strip()
    if not expected or not x_api_key:
        return None
    if hmac.compare_digest(x_api_key, expected):
        return x_api_key
    return None


def rate_limit(bucket: str):
    """FastAPI dependency factory: enforce the per-client limit for one bucket.

    Exceeding the limit raises 429 with a Retry-After header (seconds).
    Any limiter error fails open: the request is logged and let through,
    never 500ed. Auth runs first (see api.py dependency lists), so a bad
    key gets 401 without consuming quota.
    """
    if bucket not in RATE_LIMIT_BUCKETS:
        raise ValueError(f"unknown rate-limit bucket: {bucket!r}")

    # Imported here so this module stays importable without FastAPI
    # (the pure limiter is unit-tested standalone).
    from fastapi import Header, HTTPException, Request
    from fastapi import status as http_status

    def _check(
        request: Request,
        x_api_key: str | None = Header(default=None),
    ) -> None:
        try:
            from .config import get_settings

            settings = get_settings()
            if not settings.rate_limit_enabled:
                return
            limit = _limit_for(settings, bucket)
            if limit <= 0:
                return
            key = (
                f"{bucket}:{client_key(_trusted_api_key(x_api_key), request.client.host if request.client else None)}"
            )
            allowed, retry_after = _limiter.allow(key, limit)
            if not allowed:
                raise HTTPException(
                    status_code=http_status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Try again later.",
                    headers={"Retry-After": str(retry_after)},
                )
        except HTTPException:
            raise
        except Exception:
            # Fail open: a broken limiter must never take the API down.
            logger.exception("rate limiter failed open for bucket %s", bucket)

    return _check
