"""API contract test for the FastAPI backend (Part 1 of the overhaul spec).

Hits /api/chat with RAG_DEMO_FAKE=1 (deterministic fakes — no live Ollama,
no model downloads). Asserts the response shape the React frontend depends
on. Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_api.py -v
"""
import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from rag_demo.api import app

client = TestClient(app)

# --- Per-client rate limiting (Phase 1, Wave 3) ---

from contextlib import contextmanager

from rag_demo import rate_limit as _rate_limit


@contextmanager
def _limited(monkeypatch, **env):
    """Set rate-limit env vars for one test and isolate limiter state.

    Settings are read per request, so monkeypatch.setenv takes effect
    immediately. The limiter is reset before and after so tests never
    share buckets (TestClient always comes from the same client IP).
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _rate_limit.reset_rate_limiter()
    try:
        yield
    finally:
        _rate_limit.reset_rate_limiter()


def test_rate_limit_allows_requests_under_limit(monkeypatch):
    with _limited(monkeypatch, RAG_DEMO_RATE_LIMIT_CHAT_PER_MINUTE="3"):
        for _ in range(3):
            r = client.post("/api/chat", json={"question": "What is this?"})
            assert r.status_code == 200


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    with _limited(monkeypatch, RAG_DEMO_RATE_LIMIT_CHAT_PER_MINUTE="2"):
        assert client.post("/api/chat", json={"question": "q"}).status_code == 200
        assert client.post("/api/chat", json={"question": "q"}).status_code == 200
        r = client.post("/api/chat", json={"question": "q"})
        assert r.status_code == 429
        assert "retry-after" in r.headers
        assert int(r.headers["retry-after"]) >= 1
        assert r.json() == {
            "detail": "Rate limit exceeded. Try again later."
        }
        # /api/chat/stream shares the chat bucket, so it is limited too.
        r = client.post("/api/chat/stream", json={"question": "q"})
        assert r.status_code == 429


def test_rate_limit_buckets_are_per_api_key(monkeypatch):
    with _limited(monkeypatch, RAG_DEMO_RATE_LIMIT_CHAT_PER_MINUTE="1"):
        monkeypatch.setenv("RAG_DEMO_API_KEY", "key-one")
        url = "/api/chat"
        body = {"question": "q"}
        r = client.post(url, json=body, headers={"X-API-Key": "key-one"})
        assert r.status_code == 200
        r = client.post(url, json=body, headers={"X-API-Key": "key-one"})
        assert r.status_code == 429
        # A different key gets a fresh bucket.
        monkeypatch.setenv("RAG_DEMO_API_KEY", "key-two")
        r = client.post(url, json=body, headers={"X-API-Key": "key-two"})
        assert r.status_code == 200
        # And the first key is still over its own limit.
        monkeypatch.setenv("RAG_DEMO_API_KEY", "key-one")
        r = client.post(url, json=body, headers={"X-API-Key": "key-one"})
        assert r.status_code == 429


def test_rate_limit_upload_and_summarize_buckets(monkeypatch, tmp_path):
    from unittest.mock import patch

    # Upload: extraction is mocked (same pattern as the OCR test) and the
    # index dir is a temp dir, so no real index or SQLite file is touched.
    with _limited(
        monkeypatch,
        RAG_DEMO_RATE_LIMIT_UPLOAD_PER_MINUTE="1",
        RAG_DEMO_RATE_LIMIT_SUMMARIZE_PER_MINUTE="1",
        RAG_DEMO_INDEX_DIR=str(tmp_path),
    ):
        with (
            patch(
                "rag_demo.extract.extract_text_with_meta",
                return_value=("hello world " * 200, "pypdf"),
            ),
            patch("rag_demo.extract.verify_magic_bytes", return_value=True),
        ):
            files = {"file": ("doc.txt", b"hello", "text/plain")}
            r = client.post("/api/upload", files=files)
            assert r.status_code == 200
            assert r.json()["ok"] is True
            r = client.post("/api/upload", files=files)
            assert r.status_code == 429
            assert "retry-after" in r.headers
        # Summarize: unknown doc id still answers 200 (ok False) under
        # the limit, then 429; the legacy alias shares the bucket.
        r = client.post("/api/documents/nope/summarize")
        assert r.status_code == 200
        r = client.post("/api/documents/nope/summarize")
        assert r.status_code == 429
        r = client.post("/api/upload/nope/summarize")
        assert r.status_code == 429


def test_health_is_not_rate_limited(monkeypatch):
    with _limited(monkeypatch, RAG_DEMO_RATE_LIMIT_CHAT_PER_MINUTE="1"):
        assert client.post("/api/chat", json={"question": "q"}).status_code == 200
        assert client.post("/api/chat", json={"question": "q"}).status_code == 429
        for _ in range(5):
            assert client.get("/api/health").status_code == 200


def test_rate_limit_fails_open_when_limiter_errors(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("limiter down")

    _rate_limit.reset_rate_limiter()
    try:
        monkeypatch.setattr(_rate_limit._limiter, "allow", _boom)
        r = client.post("/api/chat", json={"question": "What is this?"})
        assert r.status_code == 200
    finally:
        _rate_limit.reset_rate_limiter()


def test_sliding_window_frees_after_window():
    now = [1000.0]
    limiter = _rate_limit.SlidingWindowLimiter(
        window_seconds=60, clock=lambda: now[0]
    )
    assert limiter.allow("k", 1) == (True, 0)
    allowed, retry_after = limiter.allow("k", 1)
    assert allowed is False
    assert retry_after == 60
    now[0] += 30.0
    allowed, retry_after = limiter.allow("k", 1)
    assert allowed is False
    assert retry_after == 30
    now[0] += 31.0
    assert limiter.allow("k", 1) == (True, 0)


def test_client_key_hashes_api_key():
    a = _rate_limit.client_key("secret-one", "1.2.3.4")
    b = _rate_limit.client_key("secret-two", "1.2.3.4")
    assert a != b
    assert "secret-one" not in a and "secret-two" not in b
    assert _rate_limit.client_key(None, "1.2.3.4") == _rate_limit.client_key(
        None, "1.2.3.4"
    )
    assert _rate_limit.client_key(None, "1.2.3.4") != _rate_limit.client_key(
        None, "5.6.7.8"
    )


def test_chat_shape():
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 200
    body = r.json()
    for key in ("text", "sources", "hits", "grounded", "latency_ms",
                "standalone_question"):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["sources"], list)
    assert isinstance(body["hits"], list)
    for h in body["hits"]:
        assert set(h.keys()) == {"source", "text", "score"}, \
            f"hit keys wrong: {h.keys()}"
        assert "allowed_groups" not in h  # internal only, never on the wire


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"status", "ollama_reachable", "index_present"}


def test_config_no_secrets():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    blob = str(body).lower()
    assert "token" not in blob and "secret" not in blob and "key" not in blob, \
        "config endpoint leaked a secret-bearing field"


def test_feedback_validation():
    r = client.post("/api/feedback",
                    json={"question": "q", "vote": "sideways"})
    assert r.json()["ok"] is False


def _upload(client, filename="scan.pdf", content=b"%PDF-1.4 fake scanned bytes"):
    return client.post(
        "/api/upload",
        files={"file": (filename, content, "application/pdf")},
    )


def test_upload_scanned_pdf_falls_back_to_ocr():
    """Regression test: the FastAPI rewrite dropped the OCR fallback the old
    Streamlit app had — scanned PDFs must OCR instead of 'No text found'."""
    from unittest.mock import patch

    with (
        patch(
            "rag_demo.extract.extract_text_with_meta",
            return_value=("", "pypdf"),
        ),
        patch("rag_demo.ocr.ocr_available", return_value=True),
        patch("rag_demo.ocr.ocr_pdf", return_value="OCR read this page."),
    ):
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body
    assert body["extractor"] == "ocr"
    doc_id = body["id"]
    # Cleanup: remove the test document from the index + SQLite.
    d = client.delete(f"/api/documents/{doc_id}")
    assert d.json()["ok"] is True


def test_upload_scanned_pdf_without_ocr_surfaces_install_hint():
    from unittest.mock import patch

    with (
        patch(
            "rag_demo.extract.extract_text_with_meta",
            return_value=("", "pypdf"),
        ),
        patch("rag_demo.ocr.ocr_available", return_value=False),
    ):
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "No text found" in body["error"]
    assert "tesseract" in body["error"]


def test_upload_scanned_pdf_ocr_failure_surfaces_cause():
    from unittest.mock import patch

    with (
        patch(
            "rag_demo.extract.extract_text_with_meta",
            return_value=("", "pypdf"),
        ),
        patch("rag_demo.ocr.ocr_available", return_value=True),
        patch(
            "rag_demo.ocr.ocr_pdf",
            side_effect=RuntimeError("OCR needs the tesseract binary"),
        ),
    ):
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "tesseract" in body["error"]
