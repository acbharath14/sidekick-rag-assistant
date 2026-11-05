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
    assert set(body.keys()) == {
        "status", "ollama_reachable", "index_present", "ollama",
    }


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


def test_health_reports_ollama_queue():
    """Health exposes the Ollama queue snapshot, so a backed-up Ollama
    is observable from outside the process."""
    r = client.get("/api/health")
    assert r.status_code == 200
    ollama = r.json()["ollama"]
    for key in (
        "in_flight",
        "max_in_flight",
        "calls_total",
        "timeouts_total",
        "retries_total",
        "errors_total",
        "queued_behind_total",
        "waits_total",
        "max_call_seconds",
    ):
        assert key in ollama, f"missing queue key: {key}"
    assert ollama["in_flight"] == 0


def test_chat_timeout_returns_504(monkeypatch):
    """A hung model fails fast with 504 and a plain message."""
    import time

    import rag_demo.api as api_mod

    def _slow(question, history=None, document_id=None):
        time.sleep(2)
        return None

    monkeypatch.setattr(api_mod, "ask", _slow)
    monkeypatch.setenv("RAG_DEMO_CHAT_TIMEOUT", "0.05")
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 504
    assert "took too long" in r.json()["detail"]


def test_chat_transient_failure_returns_502(monkeypatch):
    """Ollama down after retries surfaces as 502, not a bare 500."""
    import rag_demo.api as api_mod

    def _down(question, history=None, document_id=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(api_mod, "ask", _down)
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 502
    assert "unreachable" in r.json()["detail"]


def test_chat_stream_timeout_yields_error_event(monkeypatch):
    """A stalled stream ends with an error event, never a hung SSE."""
    import rag_demo.api as api_mod
    from rag_demo.ollama_guard import OllamaTimeoutError

    def _stalled(question, history=None, document_id=None):
        yield {"token": "partial"}
        raise OllamaTimeoutError("stalled")

    monkeypatch.setattr(api_mod, "stream_ask", _stalled)
    r = client.post("/api/chat/stream", json={"question": "What is this?"})
    assert r.status_code == 200
    assert "took too long" in r.text
