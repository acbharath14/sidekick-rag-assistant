"""API-key auth tests for the FastAPI backend.

RAG_DEMO_API_KEY (when set, non-empty) requires the matching X-API-Key
header on every /api/* endpoint except /api/health. When unset/empty, all
endpoints are open (local dev, backwards compatible).

Each case sets/unsets the env var explicitly — require_api_key reads it
per-request, so no app reload is needed between cases.
Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_auth.py -v
"""
import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from rag_demo.api import app

client = TestClient(app)

KEY = "test-secret-key-123"


def set_key(value):
    if value is None:
        os.environ.pop("RAG_DEMO_API_KEY", None)
    else:
        os.environ["RAG_DEMO_API_KEY"] = value


def teardown_function(_):
    # Never leak a key into other test modules.
    set_key(None)


def test_no_key_set_open_access():
    """(a) Env var unset -> everything open, no header needed."""
    set_key(None)
    for method, url in (
        ("GET", "/api/config"),
        ("GET", "/api/conversations"),
        ("GET", "/api/documents"),
        ("POST", "/api/feedback"),
    ):
        r = client.request(method, url,
                           json={"question": "q", "vote": "up"}
                           if url == "/api/feedback" else None)
        assert r.status_code == 200, f"{method} {url} -> {r.status_code}"


def test_empty_key_is_open_access():
    """Empty-string key counts as unset -> open access."""
    set_key("")
    r = client.get("/api/config")
    assert r.status_code == 200


def test_key_set_correct_header_ok():
    """(b) Key set + correct X-API-Key -> 200."""
    set_key(KEY)
    r = client.get("/api/config", headers={"X-API-Key": KEY})
    assert r.status_code == 200


def test_key_set_wrong_header_401():
    """(c) Key set + wrong key -> 401 with the exact detail body."""
    set_key(KEY)
    r = client.get("/api/config", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid or missing API key"}


def test_key_set_missing_header_401():
    """(d) Key set + no header -> 401 with the exact detail body."""
    set_key(KEY)
    r = client.get("/api/config")
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid or missing API key"}


def test_health_always_open():
    """(e) /api/health never requires auth, key set or not."""
    set_key(KEY)
    r = client.get("/api/health")
    assert r.status_code == 200
    r = client.get("/api/health", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    set_key(None)
    r = client.get("/api/health")
    assert r.status_code == 200


def test_auth_covers_write_endpoints():
    """Spot-check that mutations are protected too, not just GETs."""
    set_key(KEY)
    r = client.post("/api/conversations", json={})
    assert r.status_code == 401
    r = client.delete("/api/conversations/does-not-exist")
    assert r.status_code == 401
    r = client.post("/api/conversations", json={},
                    headers={"X-API-Key": KEY})
    assert r.status_code == 200
