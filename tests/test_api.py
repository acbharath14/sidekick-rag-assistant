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


def _upload_files(client, files):
    """POST /api/upload with several files under the repeated "file" field.

    files: list of (filename, bytes) tuples.
    """
    return client.post(
        "/api/upload",
        files=[("file", (name, content)) for name, content in files],
    )


def _delete_doc(client, doc_id):
    d = client.delete(f"/api/documents/{doc_id}")
    assert d.json()["ok"] is True, d.json()


def test_upload_two_files_both_ingest():
    r = _upload_files(client, [
        ("one.txt", b"First document about cats."),
        ("two.txt", b"Second document about dogs."),
    ])
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body
    assert body["succeeded"] == 2 and body["failed"] == 0
    assert len(body["results"]) == 2
    ids = []
    for result, name in zip(body["results"], ("one.txt", "two.txt")):
        assert result["ok"] is True, result
        assert result["filename"] == name
        assert result["extractor"] == "txt"
        ids.append(result["id"])
    assert ids[0] != ids[1]
    for doc_id in ids:
        _delete_doc(client, doc_id)


def test_upload_mixed_batch_keeps_good_file():
    r = _upload_files(client, [
        ("good.txt", b"A perfectly fine document."),
        ("bad.exe", b"MZ fake binary, not a document"),
    ])
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False  # one file failed
    assert body["succeeded"] == 1 and body["failed"] == 1
    good, bad = body["results"]
    assert good["filename"] == "good.txt" and good["ok"] is True
    assert bad["filename"] == "bad.exe" and bad["ok"] is False
    assert bad["error"]
    # The good file survived: it is retrievable, not rolled back.
    got = client.get(f"/api/documents/{good['id']}")
    assert got.json()["ok"] is True
    _delete_doc(client, good["id"])


def test_upload_single_file_keeps_legacy_shape():
    r = client.post("/api/upload", files={"file": ("solo.txt", b"Solo document.")})
    assert r.status_code == 200
    body = r.json()
    assert "results" not in body
    assert body["ok"] is True
    assert set(body.keys()) == {
        "ok", "id", "filename", "chunk_count", "char_count",
        "file_size", "extractor", "summary", "uploaded_at",
    }
    assert body["filename"] == "solo.txt"
    _delete_doc(client, body["id"])


def test_upload_single_file_failure_keeps_legacy_shape():
    r = client.post("/api/upload", files={"file": ("bad.exe", b"MZ nope")})
    body = r.json()
    assert body["ok"] is False
    assert set(body.keys()) == {"ok", "error"}


def test_upload_batch_invalidates_cache_once():
    """The cached retriever is dropped once per batch, not once per file,
    and not at all when every file fails (nothing changed on disk)."""
    from unittest.mock import patch

    with patch("rag_demo.api.clear_cache") as mock_clear:
        r = _upload_files(client, [
            ("a.txt", b"Cache test document one."),
            ("b.txt", b"Cache test document two."),
        ])
        assert r.json()["ok"] is True, r.json()
        assert mock_clear.call_count == 1
        ids = [result["id"] for result in r.json()["results"]]
        # An all-failing batch changes nothing, so no second invalidation.
        r = _upload_files(client, [
            ("bad1.exe", b"MZ nope"),
            ("bad2.exe", b"MZ nope"),
        ])
        assert r.json()["ok"] is False
        assert mock_clear.call_count == 1
    for doc_id in ids:
        _delete_doc(client, doc_id)


def test_upload_file_count_cap(monkeypatch):
    monkeypatch.setenv("RAG_DEMO_UPLOAD_MAX_FILES", "2")
    r = _upload_files(client, [
        ("1.txt", b"one"),
        ("2.txt", b"two"),
        ("3.txt", b"three"),
    ])
    body = r.json()
    assert body["ok"] is False
    assert "max 2" in body["error"]
