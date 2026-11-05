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


# --- Structured logging + metrics (observability) ---

import json as _json

from rag_demo import observability


def _log_lines(capsys):
    """Structured log lines captured on stdout, parsed as JSON dicts."""
    out = capsys.readouterr().out
    lines = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            lines.append(_json.loads(line))
        except ValueError:
            continue
    return lines


def test_question_emits_structured_log_line(capsys):
    capsys.readouterr()
    r = client.post("/api/chat",
                    json={"question": "What is the max parcel weight?"})
    assert r.status_code == 200
    lines = _log_lines(capsys)
    questions = [line for line in lines if line.get("event") == "question"]
    assert questions, f"no question log line in: {lines}"
    line = questions[-1]
    assert line["endpoint"] == "/api/chat"
    assert line["request_id"]
    assert line["model"]
    assert line["mode"] == "chat"
    assert line["question_preview"].startswith("What is the max parcel weight?")
    assert len(line["question_preview"]) <= 120
    for key in ("rewrite_ms", "episodic_ms", "retrieve_ms", "generate_ms",
                "total_ms"):
        assert key in line, f"missing timing key: {key}"
    assert isinstance(line["retrieve_ms"], (int, float))
    assert isinstance(line["total_ms"], (int, float))
    assert line["total_ms"] >= 0
    assert isinstance(line["hits"], int)
    assert isinstance(line["sources"], list)
    assert isinstance(line["grounded"], bool)
    # On main every default-path question builds a fresh retriever.
    assert line["cache_reload"] is True
    # The fake answer text must never appear in any log line.
    blob = "\n".join(_json.dumps(entry) for entry in lines)
    assert "Based on the indexed docs" not in blob


def test_metrics_move_after_question():
    before = client.get("/api/metrics").json()
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 200
    after = client.get("/api/metrics").json()
    assert after["questions"] == before["questions"] + 1
    assert (after["latency_ms"]["total"]["count"]
            == before["latency_ms"]["total"]["count"] + 1)
    assert after["latency_ms"]["total"]["mean_ms"] is not None
    assert after["latency_ms"]["total"]["p95_ms"] is not None


def test_metrics_shape():
    body = client.get("/api/metrics").json()
    for key in ("questions", "questions_streamed", "uploads", "upload_errors",
                "summarizes", "summarize_errors", "ollama_errors",
                "ollama_timeouts", "errors", "cache_hits", "cache_reloads",
                "latency_ms", "cache"):
        assert key in body, f"missing key: {key}"
    for step in ("rewrite", "episodic", "retrieve", "generate", "total",
                 "summarize", "upload"):
        summary = body["latency_ms"][step]
        assert set(summary) == {"count", "mean_ms", "p95_ms"}, \
            f"bad latency summary for {step}: {summary}"
    # Cache section is guarded: it reports available=False until the
    # caching module exposes cache_stats(), but hits/reloads are there.
    assert isinstance(body["cache"]["available"], bool)
    assert "hits" in body["cache"] and "reloads" in body["cache"]


def test_logs_never_contain_secrets_or_full_text(capsys):
    secret = "TOPSECRET-abc123"
    long_tail = "TAIL-" + "x" * 300
    question = f"api_key={secret} please explain {long_tail}"
    capsys.readouterr()
    r = client.post("/api/chat", json={"question": question})
    assert r.status_code == 200
    lines = _log_lines(capsys)
    blob = "\n".join(_json.dumps(entry) for entry in lines)
    assert secret not in blob, "secret leaked into a log line"
    assert long_tail not in blob, "full question text leaked into a log line"
    assert "x" * 200 not in blob
    previews = [entry["question_preview"] for entry in lines
                if entry.get("event") == "question"]
    assert previews
    assert len(previews[-1]) <= 120
    assert "[redacted]" in previews[-1]


def test_streamed_question_counts_as_streamed(capsys):
    before = client.get("/api/metrics").json()
    capsys.readouterr()
    r = client.post("/api/chat/stream", json={"question": "What is this?"})
    assert r.status_code == 200
    _ = r.text  # drive the SSE generator to completion
    after = client.get("/api/metrics").json()
    assert after["questions"] == before["questions"] + 1
    assert after["questions_streamed"] == before["questions_streamed"] + 1
    lines = _log_lines(capsys)
    streamed = [entry for entry in lines
                if entry.get("event") == "question"
                and entry.get("mode") == "stream"]
    assert streamed
    assert streamed[-1]["endpoint"] == "/api/chat/stream"


def test_upload_and_summarize_move_metrics_and_log(tmp_path, monkeypatch,
                                                   capsys):
    # Isolate the index and the conversations DB so this test never
    # touches the real .faiss_index or conversations.db.
    monkeypatch.setenv("RAG_DEMO_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_DEMO_CONVERSATIONS_DB",
                       str(tmp_path / "conversations.db"))
    before = client.get("/api/metrics").json()
    capsys.readouterr()
    r = client.post("/api/upload", files={
        "file": ("notes.txt", b"Parcel weights and API tokens.",
                 "text/plain"),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body
    doc_id = body["id"]
    after_upload = client.get("/api/metrics").json()
    assert after_upload["uploads"] == before["uploads"] + 1
    lines = _log_lines(capsys)
    uploads = [entry for entry in lines if entry.get("event") == "upload"]
    assert uploads, f"no upload log line in: {lines}"
    assert uploads[-1]["doc_id"] == doc_id
    assert uploads[-1]["filename"] == "notes.txt"
    assert "Parcel weights" not in _json.dumps(lines), \
        "document text leaked into a log line"

    capsys.readouterr()
    r = client.post(f"/api/documents/{doc_id}/summarize")
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True, r.json()
    after_summarize = client.get("/api/metrics").json()
    assert after_summarize["summarizes"] == before["summarizes"] + 1
    lines = _log_lines(capsys)
    summarizes = [entry for entry in lines
                  if entry.get("event") == "summarize"]
    assert summarizes, f"no summarize log line in: {lines}"
    assert summarizes[-1]["doc_id"] == doc_id
    assert summarizes[-1]["mode"] == "single"


def test_metrics_latency_summary_math():
    # Unit check of the mean/p95 computation, isolated from the shared
    # counters (reset before and after so other tests are unaffected).
    observability.reset_metrics()
    try:
        for ms in (10.0, 20.0, 30.0, 40.0):
            observability.record_question(
                {"total_ms": ms, "rewrite_ms": None, "episodic_ms": None,
                 "retrieve_ms": None, "generate_ms": None},
                streamed=False,
            )
        snap = observability.get_metrics_snapshot()
        total = snap["latency_ms"]["total"]
        assert total["count"] == 4
        assert total["mean_ms"] == 25.0
        assert total["p95_ms"] == 40.0
        assert snap["questions"] == 4
    finally:
        observability.reset_metrics()
    assert observability.get_metrics_snapshot()["questions"] == 0
