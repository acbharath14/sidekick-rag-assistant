"""API contract test for the FastAPI backend (Part 1 of the overhaul spec).

Hits /api/chat with RAG_DEMO_FAKE=1 (deterministic fakes — no live Ollama,
no model downloads). Asserts the response shape the React frontend depends
on. Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_api.py -v
"""
import asyncio
import json
import os
import time

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

import rag_demo.api as api
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


# --- Cancel-on-disconnect for the streaming endpoints ---------------------
#
# The stream driver polls request.is_disconnected() while a generation is
# in flight. These tests drive the endpoints with a stub request (the
# real Starlette probe needs a live client closing a socket, which
# TestClient cannot express deterministically) and a counting fake
# producer, and assert the producer is abandoned: pulls stop, the
# generator is closed, no done event is emitted, and no assistant
# message or summary is persisted for the cancelled generation.


class _StubRequest:
    """Minimal stand-in for starlette.requests.Request: only the
    disconnect probe the stream driver uses."""

    def __init__(self):
        self.disconnected = False
        self.probes = 0

    async def is_disconnected(self):
        self.probes += 1
        return self.disconnected


def _counting_stream(events, request, disconnect_after=None, delay=0.1):
    """Fake stream_ask / summarize_events producer.

    Yields `events` one at a time (sleeping a little per item so the
    disconnect watcher gets a chance to notice) and returns
    (generator, state). `state` records pulls, and whether the
    generator was closed via GeneratorExit (as opposed to running to
    exhaustion). Set `disconnect_after` to flip the stub request to
    disconnected once that many events have been yielded, simulating
    the client closing the tab mid-generation.
    """
    state = {"pulls": 0, "closed": False, "finished": False}

    def _gen():
        try:
            for event in events:
                state["pulls"] += 1
                time.sleep(delay)
                yield event
                if (disconnect_after is not None
                        and state["pulls"] == disconnect_after):
                    request.disconnected = True
        except GeneratorExit:
            state["closed"] = True
            raise
        state["finished"] = True

    return _gen(), state


def _collect_sse(response):
    """Read a StreamingResponse body iterator into parsed data payloads."""
    chunks = []

    async def _read():
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    asyncio.run(_read())
    events = []
    for chunk in chunks:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        for line in text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def _fake_answer(text="fake answer"):
    from rag_demo.chain import Answer

    return Answer(text=text, sources=["doc.md"], hits=[], latency_ms=1.0,
                  grounded=True, standalone_question=None)


def test_chat_stream_connected_delivers_full_sequence():
    """Connected clients see the same events as before: tokens, then done."""
    with client.stream("POST", "/api/chat/stream",
                       json={"question": "What is the max parcel weight?"}) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    tokens = [e["token"] for e in events if "token" in e]
    dones = [e["done"] for e in events if "done" in e]
    assert tokens and "".join(tokens)
    assert len(dones) == 1
    assert "".join(tokens) == dones[0]["text"]
    for key in ("text", "sources", "hits", "grounded", "latency_ms",
                "standalone_question"):
        assert key in dones[0], f"missing key: {key}"


def test_chat_stream_disconnect_before_first_token_abandons_generation(monkeypatch):
    """Client already gone when generation starts: no token is delivered,
    the producer is closed, no done event appears, and no assistant
    message is persisted (the user message, recorded before the stream
    starts, is kept as before)."""
    request = _StubRequest()
    request.disconnected = True
    stream, state = _counting_stream(
        [{"token": f"t{i}"} for i in range(10)]
        + [{"answer": _fake_answer()}],
        request,
    )
    monkeypatch.setattr(api, "stream_ask", lambda *a, **k: stream)
    monkeypatch.setattr(api, "_DISCONNECT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(api, "_DISCONNECT_GRACE_SECONDS", 1.0)
    calls = []
    monkeypatch.setattr(
        api.conv_store, "add_message",
        lambda conv_id, role, text, **k: calls.append(role),
    )
    monkeypatch.setattr(api.conv_store, "conversation_exists", lambda c: True)
    monkeypatch.setattr(api.conv_store, "maybe_autotitle", lambda c, q: None)
    monkeypatch.setattr(api.conv_store, "touch_conversation", lambda c: None)

    response = asyncio.run(api.chat_stream(
        api.ChatRequest(question="q", conversation_id="conv1"), request))
    events = _collect_sse(response)

    assert events == []
    assert state["closed"] is True
    assert state["pulls"] <= 1  # at most the in-flight first pull
    assert calls == ["user"]


def test_chat_stream_disconnect_mid_generation_stops_pulling_tokens(monkeypatch):
    """Client leaves after two tokens: pulls stop well short of the ten
    available, the producer is closed, and no done event is emitted."""
    request = _StubRequest()
    stream, state = _counting_stream(
        [{"token": f"t{i}"} for i in range(10)]
        + [{"answer": _fake_answer()}],
        request,
        disconnect_after=2,
    )
    monkeypatch.setattr(api, "stream_ask", lambda *a, **k: stream)
    monkeypatch.setattr(api, "_DISCONNECT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(api, "_DISCONNECT_GRACE_SECONDS", 1.0)

    response = asyncio.run(api.chat_stream(
        api.ChatRequest(question="q"), request))
    events = _collect_sse(response)

    assert state["closed"] is True
    assert state["pulls"] < 10
    assert not any("done" in e for e in events)
    assert all("token" in e for e in events)
    assert len(events) < 10


def test_summarize_stream_connected_delivers_full_sequence(monkeypatch):
    """Connected clients see the same summarize events as before:
    start, then done with the summary, which is then cached."""
    doc_id = "doc-cancel-test"
    monkeypatch.setattr(
        api.conv_store, "get_document",
        lambda d: {"id": doc_id, "summary": None} if d == doc_id else None,
    )
    monkeypatch.setitem(api._upload_texts, doc_id, "Short document text.")
    cached = {}
    monkeypatch.setattr(
        api.conv_store, "set_document_summary",
        lambda d, s: cached.setdefault(d, s),
    )

    response = asyncio.run(api.summarize_document_stream(doc_id, _StubRequest()))
    events = _collect_sse(response)

    assert events[0]["type"] == "start"
    dones = [e for e in events if e["type"] == "done"]
    assert len(dones) == 1
    assert dones[0]["summary"]
    assert cached.get(doc_id) == dones[0]["summary"]


def test_summarize_stream_disconnect_abandons_generation(monkeypatch):
    """Client leaves mid-summary: pulls stop, the producer is closed, no
    done event is emitted, and no partial summary is cached."""
    doc_id = "doc-cancel-test"
    request = _StubRequest()
    stream, state = _counting_stream(
        [{"type": "start", "mode": "map_reduce", "total": 10}]
        + [{"type": "progress", "done": i, "total": 10, "stage": "map"}
           for i in range(1, 10)]
        + [{"type": "done", "summary": "full summary"}],
        request,
        disconnect_after=2,
    )
    monkeypatch.setattr(
        api.conv_store, "get_document",
        lambda d: {"id": doc_id, "summary": None} if d == doc_id else None,
    )
    monkeypatch.setitem(api._upload_texts, doc_id, "Short document text.")
    cached = {}
    monkeypatch.setattr(
        api.conv_store, "set_document_summary",
        lambda d, s: cached.setdefault(d, s),
    )
    monkeypatch.setattr(
        "rag_demo.summarize.summarize_events", lambda *a, **k: stream)
    monkeypatch.setattr(api, "_DISCONNECT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(api, "_DISCONNECT_GRACE_SECONDS", 1.0)

    response = asyncio.run(api.summarize_document_stream(doc_id, request))
    events = _collect_sse(response)

    assert state["closed"] is True
    assert state["pulls"] < 11
    assert not any(e.get("type") == "done" for e in events)
    assert cached == {}


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
