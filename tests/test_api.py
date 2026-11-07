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
from rag_demo.caching import clear_cache

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
    # index dir is a temp dir, so no real index is touched. The document
    # record still goes into the shared test conversations DB, so it is
    # deleted below after the summarize assertions.
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
            doc_id = r.json()["id"]
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
        # Cleanup: the upload recorded this document in the shared test
        # conversations DB, so delete it (DELETE is not rate-limited).
        _delete_doc(client, doc_id)


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
    # Drop the cached retriever so this question builds a fresh one:
    # the retriever cache is shared across the session, so without this
    # an earlier test's question would make this one a cache hit.
    clear_cache()
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
    # The cache was cleared at the start of this test, so this question
    # builds a fresh retriever.
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


def test_summarize_after_restart_recovers_text_from_saved_chunks(
    tmp_path, monkeypatch
):
    """PR #68: Regenerate after a restart rebuilds the text from chunks.json.

    The in-memory text cache is per process; the persisted chunks are not.
    Clear the cache to simulate a restart, then regenerate both synchronously
    and over SSE: both must answer from the rebuilt text instead of the old
    "Document text no longer in memory (restarted since upload)" error.
    """
    monkeypatch.setenv("RAG_DEMO_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_DEMO_CONVERSATIONS_DB",
                       str(tmp_path / "conversations.db"))
    text = "\n\n".join(
        f"Paragraph {i}. The retention period is {i} days."
        for i in range(1, 300)
    )
    assert len(text) > 12_000  # long enough for the map-reduce path
    r = client.post("/api/upload", files={
        "file": ("policy.txt", text.encode(), "text/plain"),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body
    doc_id = body["id"]
    # A second document proves its chunks never leak into this rebuild.
    other = client.post("/api/upload", files={
        "file": ("other.txt", b"UNRELATED-OTHER-DOC-MARKER", "text/plain"),
    })
    other_id = other.json()["id"]
    try:
        api._upload_texts.pop(doc_id, None)  # simulate the restart
        api._upload_texts.pop(other_id, None)
        recovered = api._document_text(doc_id, api.get_settings())
        assert recovered is not None
        assert "Paragraph 1. " in recovered
        assert "Paragraph 159. " in recovered
        assert "UNRELATED-OTHER-DOC-MARKER" not in recovered
        # Overlap trimming is a heuristic (up to 50 chars per boundary),
        # so allow a small length delta but never a truncated document.
        assert abs(len(recovered) - len(text)) <= len(text) * 0.02

        r = client.post(f"/api/documents/{doc_id}/summarize")
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True, r.json()
        assert r.json()["summary"]
        assert api._upload_texts.get(doc_id)  # cache refilled by the rebuild

        api._upload_texts.pop(doc_id, None)
        with client.stream(
            "POST", f"/api/documents/{doc_id}/summarize/stream"
        ) as sr:
            events = [
                json.loads(line[len("data: "):])
                for line in sr.iter_lines()
                if line.startswith("data: ")
            ]
        assert any(e.get("type") == "done" for e in events), events
        assert not any(e.get("type") == "error" for e in events), events
    finally:
        api._upload_texts.pop(doc_id, None)
        api._upload_texts.pop(other_id, None)
        client.delete(f"/api/documents/{doc_id}")
        client.delete(f"/api/documents/{other_id}")


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

    def _slow(question, history=None, document_id=None,
              request_id=None, endpoint=None):
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

    def _down(question, history=None, document_id=None,
              request_id=None, endpoint=None):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(api_mod, "ask", _down)
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 502
    assert "unreachable" in r.json()["detail"]


def test_chat_stream_timeout_yields_error_event(monkeypatch):
    """A stalled stream ends with an error event, never a hung SSE."""
    import rag_demo.api as api_mod
    from rag_demo.ollama_guard import OllamaTimeoutError

    def _stalled(question, history=None, document_id=None,
                 request_id=None, endpoint=None):
        yield {"token": "partial"}
        raise OllamaTimeoutError("stalled")

    monkeypatch.setattr(api_mod, "stream_ask", _stalled)
    r = client.post("/api/chat/stream", json={"question": "What is this?"})
    assert r.status_code == 200
    assert "took too long" in r.text


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
