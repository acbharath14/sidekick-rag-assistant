"""Agentic chat wiring: /api/chat and /api/chat/stream route to the ReAct
agent when agentic=true, and stay on the classic chain otherwise.
No live Ollama: the agent entry points are monkeypatched.
Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_agentic_api.py -v
"""
import json
import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient

import rag_demo.api as api
from rag_demo.api import app
from rag_demo.chain import Answer
from rag_demo import rate_limit as _rate_limit

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    _rate_limit.reset_rate_limiter()
    yield
    _rate_limit.reset_rate_limiter()


def _fake_answer(text="agent answer"):
    return Answer(text=text, sources=["doc.md"], hits=[], latency_ms=1.0,
                  grounded=True, standalone_question=None)


def test_chat_agentic_calls_agentic_ask(monkeypatch):
    """agentic=true routes /api/chat to agentic_ask, never to ask()."""
    calls = {}

    def _agent(question, **kwargs):
        calls["question"] = question
        calls["history"] = kwargs.get("history")
        calls["retriever"] = kwargs.get("retriever")
        return _fake_answer()

    monkeypatch.setattr(api, "agentic_ask", _agent)
    monkeypatch.setattr(api, "ask", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("classic ask must not run in agentic mode")))

    r = client.post("/api/chat", json={
        "question": "What is this?",
        "history": [["q0", "a0"]],
        "agentic": True,
    })
    assert r.status_code == 200
    assert r.json()["text"] == "agent answer"
    assert r.json()["sources"] == ["doc.md"]
    assert calls["question"] == "What is this?"
    assert calls["history"] == [("q0", "a0")]  # pydantic coerces to tuples
    assert calls["retriever"] is None  # whole-corpus search, no doc scope


def test_chat_classic_default_skips_agent(monkeypatch):
    """No flag means the classic chain; agentic_ask is never called."""
    monkeypatch.setattr(api, "agentic_ask", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("agentic_ask must not run without the flag")))
    r = client.post("/api/chat", json={"question": "What is this?"})
    assert r.status_code == 200


def _collect_stream_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_chat_stream_agentic_emits_progress_tokens_done(monkeypatch):
    """Agentic stream: progress events first, then tokens, then done,
    with the tokens joining to the done text — the classic contract."""
    def _agent_stream(question, **kwargs):
        yield {"type": "progress", "phase": "thought",
               "message": "Planning search strategy..."}
        yield {"type": "progress", "phase": "action",
               "message": "Searching [search_docs]: parcels"}
        yield {"token": "agent "}
        yield {"token": "answer"}
        yield {"answer": _fake_answer("agent answer")}

    monkeypatch.setattr(api, "stream_agentic_ask", _agent_stream)
    monkeypatch.setattr(api, "stream_ask", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("classic stream_ask must not run in agentic mode")))

    with client.stream("POST", "/api/chat/stream",
                       json={"question": "q", "agentic": True}) as r:
        assert r.status_code == 200
        events = _collect_stream_events(r)

    progress = [e for e in events if e.get("type") == "progress"]
    tokens = [e["token"] for e in events if "token" in e]
    dones = [e["done"] for e in events if "done" in e]
    assert [p["phase"] for p in progress] == ["thought", "action"]
    assert progress[0]["message"] == "Planning search strategy..."
    assert "".join(tokens) == "agent answer"
    assert len(dones) == 1
    assert dones[0]["text"] == "agent answer"
    assert dones[0]["sources"] == ["doc.md"]
    for key in ("text", "sources", "hits", "grounded", "latency_ms",
                "standalone_question"):
        assert key in dones[0], f"missing key: {key}"


def test_chat_stream_classic_ignores_agent(monkeypatch):
    """Classic stream never touches the agent entry point."""
    monkeypatch.setattr(api, "stream_agentic_ask", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("stream_agentic_ask must not run without the flag")))
    with client.stream("POST", "/api/chat/stream",
                       json={"question": "What is this?"}) as r:
        assert r.status_code == 200
        events = _collect_stream_events(r)
    assert any("token" in e for e in events)
    assert any("done" in e for e in events)
    assert not any(e.get("type") == "progress" for e in events)


def test_agentic_event_stream_re_raises_agent_errors(monkeypatch):
    """A failure inside the agent surfaces as the endpoint's error event,
    not as a hung stream or a bare close."""
    from rag_demo.ollama_guard import OllamaTimeoutError

    def _failing_stream(question, **kwargs):
        yield {"type": "progress", "phase": "thought", "message": "Planning..."}
        raise OllamaTimeoutError("timed out")

    monkeypatch.setattr(api, "stream_agentic_ask", _failing_stream)
    with client.stream("POST", "/api/chat/stream",
                       json={"question": "q", "agentic": True}) as r:
        assert r.status_code == 200
        events = _collect_stream_events(r)
    assert events[0] == {"type": "progress", "phase": "thought",
                         "message": "Planning..."}
    assert events[1]["type"] == "error"
    assert "too long" in events[1]["error"]


def test_source_filtered_retriever_scopes_search():
    """The wrapper passes the upload source filter through to search()."""
    seen = {}

    class _StubRetriever:
        def search(self, query, k=None, source_filter=None):
            seen["query"] = query
            seen["k"] = k
            seen["source_filter"] = source_filter
            return ["hit"]

    wrapped = api._SourceFilteredRetriever(
        _StubRetriever(), source_filter="upload:doc1")
    assert wrapped.search("parcels", k=3) == ["hit"]
    assert seen == {"query": "parcels", "k": 3,
                    "source_filter": "upload:doc1"}


def test_agentic_retriever_none_without_doc(monkeypatch):
    """No document scope means whole-corpus search: no retriever passed."""
    assert api._agentic_retriever(None) is None


def test_agentic_retriever_scoped_with_doc(monkeypatch):
    """A document scope wraps the default retriever with its filter."""
    made = {}

    class _FakeRetriever:
        def __init__(self, settings):
            made["settings"] = settings

    monkeypatch.setattr(api, "Retriever", _FakeRetriever)
    retriever = api._agentic_retriever("doc1")
    assert isinstance(retriever, api._SourceFilteredRetriever)
    assert retriever._source_filter == "upload:doc1"
    assert isinstance(retriever._inner, _FakeRetriever)


def test_chat_stream_agentic_passes_scoped_retriever(monkeypatch):
    """Agentic stream with a document scope hands the agent a retriever;
    without one it hands None."""
    seen = {}

    def _agent_stream(question, **kwargs):
        seen["retriever"] = kwargs.get("retriever")
        yield {"answer": _fake_answer()}

    monkeypatch.setattr(api, "stream_agentic_ask", _agent_stream)
    monkeypatch.setattr(api, "Retriever", lambda settings: object())
    monkeypatch.setattr(api.conv_store, "get_document",
                        lambda doc_id: {"id": doc_id})

    with client.stream("POST", "/api/chat/stream",
                       json={"question": "q", "agentic": True,
                             "document_id": "doc1"}) as r:
        assert r.status_code == 200
        _collect_stream_events(r)
    assert isinstance(seen["retriever"], api._SourceFilteredRetriever)
    assert seen["retriever"]._source_filter == "upload:doc1"

    with client.stream("POST", "/api/chat/stream",
                       json={"question": "q", "agentic": True}) as r:
        assert r.status_code == 200
        _collect_stream_events(r)
    assert seen["retriever"] is None
