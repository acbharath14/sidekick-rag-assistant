"""Tests for rag_demo.tracing: opt-in Langfuse tracing stays off unless
the keys are set, and never breaks the app when it is off.

Run:  RAG_DEMO_FAKE=1 python -m pytest tests/test_tracing.py -v
"""
import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo import tracing


def _clear_langfuse_env(monkeypatch):
    for var in (
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_tracing_off_without_keys(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    assert tracing.tracing_enabled() is False
    assert tracing.get_callbacks() == []
    assert tracing.run_config(run_name="generate") is None


def test_tracing_off_with_url_only(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")
    assert tracing.tracing_enabled() is False
    assert tracing.get_callbacks() == []


def test_tracing_off_with_keys_only(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    assert tracing.tracing_enabled() is False
    assert tracing.get_callbacks() == []


def test_spans_are_noop_when_disabled(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    with tracing.question_span("what is this?") as span:
        assert span is None
    with tracing.span("rerank") as span:
        assert span is None


def test_question_span_accepts_trace_context_when_disabled(monkeypatch):
    """session_id/user_id/tags are optional extras: off means a no-op,
    not a TypeError."""
    _clear_langfuse_env(monkeypatch)
    with tracing.question_span(
        "what is this?", session_id="conv-1", user_id="u-1", tags=("feature:chat",)
    ) as span:
        assert span is None


def test_span_accepts_type_input_and_tags_when_disabled(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    with tracing.span(
        "rerank", as_type="retriever", input={"candidates": 3}, tags=("feature:chat",)
    ) as span:
        assert span is None


def test_set_output_flush_and_shutdown_are_safe_without_tracing(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    tracing.set_output(None, "answer")  # no observation to update
    tracing.set_output(None, None)
    tracing.flush()
    tracing.shutdown()


def test_ask_accepts_session_id(monkeypatch):
    """The chat path threads the conversation id through as session_id
    without changing the answer."""
    _clear_langfuse_env(monkeypatch)
    from rag_demo.chain import ask
    from rag_demo.config import get_settings

    answer = ask("What is this?", settings=get_settings(), session_id="conv-1")
    assert answer.text


def test_stream_ask_accepts_session_id(monkeypatch):
    _clear_langfuse_env(monkeypatch)
    from rag_demo.chain import stream_ask
    from rag_demo.config import get_settings

    events = list(
        stream_ask("What is this?", settings=get_settings(), session_id="conv-1")
    )
    assert any("answer" in event for event in events)


def test_summarize_still_emits_start_and_done(monkeypatch):
    """The summarize root span wraps the same event stream as before."""
    _clear_langfuse_env(monkeypatch)
    from rag_demo.config import get_settings
    from rag_demo.summarize import summarize_events

    events = list(summarize_events("short text", get_settings()))
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"


def test_invoke_llm_without_tracing_calls_plain_invoke(monkeypatch):
    """Hand-rolled fakes define invoke(prompt) with no config kwarg;
    invoke_llm must not pass one when tracing is off."""
    _clear_langfuse_env(monkeypatch)

    class FakeLLM:
        def invoke(self, prompt):
            return "ok"

    assert tracing.invoke_llm(FakeLLM(), "hello", run_name="rewrite") == "ok"


def test_get_callbacks_never_raises_with_keys(monkeypatch):
    """Keys set but no server (or no SDK): tracing degrades to a plain
    empty callback list instead of raising. The assertion is only that
    the call returns a list — whether the SDK is installed or not."""
    _clear_langfuse_env(monkeypatch)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    assert isinstance(tracing.get_callbacks(), list)


def test_ask_works_with_tracing_off(monkeypatch):
    """End to end through ask() with fakes and no keys: the question
    span is a no-op and the answer is unchanged."""
    _clear_langfuse_env(monkeypatch)
    from rag_demo.chain import ask
    from rag_demo.config import get_settings

    answer = ask("What is this?", settings=get_settings())
    assert answer.text
    assert answer.latency_ms is not None
