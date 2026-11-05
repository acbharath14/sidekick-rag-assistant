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
