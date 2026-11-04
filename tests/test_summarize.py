"""Tests for rag_demo.summarize: retry, raw fallback, hierarchical reduce.

Run:  python -m pytest tests/test_summarize.py -v
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.summarize import (  # noqa: E402
    REDUCE_LIMIT_CHARS,
    _invoke,
    _invoke_with_retry,
    _reduce_summaries,
    summarize,
    summarize_events,
)


class FakeLLM:
    """Scripted fake: each entry is ("ok", text), ("error", exc),
    ("timeout", sleep_seconds), or ("echo_len", None) to return the prompt
    length (lets tests inspect what the reduce step saw)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.script:
            kind, payload = self.script.pop(0)
        else:
            kind, payload = "ok", "default summary"
        if kind == "ok":
            return SimpleNamespace(content=payload)
        if kind == "error":
            raise payload
        if kind == "timeout":
            time.sleep(payload)
            return SimpleNamespace(content="too late")
        if kind == "echo_len":
            return SimpleNamespace(content=f"LEN={len(prompt)}")
        raise AssertionError(f"unknown script kind {kind}")


def _summarize_with(fake, text, **kw):
    with patch("rag_demo.summarize.get_llm", return_value=fake):
        return summarize(text, settings=object(), **kw)


def test_short_doc_single_call():
    fake = FakeLLM([("ok", "Short summary.")])
    out = _summarize_with(fake, "hello world")
    assert out == "Short summary."
    assert len(fake.calls) == 1


def test_invoke_legacy_returns_empty_on_error():
    fake = FakeLLM([("error", RuntimeError("boom"))])
    assert _invoke(fake, "p", timeout=5) == ""


def test_retry_then_success():
    fake = FakeLLM([("error", RuntimeError("blip")), ("ok", "recovered")])
    with patch("time.sleep"):
        text, status = _invoke_with_retry(fake, "p", timeout=5, backoff=(0, 0))
    assert (text, status) == ("recovered", "ok")
    assert len(fake.calls) == 2


def test_retry_exhausts_attempts():
    fake = FakeLLM([("error", RuntimeError("x"))] * 5)
    with patch("time.sleep"):
        text, status = _invoke_with_retry(fake, "p", timeout=5, attempts=3, backoff=(0, 0))
    assert (text, status) == ("", "error")
    assert len(fake.calls) == 3


def test_timeout_is_not_retried():
    fake = FakeLLM([("timeout", 5)])
    text, status = _invoke_with_retry(fake, "p", timeout=0.2, backoff=(0, 0))
    assert (text, status) == ("", "timeout")
    assert len(fake.calls) == 1  # no retry on timeout


def test_short_doc_persistent_failure_message():
    fake = FakeLLM([("error", RuntimeError("x"))] * 5)
    with patch("time.sleep"):
        out = _summarize_with(fake, "hello world")
    assert "didn't complete" in out


def _n_chunks(text):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    return len(splitter.split_text(text))


def test_long_doc_map_reduce_happy_path():
    text = "word " * 2000  # ~10k chars -> multiple ~4k chunks
    n = _n_chunks(text)
    assert n > 1
    fake = FakeLLM([("ok", f"bullets {i}") for i in range(n)] + [("echo_len", None)])
    with patch("time.sleep"):
        out = _summarize_with(fake, text)
    assert out == f"LEN={len(fake.calls[-1])}"
    assert len(fake.calls) == n + 1  # n map + 1 reduce


def test_failed_chunk_falls_back_to_raw_not_dropped():
    text = ("aaa " * 975) + ("UNIQUE-CHUNK2-MARKER " * 185) + ("bbb " * 975)
    assert len(text) > 6000
    n = _n_chunks(text)
    assert n > 1
    # Chunk 1 ok; chunk 2 fails all 3 attempts; remaining chunks ok; then reduce.
    script = [("ok", "bullets 0")]
    script += [("error", RuntimeError("x"))] * 3
    script += [("ok", f"bullets {i}") for i in range(2, n)]
    script += [("echo_len", None)]
    fake = FakeLLM(script)
    with patch("time.sleep"):
        out = _summarize_with(fake, text)
    reduce_prompt = fake.calls[-1]
    assert "[Unprocessed section" in reduce_prompt
    assert "UNIQUE-CHUNK2-MARKER" in reduce_prompt  # content preserved
    assert f"1 of {n} sections couldn't be summarized" in out


def test_reduce_recurses_on_huge_input():
    partials = ["y" * 4000] * 8  # combined ~32k > REDUCE_LIMIT_CHARS
    fake = FakeLLM([("ok", "R")] * 30)
    with patch("time.sleep"):
        out = _reduce_summaries(partials, fake, timeout=5)
    assert out == "R"
    assert all(len(c) <= REDUCE_LIMIT_CHARS + 2000 for c in fake.calls)
    # 8 partials -> two halves of 4 -> pairs fit under the limit (2x4000 < 12k),
    # so: 4 pair-reduces + 2 half-reduces + 1 final = 7 calls, not 15.
    assert len(fake.calls) == 7


def test_reduce_single_partial_passthrough():
    fake = FakeLLM([("ok", "only")])
    with patch("time.sleep"):
        assert _reduce_summaries(["single"], fake, timeout=5) == "only"


def _collect(text, fake):
    with patch("rag_demo.summarize.get_llm", return_value=fake):
        return list(summarize_events(text, settings=object()))


def test_events_single_call_sequence():
    fake = FakeLLM([("ok", "Short summary.")])
    with patch("time.sleep"):
        events = _collect("hello world", fake)
    assert events[0] == {"type": "start", "mode": "single", "total": 1}
    assert events[-1] == {"type": "done", "summary": "Short summary."}
    assert len(events) == 2


def test_events_map_reduce_sequence():
    text = "word " * 2000
    n = _n_chunks(text)
    assert n > 1
    fake = FakeLLM([("ok", f"bullets {i}") for i in range(n)] + [("ok", "Final.")])
    with patch("time.sleep"):
        events = _collect(text, fake)
    assert events[0] == {"type": "start", "mode": "map_reduce", "total": n}
    map_events = [e for e in events if e.get("stage") == "map"]
    assert [e["done"] for e in map_events] == list(range(1, n + 1))
    assert {"type": "progress", "stage": "reduce"} in events
    assert events[-1] == {"type": "done", "summary": "Final."}


def test_events_error_when_llm_unavailable():
    with patch("rag_demo.summarize.get_llm", side_effect=RuntimeError("no model")):
        events = list(summarize_events("hello", settings=object()))
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "no model" in events[0]["error"]


def test_summarize_matches_event_stream():
    text = "word " * 2000
    n = _n_chunks(text)
    script = [("ok", f"bullets {i}") for i in range(n)] + [("ok", "Final.")]
    with patch("time.sleep"):
        via_events = _collect(text, FakeLLM(list(script)))[-1]["summary"]
        direct = _summarize_with(FakeLLM(list(script)), text)
    assert direct == via_events == "Final."


def test_summarize_raises_on_error_event():
    import pytest

    with patch("rag_demo.summarize.get_llm", side_effect=RuntimeError("no model")):
        with pytest.raises(RuntimeError, match="no model"):
            summarize("hello", settings=object())
