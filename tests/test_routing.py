"""Tests for the Jev routing trial (src/rag_demo/routing.py + eval scorer).

Hermetic: fake decision clients only, no Ollama, no network, no
jev_demo checkout. The only tests that touch the client resolver use a
stub jev_demo package built in a temp dir (or an empty dir for the
missing-client case).

The wiring tests run the fake tier (RAG_DEMO_FAKE=1): deterministic
fakes, no model downloads. The wire-format test skips where FastAPI is
missing (sandbox) and runs on the iMac.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["RAG_DEMO_FAKE"] = "1"

import pytest

from rag_demo import ingest
from rag_demo.chain import ask, stream_ask
from rag_demo.config import Settings, get_settings
from rag_demo.routing import (
    ROUTE_CORPUS,
    ROUTE_GENERAL,
    RouteDecision,
    RoutingError,
    decide_route,
    load_client,
    routing_enabled,
)


def _fake_client(choice, confidence=None, probabilities=None, error=None):
    """A client whose decide() returns one canned choice answer."""

    class FakeClient:
        def decide(self, state, questions, model=None):
            if error is not None:
                raise error
            answer = SimpleNamespace(
                choice=choice,
                confidence=confidence,
                probabilities=probabilities or {choice: 0.9},
            )
            return SimpleNamespace(
                answers={"route": answer}, elapsed_ms=123.0
            )

    return FakeClient()


def test_general_knowledge_high_confidence_routes_general():
    decision = decide_route(
        "What is the capital of France?",
        client=_fake_client(ROUTE_GENERAL, confidence=0.9),
    )
    assert decision.route == ROUTE_GENERAL
    assert decision.confidence == 0.9
    assert decision.fallback is False


def test_general_knowledge_below_threshold_routes_corpus():
    decision = decide_route(
        "What is the capital of France?",
        client=_fake_client(ROUTE_GENERAL, confidence=0.5),
    )
    assert decision.route == ROUTE_CORPUS
    assert decision.fallback is False


def test_threshold_boundary_is_inclusive():
    decision = decide_route(
        "What is the capital of France?",
        client=_fake_client(ROUTE_GENERAL, confidence=0.6),
        threshold=0.6,
    )
    assert decision.route == ROUTE_GENERAL


def test_corpus_choice_routes_corpus_regardless_of_confidence():
    decision = decide_route(
        "How do I create a shipment via the API?",
        client=_fake_client(ROUTE_CORPUS, confidence=0.99),
    )
    assert decision.route == ROUTE_CORPUS


def test_missing_confidence_falls_back_to_max_probability():
    decision = decide_route(
        "What is the capital of France?",
        client=_fake_client(
            ROUTE_GENERAL,
            confidence=None,
            probabilities={ROUTE_GENERAL: 0.75, ROUTE_CORPUS: 0.25},
        ),
    )
    assert decision.route == ROUTE_GENERAL
    assert decision.confidence == 0.75


def test_client_error_routes_corpus_with_fallback():
    decision = decide_route(
        "How do I create a shipment via the API?",
        client=_fake_client(ROUTE_CORPUS, error=RuntimeError("boom")),
    )
    assert decision.route == ROUTE_CORPUS
    assert decision.fallback is True
    assert "boom" in (decision.error or "")


def test_empty_question_routes_corpus_with_fallback():
    decision = decide_route("   ", client=_fake_client(ROUTE_GENERAL, 0.9))
    assert decision.route == ROUTE_CORPUS
    assert decision.fallback is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("on", True),
    ],
)
def test_routing_enabled_env(monkeypatch, value, expected):
    monkeypatch.delenv("RAG_DEMO_JEV_ROUTING", raising=False)
    if value is not None:
        monkeypatch.setenv("RAG_DEMO_JEV_ROUTING", value)
    assert routing_enabled() is expected


def _clear_jev_demo(monkeypatch):
    """Drop any imported jev_demo so the resolver really has to find it."""
    for name in [n for n in sys.modules if n == "jev_demo" or n.startswith("jev_demo.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_load_client_missing_raises_helpful_error(monkeypatch, tmp_path):
    _clear_jev_demo(monkeypatch)
    monkeypatch.setenv("JEV_DEMO_SRC", str(tmp_path))
    with pytest.raises(RoutingError, match="JEV_DEMO_SRC"):
        load_client()


def test_load_client_finds_stub_package(monkeypatch, tmp_path):
    _clear_jev_demo(monkeypatch)
    pkg = tmp_path / "jev_demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "client.py").write_text(
        "from types import SimpleNamespace\n"
        "\n"
        "class SystemOneClient:\n"
        "    def __init__(self, base_url=None, model=None, timeout=None):\n"
        "        self.model = model\n"
        "    def decide(self, state, questions, model=None):\n"
        "        answer = SimpleNamespace(\n"
        "            choice='general_knowledge', confidence=0.8,\n"
        "            probabilities={'general_knowledge': 0.8, 'corpus': 0.2},\n"
        "        )\n"
        "        return SimpleNamespace(\n"
        "            answers={'route': answer}, elapsed_ms=42.0\n"
        "        )\n"
    )
    monkeypatch.setenv("JEV_DEMO_SRC", str(tmp_path))
    client = load_client()
    decision = decide_route("What is the capital of France?", client=client)
    assert decision.route == ROUTE_GENERAL
    assert decision.elapsed_ms == 42.0


def _load_eval_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "evals"
        / "jev_routing_eval.py"
    )
    spec = importlib.util.spec_from_file_location("jev_routing_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_cases_counts_accuracy_per_class():
    eval_mod = _load_eval_module()
    cases = [
        {"id": "c1", "question": "corpus q", "expected": ROUTE_CORPUS},
        {"id": "c2", "question": "general q", "expected": ROUTE_GENERAL},
        {"id": "c3", "question": "general q 2", "expected": ROUTE_GENERAL},
    ]

    def decider(question):
        route = ROUTE_CORPUS if "corpus" in question else ROUTE_GENERAL
        return SimpleNamespace(
            route=route, confidence=0.9, elapsed_ms=100.0,
            fallback=False, error=None,
        )

    report = eval_mod.score_cases(cases, decider)
    assert report["total"] == 3
    assert report["correct"] == 3
    assert report["accuracy"] == 1.0
    assert report["corpus_correct"] == 1
    assert report["general_correct"] == 2
    assert report["wrong"] == []


def test_score_cases_lists_wrong_routes():
    eval_mod = _load_eval_module()
    cases = [
        {"id": "c1", "question": "corpus q", "expected": ROUTE_CORPUS},
        {"id": "c2", "question": "general q", "expected": ROUTE_GENERAL},
    ]

    def always_corpus(question):
        return SimpleNamespace(
            route=ROUTE_CORPUS, confidence=0.9, elapsed_ms=100.0,
            fallback=False, error=None,
        )

    report = eval_mod.score_cases(cases, always_corpus)
    assert report["correct"] == 1
    assert len(report["wrong"]) == 1
    assert report["wrong"][0]["id"] == "c2"
    assert report["general_correct"] == 0


# --- Part 2: wiring ask()/stream_ask() behind the flag --------------------
# Fake tier (RAG_DEMO_FAKE=1, set at the top of this file): deterministic
# fakes, no model downloads. The decider is monkeypatched so no Jev
# client or Ollama is involved.


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    s = get_settings()
    s.index_dir = tmp_path_factory.mktemp("index")
    return s


@pytest.fixture(scope="module")
def _indexed(settings):
    ingest.build_index(settings)
    return settings


def _fake_decider(route, confidence=0.9, elapsed_ms=100.0, fallback=False, error=None):
    def decider(question, **kwargs):
        return RouteDecision(
            route=route,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            fallback=fallback,
            error=error,
        )

    return decider


def test_jev_routing_flag_defaults_off():
    assert Settings().jev_routing is False


def test_jev_routing_flag_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_DEMO_JEV_ROUTING", "1")
    assert get_settings().jev_routing is True


def test_ask_routes_general_knowledge(monkeypatch, _indexed):
    monkeypatch.setattr(_indexed, "jev_routing", True)
    monkeypatch.setattr(
        "rag_demo.chain.decide_route",
        _fake_decider(ROUTE_GENERAL, confidence=0.9, elapsed_ms=100.0),
    )
    answer = ask("What is the capital of France?", _indexed)
    assert answer.grounded is False
    assert answer.route == ROUTE_GENERAL
    assert answer.route_confidence == 0.9
    assert answer.route_ms == 100.0
    assert answer.route_fallback is False
    assert answer.sources == []
    assert "General knowledge" in answer.text


def test_ask_corpus_route_runs_retrieval(monkeypatch, _indexed):
    monkeypatch.setattr(_indexed, "jev_routing", True)
    monkeypatch.setattr(
        "rag_demo.chain.decide_route",
        _fake_decider(ROUTE_CORPUS, confidence=0.8, elapsed_ms=50.0),
    )
    answer = ask(
        "How long are API tokens valid, and what happens when one expires?",
        _indexed,
    )
    assert answer.route == ROUTE_CORPUS
    assert answer.route_confidence == 0.8
    assert answer.route_ms == 50.0
    assert answer.route_fallback is False
    assert answer.sources  # retrieval ran


def test_ask_routing_failure_falls_back_to_corpus(monkeypatch, _indexed):
    monkeypatch.setattr(_indexed, "jev_routing", True)
    monkeypatch.setattr(
        "rag_demo.chain.decide_route",
        _fake_decider(
            ROUTE_CORPUS,
            confidence=0.0,
            elapsed_ms=10.0,
            fallback=True,
            error="boom",
        ),
    )
    answer = ask(
        "How long are API tokens valid, and what happens when one expires?",
        _indexed,
    )
    assert answer.route == ROUTE_CORPUS
    assert answer.route_fallback is True
    assert answer.sources


def test_ask_flag_off_never_calls_decider(monkeypatch, _indexed):
    monkeypatch.setattr(_indexed, "jev_routing", False)

    def _boom(question, **kwargs):
        raise AssertionError("decide_route must not run when routing is off")

    monkeypatch.setattr("rag_demo.chain.decide_route", _boom)
    answer = ask(
        "How long are API tokens valid, and what happens when one expires?",
        _indexed,
    )
    assert answer.route is None
    assert answer.sources


def test_stream_ask_routes_general_knowledge(monkeypatch, _indexed):
    monkeypatch.setattr(_indexed, "jev_routing", True)
    monkeypatch.setattr(
        "rag_demo.chain.decide_route",
        _fake_decider(ROUTE_GENERAL, confidence=0.9, elapsed_ms=100.0),
    )
    events = list(stream_ask("What is the capital of France?", _indexed))
    assert len(events) == 2
    final = events[1]["answer"]
    assert events[0]["token"] == final.text
    assert final.route == ROUTE_GENERAL
    assert final.grounded is False


def test_audit_log_captures_route(monkeypatch, tmp_path, _indexed):
    monkeypatch.setattr(_indexed, "jev_routing", True)
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(_indexed, "audit_log_path", str(log_path))
    monkeypatch.setattr(
        "rag_demo.chain.decide_route",
        _fake_decider(ROUTE_GENERAL, confidence=0.9, elapsed_ms=100.0),
    )
    ask("What is the capital of France?", _indexed)
    entry = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert entry["route"] == ROUTE_GENERAL
    assert entry["route_confidence"] == 0.9
    assert entry["route_ms"] == 100.0
    assert entry["route_fallback"] is False


def test_answer_to_wire_includes_route():
    api = pytest.importorskip("rag_demo.api")
    from rag_demo.chain import Answer

    answer = Answer(
        text="x", sources=[], hits=[], route=ROUTE_GENERAL, route_confidence=0.9
    )
    wire = api._answer_to_wire(answer)
    assert wire["route"] == ROUTE_GENERAL
    assert wire["route_confidence"] == 0.9
