"""Tests for the Jev routing trial (src/rag_demo/routing.py + eval scorer).

Hermetic: fake decision clients only, no Ollama, no network, no
jev_demo checkout. The only tests that touch the client resolver use a
stub jev_demo package built in a temp dir (or an empty dir for the
missing-client case).
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from rag_demo.routing import (
    ROUTE_CORPUS,
    ROUTE_GENERAL,
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
