"""Tests for the LangGraph wrap (rag_demo/graph.py).

Run with RAG_DEMO_FAKE=1: deterministic fakes, no model downloads.
The graph must behave exactly like the chain for the same inputs: same
steps in the same order, same answer. The chain is the reference; the
graph is the system under test.
"""

import json
import os
import time

os.environ["RAG_DEMO_FAKE"] = "1"

import pytest

from langchain_core.language_models.fake import FakeListLLM

import rag_demo.chain as chain_mod
import rag_demo.graph as graph_mod
from rag_demo.chain import ask, stream_ask
from rag_demo.config import get_settings
from rag_demo.graph import ask_via_graph, stream_ask_via_graph
from rag_demo.retriever import Hit
from rag_demo.routing import ROUTE_CORPUS, ROUTE_GENERAL, RouteDecision


class StubEpisodic:
    """Episodic memory stub: records searches, never touches LanceDB."""

    def __init__(self, episodes=None):
        self.episodes = episodes or []
        self.calls = []
        self.saved = []

    def search(self, question, k=3):
        self.calls.append(question)
        return self.episodes[:k]

    def add(self, question, answer):
        self.saved.append((question, answer))


class FakeRetriever:
    """Retriever stub: returns canned hits, records every search."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []
        self.last_candidates = len(hits)
        self.last_floor_dropped = 0

    def search(self, query, user_groups=None, source_filter=None):
        self.calls.append((query, user_groups, source_filter))
        return list(self.hits)


def _hits():
    return [
        Hit(text="Express parcels max 1200 kg.", source="api-reference.md", score=0.9),
        Hit(text="Standard parcels max 30 kg.", source="shipping-guide.md", score=0.8),
    ]


def _fake_llm(responses):
    return FakeListLLM(responses=list(responses))


@pytest.fixture
def _fakes(monkeypatch):
    """Patch both the chain and the graph to use the same fakes.

    Returns a dict with the fake llm, the episodic stub, and the
    retriever, so each test can inspect what the run actually called.
    """
    episodic = StubEpisodic()
    llm = _fake_llm(["Based on the indexed docs: express parcels max 1200 kg."])
    monkeypatch.setattr(chain_mod, "get_llm", lambda settings: llm)
    monkeypatch.setattr(graph_mod, "get_llm", lambda settings: llm)
    monkeypatch.setattr(
        chain_mod, "get_cached_episodic_memory", lambda settings: episodic
    )
    monkeypatch.setattr(
        graph_mod, "get_cached_episodic_memory", lambda settings: episodic
    )
    return {"llm": llm, "episodic": episodic}


def test_graph_matches_chain_answer(_fakes):
    """Equivalence: same question, same fakes, same answer as the chain."""
    settings = get_settings()
    history = [("What is the max parcel weight?", "1200 kg per parcel")]
    chain_answer = ask(
        "what about express?",
        settings,
        history=history,
        retriever=FakeRetriever(_hits()),
    )
    graph_answer = ask_via_graph(
        "what about express?",
        settings,
        history=history,
        retriever=FakeRetriever(_hits()),
    )
    assert graph_answer.text == chain_answer.text
    assert graph_answer.sources == chain_answer.sources
    assert [h.source for h in graph_answer.hits] == [
        h.source for h in chain_answer.hits
    ]
    assert graph_answer.standalone_question == chain_answer.standalone_question
    assert graph_answer.grounded == chain_answer.grounded
    assert graph_answer.weak_evidence == chain_answer.weak_evidence


def test_graph_runs_nodes_in_order(monkeypatch):
    """The graph runs episodic -> rewrite -> retrieve -> generate."""
    order = []
    episodic = StubEpisodic()

    class RecordingLLM(FakeListLLM):
        def invoke(self, input, config=None, **kwargs):
            if isinstance(input, str) and "Standalone question" in input:
                order.append("rewrite")
            elif isinstance(input, list) or hasattr(input, "to_messages"):
                # The generate node calls the prompt|llm fragment, which
                # passes a ChatPromptValue (not a plain list) to the LLM.
                order.append("generate")
            else:
                order.append("invoke")
            return super().invoke(input, config=config, **kwargs)

    llm = RecordingLLM(
        responses=["What is the max parcel weight for express?"]
    )
    monkeypatch.setattr(graph_mod, "get_llm", lambda settings: llm)
    monkeypatch.setattr(
        graph_mod, "get_cached_episodic_memory", lambda settings: episodic
    )
    orig_search = StubEpisodic.search

    def _recorded_search(self, question, k=3):
        order.append("episodic")
        return orig_search(self, question, k)

    monkeypatch.setattr(StubEpisodic, "search", _recorded_search)
    retriever = FakeRetriever(_hits())
    orig_retrieve = FakeRetriever.search

    def _recorded_retrieve(self, query, user_groups=None, source_filter=None):
        order.append("retrieve")
        return orig_retrieve(self, query, user_groups, source_filter)

    monkeypatch.setattr(FakeRetriever, "search", _recorded_retrieve)
    answer = ask_via_graph(
        "what about express?",
        get_settings(),
        history=[("What is the max parcel weight?", "1200 kg per parcel")],
        retriever=retriever,
    )
    assert answer.text
    assert order == ["episodic", "rewrite", "retrieve", "generate"]


def test_graph_small_talk_bypasses_retrieval(_fakes):
    """Small talk never touches the retriever, as in the chain."""
    retriever = FakeRetriever(_hits())
    answer = ask_via_graph("hi", get_settings(), retriever=retriever)
    assert answer.text == "Based on the indexed docs: express parcels max 1200 kg."
    assert answer.sources == []
    assert answer.hits == []
    assert answer.grounded is True
    assert retriever.calls == []


def test_graph_episodic_context_reaches_generate(monkeypatch):
    """Episodic hits are supplementary context in the generate prompt."""
    seen_prompts = []
    episodic = StubEpisodic(
        [{"question": "prior q", "answer": "prior a"}]
    )

    class CapturingLLM(FakeListLLM):
        def invoke(self, input, config=None, **kwargs):
            if hasattr(input, "to_messages"):
                seen_prompts.append(input.to_messages())
            elif isinstance(input, list):
                seen_prompts.append(input)
            return super().invoke(input, config=config, **kwargs)

    llm = CapturingLLM(responses=["answer text"])
    monkeypatch.setattr(graph_mod, "get_llm", lambda settings: llm)
    monkeypatch.setattr(
        graph_mod, "get_cached_episodic_memory", lambda settings: episodic
    )
    answer = ask_via_graph(
        "What is the max parcel weight?",
        get_settings(),
        retriever=FakeRetriever(_hits()),
    )
    assert answer.text == "answer text"
    assert len(seen_prompts) == 1
    prompt_text = "\n".join(m.content for m in seen_prompts[0])
    assert "Prior conversations" in prompt_text
    assert "prior q" in prompt_text


def test_graph_stream_matches_chain_stream(_fakes):
    """Streaming equivalence: same tokens, same final answer as the chain."""
    settings = get_settings()
    history = [("What is the max parcel weight?", "1200 kg per parcel")]

    def _collect(gen):
        tokens, answer = [], None
        for event in gen:
            if "token" in event:
                tokens.append(event["token"])
            else:
                answer = event["answer"]
        return tokens, answer

    chain_tokens, chain_answer = _collect(
        stream_ask(
            "what about express?",
            settings,
            history=history,
            retriever=FakeRetriever(_hits()),
        )
    )
    graph_tokens, graph_answer = _collect(
        stream_ask_via_graph(
            "what about express?",
            settings,
            history=history,
            retriever=FakeRetriever(_hits()),
        )
    )
    assert "".join(graph_tokens) == graph_answer.text
    assert graph_answer.text == chain_answer.text
    assert graph_tokens == chain_tokens


def test_graph_stream_small_talk_is_one_token(_fakes):
    """Small talk streams as a single token, then the answer."""
    events = list(stream_ask_via_graph("hello", get_settings()))
    assert len(events) == 2
    assert events[0] == {
        "token": "Based on the indexed docs: express parcels max 1200 kg."
    }
    assert events[1]["answer"].sources == []
    assert events[1]["answer"].grounded is True


def test_graph_stream_close_abandons_generation(monkeypatch):
    """Closing the stream stops the generate node pulling (disconnect).

    Eight 60-char tokens: the head buffer flushes after four pulls, so
    the consumer sees two tokens and closes while the fifth pull is in
    flight. The node must stop there instead of running to all eight.
    """
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.outputs import Generation, GenerationChunk, LLMResult

    class SlowStreamingLLM(BaseLLM):
        text: str = "x" * 60
        pulls: int = 0

        @property
        def _llm_type(self) -> str:
            return "slow-streaming"

        def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
            return LLMResult(generations=[[Generation(text="unused")]])

        def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
            for _ in range(8):
                time.sleep(0.2)
                self.pulls += 1
                chunk = GenerationChunk(text="x" * 60)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk

    llm = SlowStreamingLLM()
    monkeypatch.setattr(graph_mod, "get_llm", lambda settings: llm)
    monkeypatch.setattr(
        graph_mod,
        "get_cached_episodic_memory",
        lambda settings: StubEpisodic(),
    )
    gen = stream_ask_via_graph(
        "What is the max parcel weight?",
        get_settings(),
        retriever=FakeRetriever(_hits()),
    )
    assert "token" in next(gen)
    assert "token" in next(gen)
    gen.close()
    pulls_at_close = llm.pulls
    # At most the in-flight pull finishes (4 buffered pulls + 1 in
    # flight); the node must not run on to all eight.
    assert pulls_at_close <= 5
    time.sleep(1.0)
    assert llm.pulls == pulls_at_close


def test_chain_dispatch_routes_to_graph(monkeypatch, _fakes):
    """RAG_DEMO_USE_LANGGRAPH=1 routes chain.ask/stream_ask to the graph."""
    sentinel = chain_mod.Answer(text="sentinel", sources=[], hits=[])
    called = {}

    def _fake_ask(*args, **kwargs):
        called["ask"] = True
        return sentinel

    def _fake_stream(*args, **kwargs):
        called["stream"] = True
        yield {"answer": sentinel}

    monkeypatch.setattr(graph_mod, "ask_via_graph", _fake_ask)
    monkeypatch.setattr(graph_mod, "stream_ask_via_graph", _fake_stream)
    settings = get_settings()
    settings.use_langgraph = True
    assert ask("hi", settings) is sentinel
    assert called.get("ask") is True
    events = list(stream_ask("hi", settings))
    assert called.get("stream") is True
    assert events[0]["answer"] is sentinel
    # Flag off: the graph is not called; the real chain answers.
    called.clear()
    settings.use_langgraph = False
    answer = ask(
        "What is the max parcel weight?",
        settings,
        retriever=FakeRetriever(_hits()),
    )
    assert called == {}
    assert answer.text


def test_use_langgraph_flag_defaults_off(monkeypatch):
    """The flag is off unless RAG_DEMO_USE_LANGGRAPH=1 is set."""
    monkeypatch.delenv("RAG_DEMO_USE_LANGGRAPH", raising=False)
    assert get_settings().use_langgraph is False
    monkeypatch.setenv("RAG_DEMO_USE_LANGGRAPH", "1")
    assert get_settings().use_langgraph is True

# --- Conditional retrieval (route node + general node) ---


def _fake_decider(route, confidence=0.9, elapsed_ms=100.0, fallback=False):
    """A decide_route stand-in returning one canned verdict."""

    def _decide(question, **kwargs):
        return RouteDecision(
            route=route,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            fallback=fallback,
        )

    return _decide


def _raising_decider(question, **kwargs):
    raise AssertionError("decide_route must not run here")


def test_graph_flag_off_routes_corpus_without_router(monkeypatch, _fakes):
    """Flag off: retrieve runs, the router never runs, route is corpus.

    The Answer carries route=None when routing is off, exactly as on
    the chain (see Answer); the route node's flow decision is corpus,
    asserted on the graph state directly.
    """
    settings = get_settings()
    settings.jev_routing = False
    monkeypatch.setattr(graph_mod, "decide_route", _raising_decider)
    # State-level: the route node resolves corpus with routing off.
    graph = graph_mod.build_question_graph(
        settings,
        llm=_fake_llm(["Based on the indexed docs: express parcels max 1200 kg."]),
        retriever=FakeRetriever(_hits()),
    )
    final = graph.invoke(
        {
            "question": "What is the max parcel weight?",
            "history": [],
            "document_id": None,
            "user_groups": None,
        }
    )
    assert final["route"] == ROUTE_CORPUS
    assert final["route_fallback"] is False
    # Answer-level: retrieval ran, no routing verdict on the Answer.
    retriever = FakeRetriever(_hits())
    answer = ask_via_graph(
        "What is the max parcel weight?", settings, retriever=retriever
    )
    assert retriever.calls != []
    assert answer.text == "Based on the indexed docs: express parcels max 1200 kg."
    assert answer.route is None
    assert answer.route_fallback is False


def test_graph_flag_on_general_skips_retrieval(monkeypatch, _fakes):
    """Flag on, router says general: no retrieval, general answer, route fields set."""
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(
        graph_mod,
        "decide_route",
        _fake_decider(ROUTE_GENERAL, confidence=0.85, elapsed_ms=123.0),
    )
    retriever = FakeRetriever(_hits())
    answer = ask_via_graph(
        "What is the capital of Australia?", settings, retriever=retriever
    )
    assert retriever.calls == []
    assert answer.text.startswith("🌐 General knowledge (not from your corpus).")
    assert answer.sources == []
    assert answer.hits == []
    assert answer.grounded is False
    assert answer.route == ROUTE_GENERAL
    assert answer.route_confidence == 0.85
    assert answer.route_ms == 123.0
    assert answer.route_fallback is False


def test_graph_flag_on_corpus_retrieves(monkeypatch, _fakes):
    """Flag on, router says corpus: retrieval runs, route fields set."""
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(
        graph_mod,
        "decide_route",
        _fake_decider(ROUTE_CORPUS, confidence=0.7, elapsed_ms=90.0),
    )
    retriever = FakeRetriever(_hits())
    answer = ask_via_graph(
        "What is the max parcel weight?", settings, retriever=retriever
    )
    assert len(retriever.calls) == 1
    assert answer.text == "Based on the indexed docs: express parcels max 1200 kg."
    assert answer.route == ROUTE_CORPUS
    assert answer.route_confidence == 0.7
    assert answer.route_ms == 90.0
    assert answer.route_fallback is False
    assert answer.grounded is True
    assert len(_fakes["episodic"].saved) == 1


def test_graph_router_fallback_goes_to_corpus(monkeypatch, _fakes):
    """A router failure degrades to the corpus path, never to a wrong answer."""
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(
        graph_mod,
        "decide_route",
        _fake_decider(
            ROUTE_CORPUS, confidence=0.0, elapsed_ms=5.0, fallback=True
        ),
    )
    retriever = FakeRetriever(_hits())
    answer = ask_via_graph(
        "What is the max parcel weight?", settings, retriever=retriever
    )
    assert len(retriever.calls) == 1
    assert answer.text == "Based on the indexed docs: express parcels max 1200 kg."
    assert answer.route == ROUTE_CORPUS
    assert answer.route_fallback is True


def test_graph_document_chat_never_skips_retrieval(monkeypatch, _fakes):
    """An attached document forces the corpus path even when the router says general."""
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(graph_mod, "decide_route", _raising_decider)
    retriever = FakeRetriever(_hits())
    answer = ask_via_graph(
        "What does this document say?",
        settings,
        retriever=retriever,
        document_id="doc-1",
    )
    assert len(retriever.calls) == 1
    assert retriever.calls[0][2] == "upload:doc-1"
    assert answer.route == ROUTE_CORPUS
    assert answer.route_ms == 0.0
    assert answer.route_fallback is False


def test_graph_stream_routed_general_streams_tokens(monkeypatch, _fakes):
    """Routed-general streams tokens, then the final Answer with route fields."""
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(
        graph_mod,
        "decide_route",
        _fake_decider(ROUTE_GENERAL, confidence=0.9, elapsed_ms=80.0),
    )
    retriever = FakeRetriever(_hits())
    tokens, answer = [], None
    for event in stream_ask_via_graph(
        "What is the capital of Australia?", settings, retriever=retriever
    ):
        if "token" in event:
            tokens.append(event["token"])
        else:
            answer = event["answer"]
    assert retriever.calls == []
    assert "".join(tokens) == answer.text
    assert answer.text.startswith("🌐 General knowledge (not from your corpus).")
    assert answer.route == ROUTE_GENERAL
    assert answer.route_confidence == 0.9
    assert answer.sources == []
    assert answer.hits == []


def test_graph_route_fields_in_audit_and_event_log(monkeypatch, tmp_path, _fakes):
    """Route fields land in the audit log and the question event."""
    settings = get_settings()
    settings.jev_routing = True
    log_path = tmp_path / "audit.jsonl"
    settings.audit_log_path = str(log_path)
    monkeypatch.setattr(
        graph_mod,
        "decide_route",
        _fake_decider(ROUTE_GENERAL, confidence=0.9, elapsed_ms=100.0),
    )
    events = []
    monkeypatch.setattr(
        graph_mod, "log_question_event", lambda **kwargs: events.append(kwargs)
    )
    answer = ask_via_graph(
        "What is the capital of France?", settings, retriever=FakeRetriever(_hits())
    )
    assert answer.route == ROUTE_GENERAL
    entry = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert entry["route"] == ROUTE_GENERAL
    assert entry["route_confidence"] == 0.9
    assert entry["route_ms"] == 100.0
    assert entry["route_fallback"] is False
    assert events and events[-1]["route"] == ROUTE_GENERAL
    assert events[-1]["route_confidence"] == 0.9
    assert events[-1]["route_ms"] == 100.0


def test_graph_routed_general_not_saved_to_episodic(monkeypatch, _fakes):
    """A general-knowledge answer is ungrounded, so it is not saved to episodic memory."""
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(graph_mod, "decide_route", _fake_decider(ROUTE_GENERAL))
    episodic = _fakes["episodic"]
    answer = ask_via_graph(
        "What is the capital of Australia?", settings, retriever=FakeRetriever(_hits())
    )
    assert answer.grounded is False
    assert episodic.saved == []


def test_graph_stream_routed_general_ignores_abstention_phrasing(monkeypatch):
    """A routed-general answer containing abstention phrasing streams whole.

    The hybrid fallback is a corpus-branch safety net: it must not
    truncate a general answer whose head happens to say "I don't know".
    """
    settings = get_settings()
    settings.jev_routing = True
    monkeypatch.setattr(graph_mod, "decide_route", _fake_decider(ROUTE_GENERAL))
    long_text = "I don't know the exact figure off hand, " + (
        "but the general idea is clear. " * 12
    )
    llm = _fake_llm([long_text])
    monkeypatch.setattr(graph_mod, "get_llm", lambda settings: llm)
    monkeypatch.setattr(
        graph_mod, "get_cached_episodic_memory", lambda settings: StubEpisodic()
    )
    tokens, answer = [], None
    for event in stream_ask_via_graph(
        "What is the capital of Australia?",
        settings,
        retriever=FakeRetriever(_hits()),
    ):
        if "token" in event:
            tokens.append(event["token"])
        else:
            answer = event["answer"]
    assert "".join(tokens) == answer.text
    assert answer.text.startswith("🌐 General knowledge (not from your corpus).")
    assert "I don't know" in answer.text
    assert answer.route == ROUTE_GENERAL
