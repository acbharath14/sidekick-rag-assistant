"""Jev routing trial: one decision-model question in front of retrieval.

Trial, part 1 (eval first): this module is UNWIRED. Nothing in `chain.py`
or `config.py` imports it yet, and the `RAG_DEMO_JEV_ROUTING` flag defaults
to off. Wiring it into `ask()` waits on the eval in
`evals/jev_routing_eval.py` (see the trial spec at
`~/workspace/your_files/sidekick-jev-routing-trial-spec.md`).

The decision is one Jev `choice` question: does this question need the
indexed corpus, or can it be answered from general knowledge? The Jev
client is imported from the ollama-jev-decision-demo repo, never copied:
`routing.load_client()` resolves it from `JEV_DEMO_SRC`, a sibling
checkout (`../ollama-jev-decision-demo/src`), or
`~/Documents/ollama-jev-decision-demo/src`, and raises RoutingError with
a clear message when it cannot be found.

Fallback contract (the whole point of the trial): any failure — client
missing, DecisionError, 404, timeout — or a confidence below threshold
routes to the corpus, i.e. today's behaviour. A wrong or uncertain route
must degrade to current behaviour, never to a wrong answer.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROUTE_CORPUS = "corpus"
ROUTE_GENERAL = "general_knowledge"

DEFAULT_MODEL = "tev1:0.8b"
DEFAULT_BASE_URL = "http://localhost:11434"
# Routing must stay cheap: ~1s per question on the iMac. 30s bounds a
# wedged call instead of letting it stall a chat turn for two minutes.
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

FLAG_ENV = "RAG_DEMO_JEV_ROUTING"
CLIENT_SRC_ENV = "JEV_DEMO_SRC"


class RoutingError(Exception):
    """The routing decision could not be made (client missing/unreachable)."""


@dataclass
class RouteDecision:
    """One routing verdict. `fallback` means "route corpus, as today"."""

    route: str
    confidence: float
    elapsed_ms: float = 0.0
    fallback: bool = False
    error: Optional[str] = None


def routing_enabled() -> bool:
    """True only when RAG_DEMO_JEV_ROUTING is set to a truthy value."""
    return os.environ.get(FLAG_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def route_question() -> dict:
    """Build a fresh routing question (never mutate a shared constant)."""
    return {
        "route": {
            "type": "choice",
            "instructions": (
                "Does this question need the indexed corpus, or can it "
                "be answered from general knowledge?"
            ),
            "criteria": {
                "corpus": (
                    "The answer is in the indexed enterprise corpus "
                    "(Meridian API docs, runbooks, Playwright docs, "
                    "GitHub docs, Confluence pages)"
                ),
                "general_knowledge": (
                    "The answer comes from general knowledge, not from "
                    "any indexed document"
                ),
            },
        }
    }


def _candidate_client_srcs() -> list[str]:
    """Where to look for the ollama-jev-decision-demo checkout."""
    srcs: list[str] = []
    env_src = os.environ.get(CLIENT_SRC_ENV, "").strip()
    if env_src:
        srcs.append(env_src)
    here = Path(__file__).resolve()
    srcs.append(str(here.parent.parent.parent.parent / "ollama-jev-decision-demo" / "src"))
    srcs.append(str(Path.home() / "Documents" / "ollama-jev-decision-demo" / "src"))
    return srcs


def load_client(
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
):
    """Return a SystemOneClient, importing it from the Jev demo repo.

    Raises RoutingError when the client cannot be found, telling the
    user exactly where it looked and how to point at it.
    """
    try:
        from jev_demo.client import SystemOneClient  # noqa: F401
    except ImportError:
        looked: list[str] = []
        for src in _candidate_client_srcs():
            looked.append(src)
            if src not in sys.path:
                sys.path.insert(0, src)
            try:
                from jev_demo.client import SystemOneClient  # noqa: F401

                break
            except ImportError:
                continue
        else:
            raise RoutingError(
                "jev_demo client not found. Clone ollama-jev-decision-demo "
                "next to this repo, or set JEV_DEMO_SRC to its src/ "
                "directory. Looked in: " + "; ".join(looked)
            ) from None
    return SystemOneClient(base_url=base_url, model=model, timeout=timeout)


def _confidence(answer: Any) -> float:
    """Confidence for a choice answer, else the best probability."""
    conf = getattr(answer, "confidence", None)
    if conf is not None:
        return float(conf)
    probs = getattr(answer, "probabilities", {}) or {}
    return float(max(probs.values())) if probs else 0.0


def decide_route(
    question: str,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> RouteDecision:
    """Route one question to the corpus or to general knowledge.

    `client` is injectable for tests; when omitted, the real Jev client
    is loaded. Any failure routes to the corpus with fallback=True.
    """
    started = time.perf_counter()
    if not (question or "").strip():
        return RouteDecision(
            route=ROUTE_CORPUS,
            confidence=0.0,
            elapsed_ms=0.0,
            fallback=True,
            error="empty question",
        )
    try:
        if client is None:
            client = load_client(base_url=base_url, model=model, timeout=timeout)
        result = client.decide(
            {"question": question}, route_question(), model=model
        )
        answer = result.answers.get("route")
        if answer is None or getattr(answer, "choice", None) is None:
            raise RoutingError("routing answer missing 'route' choice")
        confidence = _confidence(answer)
        elapsed = float(getattr(result, "elapsed_ms", 0.0)) or (
            (time.perf_counter() - started) * 1000.0
        )
        if answer.choice == ROUTE_GENERAL and confidence >= threshold:
            return RouteDecision(
                route=ROUTE_GENERAL, confidence=confidence, elapsed_ms=elapsed
            )
        return RouteDecision(
            route=ROUTE_CORPUS, confidence=confidence, elapsed_ms=elapsed
        )
    except Exception as exc:  # noqa: BLE001 — any failure degrades to corpus
        return RouteDecision(
            route=ROUTE_CORPUS,
            confidence=0.0,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            fallback=True,
            error=str(exc),
        )
