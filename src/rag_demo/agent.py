"""Agentic RAG: plan → act → verify → refine loop.

Unlike classic RAG (single retrieve → generate), the agentic loop lets the
LLM:
  1. PLAN: break the question into sub-questions, decide what to search for
  2. ACT: execute tool calls (search_docs, etc.)
  3. VERIFY: evaluate if results answer the question, identify gaps
  4. REFINE: if gaps, refine queries and loop back (max N iterations)
  5. SYNTHESIZE: combine all evidence into a final answer

Tools are simple Python functions. The LLM drives the loop via structured
prompts — no external agent framework required.
"""

from __future__ import annotations

import json
import re

from .chain import (
    Answer,
    get_llm,
    is_abstention,
    is_fallback_answer,
    strip_think,
)
from .config import Settings, get_settings
from .retriever import Retriever


MAX_ITERATIONS = 4


def _search_docs_tool(query: str, k: int = 4, settings: Settings | None = None) -> str:
    """Tool: search the corpus. Returns formatted passages."""
    settings = settings or get_settings()
    retriever = Retriever(settings)
    hits = retriever.search(query, k=k)
    if not hits:
        return "No matching passages found."
    return "\n\n".join(
        f"[{h.source}]\n{h.text}" for h in hits
    )


PLAN_PROMPT = """You are a research planner. Break the user's question into \
searchable sub-questions.

User question: {question}

Reply with a JSON list of 1-3 specific search queries that would find the \
information needed. Example: ["API token expiry duration", "token refresh procedure"]

JSON:"""

VERIFY_PROMPT = """You are a research verifier. Given the user's question and \
the evidence gathered so far, decide if the question can be answered.

User question: {question}

Evidence:
{evidence}

Reply with JSON: {{"sufficient": true/false, "gaps": ["what's still missing"], \
"refined_queries": ["better search queries if gaps exist"]}}

JSON:"""

SYNTHESIZE_PROMPT = """Answer the user's question using ONLY the evidence below. \
Cite sources with [filename] markers.

User question: {question}

Evidence:
{evidence}

Answer concisely with citations:"""


def _parse_json_list(text: str) -> list[str]:
    """Extract a JSON list from LLM output (handles markdown fences)."""
    text = strip_think(text).strip()
    # Remove markdown code fences.
    text = re.sub(r"```(?:json)?\n?", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    # Fallback: treat each line as a query.
    lines = [ln.strip("- *").strip() for ln in text.splitlines() if ln.strip()]
    return lines[:3]


def _parse_verify(text: str) -> dict:
    """Parse the verify step's JSON response."""
    text = strip_think(text).strip()
    text = re.sub(r"```(?:json)?\n?", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "sufficient": bool(data.get("sufficient", False)),
                "gaps": data.get("gaps", []),
                "refined_queries": data.get("refined_queries", []),
            }
    except json.JSONDecodeError:
        pass
    return {"sufficient": True, "gaps": [], "refined_queries": []}


def agentic_ask(
    question: str,
    settings: Settings | None = None,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = False,
) -> Answer:
    """Run the plan → act → verify → refine loop.

    Returns an Answer with sources from all tool calls.
    """
    import time

    settings = settings or get_settings()
    llm = get_llm(settings)
    started = time.perf_counter()

    def _invoke(prompt: str) -> str:
        out = llm.invoke(prompt)
        text = out.content if hasattr(out, "content") else str(out)
        return strip_think(text.strip())

    # ---- PLAN ----
    plan_text = _invoke(PLAN_PROMPT.format(question=question))
    queries = _parse_json_list(plan_text)
    if not queries:
        queries = [question]
    if verbose:
        print(f"[plan] queries: {queries}")

    # ---- ACT → VERIFY → REFINE loop ----
    all_evidence: list[str] = []
    all_sources: set[str] = set()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"[act] iteration {iteration}, searching: {queries}")

        # ACT: execute searches.
        for q in queries:
            result = _search_docs_tool(q, k=4, settings=settings)
            all_evidence.append(f"--- Query: {q} ---\n{result}")
            # Extract sources for citations.
            for m in re.finditer(r"\[([^\]]+)\]", result):
                all_sources.add(m.group(1))

        # VERIFY: is the evidence sufficient?
        evidence_text = "\n\n".join(all_evidence)
        verify_text = _invoke(VERIFY_PROMPT.format(
            question=question, evidence=evidence_text
        ))
        verdict = _parse_verify(verify_text)
        if verbose:
            print(f"[verify] sufficient={verdict['sufficient']}, "
                  f"gaps={verdict['gaps']}")

        if verdict["sufficient"]:
            break

        # REFINE: use refined queries for the next iteration.
        refined = verdict.get("refined_queries", [])
        if not refined:
            break
        queries = refined[:3]

    # ---- SYNTHESIZE ----
    evidence_text = "\n\n".join(all_evidence)
    answer_text = _invoke(SYNTHESIZE_PROMPT.format(
        question=question, evidence=evidence_text
    ))

    latency_ms = (time.perf_counter() - started) * 1000
    return Answer(
        text=answer_text,
        sources=sorted(all_sources),
        hits=[],
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(answer_text),
        standalone_question=None,
    )
