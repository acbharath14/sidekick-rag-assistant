"""Agentic RAG via the ReAct pattern (Thought → Action → Observation).

Unlike classic RAG (single retrieve → generate), the agent iterates:
  1. THOUGHT: reason about what information is needed, plan search queries
  2. ACTION: execute tool calls (search_docs, etc.)
  3. OBSERVATION: evaluate results, decide if more searching is needed
  4. Repeat until sufficient evidence or max iterations, then synthesize

This follows Yao et al.'s ReAct framework, with a corrective-RAG style
evaluation step (assess retrieval quality, refine queries if gaps remain).

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


def _search_web_tool(query: str, k: int = 5) -> str:
    """Tool: search the web via DuckDuckGo. Returns titles, URLs, snippets."""
    import requests
    from html.parser import HTMLParser
    from urllib.parse import parse_qs, urlparse

    url = "https://html.duckduckgo.com/html/"
    try:
        resp = requests.post(
            url, data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Web search failed: {e}"

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results: list[dict] = []
            self._in_a = False
            self._href = ""
            self._text = ""

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "a" and "result__a" in attrs.get("class", ""):
                self._in_a = True
                self._href = attrs.get("href", "")

        def handle_data(self, data):
            if self._in_a:
                self._text += data

        def handle_endtag(self, tag):
            if tag == "a" and self._in_a:
                self._in_a = False
                href = self._href
                if "uddg=" in href:
                    qs = parse_qs(urlparse(href).query)
                    href = qs.get("uddg", [href])[0]
                self.results.append({"title": self._text.strip(), "url": href})
                self._text = ""
                self._href = ""

    parser = _Parser()
    parser.feed(resp.text)
    results = parser.results[:k]
    if not results:
        return "No web results found."
    return "\n\n".join(f"[{r['title']}]({r['url']})" for r in results)


# Available tools for the agent. The THOUGHT step chooses which to use.
TOOLS = {
    "search_docs": "Search the local document corpus (use for questions about the indexed docs).",
    "search_web": "Search the web via DuckDuckGo (use for current events, prices, deals, general knowledge).",
}


THOUGHT_PROMPT = """You are a research planner. Break the user's question into \
searchable sub-questions and choose the right tool for each.

Available tools:
- search_docs: Search the local document corpus (indexed docs, PDFs, etc.)
- search_web: Search the web (current events, prices, deals, general knowledge)

User question: {question}

Reply with JSON: a list of objects like \
[{{"tool": "search_docs", "query": "..."}}, {{"tool": "search_web", "query": "..."}}]
Use 1-3 tool calls. Prefer search_docs for questions about the indexed corpus; \
search_web for anything requiring current/external information.

JSON:"""

OBSERVATION_PROMPT = """You are evaluating retrieved evidence. Given the user's question and \
the evidence gathered so far, decide if the question can be answered.

User question: {question}

Evidence:
{evidence}

Reply with JSON: {{"sufficient": true/false, "gaps": ["what's still missing"], \
"refined_queries": [{{"tool": "search_docs"|"search_web", "query": "..."}}]}}
Use search_web for gaps needing current/external info, search_docs for corpus gaps.

JSON:"""

SYNTHESIZE_PROMPT = """Answer the user's question using ONLY the evidence below. \
Cite sources with [filename] markers.

User question: {question}

Evidence:
{evidence}

Answer concisely with citations:"""


def _parse_thought(text: str) -> list[dict]:
    """Parse the THOUGHT step: list of {tool, query}."""
    text = strip_think(text).strip()
    text = re.sub(r"```(?:json)?\n?", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, dict) and "query" in item:
                    tool = item.get("tool", "search_docs")
                    if tool not in TOOLS:
                        tool = "search_docs"
                    result.append({"tool": tool, "query": str(item["query"])})
                elif isinstance(item, str):
                    # Backwards compat: plain query strings → search_docs.
                    result.append({"tool": "search_docs", "query": item})
            return result[:3]
    except json.JSONDecodeError:
        pass
    # Fallback: single search_docs query.
    lines = [ln.strip("- *").strip() for ln in text.splitlines() if ln.strip()]
    return [{"tool": "search_docs", "query": lines[0]}] if lines else []


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
    """Run the ReAct loop (Thought → Action → Observation).

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

    # ---- THOUGHT: plan tool calls ----
    thought_text = _invoke(THOUGHT_PROMPT.format(question=question))
    plan = _parse_thought(thought_text)
    if not plan:
        plan = [{"tool": "search_docs", "query": question}]
    if verbose:
        print(f"[thought] plan: {plan}")

    # ---- ACTION → OBSERVATION loop ----
    all_evidence: list[str] = []
    all_sources: set[str] = set()
    iteration = 0

    def _dispatch(tool: str, query: str) -> str:
        if tool == "search_web":
            return _search_web_tool(query)
        return _search_docs_tool(query, settings=settings)

    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"[action] iteration {iteration}, plan: {plan}")

        # ACTION: execute tool calls.
        for step in plan:
            tool, q = step["tool"], step["query"]
            result = _dispatch(tool, q)
            all_evidence.append(f"--- {tool}: {q} ---\n{result}")
            for m in re.finditer(r"\[([^\]]+)\]", result):
                all_sources.add(m.group(1))

        # OBSERVATION: evaluate sufficiency.
        evidence_text = "\n\n".join(all_evidence)
        obs_text = _invoke(OBSERVATION_PROMPT.format(
            question=question, evidence=evidence_text
        ))
        verdict = _parse_verify(obs_text)
        if verbose:
            print(f"[observation] sufficient={verdict['sufficient']}, "
                  f"gaps={verdict['gaps']}")

        if verdict["sufficient"]:
            break

        # Refine: parse refined queries into tool calls for the next iteration.
        refined = verdict.get("refined_queries", [])
        if not refined:
            break
        # refined_queries may be plain strings or {tool, query} dicts.
        new_plan = []
        for r in refined[:3]:
            if isinstance(r, dict) and "query" in r:
                tool = r.get("tool", "search_docs")
                if tool not in TOOLS:
                    tool = "search_docs"
                new_plan.append({"tool": tool, "query": str(r["query"])})
            elif isinstance(r, str):
                new_plan.append({"tool": "search_web", "query": r})
        if not new_plan:
            break
        plan = new_plan

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
