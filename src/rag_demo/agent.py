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
    """Tool: search the web. Uses Tavily API if RAG_DEMO_TAVILY_API_KEY is set,
    otherwise falls back to DuckDuckGo HTML scraping (unreliable, often blocked).

    Returns titles, URLs, snippets.
    """
    import os
    import requests

    # Check session cache first (defined in agentic_ask).
    global _search_cache
    cache = globals().get("_search_cache", {})
    key = query.lower().strip()
    if key in cache:
        import time
        entry = cache[key]
        age = time.time() - entry["timestamp"]
        if age <= 1800:  # 30-min TTL
            return entry["result"] + f"\n\n[CACHED {int(age)}s ago]"
        del cache[key]  # Stale — re-fetch.

    tavily_key = os.environ.get("RAG_DEMO_TAVILY_API_KEY")
    if tavily_key:
        result = _tavily_search(query, k, tavily_key)
        backend = "tavily"
    else:
        result = _duckduckgo_search(query, k)
        backend = "duckduckgo"

    # Cache successful results.
    if not result.startswith("[TOOL_ERROR]"):
        import time
        cache[key] = {"result": result, "timestamp": time.time(),
                      "backend": backend}
    return result


def _tavily_search(query: str, k: int, api_key: str,
                   include_content: bool = False) -> str:
    """Search via Tavily API (reliable, AI-optimized).

    If include_content=True, returns full page markdown (truncated to ~3000
    chars per result). Use for deals/prices where snippets aren't enough.
    """
    import requests

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": k,
        "search_depth": "basic",
        "include_answer": False,
    }
    if include_content:
        payload["include_raw_content"] = "markdown"

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30 if include_content else 20,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        return "[TOOL_ERROR] Tavily search timed out"
    except requests.ConnectionError as e:
        return f"[TOOL_ERROR] Tavily connection failed: {e}"
    except Exception as e:
        return f"[TOOL_ERROR] Tavily search failed: {e}"

    results = data.get("results", [])
    if not results:
        return "[NO_RESULTS] Tavily returned no results for this query."
    parts = []
    for r in results[:k]:
        title = r.get('title', 'No title')
        url = r.get('url', '')
        if include_content:
            content = r.get('raw_content', '') or r.get('content', '')
            # Truncate to bound LLM tokens (~3000 chars ≈ 750 tokens).
            content = content[:3000]
        else:
            content = r.get('content', '')[:500]
        parts.append(f"[{title}]({url})\n{content}".strip())
    return "\n\n".join(parts)


def _extract_pages_tool(urls: list[str], query: str = "") -> str:
    """Tool: fetch full page content from URLs via Tavily Extract.

    Use after search_web finds relevant pages but snippets lack details
    (prices, dates, specs). Max 5 URLs per call.
    """
    import os
    import requests

    api_key = os.environ.get("RAG_DEMO_TAVILY_API_KEY")
    if not api_key:
        return "[TOOL_ERROR] Tavily API key not set — cannot extract pages."

    urls = urls[:5]  # Hard cap.
    try:
        resp = requests.post(
            "https://api.tavily.com/extract",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": api_key,
                "urls": urls,
                "query": query,
                "chunks_per_source": 3,
                "extract_depth": "basic",
                "format": "markdown",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        return "[TOOL_ERROR] Tavily extract timed out after 30s"
    except Exception as e:
        return f"[TOOL_ERROR] Tavily extract failed: {e}"

    parts = []
    for r in data.get("results", []):
        url = r.get("url", "")
        content = r.get("raw_content", "")[:4000]  # Bound tokens.
        parts.append(f"--- {url} ---\n{content}".strip())
    for f in data.get("failed_results", []):
        parts.append(f"--- {f} ---\n[EXTRACTION FAILED]")
    if not parts:
        return "[NO_RESULTS] No content extracted from the provided URLs."
    return "\n\n".join(parts)


def _duckduckgo_search(query: str, k: int = 5) -> str:
    """Fallback: DuckDuckGo HTML scraping (unreliable, often 403/blocked)."""
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
    except requests.Timeout:
        return "[TOOL_ERROR] Web search timed out after 15s (network slow or blocked)"
    except requests.ConnectionError as e:
        return f"[TOOL_ERROR] Web search connection failed: {e}"
    except Exception as e:
        return f"[TOOL_ERROR] Web search failed: {e}"

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
        return "[NO_RESULTS] Web search returned no results for this query."
    return "\n\n".join(f"[{r['title']}]({r['url']})" for r in results)


# Available tools for the agent. The THOUGHT step chooses which to use.
TOOLS = {
    "search_docs": "Search the local document corpus (use for questions about the indexed docs).",
    "search_web": "Search the web via Tavily (use for current events, prices, deals, general knowledge). Returns snippets.",
    "extract_pages": "Fetch FULL page content from URLs (use after search_web when snippets lack prices, dates, or details).",
}


THOUGHT_PROMPT = """You are a research planner. Break the user's question into \
searchable sub-questions and choose the right tool for each.

Current date: {current_date}

Conversation history (for follow-up context):
{history}

Available tools:
- search_docs: Search the local document corpus. The corpus contains ONLY \
technical documentation (APIs, deployment guides, troubleshooting). Use this \
ONLY if the question is about software, APIs, or technical documentation.
- search_web: Search the web. Use this for EVERYTHING else: current events, \
prices, deals, products, companies, people, news, shopping, reviews, \
general knowledge, or any question not about the technical docs. Returns snippets.
- extract_pages: Fetch FULL page content from specific URLs. Use AFTER search_web \
when snippets lack the details you need (exact prices, dates, specs, tables). \
Pass comma-separated URLs as the query.

RULES:
- If the question mentions a retailer, product, deal, price, company, celebrity, \
news event, or anything not in technical docs → use search_web.
- If the question is about APIs, code, deployment, or technical troubleshooting \
→ use search_docs.
- When in doubt, use search_web. The local corpus is very narrow.
- Use conversation history to resolve ambiguous follow-ups.
- For shopping/deals questions: search_web first, then extract_pages on the \
top 2-3 most relevant URLs to get actual prices.

Examples:
- "Walmart Black Friday deals" → search_web (retail, not tech docs)
- "How do I create a shipment via API?" → search_docs (API question)
- "What is the capital of France?" → search_web (general knowledge)
- "API token expiry" → search_docs (technical)

User question: {question}

Reply with JSON: a list of objects like \
[{{"tool": "search_docs", "query": "..."}}, {{"tool": "search_web", "query": "..."}}]
Use 1-3 tool calls.

JSON:"""

OBSERVATION_PROMPT = """You are evaluating retrieved evidence. Given the user's question and \
the evidence gathered so far, decide if the question can be answered.

Current date: {current_date}

Conversation history:
{history}

User question: {question}

Evidence:
{evidence}

IMPORTANT - Tool status markers:
- [TOOL_ERROR]: The search tool itself failed (network, timeout, blocked). \
This is NOT evidence that no information exists. Do NOT conclude "doesn't exist" \
from a tool error. Instead, note the tool failure in gaps and suggest retrying \
or trying a different query.
- [NO_RESULTS]: The search completed but found nothing. This suggests the \
information may not exist, but try 1-2 alternative phrasings before concluding.
- [CACHED Xs ago]: This result came from the session cache, not a fresh search. \
If the question needs up-to-the-minute data and the cache is old, consider \
re-searching.

Reply with JSON: {{"sufficient": true/false, "gaps": ["what's still missing"], \
"refined_queries": [{{"tool": "search_docs"|"search_web", "query": "..."}}], \
"tool_errors": ["describe any tool failures separately"]}}
Use search_web for gaps needing current/external info, search_docs for corpus gaps.
If evidence contains [TOOL_ERROR], ALWAYS set sufficient=false and explain the \
tool failure in gaps — never claim the information doesn't exist.

JSON:"""

SYNTHESIZE_PROMPT = """Answer the user's question using ONLY the evidence below. \
Cite sources with [filename] markers.

Current date: {current_date}

Conversation history (for follow-up coherence):
{history}

User question: {question}

Evidence:
{evidence}

IMPORTANT: If the evidence contains [TOOL_ERROR] markers, do NOT claim the \
information doesn't exist. Instead, say: "I couldn't retrieve [X] because the \
search tool failed: [reason]." Be honest about tool limitations.

If this is a follow-up, frame the answer in the context of the prior conversation.

For shopping/deals questions, present results as a structured table:
| Product | Retailer | Regular Price | Deal Price | Discount | Status | Source |
Status is one of: announced (retailer confirmed), predicted (analyst/blog),
leaked (unofficial early info), historical (prior year reference).

Answer concisely with citations:"""


def format_plan_as_mermaid(plan: list[dict], question: str) -> str:
    """Render the agent's plan as a Mermaid flowchart."""
    lines = ["flowchart TD"]
    lines.append(f'    Q["❓ {question[:50]}"]')
    for i, step in enumerate(plan):
        tool = step["tool"]
        query = step["query"][:40]
        icon = "🌐" if tool == "search_web" else "📚"
        node_id = f"S{i+1}"
        lines.append(f'    {node_id}["{icon} {tool}<br/>{query}"]')
        if i == 0:
            lines.append(f"    Q --> {node_id}")
        else:
            lines.append(f"    S{i} --> {node_id}")
    lines.append(f'    S{len(plan)} --> SYN["📝 Synthesize answer"]')
    return "\n".join(lines)


def format_plan_as_ascii(plan: list[dict], question: str) -> str:
    """Render the plan as an ASCII flowchart for terminals."""
    lines = []
    lines.append("┌─ Plan ─────────────────────────────────────")
    lines.append(f"│ ❓ Question: {question[:60]}")
    lines.append("│")
    for i, step in enumerate(plan):
        tool = step["tool"]
        query = step["query"][:55]
        icon = "🌐" if tool == "search_web" else "📚"
        lines.append(f"│  {i+1}. {icon} [{tool}]")
        lines.append(f"│     └─ {query}")
        if i < len(plan) - 1:
            lines.append("│     ↓")
    lines.append("│")
    lines.append("│  📝 → Synthesize final answer")
    lines.append("└────────────────────────────────────────────")
    return "\n".join(lines)


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
    # Fallback: if JSON parsing failed, return empty to trigger safe default.
    # Never use raw malformed text (e.g. "{") as a search query.
    return []


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
    show_plan: bool = False,
    on_progress: object = None,
    history: list[tuple[str, str]] | None = None,
) -> Answer:
    """Run the ReAct loop (Thought → Action → Observation).

    If show_plan is True, displays the plan as a flowchart and asks for
    confirmation before executing.

    on_progress: optional callable(phase, message) for live UI updates.
        phase is one of "thought", "action", "observation", "synthesize".

    history: optional list of (question, answer) tuples for follow-up context.
        Follow-ups like "what about Target?" are rewritten to standalone
        questions using this history.

    Returns an Answer with sources from all tool calls.
    """
    import time

    # ---- Search result cache (session-scoped, TTL-monitored) ----
    # Key: normalized query. Value: {result, timestamp, backend}.
    # Staleness mitigation: entries older than CACHE_TTL are re-fetched;
    # cache hits are marked with age so the LLM knows they're not fresh.
    global _search_cache
    if "_search_cache" not in globals():
        _search_cache = {}
    CACHE_TTL_SECONDS = 1800  # 30 minutes

    def _cache_get(query: str) -> str | None:
        key = query.lower().strip()
        entry = _search_cache.get(key)
        if not entry:
            return None
        age = time.time() - entry["timestamp"]
        if age > CACHE_TTL_SECONDS:
            # Stale — evict and re-fetch.
            del _search_cache[key]
            return None
        return entry["result"] + f"\n\n[CACHED {int(age)}s ago]"

    def _cache_put(query: str, result: str, backend: str):
        # Don't cache errors — only successful results.
        if result.startswith("[TOOL_ERROR]"):
            return
        _search_cache[query.lower().strip()] = {
            "result": result,
            "timestamp": time.time(),
            "backend": backend,
        }

    def clear_search_cache():
        """Clear the session search cache. Called from UI."""
        _search_cache.clear()

    settings = settings or get_settings()
    llm = get_llm(settings)
    started = time.perf_counter()

    def _progress(phase: str, message: str):
        if on_progress:
            try:
                on_progress(phase, message)
            except Exception:
                pass
        if verbose:
            print(f"[{phase}] {message}")

    def _invoke(prompt: str) -> str:
        out = llm.invoke(prompt)
        text = out.content if hasattr(out, "content") else str(out)
        return strip_think(text.strip())

    # ---- THOUGHT: plan tool calls ----
    from datetime import date
    current_date = date.today().isoformat()

    # Rewrite follow-ups to standalone questions using conversation history.
    # "what about Target?" + history("Walmart Black Friday deals") →
    # "Target Black Friday deals 2026". Must run BEFORE the fast-path so the
    # keyword check sees the resolved query, not the vague follow-up.
    effective_question = question
    if history:
        _progress("thought", "Rewriting follow-up with conversation context...")
        try:
            from .rewrite import rewrite_query
            rewritten = rewrite_query(question, history, llm)
            if rewritten and rewritten.strip() and rewritten != question:
                effective_question = rewritten.strip()
                _progress("thought", f"Rewritten: {effective_question[:60]}")
        except Exception:
            pass  # Fall back to original question.

    # Format history for prompts.
    history_text = ""
    if history:
        history_text = "\n".join(
            f"Q: {q}\nA: {a[:300]}" for q, a in history[-3:]
        )

    # Fast path: obvious web queries skip the LLM planning call.
    # Saves 20-30s on CPU by going straight to search_web.
    # Uses effective_question (rewritten) so follow-ups route correctly.
    _web_keywords = {
        "black friday", "deals", "deal ", "price", "walmart", "amazon",
        "target", "best buy", "costco", "sale", "discount", "coupon",
        "news", "weather", "temperature", "stock price", "election", "president",
        "celebrity", "movie", "sports", "game score",
    }
    _q_lower = effective_question.lower()
    if any(kw in _q_lower for kw in _web_keywords):
        _progress("thought", "Fast path: web query detected, skipping LLM planning")
        plan = [{"tool": "search_web", "query": effective_question}]
    else:
        _progress("thought", "Planning search strategy...")
        thought_text = _invoke(THOUGHT_PROMPT.format(
            question=effective_question, current_date=current_date,
            history=history_text or "No prior conversation."
        ))
        plan = _parse_thought(thought_text)
        if not plan:
            plan = [{"tool": "search_docs", "query": question}]
    _progress("thought", f"Plan: {len(plan)} tool calls")

    # ---- Show plan and get approval (if requested) ----
    if show_plan:
        print("\n" + format_plan_as_ascii(plan, question))
        print("\nMermaid (for docs/dashboards):")
        print(format_plan_as_mermaid(plan, question))
        try:
            confirm = input("\nExecute this plan? [Y/n/edit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"
        if confirm == "edit":
            # Let user modify the plan.
            print("Enter tool calls as 'tool: query' (empty line to finish):")
            print("Tools: search_docs, search_web")
            new_plan = []
            while True:
                try:
                    line = input(f"  {len(new_plan)+1}. ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    break
                if ":" in line:
                    tool, query = line.split(":", 1)
                    tool = tool.strip()
                    query = query.strip()
                    if tool in TOOLS and query:
                        new_plan.append({"tool": tool, "query": query})
                else:
                    print("  Format: tool: query")
            if new_plan:
                plan = new_plan
                print("\nUpdated plan:")
                print(format_plan_as_ascii(plan, question))
        elif confirm in ("n", "no", "q", "quit"):
            print("Plan rejected.")
            return Answer(
                text="Plan rejected by user.",
                sources=[], hits=[], latency_ms=0,
                grounded=False, standalone_question=None,
            )
        print()  # blank line before execution

    # ---- ACTION → OBSERVATION loop ----
    all_evidence: list[str] = []
    all_sources: set[str] = set()
    iteration = 0

    # Credit guardrail: max page extractions per agent run.
    import os
    _max_extracts = int(os.environ.get("RAG_DEMO_MAX_EXTRACTS", "3"))
    _extract_count = 0

    def _dispatch(tool: str, query: str) -> str:
        nonlocal _extract_count
        if tool == "search_web":
            return _search_web_tool(query)
        if tool == "extract_pages":
            # query is a comma-separated URL list for this tool.
            if _extract_count >= _max_extracts:
                return (f"[TOOL_ERROR] Extraction limit reached "
                        f"({_max_extracts} per run). Use search results.")
            urls = [u.strip() for u in query.split(",") if u.strip()]
            _extract_count += 1
            _progress("action",
                      f"Extracting {len(urls)} page(s) "
                      f"({_extract_count}/{_max_extracts})")
            return _extract_pages_tool(urls, effective_question)
        return _search_docs_tool(query, settings=settings)

    while iteration < max_iterations:
        iteration += 1
        _progress("action", f"Iteration {iteration}: executing {len(plan)} searches...")

        # ACTION: execute tool calls.
        for step in plan:
            tool, q = step["tool"], step["query"]
            _progress("action", f"Searching [{tool}]: {q[:60]}")
            result = _dispatch(tool, q)
            all_evidence.append(f"--- {tool}: {q} ---\n{result}")
            for m in re.finditer(r"\[([^\]]+)\]", result):
                all_sources.add(m.group(1))

        # OBSERVATION: evaluate sufficiency.
        _progress("observation", "Evaluating if evidence is sufficient...")
        evidence_text = "\n\n".join(all_evidence)
        obs_text = _invoke(OBSERVATION_PROMPT.format(
            question=effective_question, evidence=evidence_text,
            current_date=current_date,
            history=history_text or "No prior conversation."
        ))
        verdict = _parse_verify(obs_text)
        _progress(
            "observation",
            f"Sufficient: {verdict['sufficient']}"
            + (f", gaps: {verdict['gaps']}" if verdict.get("gaps") else "")
        )

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
    _progress("synthesize", "Synthesizing final answer from all evidence...")
    evidence_text = "\n\n".join(all_evidence)
    answer_text = _invoke(SYNTHESIZE_PROMPT.format(
        question=effective_question, evidence=evidence_text,
        current_date=current_date,
        history=history_text or "No prior conversation."
    ))
    _progress("synthesize", "Done.")

    latency_ms = (time.perf_counter() - started) * 1000
    return Answer(
        text=answer_text,
        sources=sorted(all_sources),
        hits=[],
        latency_ms=latency_ms,
        grounded=not is_fallback_answer(answer_text),
        standalone_question=None,
    )
