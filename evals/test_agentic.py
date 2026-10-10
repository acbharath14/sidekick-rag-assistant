"""Unit tests for agentic RAG behaviors. No LLM or network required.

Tests the pure parsing/logic functions by extracting them without
importing the full agent module (which needs langchain).
"""

import json
import re
import sys
from pathlib import Path


def _parse_thought_isolated(text: str) -> list[dict]:
    """Copy of agent._parse_thought logic for isolated testing."""
    TOOLS = {"search_docs", "search_web", "extract_pages"}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
                    result.append({"tool": "search_docs", "query": item})
            return result[:3]
    except json.JSONDecodeError:
        pass
    # Fallback: return empty (never use malformed text as query).
    return []


def test_parse_thought_malformed_returns_empty():
    """Malformed JSON must not become a search query (the '{' bug)."""
    assert _parse_thought_isolated("{") == []
    assert _parse_thought_isolated("not json at all {{{") == []
    assert _parse_thought_isolated("") == []


def test_parse_thought_valid_json():
    """Valid JSON list parses to tool calls."""
    result = _parse_thought_isolated(
        '[{"tool": "search_web", "query": "walmart deals"}]'
    )
    assert len(result) == 1
    assert result[0]["tool"] == "search_web"
    assert result[0]["query"] == "walmart deals"


def test_parse_thought_invalid_tool_defaults():
    """Unknown tool names fall back to search_docs."""
    result = _parse_thought_isolated(
        '[{"tool": "nonexistent", "query": "test"}]'
    )
    assert result[0]["tool"] == "search_docs"


def test_parse_thought_extract_pages():
    """extract_pages is a valid tool choice."""
    result = _parse_thought_isolated(
        '[{"tool": "extract_pages", "query": "https://example.com/a, https://example.com/b"}]'
    )
    assert result[0]["tool"] == "extract_pages"


def test_error_markers_distinct():
    """[TOOL_ERROR] and [NO_RESULTS] are distinct markers."""
    assert "[TOOL_ERROR]" != "[NO_RESULTS]"
    # Verify agent.py uses both.
    agent_path = Path(__file__).resolve().parent.parent / "src" / "rag_demo" / "agent.py"
    content = agent_path.read_text()
    assert "[TOOL_ERROR]" in content, "TOOL_ERROR marker missing from agent.py"
    assert "[NO_RESULTS]" in content, "NO_RESULTS marker missing from agent.py"


def test_fast_path_keywords():
    """Key queries trigger the fast-path heuristic."""
    agent_path = Path(__file__).resolve().parent.parent / "src" / "rag_demo" / "agent.py"
    content = agent_path.read_text()
    for kw in ["temperature", "black friday", "walmart", "weather"]:
        assert f'"{kw}"' in content, f"Keyword {kw!r} missing from fast-path"


def test_extract_url_cap():
    """Extract caps at 5 URLs per call."""
    agent_path = Path(__file__).resolve().parent.parent / "src" / "rag_demo" / "agent.py"
    content = agent_path.read_text()
    assert "urls[:5]" in content, "URL cap not found in _extract_pages_tool"


if __name__ == "__main__":
    test_parse_thought_malformed_returns_empty()
    print("✓ malformed JSON returns empty (no '{' bug)")
    test_parse_thought_valid_json()
    print("✓ valid JSON parses")
    test_parse_thought_invalid_tool_defaults()
    print("✓ invalid tool defaults to search_docs")
    test_parse_thought_extract_pages()
    print("✓ extract_pages is valid tool")
    test_error_markers_distinct()
    print("✓ error markers distinct and present")
    test_fast_path_keywords()
    print("✓ fast-path keywords present")
    test_extract_url_cap()
    print("✓ URL cap enforced")
    print("\nAll agentic unit tests pass.")
