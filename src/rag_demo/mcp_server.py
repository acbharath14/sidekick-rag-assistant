"""MCP server exposing the docs search as a tool.

Run:  python -m rag_demo.mcp_server
Then point any MCP client at this process over stdio.
"""

from __future__ import annotations

try:  # mcp>=2 renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp<2
    from mcp.server.fastmcp import FastMCP as _Server

from .config import get_settings
from .retriever import Retriever

mcp = _Server("meridian-docs")
_settings = get_settings()
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(_settings)
    return _retriever


@mcp.tool()
def search_docs(query: str, k: int = 4) -> str:
    """Search the Meridian Logistics documentation corpus.

    Args:
        query: natural-language search query.
        k: number of passages to return (default 4).
    """
    hits = _get_retriever().search(query, k=k)
    if not hits:
        return "No matching passages found."
    return "\n\n".join(
        f"[{h.source} | score={h.score:.3f}]\n{h.text}" for h in hits
    )


@mcp.tool()
def search_web(query: str, k: int = 5) -> str:
    """Search the web via DuckDuckGo (no API key needed).

    Args:
        query: search query.
        k: max results to return (default 5).
    """
    import requests
    from html.parser import HTMLParser

    # DuckDuckGo HTML endpoint (no key required).
    url = "https://html.duckduckgo.com/html/"
    try:
        resp = requests.post(
            url,
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
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
            self._in_snippet = False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "a" and "result__a" in attrs.get("class", ""):
                self._in_a = True
                self._href = attrs.get("href", "")
            elif tag == "a" and "result__snippet" in attrs.get("class", ""):
                self._in_snippet = True

        def handle_data(self, data):
            if self._in_a:
                self._text += data
            elif self._in_snippet:
                # Snippet text follows the title link.
                if self.results:
                    self.results[-1]["snippet"] = self.results[-1].get("snippet", "") + data

        def handle_endtag(self, tag):
            if tag == "a" and self._in_a:
                self._in_a = False
                # DuckDuckGo wraps URLs in a redirect; extract the real one.
                href = self._href
                if "uddg=" in href:
                    from urllib.parse import parse_qs, urlparse
                    qs = parse_qs(urlparse(href).query)
                    href = qs.get("uddg", [href])[0]
                self.results.append({"title": self._text.strip(), "url": href})
                self._text = ""
                self._href = ""
            elif tag == "a" and self._in_snippet:
                self._in_snippet = False

    parser = _Parser()
    parser.feed(resp.text)
    results = parser.results[:k]
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r.get('snippet', '')}".strip()
        for r in results
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
