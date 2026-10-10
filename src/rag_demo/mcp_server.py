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
    """Search the web. Uses Tavily API if RAG_DEMO_TAVILY_API_KEY is set,
    otherwise falls back to DuckDuckGo (unreliable).

    Args:
        query: search query.
        k: max results to return (default 5).
    """
    # Reuse the agent's web search implementation.
    from .agent import _search_web_tool
    return _search_web_tool(query, k=k)


@mcp.tool()
def extract_pages(urls: str, query: str = "") -> str:
    """Fetch full page content from URLs via Tavily Extract.

    Args:
        urls: comma-separated URLs to extract.
        query: optional query for reranking extracted chunks.
    """
    from .agent import _extract_pages_tool
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    return _extract_pages_tool(url_list, query)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
