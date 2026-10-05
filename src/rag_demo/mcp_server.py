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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
