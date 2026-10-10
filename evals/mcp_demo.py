"""MCP demo: act as a client, call search_docs, show the round-trip.

Runs the MCP server as a subprocess over stdio, lists its tools,
calls search_docs, and prints the results.

Usage:
    PYTHONPATH=src python -m evals.mcp_demo "How long are API tokens valid?"
    PYTHONPATH=src python -m evals.mcp_demo  # uses a default query

Requires: mcp package (in requirements.txt), built FAISS index.
"""

from __future__ import annotations

import asyncio
import sys


async def _run_mcp_client(query: str, k: int = 4) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rag_demo.mcp_server"],
        env=None,
    )
    print(f"MCP client: connecting to rag_demo.mcp_server over stdio...\n")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected. Listing tools...\n")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"  Tool: {t.name}")
                print(f"  Description: {t.description}\n")

            print(f"→ Calling search_docs(query={query!r}, k={k})...\n")
            result = await session.call_tool(
                "search_docs", arguments={"query": query, "k": k}
            )
            print("=" * 60)
            print("RESULT FROM MCP SERVER:")
            print("=" * 60)
            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
            print("=" * 60)
            print("\n✓ Round-trip complete.")


def _run_direct(query: str, k: int = 4) -> None:
    """Fallback: call the tool function directly (no MCP transport).

    Used when the mcp package isn't installed. Demonstrates the same
    search_docs logic the MCP server exposes.
    """
    print("mcp package not found — running direct function call demo.\n")
    print("(Install with: pip install mcp)\n")
    from rag_demo.mcp_server import search_docs

    print(f"→ Calling search_docs(query={query!r}, k={k})...\n")
    print("=" * 60)
    print("RESULT:")
    print("=" * 60)
    # search_docs is decorated; call the underlying function.
    fn = getattr(search_docs, "fn", search_docs)
    print(fn(query, k))
    print("=" * 60)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "How long are API tokens valid?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    try:
        import mcp  # noqa: F401
        asyncio.run(_run_mcp_client(query, k))
    except ImportError:
        _run_direct(query, k)


if __name__ == "__main__":
    main()
