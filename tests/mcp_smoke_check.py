"""Ad-hoc MCP connectivity check — run directly, not part of the test suite.

Usage:  python tests/mcp_smoke_check.py
Exits 0 when tavily-mcp connects and returns search results; 1 otherwise.
"""

import asyncio
import os
import sys


async def main() -> int:
    from agentic_rag.core.mcp.client import MCPClient

    client = MCPClient()
    env = dict(os.environ)
    env["TAVILY_API_KEY"] = (
        "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"
    )
    try:
        tools = await client.connect_stdio(
            server_name="tavily-mcp",
            command="npx",
            args=["-y", "tavily-mcp@0.2.20"],
            env=env,
        )
    except Exception as e:
        print(f"CONNECT FAILED: {type(e).__name__}: {e}")
        return 1

    print(f"CONNECTED — {len(tools)} tool(s):")
    for t in tools:
        print(f"  - {t.name}: {t.description[:70]}")

    if not tools:
        print("NO TOOLS registered — server is useless")
        return 1

    try:
        result = await tools[0].execute(query="2026 FIFA World Cup host cities")
    except Exception as e:
        print(f"SEARCH FAILED: {type(e).__name__}: {e}")
        return 1

    text = str(result)
    print(f"\nSEARCH OK — {len(text)} chars. First 400:")
    print(text[:400])
    await client.disconnect_all()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
