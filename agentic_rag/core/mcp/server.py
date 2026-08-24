"""MCP (Model Context Protocol) Server.

Exposes this system's tools and resources to external MCP clients via stdio.
"""

import asyncio
import json
import sys
from typing import Any


class MCPServer:
    """Expose Agentic RAG tools as an MCP server.

    Runs as a stdio-based JSON-RPC server that external MCP clients can connect to.
    """

    def __init__(self, tool_registry=None, name: str = "agentic_rag",
                 version: str = "0.1.0"):
        self.name = name
        self.version = version
        self.tool_registry = tool_registry
        self._running = False

    async def run_stdio(self) -> None:
        """Run the MCP server over stdio (stdin/stdout).

        Reads JSON-RPC requests from stdin and writes responses to stdout.
        """
        self._running = True
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while self._running:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=300.0)
                if not line:
                    break

                request = json.loads(line.decode())
                response = await self._handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except asyncio.TimeoutError:
                break
            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if "request" in dir() else None,
                    "error": {"code": -32603, "message": str(e)},
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

    async def _handle_request(self, request: dict) -> dict:
        """Handle a single JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
        }

        handler = handlers.get(method)
        if handler:
            result = await handler(request.get("params", {}))
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    async def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": self.name, "version": self.version},
        }

    async def _handle_tools_list(self, params: dict) -> dict:
        tools = self.tool_registry.get_all() if self.tool_registry else []
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters_schema,
                }
                for t in tools
            ]
        }

    async def _handle_tools_call(self, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool = self.tool_registry.get(tool_name) if self.tool_registry else None
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Tool not found: {tool_name}"}],
                "isError": True,
            }

        try:
            result = await tool.execute(**arguments)
            return {
                "content": [{"type": "text", "text": str(result)}],
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    async def _handle_resources_list(self, params: dict) -> dict:
        return {"resources": []}

    def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False
