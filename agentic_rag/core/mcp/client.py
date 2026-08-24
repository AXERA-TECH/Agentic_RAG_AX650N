"""MCP (Model Context Protocol) Client.

Connects to external MCP servers and exposes their tools to the agent.
"""

import asyncio
import json
from typing import Any, Optional

from agentic_rag.orchestration.l1_tools.base import BaseTool


class MCPTool(BaseTool):
    """Proxy tool that wraps an MCP server tool."""

    _CAPABILITY_HINTS = {
        "fresh_information": (
            "search", "web", "internet", "browser", "browse", "news",
            "current", "latest", "real-time", "realtime", "搜索", "网页",
            "网络", "新闻", "最新", "实时",
        ),
    }

    def __init__(self, name: str, description: str, parameters_schema: dict,
                 server_name: str, client: "MCPClient"):
        self.name = f"mcp__{server_name}__{name}"
        self.description = f"[MCP/{server_name}] {description}"
        self.parameters_schema = parameters_schema
        self.capabilities = self._infer_capabilities(name, description)
        self.source = "mcp"
        self._original_name = name
        self._server_name = server_name
        self._client = client
        self.requires_confirmation = False

    @classmethod
    def _infer_capabilities(cls, name: str, description: str) -> frozenset[str]:
        """Infer semantic capabilities from MCP tool metadata."""
        metadata = f"{name} {description}".lower()
        return frozenset(
            capability
            for capability, hints in cls._CAPABILITY_HINTS.items()
            if any(hint in metadata for hint in hints)
        )

    async def execute(self, **kwargs) -> Any:
        """Execute the MCP tool via the connected client."""
        return await self._client.call_tool(self._server_name, self._original_name, kwargs)


class MCPClient:
    """MCP Client for connecting to external MCP servers.

    Supports stdio transport (subprocess-based) for connecting to MCP servers.
    """

    def __init__(self):
        self._connections: dict[str, dict] = {}  # server_name -> {process, tools}

    @property
    def connected_servers(self) -> list[str]:
        return list(self._connections.keys())

    async def connect_stdio(self, server_name: str, command: str,
                             args: list[str] | None = None,
                             env: dict[str, str] | None = None) -> list[MCPTool]:
        """Connect to an MCP server via stdio subprocess.

        Args:
            server_name: Logical name for this server.
            command: Command to launch the MCP server.
            args: Arguments for the command.
            env: Environment variables.

        Returns:
            List of MCPTool instances exposed by the server.
        """
        proc = await asyncio.create_subprocess_exec(
            command, *(args or []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Initialize MCP session
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agentic_rag", "version": "0.1.0"},
            },
        }
        await self._send_request(proc, init_request)

        # Discover tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        response = await self._send_request(proc, tools_request)

        tools = []
        for tool_def in response.get("result", {}).get("tools", []):
            tool = MCPTool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                parameters_schema=tool_def.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
                server_name=server_name,
                client=self,
            )
            tools.append(tool)

        self._connections[server_name] = {
            "process": proc,
            "tools": {t._original_name: t for t in tools},
            "next_id": 3,
            "lock": asyncio.Lock(),
        }

        return tools

    async def call_tool(self, server_name: str, name: str, arguments: dict) -> str:
        """Call a tool on its owning MCP server and serialize stdio access."""
        conn = self._connections.get(server_name)
        if conn and name in conn["tools"]:
            async with conn["lock"]:
                request = {
                    "jsonrpc": "2.0",
                    "id": conn["next_id"],
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
                conn["next_id"] += 1
                response = await self._send_request(conn["process"], request)

                if response.get("error"):
                    return f"MCP error: {json.dumps(response['error'], ensure_ascii=False)}"

                content = response.get("result", {}).get("content", [])
                if content:
                    return "\n".join(
                        c.get("text", str(c)) for c in content
                        if isinstance(c, dict)
                    )
                return json.dumps(response.get("result", {}), ensure_ascii=False)

        return f"MCP tool '{name}' not found on server '{server_name}'."

    async def list_resources(self, server_name: str) -> list[dict]:
        """List resources from an MCP server."""
        conn = self._connections.get(server_name)
        if not conn:
            return []

        request = {
            "jsonrpc": "2.0",
            "id": conn["next_id"],
            "method": "resources/list",
            "params": {},
        }
        conn["next_id"] += 1
        response = await self._send_request(conn["process"], request)
        return response.get("result", {}).get("resources", [])

    async def read_resource(self, server_name: str, uri: str) -> Any:
        """Read a resource from an MCP server."""
        conn = self._connections.get(server_name)
        if not conn:
            return f"MCP server '{server_name}' not connected."

        request = {
            "jsonrpc": "2.0",
            "id": conn["next_id"],
            "method": "resources/read",
            "params": {"uri": uri},
        }
        conn["next_id"] += 1
        response = await self._send_request(conn["process"], request)
        return response.get("result", {})

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server."""
        conn = self._connections.pop(server_name, None)
        if conn:
            proc = conn["process"]
            proc.stdin.close()
            await proc.wait()

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    async def _send_request(self, proc: asyncio.subprocess.Process, request: dict) -> dict:
        """Send a JSON-RPC request and receive the response."""
        msg = json.dumps(request) + "\n"
        proc.stdin.write(msg.encode())
        await proc.stdin.drain()

        expected_id = request.get("id")
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
            if not line:
                return {"error": "No response from MCP server"}
            response = json.loads(line.decode())
            # MCP notifications have no id and may arrive between responses.
            if response.get("id") == expected_id:
                return response
