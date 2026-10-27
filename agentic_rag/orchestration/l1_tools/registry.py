"""Tool Registry — manages tool registration, discovery, and MCP integration."""

from typing import Optional

from agentic_rag.orchestration.l1_tools.base import BaseTool


class ToolRegistry:
    """Central registry for all tools (built-in + MCP).

    Supports:
    - Global tools (available to all agents)
    - Optionally namespaced tools
    - Dynamic MCP tool registration
    - Plugin-based tool discovery
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._mcp_tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, scope: Optional[str] = None) -> None:
        """Register a tool. If scope is provided, tool name is prefixed."""
        key = f"{scope}__{tool.name}" if scope else tool.name
        self._tools[key] = tool

    def register_mcp(self, tool: BaseTool, server_name: str) -> None:
        """Register a tool from an MCP server.

        The tool's ``.name`` already includes the ``mcp__<server>__`` prefix
        (set by MCPTool), so we use it as-is.
        """
        self._tools[tool.name] = tool
        self._mcp_tools[tool.name] = tool

    def unregister_mcp_server(self, server_name: str) -> None:
        """Remove all tools from a specific MCP server."""
        prefix = f"mcp__{server_name}__"
        keys_to_remove = [k for k in self._tools if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._tools[key]
            if key in self._mcp_tools:
                del self._mcp_tools[key]

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def filter(self, tool_names: list[str]) -> list[BaseTool]:
        """Get multiple tools by name. Silently skips missing tools."""
        return [self._tools[n] for n in tool_names if n in self._tools]

    def get_all(self, include_mcp: bool = True) -> list[BaseTool]:
        """Get all registered tools."""
        if include_mcp:
            return list(self._tools.values())
        return [t for k, t in self._tools.items() if k not in self._mcp_tools]

    def get_mcp_tools(self) -> list[BaseTool]:
        """Get only MCP-registered tools."""
        return list(self._mcp_tools.values())

    def list_names(self, include_mcp: bool = True) -> list[str]:
        """List all tool names."""
        if include_mcp:
            return list(self._tools.keys())
        return [k for k in self._tools if k not in self._mcp_tools]

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def mcp_tool_count(self) -> int:
        return len(self._mcp_tools)

    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()
        self._mcp_tools.clear()


# Global registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
