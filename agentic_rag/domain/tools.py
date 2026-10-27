"""Tool contracts shared by tools, MCP integration, and the agent engine.

These classes used to live in ``orchestration.l1_tools.base``, which forced
``core.mcp`` to import from the orchestration layer.  They have no
dependencies beyond the data models, so they belong in the domain kernel.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from agentic_rag.data.models import ToolDefinition


class BaseTool(ABC):
    """Abstract base for all atomic tools.

    Each tool has:
    - name: Unique identifier
    - description: Human-readable description for LLM context
    - parameters_schema: JSON Schema for parameters
    - requires_confirmation: If True, the ReAct executor refuses to run the
      tool until an explicit user approval flow exists
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    requires_confirmation: bool = False
    capabilities: frozenset[str] = frozenset()
    source: str = "builtin"

    def has_capability(self, capability: str) -> bool:
        """Return whether the tool advertises a semantic capability."""
        return capability in self.capabilities

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters."""
        ...

    def to_definition(self) -> ToolDefinition:
        """Convert to a ToolDefinition for LLM function calling."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


class ToolOutput(BaseModel):
    """Standardized tool output."""
    success: bool
    result: Any = None
    error: str | None = None
