"""Tool resolution and execution for the ReAct loop."""

import asyncio
import json
import uuid

from agentic_rag.data.models import ToolCallResult


class ToolExecutor:
    """Resolve and execute tool calls, guarding against duplicates and misses.

    By default a call is refused when the tool isn't in the engine's filtered
    set — that set IS the agent's capability boundary chosen by the router.
    ``allow_registry_fallback`` opts back into the old behaviour (resolve
    against the global tool registry), for callers that deliberately want the
    model to reach tools it wasn't handed.
    """

    def __init__(
        self,
        tools: list,
        allow_registry_fallback: bool = False,
        max_calls: dict[str, int] | None = None,
    ):
        self.tools = tools
        self._tool_map = {t.name: t for t in tools}
        self.allow_registry_fallback = allow_registry_fallback
        self.executed_sigs: set[tuple[str, str]] = set()
        # Per-run call budgets (e.g. rag_search at most twice).  The
        # deterministic seed counts as the first call.
        self.max_calls = dict(max_calls or {})
        self.call_counts: dict[str, int] = {}

    @property
    def names(self) -> list[str]:
        return list(self._tool_map.keys())

    def signature(self, name: str, arguments: dict) -> tuple[str, str]:
        return (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))

    def is_duplicate(self, name: str, arguments: dict) -> bool:
        return self.signature(name, arguments) in self.executed_sigs

    def is_budget_exhausted(self, name: str) -> bool:
        limit = self.max_calls.get(name)
        return limit is not None and self.call_counts.get(name, 0) >= limit

    def budget_remaining(self, name: str) -> int | None:
        limit = self.max_calls.get(name)
        if limit is None:
            return None
        return max(0, limit - self.call_counts.get(name, 0))

    async def execute(self, name: str, arguments: dict) -> ToolCallResult:
        """Execute a tool by name with arguments (60s timeout)."""
        call_id = f"call_{uuid.uuid4().hex[:12]}"

        tool = self._tool_map.get(name)
        if tool is None and self.allow_registry_fallback:
            try:
                from agentic_rag.orchestration.l1_tools.registry import get_tool_registry
                tool = get_tool_registry().get(name)
            except Exception:
                tool = None
        if tool is None:
            return ToolCallResult(
                call_id=call_id,
                name=name,
                result=None,
                error=f"Tool '{name}' not found. Available: {list(self._tool_map.keys())}",
            )

        # Fail closed: a tool that declares ``requires_confirmation`` must
        # never run unattended.  There is no approval UI yet, so refusal is
        # the only safe behaviour (and it keeps the flag from being a lie).
        if getattr(tool, "requires_confirmation", False):
            return ToolCallResult(
                call_id=call_id,
                name=name,
                result=None,
                error=(
                    f"Tool '{name}' requires explicit user confirmation and "
                    "was not executed. Do not retry it; answer without it or "
                    "ask the user to approve the action."
                ),
            )

        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        try:
            result = await asyncio.wait_for(tool.execute(**arguments), timeout=60.0)
            return ToolCallResult(call_id=call_id, name=name, result=result)
        except asyncio.TimeoutError:
            return ToolCallResult(
                call_id=call_id,
                name=name,
                result=None,
                error=f"Tool '{name}' execution timed out after 60 seconds",
            )
        except Exception as e:
            return ToolCallResult(call_id=call_id, name=name, result=None, error=str(e))
