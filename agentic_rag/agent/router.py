"""Agent Router — selects the smallest execution path for one chat endpoint."""

from agentic_rag.agent.react_engine import ReActEngine
from agentic_rag.agent.react_prompt import SYSTEM_PROMPT
from agentic_rag.agent.request_policy import RequestPolicy, RequestStrategy
from agentic_rag.agent.single_pass_engine import SinglePassEngine
from agentic_rag.services.llm.base import BaseLLMProvider


class AgentRouter:
    """Create one chat agent without exposing user-selectable agent modes."""

    def __init__(self, llm: BaseLLMProvider, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry
        # Read native-tool-calling preference from the active LLM provider config
        try:
            from agentic_rag.config.settings import get_settings
            settings = get_settings()
            provider_cfg = settings.llm_providers.get(settings.default_provider)
            self._native_tool_calls = provider_cfg.enable_native_tool_calls if provider_cfg else True
        except Exception:
            self._native_tool_calls = True

    async def route(
        self,
        query: str,
        has_media: bool = False,
    ):
        """Build the minimal engine required for this query.

        Args:
            query: User's query text.
            has_media: Whether the LLM receives media content directly.

        Returns:
            A direct, retrieval-first, or ReAct engine.
        """
        decision = RequestPolicy.decide(query, has_media=has_media)

        if decision.strategy == RequestStrategy.DIRECT:
            return SinglePassEngine(
                llm=self.llm,
                strategy=decision.strategy,
                direct_answer=decision.direct_answer,
                route_reason=decision.reason,
            )

        rag_tool = self.tool_registry.get("rag_search")
        mcp_tools = self.tool_registry.get_mcp_tools()

        # RAG is always seeded first (see ``initial_tool_name`` below); every
        # MCP tool — web search, weather, … — is offered to the model as a
        # fallback/supplement it can pick as needed. The ReAct loop, tool-name
        # resolution and evidence gating are all N-tool general, so no MCP tool
        # is singled out here.
        tools = [tool for tool in (rag_tool, *mcp_tools) if tool is not None]
        return ReActEngine(
            llm=self.llm,
            tools=tools,
            system_prompt_template=SYSTEM_PROMPT,
            max_iterations=6,
            enable_native_tool_calls=self._native_tool_calls,
            # Retrieval-first: run the knowledge base before the first LLM turn.
            initial_tool_name="rag_search" if rag_tool is not None else "",
            # Never answer a tool-use request from model memory alone — require
            # usable evidence from RAG or any MCP tool.
            require_tool_call=True,
            required_tool_names={tool.name for tool in tools},
            route_reason=decision.reason,
        )
