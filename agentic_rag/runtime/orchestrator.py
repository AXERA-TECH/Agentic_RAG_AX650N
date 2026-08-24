"""ChatOrchestrator — central coordinator for all agent interactions."""

import uuid
from typing import AsyncIterator, Optional

from agentic_rag.data.models import AgentEvent, AgentInput, AgentOutput, Message
from agentic_rag.runtime.stream_bus import get_stream_bus
from agentic_rag.runtime.turn_manager import get_turn_manager
from agentic_rag.runtime.unified_context import UnifiedContext


class ChatOrchestrator:
    """The central conductor for the Agentic RAG system.

    Orchestrates the full lifecycle of each user request:
    1. Create/set up UnifiedContext
    2. Select a local, direct, retrieval-first, or ReAct path via AgentRouter
    3. Execute the agent (streaming or non-streaming)
    4. Persist results to memory
    5. Emit events via StreamBus
    """

    def __init__(self):
        self.stream_bus = get_stream_bus()
        self.turn_manager = get_turn_manager()

    async def process(
        self,
        query: str,
        session_id: str = "",
        has_media: bool = False,
        input_data: AgentInput | None = None,
        context: UnifiedContext | None = None,
        provider: str = "",
    ) -> AgentOutput:
        """Process a query through the full pipeline (non-streaming)."""
        if context is None:
            context = UnifiedContext.create(session_id=session_id)

        # Start turn
        turn = self.turn_manager.start_turn(context.session_id)

        # Route and execute
        engine = await self._route(query, has_media, provider)
        engine_input = input_data or AgentInput(query=query)

        try:
            output = await engine.run(engine_input, turn_id=turn.turn_id)

            # Update turn stats
            self.turn_manager.update_usage(
                turn.turn_id,
                prompt_tokens=output.usage.get("prompt_tokens", 0),
                completion_tokens=output.usage.get("completion_tokens", 0),
            )
            self.turn_manager.finish_turn(turn.turn_id)

            # Persist to memory
            await self._persist_memory(context, query, output.final_answer)

            return output
        except Exception as e:
            self.turn_manager.finish_turn(turn.turn_id, error=str(e))
            raise

    async def process_stream(
        self,
        query: str,
        session_id: str = "",
        has_media: bool = False,
        input_data: AgentInput | None = None,
        context: UnifiedContext | None = None,
        provider: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Process a query with streaming output."""
        if context is None:
            context = UnifiedContext.create(session_id=session_id)

        turn = self.turn_manager.start_turn(context.session_id)
        engine = await self._route(query, has_media, provider)
        engine_input = input_data or AgentInput(query=query)

        full_answer = ""
        try:
            async for event in engine.stream(engine_input, turn_id=turn.turn_id):
                self.stream_bus.publish(turn.turn_id, event)
                if event.event_type.value == "text_delta":
                    full_answer += event.data.get("content", "")
                elif event.event_type.value == "done":
                    full_answer = event.data.get("final_answer", "") or full_answer
                yield event

            self.turn_manager.finish_turn(turn.turn_id)
            await self._persist_memory(context, query, full_answer)

        except Exception as e:
            self.turn_manager.finish_turn(turn.turn_id, error=str(e))
            yield AgentEvent(event_type="error", data={"message": str(e)})

    async def _route(self, query: str, has_media: bool, provider: str = ""):
        """Route query to the smallest suitable execution engine."""
        from agentic_rag.services.llm.factory import get_llm
        from agentic_rag.orchestration.l1_tools.registry import get_tool_registry
        from agentic_rag.agent.router import AgentRouter

        llm = get_llm(provider) if provider else get_llm()
        tool_registry = get_tool_registry()
        self._ensure_tools(tool_registry)

        router = AgentRouter(llm, tool_registry)
        return await router.route(query=query, has_media=has_media)

    async def _persist_memory(self, context: UnifiedContext, query: str, answer: str) -> None:
        """Save the interaction to memory."""
        try:
            memory = context.memory_manager
            if memory:
                memory.add_message(context.session_id, Message.user(query))
                memory.add_message(context.session_id, Message.assistant(answer))
        except Exception:
            pass  # Memory persistence should never fail the main flow

    @staticmethod
    def _ensure_tools(registry) -> None:
        """Ensure the built-in knowledge search tool is registered."""
        if registry.get("rag_search") is None:
            from agentic_rag.orchestration.l1_tools.rag_tools import RAGSearchTool
            registry.register(RAGSearchTool())


# Global instance
_orchestrator: Optional[ChatOrchestrator] = None


def get_orchestrator() -> ChatOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChatOrchestrator()
    return _orchestrator
