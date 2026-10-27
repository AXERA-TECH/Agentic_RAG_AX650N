"""ChatOrchestrator — central coordinator for all agent interactions."""

import asyncio
import threading
from typing import AsyncIterator, Callable, Optional

from agentic_rag.data.models import AgentEvent, AgentInput, AgentOutput
from agentic_rag.runtime.turn_manager import get_turn_manager
from agentic_rag.runtime.unified_context import UnifiedContext
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _default_llm_factory(provider: str = ""):
    from agentic_rag.services.llm.factory import get_llm
    return get_llm(provider) if provider else get_llm()


def _default_tool_registry():
    from agentic_rag.orchestration.l1_tools.registry import get_tool_registry
    return get_tool_registry()


def _default_router_factory(llm, tool_registry):
    from agentic_rag.agent.router import AgentRouter
    return AgentRouter(llm, tool_registry)


def _default_repo_factory():
    from agentic_rag.data.db.session_repo import SessionRepo
    return SessionRepo()


class ChatOrchestrator:
    """The central conductor for the Agentic RAG system.

    Orchestrates the full lifecycle of each user request:
    1. Create/set up UnifiedContext
    2. Select a local, direct, retrieval-first, or ReAct path via AgentRouter
    3. Execute the agent (streaming or non-streaming)
    4. Persist the turn to SQLite (the single conversation store)

    Dependencies default to the process-wide factories but can be injected,
    which is what the application composition root does.
    """

    def __init__(
        self,
        *,
        llm_factory: Callable[..., object] | None = None,
        tool_registry: Callable[[], object] | None = None,
        router_factory: Callable[..., object] | None = None,
        repo_factory: Callable[[], object] | None = None,
    ):
        self.turn_manager = get_turn_manager()
        self._llm_factory = llm_factory or _default_llm_factory
        self._tool_registry = tool_registry or _default_tool_registry
        self._router_factory = router_factory or _default_router_factory
        self._repo_factory = repo_factory or _default_repo_factory

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

        # Route and execute.  Attachments are request-scoped capability IDs
        # carried on AgentInput; the router turns them into media tools.
        engine_input = input_data or AgentInput(query=query)
        engine = await self._route(
            query, has_media, provider, engine_input=engine_input,
        )

        try:
            output = await engine.run(engine_input, turn_id=turn.turn_id)

            # Update turn stats
            self.turn_manager.update_usage(
                turn.turn_id,
                prompt_tokens=output.usage.get("prompt_tokens", 0),
                completion_tokens=output.usage.get("completion_tokens", 0),
            )
            self.turn_manager.finish_turn(turn.turn_id)

            # Persist the finished turn to SQLite without blocking the loop.
            if session_id:
                await self._persist_session(
                    context.session_id, query, output.final_answer,
                    engine_input,
                )
            report = (engine_input.parameters or {}).get("context_report")
            if report:
                output.diagnostics["context"] = report

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
        engine_input = input_data or AgentInput(query=query)
        engine = await self._route(
            query, has_media, provider, engine_input=engine_input,
        )

        full_answer = ""
        try:
            async for event in engine.stream(engine_input, turn_id=turn.turn_id):
                if event.event_type.value == "done":
                    report = (engine_input.parameters or {}).get("context_report")
                    if report:
                        event.data["context_report"] = report
                if event.event_type.value == "text_delta":
                    full_answer += event.data.get("content", "")
                elif event.event_type.value == "done":
                    full_answer = event.data.get("final_answer", "") or full_answer
                yield event

            self.turn_manager.finish_turn(turn.turn_id)
            if session_id:
                await self._persist_session(
                    context.session_id, query, full_answer, engine_input,
                )

        except Exception as e:
            self.turn_manager.finish_turn(turn.turn_id, error=str(e))
            yield AgentEvent(event_type="error", data={"message": str(e)})

    async def _route(
        self,
        query: str,
        has_media: bool,
        provider: str = "",
        engine_input: AgentInput | None = None,
    ):
        """Route query to the smallest suitable execution engine."""
        llm = self._llm_factory(provider)
        tool_registry = self._tool_registry()
        self._ensure_tools(tool_registry)

        router = self._router_factory(llm, tool_registry)
        attachments = list(engine_input.attachments) if engine_input else []
        return await router.route(
            query=query,
            has_media=has_media or bool(attachments),
            attachments=attachments,
        )

    async def _persist_session(
        self,
        session_id: str,
        query: str,
        answer: str,
        input_data: AgentInput | None = None,
    ) -> None:
        """Persist the finished turn off the event loop."""
        if not session_id:
            return
        await asyncio.to_thread(
            self._persist_session_sync, session_id, query, answer, input_data,
        )

    def _persist_session_sync(
        self,
        session_id: str,
        query: str,
        answer: str,
        input_data: AgentInput | None = None,
    ) -> None:
        """Persist the finished turn to SQLite exactly once.

        All entrypoints (REST, voice, WebSocket, messaging gateways) go
        through the orchestrator, so this is the only writer.  The context
        builder reads these rows on the next request; persisting in one place
        avoids the previous split between ``build_chat_input`` writing the
        user message and each route writing the assistant reply.
        """
        if not session_id:
            return
        try:
            repo = self._repo_factory()
            if not repo.get(session_id):
                user_id = "default"
                if input_data is not None:
                    user_id = (input_data.parameters or {}).get(
                        "user_id", "default",
                    )
                repo.create_with_id(session_id, user_id=user_id)
            attachments = []
            if input_data is not None:
                attachments = [
                    ref.model_dump(mode="json")
                    for ref in input_data.attachments
                ]
            repo.add_message(
                session_id, role="user", content=query,
                attachments=attachments,
            )
            if answer and answer.strip():
                repo.add_message(session_id, role="assistant", content=answer)
        except Exception:
            logger.exception("Failed to persist session %s", session_id)

    @staticmethod
    def _ensure_tools(registry) -> None:
        """Ensure the built-in knowledge search tool is registered."""
        if registry.get("rag_search") is None:
            from agentic_rag.orchestration.l1_tools.rag_tools import RAGSearchTool
            registry.register(RAGSearchTool())


# Global instance
_orchestrator: Optional[ChatOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> ChatOrchestrator:
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = ChatOrchestrator()
        return _orchestrator


def set_orchestrator(instance: ChatOrchestrator) -> None:
    """Install the instance built by the composition root."""
    global _orchestrator
    with _orchestrator_lock:
        _orchestrator = instance

