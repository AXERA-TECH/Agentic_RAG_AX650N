"""L2 Chat Capability — policy-routed multi-turn conversation."""

from typing import AsyncIterator

from agentic_rag.data.models import AgentEvent, AgentOutput, Message, MessageRole
from agentic_rag.runtime.orchestrator import ChatOrchestrator
from agentic_rag.runtime.unified_context import UnifiedContext


class ChatCapability:
    """L2 Chat capability — orchestrates policy-routed conversational interactions.

    This is the main user-facing capability that wraps the full pipeline:
    multimodal processing → request policy → execution → memory persistence.
    """

    def __init__(self, orchestrator: ChatOrchestrator | None = None):
        self.orchestrator = orchestrator or ChatOrchestrator()

    async def chat(
        self,
        message: str,
        session_id: str = "",
        has_media: bool = False,
        context: UnifiedContext | None = None,
        attachment_ids: list[str] | None = None,
    ) -> AgentOutput:
        """Process a chat message and return the agent's response (non-streaming).

        Args:
            message: User's message text.
            session_id: Session identifier for conversation continuity.
            has_media: Whether the request includes media attachments.
            context: Optional pre-built UnifiedContext.
            attachment_ids: Uploaded attachments for this turn.

        Returns:
            AgentOutput with final_answer and execution metadata.
        """
        if context is None:
            context = UnifiedContext.create(session_id=session_id)

        # Load summary + recent history from SQLite through the shared
        # context builder (same path as /chat and the gateways).
        from agentic_rag.runtime.session_input import build_chat_input
        sid = context.session_id or session_id
        input_data = await build_chat_input(
            sid, message, attachment_ids or [], user_id=context.user_id,
        )

        return await self.orchestrator.process(
            query=message,
            session_id=sid,
            has_media=has_media or bool(attachment_ids),
            input_data=input_data,
            context=context,
        )

    async def chat_stream(
        self,
        message: str,
        session_id: str = "",
        context: UnifiedContext | None = None,
        attachment_ids: list[str] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Process a chat message with streaming output.

        Yields AgentEvents as the selected execution path progresses:
        - thought: Agent's reasoning step
        - text_delta: Incremental response text
        - tool_call_start: Tool invocation beginning
        - tool_call_result: Tool execution result
        - done: Final answer delivered
        """
        if context is None:
            context = UnifiedContext.create(session_id=session_id)
        from agentic_rag.runtime.session_input import build_chat_input
        sid = context.session_id or session_id
        input_data = await build_chat_input(
            sid, message, attachment_ids or [], user_id=context.user_id,
        )

        async for event in self.orchestrator.process_stream(
            query=message,
            session_id=sid,
            has_media=bool(attachment_ids),
            input_data=input_data,
            context=context,
        ):
            yield event

    async def chat_multimodal(
        self,
        text: str = "",
        attachment_ids: list[str] | None = None,
        session_id: str = "",
        context: UnifiedContext | None = None,
        **legacy_media,
    ) -> AgentOutput:
        """Process a message plus previously uploaded attachments.

        Raw paths/URLs are intentionally unsupported: callers upload through
        ``POST /attachments`` and pass the resulting IDs.
        """
        if legacy_media:
            raise ValueError(
                "chat_multimodal no longer accepts raw media paths; upload "
                "files via /attachments and pass attachment_ids instead."
            )
        return await self.chat(
            message=text,
            session_id=session_id,
            has_media=bool(attachment_ids),
            context=context,
            attachment_ids=attachment_ids or [],
        )

    def get_history(self, session_id: str = "") -> list[Message]:
        """Get persisted conversation history for a session.

        Uses the same SQLite source as the context builder.  The previous
        implementation read a process-local in-memory store, which made
        WebSocket history diverge from REST history.
        """
        from agentic_rag.data.db.session_repo import SessionRepo
        from agentic_rag.services.memory.context_builder import _row_text

        messages: list[Message] = []
        for row in SessionRepo().get_messages(session_id, limit=50):
            try:
                role = MessageRole(row.get("role", "user"))
            except ValueError:
                role = MessageRole.USER
            messages.append(Message(
                role=role,
                content=_row_text(row),
            ))
        return messages


# Global instance
_chat_capability: ChatCapability | None = None


def get_chat_capability() -> ChatCapability:
    global _chat_capability
    if _chat_capability is None:
        _chat_capability = ChatCapability()
    return _chat_capability
