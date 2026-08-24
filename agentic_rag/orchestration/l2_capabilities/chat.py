"""L2 Chat Capability — policy-routed multi-turn conversation."""

from typing import AsyncIterator

from agentic_rag.data.models import AgentEvent, AgentInput, AgentOutput, Message
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
    ) -> AgentOutput:
        """Process a chat message and return the agent's response (non-streaming).

        Args:
            message: User's message text.
            session_id: Session identifier for conversation continuity.
            has_media: Whether the request includes media attachments.
            context: Optional pre-built UnifiedContext.

        Returns:
            AgentOutput with final_answer and execution metadata.
        """
        if context is None:
            context = UnifiedContext.create(session_id=session_id)

        # Load conversation history
        history = self._load_history(context)
        input_data = AgentInput(query=message, messages=history)

        return await self.orchestrator.process(
            query=message,
            session_id=context.session_id,
            has_media=has_media,
            input_data=input_data,
            context=context,
        )

    async def chat_stream(
        self,
        message: str,
        session_id: str = "",
        context: UnifiedContext | None = None,
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
        history = self._load_history(context)
        input_data = AgentInput(query=message, messages=history)

        async for event in self.orchestrator.process_stream(
            query=message,
            session_id=context.session_id,
            input_data=input_data,
            context=context,
        ):
            yield event

    async def chat_multimodal(
        self,
        text: str = "",
        images: list[str] | None = None,
        audio: str | None = None,
        video: str | None = None,
        session_id: str = "",
        context: UnifiedContext | None = None,
    ) -> AgentOutput:
        """Process a multimodal message (text + optional media).

        Images, audio, and video are processed before passing to the agent.
        """
        # Process multimodal content
        processed_text = text or ""
        has_media = bool(images or audio or video)

        if images:
            for img in images:
                processed_text += f"\n[Image: {img}]"
        if audio:
            processed_text += f"\n[Audio: {audio}]"
        if video:
            processed_text += f"\n[Video: {video}]"

        return await self.chat(
            message=processed_text,
            session_id=session_id,
            has_media=has_media,
            context=context,
        )

    def _load_history(self, context: UnifiedContext) -> list[Message]:
        """Load conversation history from memory."""
        memory = context.memory_manager
        if memory:
            return memory.get_messages(context.session_id, limit=20)
        return []

    def get_history(self, session_id: str = "") -> list[Message]:
        """Get conversation history for a session."""
        from agentic_rag.services.memory.manager import get_memory_manager
        return get_memory_manager().get_messages(session_id, limit=50)


# Global instance
_chat_capability: ChatCapability | None = None


def get_chat_capability() -> ChatCapability:
    global _chat_capability
    if _chat_capability is None:
        _chat_capability = ChatCapability()
    return _chat_capability
