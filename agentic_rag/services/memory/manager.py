"""Memory Manager — coordinates short-term, long-term, and working memory."""

from typing import Optional

from agentic_rag.data.models import Message


class MemoryManager:
    """Coordinates the three-tier memory system.

    - ShortTermMemory: Sliding window of recent messages
    - LongTermMemory: Vector-indexed semantic memories (Milvus-backed, future)
    - WorkingMemory: Ephemeral scratchpad per turn
    """

    def __init__(self, max_short_term_tokens: int = 8000):
        self.max_short_term_tokens = max_short_term_tokens
        self._short_term: dict[str, list[Message]] = {}  # session_id -> messages
        self._working: dict[str, dict] = {}               # session_id -> scratchpad

    # ── Short-term Memory ──────────────────────────

    def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to short-term memory."""
        if session_id not in self._short_term:
            self._short_term[session_id] = []
        self._short_term[session_id].append(message)
        self._trim(session_id)

    def get_messages(self, session_id: str, limit: int = 50) -> list[Message]:
        """Get recent messages for a session."""
        messages = self._short_term.get(session_id, [])
        return messages[-limit:] if len(messages) > limit else messages

    def clear_short_term(self, session_id: str) -> None:
        """Clear short-term memory for a session."""
        self._short_term.pop(session_id, None)

    def _trim(self, session_id: str) -> None:
        """Trim messages to fit within token budget (approximate)."""
        messages = self._short_term.get(session_id, [])
        total_tokens = sum(len(str(m.content)) // 4 for m in messages)
        while total_tokens > self.max_short_term_tokens and len(messages) > 2:
            removed = messages.pop(0)
            total_tokens -= len(str(removed.content)) // 4

    # ── Working Memory ─────────────────────────────

    def set_working(self, session_id: str, key: str, value) -> None:
        """Set a working memory value for the current turn."""
        if session_id not in self._working:
            self._working[session_id] = {}
        self._working[session_id][key] = value

    def get_working(self, session_id: str, key: str, default=None):
        """Get a working memory value."""
        return self._working.get(session_id, {}).get(key, default)

    def clear_working(self, session_id: str) -> None:
        """Clear working memory at end of turn."""
        self._working.pop(session_id, None)

    # ── Long-term Memory (future) ──────────────────

    async def search_long_term(self, session_id: str, query: str, top_k: int = 10) -> list[dict]:
        """Search long-term memory for relevant past interactions."""
        # Placeholder — will be implemented with Milvus in Phase 3
        return []

    async def store_long_term(self, session_id: str, fact: str, metadata: dict | None = None) -> None:
        """Store a fact in long-term memory."""
        # Placeholder — will be implemented with Milvus in Phase 3
        pass


# Global instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        from agentic_rag.config.settings import get_settings
        _memory_manager = MemoryManager(
            max_short_term_tokens=get_settings().memory.short_term_max_tokens,
        )
    return _memory_manager
