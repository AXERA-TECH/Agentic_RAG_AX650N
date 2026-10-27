"""TurnRuntimeManager — manages the lifecycle of a single conversational turn."""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TurnRecord:
    """Record of a single conversational turn."""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls_count: int = 0
    iterations: int = 0
    error: str | None = None


class TurnRuntimeManager:
    """Manages turn lifecycle — start, track, finish."""

    def __init__(self):
        self._active_turns: dict[str, TurnRecord] = {}
        self._history: list[TurnRecord] = []

    def start_turn(self, session_id: str) -> TurnRecord:
        """Begin a new turn."""
        turn = TurnRecord(session_id=session_id)
        self._active_turns[turn.turn_id] = turn
        return turn

    def update_usage(self, turn_id: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Update token counts for a turn."""
        turn = self._active_turns.get(turn_id)
        if turn:
            turn.prompt_tokens += prompt_tokens
            turn.completion_tokens += completion_tokens

    def record_tool_call(self, turn_id: str) -> None:
        """Record a tool call in the active turn."""
        turn = self._active_turns.get(turn_id)
        if turn:
            turn.tool_calls_count += 1

    def finish_turn(self, turn_id: str, error: str | None = None) -> TurnRecord | None:
        """Complete a turn and archive it."""
        turn = self._active_turns.pop(turn_id, None)
        if turn:
            turn.finished_at = time.time()
            turn.error = error
            self._history.append(turn)
        return turn

    def get_active_turn(self, turn_id: str) -> TurnRecord | None:
        """Get an active turn by ID."""
        return self._active_turns.get(turn_id)

    @property
    def active_count(self) -> int:
        return len(self._active_turns)

    @property
    def total_turns(self) -> int:
        return len(self._history)

    def get_history(self, limit: int = 20) -> list[TurnRecord]:
        """Get recent turn history."""
        return self._history[-limit:]


# Global instance
_turn_manager: TurnRuntimeManager | None = None


def get_turn_manager() -> TurnRuntimeManager:
    global _turn_manager
    if _turn_manager is None:
        _turn_manager = TurnRuntimeManager()
    return _turn_manager
