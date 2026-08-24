"""StreamBus — internal pub/sub event bus for streaming events to clients."""

import asyncio
from typing import AsyncIterator, Optional

from agentic_rag.data.models import AgentEvent


class StreamBus:
    """Async pub/sub event bus for agent streaming.

    Agents publish events to topics (keyed by turn_id).
    Entry points subscribe to relay events to SSE/WebSocket clients.
    """

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def publish(self, turn_id: str, event: AgentEvent) -> None:
        """Publish an event to all subscribers of a turn."""
        event.turn_id = turn_id
        queues = self._subscribers.get(turn_id, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop event if subscriber is too slow

    async def subscribe(self, turn_id: str, buffer_size: int = 100) -> AsyncIterator[AgentEvent]:
        """Subscribe to events for a turn. Yields events as they arrive."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        if turn_id not in self._subscribers:
            self._subscribers[turn_id] = []
        self._subscribers[turn_id].append(queue)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                    yield event
                    if event.event_type.value == "done":
                        break
                except asyncio.TimeoutError:
                    yield AgentEvent(event_type="error", data={"message": "Stream timeout"})
                    break
        finally:
            self._subscribers[turn_id].remove(queue)
            if not self._subscribers[turn_id]:
                del self._subscribers[turn_id]

    def cleanup(self, turn_id: str) -> None:
        """Remove all subscribers for a turn."""
        self._subscribers.pop(turn_id, None)


# Global instance
_stream_bus: Optional[StreamBus] = None


def get_stream_bus() -> StreamBus:
    global _stream_bus
    if _stream_bus is None:
        _stream_bus = StreamBus()
    return _stream_bus
