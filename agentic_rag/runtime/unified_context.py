"""UnifiedContext — immutable request-scoped context carrying session, user, and service references."""

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnifiedContext:
    """Immutable context object for a single request/turn.

    Carries everything the agent needs to know about the current request:
    - Session and user identity
    - Service container references
    - Request metadata
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = "default"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Service references (set at runtime)
    settings: object = None
    session_manager: object = None
    memory_manager: object = None
    tool_registry: object = None

    @classmethod
    def create(cls, session_id: str = "", user_id: str = "default") -> "UnifiedContext":
        """Create a UnifiedContext with service references resolved."""
        from agentic_rag.config.settings import get_settings
        from agentic_rag.services.session.manager import get_session_manager
        from agentic_rag.services.memory.manager import get_memory_manager
        from agentic_rag.orchestration.l1_tools.registry import get_tool_registry

        return cls(
            session_id=session_id or uuid.uuid4().hex,
            user_id=user_id,
            request_id=uuid.uuid4().hex[:12],
            settings=get_settings(),
            session_manager=get_session_manager(),
            memory_manager=get_memory_manager(),
            tool_registry=get_tool_registry(),
        )
