"""Session lifecycle manager."""

import time
import uuid
from typing import Optional

from agentic_rag.data.models import Session


class SessionManager:
    """Manages session creation, retrieval, and expiration.

    Uses in-memory storage (will be upgraded to SQLite in a later phase).
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}

    def create(self, user_id: str = "default") -> Session:
        """Create a new session."""
        session = Session(user_id=user_id)
        session.expires_at = time.time() + self.ttl_seconds
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID. Returns None if expired or not found."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() > session.expires_at:
            del self._sessions[session_id]
            return None
        return session

    def get_or_create(self, session_id: str | None, user_id: str = "default") -> Session:
        """Get existing session or create a new one."""
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        return self.create(user_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def extend(self, session_id: str) -> bool:
        """Extend session TTL."""
        session = self._sessions.get(session_id)
        if session:
            session.expires_at = time.time() + self.ttl_seconds
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now > s.expires_at]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Global instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        from agentic_rag.config.settings import get_settings
        _session_manager = SessionManager(ttl_seconds=get_settings().session.ttl_seconds)
    return _session_manager
