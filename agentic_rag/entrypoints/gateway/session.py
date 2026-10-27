"""Platform session mapping — binds (platform, user_id, chat_id) → internal session_id.

Uses an in-memory LRU cache for hot-path lookups, backed by SQLite for persistence.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from agentic_rag.data.db.session_repo import SessionRepo


class PlatformSessionMap:
    """Maps (platform, platform_user_id, chat_id) → internal session_id."""

    def __init__(self, max_cache_size: int = 1000) -> None:
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._max_cache = max_cache_size

    # ── Public API ──────────────────────────────────────────

    def get_or_create(self, platform: str, user_id: str, chat_id: str = "") -> str:
        """Look up an existing session or create a new one."""
        cache_key = (platform, user_id, chat_id)

        # 1. Cache hit
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # 2. DB lookup
        sid = self._db_get(platform, user_id, chat_id)
        if sid:
            self._set_cache(cache_key, sid)
            return sid

        # 3. Create new session
        repo = SessionRepo()
        display_name = f"{platform}:{user_id}"
        if chat_id:
            display_name += f":{chat_id}"
        session = repo.create(user_id=display_name)
        sid = session["id"]
        self._db_set(platform, user_id, chat_id, sid)

        self._set_cache(cache_key, sid)
        return sid

    def invalidate(self, platform: str, user_id: str, chat_id: str = "") -> None:
        """Remove a cached mapping (on session deletion, etc.)."""
        cache_key = (platform, user_id, chat_id)
        self._cache.pop(cache_key, None)

    # ── Internal cache helpers ───────────────────────────────

    def _set_cache(self, key: tuple[str, str, str], sid: str) -> None:
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)  # evict oldest
        self._cache[key] = sid

    # ── DB helpers (delegated to SessionRepo) ────────────────

    @staticmethod
    def _db_get(platform: str, user_id: str, chat_id: str) -> Optional[str]:
        try:
            return SessionRepo().get_platform_session(platform, user_id, chat_id)
        except Exception:
            return None

    @staticmethod
    def _db_set(platform: str, user_id: str, chat_id: str, session_id: str) -> None:
        try:
            SessionRepo().bind_platform_session(platform, user_id, chat_id, session_id)
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────

_platform_session_map: Optional[PlatformSessionMap] = None


def get_platform_session_map() -> PlatformSessionMap:
    """Get the global PlatformSessionMap singleton."""
    global _platform_session_map
    if _platform_session_map is None:
        _platform_session_map = PlatformSessionMap()
    return _platform_session_map
