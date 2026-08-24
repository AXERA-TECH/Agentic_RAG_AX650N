"""Session repository — persistence layer for chat sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from agentic_rag.data.db import get_db


class SessionRepo:
    """CRUD operations for sessions."""

    def __init__(self) -> None:
        self._db = get_db()

    # ── Session CRUD ──────────────────────────────

    def create(self, user_id: str = "default", title: str = "") -> dict:
        sid = uuid.uuid4().hex
        return self.create_with_id(sid, user_id, title)

    def create_with_id(self, sid: str, user_id: str = "default", title: str = "") -> dict:
        now = _now()
        self._db.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, user_id, title, now, now),
        )
        self._db.commit()
        return {"id": sid, "user_id": user_id, "title": title,
                "created_at": now, "updated_at": now}

    def get(self, session_id: str) -> Optional[dict]:
        row = self._db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,),
        )
        return dict(row) if row else None

    def list(self, user_id: str = "default", limit: int = 50) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM sessions WHERE user_id = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in rows]

    def update(self, session_id: str, **fields) -> bool:
        if not fields:
            return False
        sets = [f"{k} = ?" for k in fields]
        values = list(fields.values())
        values.append(_now())
        values.append(session_id)
        self._db.execute(
            f"UPDATE sessions SET {', '.join(sets)}, updated_at = ? "
            f"WHERE id = ?",
            tuple(values),
        )
        self._db.commit()
        return True

    def delete(self, session_id: str) -> bool:
        self._db.execute("DELETE FROM platform_sessions WHERE session_id = ?", (session_id,))
        self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._db.commit()
        return True

    def delete_all(self, user_id: str = "default") -> int:
        """Delete all sessions and messages for a user. Returns count of deleted sessions."""
        sessions = self._db.fetchall(
            "SELECT id FROM sessions WHERE user_id = ?", (user_id,),
        )
        count = len(sessions)
        for s in sessions:
            self._db.execute("DELETE FROM platform_sessions WHERE session_id = ?", (s["id"],))
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (s["id"],))
        self._db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self._db.commit()
        return count

    # ── Platform Session Binding ──────────────────

    def get_platform_session(
        self, platform: str, platform_user_id: str, chat_id: str = ""
    ) -> Optional[str]:
        """Get the internal session_id for a platform user+chat combination."""
        row = self._db.fetchone(
            "SELECT session_id FROM platform_sessions "
            "WHERE platform = ? AND platform_user_id = ? AND chat_id = ?",
            (platform, platform_user_id, chat_id),
        )
        return row["session_id"] if row else None

    def clear_messages(self, session_id: str) -> int:
        """Delete all messages from a session, keep the session. Returns count of deleted messages."""
        cursor = self._db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
            (session_id,),
        )
        count = cursor.fetchone()["cnt"]
        self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._db.commit()
        return count

    def bind_platform_session(
        self, platform: str, platform_user_id: str,
        chat_id: str, session_id: str,
    ) -> None:
        """Bind a platform user+chat to an internal session (idempotent upsert)."""
        self._db.execute(
            "INSERT OR REPLACE INTO platform_sessions "
            "(platform, platform_user_id, chat_id, session_id) "
            "VALUES (?, ?, ?, ?)",
            (platform, platform_user_id, chat_id, session_id),
        )
        self._db.commit()
        """Delete all sessions and messages for a user. Returns count of deleted sessions."""
        sessions = self._db.fetchall(
            "SELECT id FROM sessions WHERE user_id = ?", (user_id,),
        )
        count = len(sessions)
        for s in sessions:
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (s["id"],))
        self._db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self._db.commit()
        return count

    # ── Messages ──────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str,
                    tool_calls: list | None = None,
                    tool_call_id: str | None = None) -> int:
        self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        cursor = self._db.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content,
             json.dumps(tool_calls or []),
             tool_call_id),
        )
        self._db.commit()
        return cursor.lastrowid

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM messages WHERE session_id = ? "
            "ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        )
        return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
