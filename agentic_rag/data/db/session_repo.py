"""Session repository — persistence layer for chat sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from agentic_rag.data.db import get_db


class SessionRepo:
    """CRUD operations for sessions."""

    def __init__(self, db=None) -> None:
        self._db = db or get_db()

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
        self._db.execute("DELETE FROM session_context WHERE session_id = ?", (session_id,))
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
            self._db.execute("DELETE FROM session_context WHERE session_id = ?", (s["id"],))
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
        # The rolling summary refers to messages that no longer exist. Reset it
        # so the next request starts from a clean, empty history.
        self._db.execute("DELETE FROM session_context WHERE session_id = ?", (session_id,))
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

    # ── Messages ──────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str,
                    tool_calls: list | None = None,
                    tool_call_id: str | None = None,
                    attachments: list[dict] | None = None) -> int:
        self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        cursor = self._db.execute(
            "INSERT INTO messages "
            "(session_id, role, content, tool_calls, tool_call_id, attachments) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content,
             json.dumps(tool_calls or []),
             tool_call_id,
             json.dumps(attachments or [])),
        )
        self._db.commit()
        return cursor.lastrowid

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        """Return the most recent ``limit`` messages, oldest-first.

        The previous implementation used ``ORDER BY id ASC LIMIT``, which
        returned the *oldest* messages once a session grew past the limit —
        the recent turns silently disappeared.
        """
        rows = self._db.fetchall(
            "SELECT * FROM ("
            "  SELECT * FROM messages WHERE session_id = ? "
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (session_id, limit),
        )
        return [_message_row(row) for row in rows]

    def get_messages_after(self, session_id: str, after_id: int) -> list[dict]:
        """Return messages newer than ``after_id``, oldest-first.

        Used by the context builder: everything at or before
        ``covered_until_id`` is already folded into the rolling summary.
        """
        rows = self._db.fetchall(
            "SELECT * FROM messages WHERE session_id = ? AND id > ? "
            "ORDER BY id ASC",
            (session_id, after_id),
        )
        return [_message_row(row) for row in rows]

    # ── Rolling summary state ─────────────────────

    def get_context(self, session_id: str) -> dict:
        """Return the session's rolling-summary state (empty default)."""
        row = self._db.fetchone(
            "SELECT * FROM session_context WHERE session_id = ?",
            (session_id,),
        )
        if row:
            return dict(row)
        return {
            "session_id": session_id,
            "summary": "",
            "summary_tokens": 0,
            "covered_until_id": 0,
            "summary_version": 0,
        }

    def save_context(
        self,
        session_id: str,
        summary: str,
        summary_tokens: int,
        covered_until_id: int,
    ) -> bool:
        """Persist a rebuilt summary and advance ``covered_until_id``.

        The ``WHERE`` clause makes the update monotonic: when two requests
        rebuild concurrently, only the one covering more history wins.  The
        summary row is only ever advanced after a successful summarization,
        so evicted messages are never lost without a persisted record.
        """
        with self._db.conn:
            cursor = self._db.execute(
                """INSERT INTO session_context
                       (session_id, summary, summary_tokens, covered_until_id,
                        summary_version, updated_at)
                   VALUES (?, ?, ?, ?, 1, datetime('now'))
                   ON CONFLICT(session_id) DO UPDATE SET
                       summary = excluded.summary,
                       summary_tokens = excluded.summary_tokens,
                       covered_until_id = excluded.covered_until_id,
                       summary_version = session_context.summary_version + 1,
                       updated_at = datetime('now')
                   WHERE excluded.covered_until_id
                         > session_context.covered_until_id""",
                (session_id, summary, summary_tokens, covered_until_id),
            )
            return cursor.rowcount > 0


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _message_row(row) -> dict:
    """Decode JSON columns without exposing raw serialized blobs to callers."""
    data = dict(row)
    raw = data.get("attachments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "[]")
        except ValueError:
            parsed = []
        data["attachments"] = parsed if isinstance(parsed, list) else []
    else:
        data["attachments"] = raw or []
    return data
