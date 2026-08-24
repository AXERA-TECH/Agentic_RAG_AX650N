"""Database layer — SQLite-backed persistence for sessions, documents, and metadata.

Replaces the in-memory dicts used during early development with a proper
persistent store.  Uses aiosqlite for async access so it integrates cleanly
with the FastAPI / asyncio stack.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Connection management
# ═══════════════════════════════════════════════════════════════

class Database:
    """Thin wrapper around a SQLite connection with schema management."""

    def __init__(self, db_path: str | Path = "data/agentic_rag.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Schema ────────────────────────────────────

    SCHEMA_VERSION = 1

    def init_schema(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        c = self.conn
        c.executescript("""
            -- Schema versioning
            CREATE TABLE IF NOT EXISTS _schema (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- Sessions (chat conversation containers)
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                user_id     TEXT    NOT NULL DEFAULT 'default',
                title       TEXT    NOT NULL DEFAULT '',
                metadata    TEXT    NOT NULL DEFAULT '{}',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- Messages within a session
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT    NOT NULL,  -- 'system' | 'user' | 'assistant' | 'tool'
                content     TEXT    NOT NULL DEFAULT '',
                tool_calls  TEXT    NOT NULL DEFAULT '[]',
                tool_call_id TEXT   DEFAULT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);

            -- Documents ingested into the knowledge base
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                source      TEXT    NOT NULL DEFAULT '',
                source_type TEXT    NOT NULL DEFAULT 'text',
                content_preview TEXT NOT NULL DEFAULT '',
                item_count  INTEGER NOT NULL DEFAULT 0,
                metadata    TEXT    NOT NULL DEFAULT '{}',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- Content items (individual chunks / images / tables)
            CREATE TABLE IF NOT EXISTS content_items (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                type        TEXT    NOT NULL DEFAULT 'text',
                text        TEXT    NOT NULL DEFAULT '',
                img_path    TEXT    NOT NULL DEFAULT '',
                table_body  TEXT    NOT NULL DEFAULT '',
                page_idx    INTEGER NOT NULL DEFAULT 0,
                metadata    TEXT    NOT NULL DEFAULT '{}',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_content_doc
                ON content_items(doc_id);
            CREATE INDEX IF NOT EXISTS idx_content_type
                ON content_items(type);

            -- Platform-to-session bindings (for messaging gateway)
            CREATE TABLE IF NOT EXISTS platform_sessions (
                platform         TEXT NOT NULL,
                platform_user_id TEXT NOT NULL,
                chat_id          TEXT NOT NULL DEFAULT '',
                session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (platform, platform_user_id, chat_id)
            );

            -- Key-value store for application config / cache
            CREATE TABLE IF NOT EXISTS kv_store (
                key         TEXT PRIMARY KEY,
                value       TEXT    NOT NULL DEFAULT '',
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)

        # Record schema version
        existing = c.execute(
            "SELECT version FROM _schema WHERE version = ?",
            (self.SCHEMA_VERSION,),
        ).fetchone()
        if not existing:
            c.execute(
                "INSERT INTO _schema (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )
            c.commit()

    # ── Raw SQL helpers ───────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_list)

    def commit(self) -> None:
        self.conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()


# ═══════════════════════════════════════════════════════════════
# Global instance
# ═══════════════════════════════════════════════════════════════

_db: Optional[Database] = None


def get_db(db_path: str | Path = "") -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        path = db_path or "data/agentic_rag.db"
        try:
            from agentic_rag.config.settings import get_settings
            path = get_settings().db_path
        except Exception:
            pass
        _db = Database(path)
        _db.init_schema()
    return _db


def init_db(db_path: str | Path = "") -> Database:
    """Explicitly initialise the database (call at app startup)."""
    global _db
    _db = Database(db_path or "data/agentic_rag.db")
    _db.init_schema()
    return _db
