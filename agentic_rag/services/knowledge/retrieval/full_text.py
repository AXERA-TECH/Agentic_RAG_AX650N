"""Persistent SQLite FTS5 index for independent BM25 retrieval."""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Iterable


class FullTextIndex:
    """Small persistent full-text index backed by SQLite FTS5 trigram tokens."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                id UNINDEXED,
                text,
                source UNINDEXED,
                content_type UNINDEXED,
                image_path UNINDEXED,
                video_path UNINDEXED,
                audio_path UNINDEXED,
                entity_id UNINDEXED,
                tokenize='trigram'
            )
            """
        )
        self._db.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def upsert(self, rows: Iterable[dict]) -> None:
        columns = (
            "id", "text", "source", "content_type", "image_path",
            "video_path", "audio_path", "entity_id",
        )
        with self._lock, self._db:
            for row in rows:
                chunk_id = str(row.get("id", ""))
                if not chunk_id:
                    continue
                self._db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
                self._db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(str(row.get(column, "") or "") for column in columns),
                )

    def delete_ids(self, ids: Iterable[str]) -> None:
        values = [(str(chunk_id),) for chunk_id in ids if chunk_id]
        if not values:
            return
        with self._lock, self._db:
            self._db.executemany("DELETE FROM chunks WHERE id = ?", values)

    def delete_source(self, doc_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE source = ?", (doc_id,))

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks")

    def search(self, query: str, limit: int = 30,
               content_types: list[str] | None = None) -> list[dict]:
        match_query = self._match_query(query)
        if not match_query:
            return self._like_search(query, limit, content_types)

        where = ["chunks MATCH ?"]
        params: list[object] = [match_query]
        if content_types:
            placeholders = ",".join("?" for _ in content_types)
            where.append(f"content_type IN ({placeholders})")
            params.extend(content_types)
        params.append(max(1, int(limit)))
        sql = f"""
            SELECT id, text, source, content_type, image_path, video_path,
                   audio_path, entity_id, bm25(chunks) AS rank
            FROM chunks
            WHERE {' AND '.join(where)}
            ORDER BY rank
            LIMIT ?
        """
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _match_query(query: str) -> str:
        normalized = " ".join(query.lower().split())
        segments = re.findall(r"[\w\u3400-\u9fff]+", normalized, flags=re.UNICODE)
        terms: list[str] = []
        seen = set()
        for segment in segments:
            if len(segment) < 3:
                continue
            for index in range(len(segment) - 2):
                term = segment[index:index + 3]
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
                if len(terms) >= 64:
                    break
            if len(terms) >= 64:
                break
        return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

    def _like_search(self, query: str, limit: int,
                     content_types: list[str] | None) -> list[dict]:
        needle = "".join(query.split())
        if not needle:
            return []
        where = ["replace(text, ' ', '') LIKE ?"]
        params: list[object] = [f"%{needle}%"]
        if content_types:
            placeholders = ",".join("?" for _ in content_types)
            where.append(f"content_type IN ({placeholders})")
            params.extend(content_types)
        params.append(max(1, int(limit)))
        sql = f"""
            SELECT id, text, source, content_type, image_path, video_path,
                   audio_path, entity_id, 0.0 AS rank
            FROM chunks
            WHERE {' AND '.join(where)}
            LIMIT ?
        """
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        fields = (
            "id", "text", "source", "content_type", "image_path",
            "video_path", "audio_path", "entity_id", "rank",
        )
        return dict(zip(fields, row))
