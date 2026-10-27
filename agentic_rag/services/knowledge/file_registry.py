"""Persistent source-file registry for the knowledge base.

Kept separate from ``KnowledgePipeline`` so index/ingestion code does not own
JSON persistence, and so the registry can be constructed with explicit paths
in tests or alternative deployments.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeFileRegistry:
    """Map document IDs to display metadata, persisted as JSON."""

    def __init__(
        self,
        path_provider: Callable[[], Path],
        graph_path_provider: Callable[[], Path] | None = None,
        vector_store=None,
    ) -> None:
        self._path_provider = path_provider
        self._graph_path_provider = graph_path_provider
        self._vector_store = vector_store
        self._entries: dict[str, dict] = {}
        self._lock = threading.Lock()

    @property
    def entries(self) -> dict[str, dict]:
        """Return the live mapping (callers must not mutate it directly)."""
        return self._entries

    def save(self) -> None:
        """Persist source metadata so the UI can restore it after a restart."""
        path = self._path_provider()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(self.entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except Exception:
            logger.exception("Failed to save file registry")

    def load(self) -> None:
        """Restore source metadata from disk when the server starts."""
        path = self._path_provider()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = {}
            if isinstance(data, dict):
                self._entries = {
                    str(doc_id): meta
                    for doc_id, meta in data.items()
                    if isinstance(meta, dict) and meta.get("name")
                }

            self._recover_from_graph()
            self._recover_from_vector_store()
        except Exception:
            logger.exception("Failed to load file registry")

    def _recover_from_graph(self) -> None:
        """Recover names from KG document entities written by older versions."""
        if self._graph_path_provider is None:
            return
        graph_path = self._graph_path_provider()
        if not graph_path.exists():
            return
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        changed = False
        for entity in graph_data.get("entities", {}).values():
            if entity.get("type") != "document":
                continue
            props = entity.get("properties", {})
            doc_id, name = props.get("doc_id"), entity.get("name")
            if doc_id and name and doc_id not in self._entries:
                self._entries[doc_id] = {
                    "name": name,
                    "size": 0,
                    "type": Path(name).suffix.lower().lstrip(".") or "text",
                    "time": "",
                }
                changed = True
        if changed:
            self.save()

    def _recover_from_vector_store(self) -> None:
        """Last-resort recovery for indexes created before this registry."""
        if self._entries or self._vector_store is None:
            return
        client = getattr(self._vector_store, "c", None)
        collection = getattr(self._vector_store, "col", "knowledge")
        if client is None:
            return
        try:
            client.load_collection(collection)
            rows = client.query(
                collection,
                filter="source != ''",
                output_fields=["source", "content_type"],
                limit=10000,
            )
            for row in rows or []:
                doc_id = row.get("source", "")
                if doc_id and doc_id not in self._entries:
                    self._entries[doc_id] = {
                        "name": doc_id,
                        "size": 0,
                        "type": row.get("content_type", "text"),
                        "time": "",
                    }
            if self._entries:
                self.save()
        except Exception:
            logger.exception("Failed to recover files from vector index")

    def register(
        self, doc_id: str, name: str, size: int, content_type: str,
    ) -> None:
        """Record file metadata for later enumeration."""
        with self._lock:
            self._entries[doc_id] = {
                "name": name,
                "size": size,
                "type": content_type,
                "time": datetime.now().isoformat(),
            }
            self.save()

    def list(self) -> list[dict]:
        """Return all registered files with their doc_ids."""
        with self._lock:
            snapshot = list(self._entries.items())
        return [{"doc_id": did, **meta} for did, meta in snapshot]

    def unregister(self, doc_id: str) -> None:
        """Remove a file from the registry."""
        with self._lock:
            if doc_id in self._entries:
                self._entries.pop(doc_id, None)
                self.save()

    def clear(self) -> None:
        """Remove all persisted source metadata."""
        with self._lock:
            self._entries.clear()
            try:
                self._path_provider().unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to clear file registry")
