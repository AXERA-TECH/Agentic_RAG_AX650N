"""Persistent MCP server configuration file."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


class MCPConfigStore:
    """Read/write ``mcp_servers.json`` without leaking I/O details to routes."""

    def __init__(
        self,
        path_provider: Callable[[], Path] | None = None,
    ) -> None:
        self._path_provider = path_provider or (
            lambda: Path("mcp_servers.json")
        )
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path_provider()

    def read(self) -> dict:
        """Return the full config document, defaulting to an empty map."""
        path = self.path
        if not path.exists():
            return {"mcpServers": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read MCP config %s", path)
            return {"mcpServers": {}}
        if not isinstance(data, dict):
            return {"mcpServers": {}}
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            data["mcpServers"] = {}
        return data

    def read_servers(self) -> dict:
        return self.read().get("mcpServers", {})

    def write(self, data: dict) -> None:
        """Atomically replace the config file."""
        path = self.path
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)

    def replace_servers(self, servers: dict) -> int:
        self.write({"mcpServers": servers})
        return len(servers)

    def delete_server(self, name: str) -> bool:
        data = self.read()
        servers = data.get("mcpServers", {})
        if name not in servers:
            return False
        del servers[name]
        self.write(data)
        return True


_store: MCPConfigStore | None = None


def get_mcp_config_store() -> MCPConfigStore:
    global _store
    if _store is None:
        _store = MCPConfigStore()
    return _store
