"""MCP server configuration management endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

MCP_CONFIG_PATH = Path("mcp_servers.json")


def _read_config() -> dict:
    """Read current MCP config from JSON file."""
    if MCP_CONFIG_PATH.exists():
        try:
            return json.loads(MCP_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"mcpServers": {}}


def _write_config(data: dict) -> None:
    """Write MCP config to JSON file."""
    MCP_CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class MCPServerEntry(BaseModel):
    """A single MCP server config entry."""
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    disabled: bool = False


class MCPServersUpdate(BaseModel):
    """Full MCP servers config update."""
    servers: dict[str, MCPServerEntry]


@router.get("/mcp/servers")
async def get_mcp_servers():
    """Return the current MCP server configuration."""
    config = _read_config()
    return {"servers": config.get("mcpServers", {})}


@router.put("/mcp/servers")
async def update_mcp_servers(data: MCPServersUpdate):
    """Replace the entire MCP server configuration and write to disk."""
    config = {"mcpServers": {}}
    for name, entry in data.servers.items():
        config["mcpServers"][name] = {
            "command": entry.command,
            "args": entry.args if isinstance(entry.args, list) else entry.args.split(),
            "env": entry.env or {},
            "disabled": entry.disabled,
        }
    _write_config(config)
    return {"status": "saved", "servers": len(config["mcpServers"])}


@router.delete("/mcp/servers/{name}")
async def delete_mcp_server(name: str):
    """Delete a single MCP server from the config."""
    config = _read_config()
    servers = config.get("mcpServers", {})
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    del servers[name]
    _write_config(config)
    return {"status": "deleted", "name": name}
