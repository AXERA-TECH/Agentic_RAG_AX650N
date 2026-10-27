"""MCP server configuration management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agentic_rag.services.config.mcp_config import get_mcp_config_store

router = APIRouter()


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
    return {"servers": get_mcp_config_store().read_servers()}


@router.put("/mcp/servers")
async def update_mcp_servers(data: MCPServersUpdate):
    """Replace the entire MCP server configuration and write to disk."""
    servers = {}
    for name, entry in data.servers.items():
        servers[name] = {
            "command": entry.command,
            "args": (
                entry.args if isinstance(entry.args, list)
                else entry.args.split()
            ),
            "env": entry.env or {},
            "disabled": entry.disabled,
        }
    count = get_mcp_config_store().replace_servers(servers)
    return {"status": "saved", "servers": count}


@router.delete("/mcp/servers/{name}")
async def delete_mcp_server(name: str):
    """Delete a single MCP server from the config."""
    if not get_mcp_config_store().delete_server(name):
        raise HTTPException(
            status_code=404, detail=f"MCP server '{name}' not found",
        )
    return {"status": "deleted", "name": name}
