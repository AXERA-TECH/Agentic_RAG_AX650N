"""Application composition root.

Entrypoints should not assemble the database, tool registry, MCP client,
knowledge pipeline, and orchestrator themselves.  They call the builders here
and receive one :class:`ApplicationServices` object; the legacy module-level
getters remain as compatibility shims for code that has not been migrated yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.data.db import Database, init_db
from agentic_rag.runtime.orchestrator import ChatOrchestrator, set_orchestrator
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ApplicationServices:
    """Objects owned by one application process."""

    settings: Settings
    db: Database | None
    tool_registry: object | None
    orchestrator: ChatOrchestrator | None
    knowledge_pipeline: object | None
    mcp_client: object | None


async def build_application_services(
    settings: Optional[Settings] = None,
) -> ApplicationServices:
    """Wire process-wide services in the order their constructors require.

    MCP servers must start before the knowledge pipeline because the latter
    may initialize Milvus/gRPC, and forking after gRPC starts is unsafe.
    """
    settings = settings or get_settings()

    db = None
    try:
        db = init_db(settings.db_path)
        logger.info("Database ready (%s)", settings.db_path)
    except Exception:
        logger.exception("Database unavailable")

    registry = None
    try:
        from agentic_rag.orchestration.l1_tools.registry import get_tool_registry
        registry = get_tool_registry()
    except Exception:
        logger.exception("Tool registry unavailable")

    mcp_client = await _connect_mcp_servers(settings, registry)

    knowledge_pipeline = None
    try:
        from agentic_rag.services.knowledge.pipeline import init_knowledge_pipeline
        knowledge_pipeline = init_knowledge_pipeline()
        logger.info(
            "Knowledge pipeline ready (embedding=%s, dim=%s)",
            settings.embedding.model, settings.embedding.dim,
        )
    except Exception:
        logger.exception("Knowledge pipeline unavailable")

    orchestrator = ChatOrchestrator(
        tool_registry=(lambda: registry) if registry is not None else None,
    )
    set_orchestrator(orchestrator)

    return ApplicationServices(
        settings=settings,
        db=db,
        tool_registry=registry,
        orchestrator=orchestrator,
        knowledge_pipeline=knowledge_pipeline,
        mcp_client=mcp_client,
    )


async def _connect_mcp_servers(settings: Settings, registry) -> object | None:
    """Connect every configured MCP server and register its tools."""
    mcp_servers = settings.mcp_servers
    if not mcp_servers or registry is None:
        return None

    import os

    from agentic_rag.integrations.mcp.client import MCPClient

    client = MCPClient()
    for name, config in mcp_servers.items():
        try:
            cmd = config.get("command", "")
            args_raw = config.get("args", "")
            if isinstance(args_raw, list):
                arg_list = args_raw
            else:
                arg_list = args_raw.split() if args_raw else []
            # Merge MCP env with the current process env (needs PATH etc.).
            env = dict(os.environ)
            nested_env = config.get("env", {})
            if isinstance(nested_env, dict):
                env.update(
                    {str(k).upper(): str(v) for k, v in nested_env.items() if v}
                )
            tools = await client.connect_stdio(
                server_name=name, command=cmd,
                args=arg_list, env=env if env else None,
            )
            for tool in tools:
                registry.register_mcp(tool, name)
            logger.info(
                "MCP/%s connected — %s tools (%s %s...)",
                name, len(tools), cmd, " ".join(arg_list[:2]),
            )
        except Exception:
            logger.exception("MCP/%s failed", name)
    return client


async def shutdown_application_services(services: ApplicationServices) -> None:
    """Release process-wide services in reverse dependency order."""
    if services.mcp_client is not None:
        try:
            await services.mcp_client.disconnect_all()
        except Exception:
            logger.exception("MCP shutdown failed")

    if services.db is not None:
        try:
            services.db.close()
        except Exception:
            logger.exception("Database shutdown failed")

    # Flush batched LLM observations before the process exits.  Langfuse is
    # optional, so shutdown must remain safe when it is absent or unconfigured.
    try:
        from agentic_rag.config.settings import configure_langfuse_environment
        if configure_langfuse_environment():
            from langfuse import get_client
            get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed")
