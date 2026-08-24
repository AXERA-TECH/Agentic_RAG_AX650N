"""FastAPI application — the main REST entry point."""

import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    settings = get_settings()
    print(f"[{settings.app_name}] Starting on {settings.api.host}:{settings.api.port}")

    # Initialize database
    try:
        from agentic_rag.data.db import init_db
        init_db(settings.db_path)
        print(f"[{settings.app_name}] Database ready ({settings.db_path})")
    except Exception as e:
        print(f"[{settings.app_name}] ⚠ Database unavailable: {e}")

    # Connect MCP servers FIRST (before gRPC/Milvus to avoid fork conflicts)
    mcp_servers = settings.mcp_servers
    if mcp_servers:
        from agentic_rag.core.mcp.client import MCPClient
        from agentic_rag.orchestration.l1_tools.registry import get_tool_registry
        mcp_client = MCPClient()
        registry = get_tool_registry()
        for name, config in mcp_servers.items():
            try:
                cmd = config.get("command", "")
                args_raw = config.get("args", "")
                # args can be a JSON array or space-separated string
                if isinstance(args_raw, list):
                    arg_list = args_raw
                else:
                    arg_list = args_raw.split() if args_raw else []
                # Merge MCP env with current process env (needs PATH etc.)
                import os as _os
                env = dict(_os.environ)
                nested_env = config.get("env", {})
                if isinstance(nested_env, dict):
                    env.update({str(k).upper(): str(v) for k, v in nested_env.items() if v})
                tools = await mcp_client.connect_stdio(
                    server_name=name, command=cmd,
                    args=arg_list, env=env if env else None,
                )
                for tool in tools:
                    registry.register_mcp(tool, name)
                print(f"[{settings.app_name}] MCP/{name} connected — {len(tools)} tools "
                      f"({cmd} {' '.join(arg_list[:2])}...)")
            except Exception as e:
                print(f"[{settings.app_name}] ⚠ MCP/{name} failed: {e}")

    # Initialize knowledge pipeline AFTER MCP (avoids gRPC fork conflicts)
    try:
        from agentic_rag.services.knowledge.pipeline import init_knowledge_pipeline
        init_knowledge_pipeline()
        print(f"[{settings.app_name}] Knowledge pipeline ready  "
              f"(embedding={settings.embedding.model}, dim={settings.embedding.dim})")
    except Exception as e:
        print(f"[{settings.app_name}] ⚠ Knowledge pipeline unavailable: {e}")

    # Start QQ Bot (WebSocket client — not a webhook)
    if settings.gateway.qqbot.enabled:
        from agentic_rag.entrypoints.gateway.qqbot import start_qqbot
        await start_qqbot()

    yield

    # Shutdown: stop QQ Bot
    if settings.gateway.qqbot.enabled:
        from agentic_rag.entrypoints.gateway.qqbot import stop_qqbot
        await stop_qqbot()

    try:
        from agentic_rag.data.db import _db
        if _db:
            _db.close()
    except Exception:
        pass

    # Flush batched LLM observations before the process exits. Langfuse is
    # optional, so shutdown must remain safe when it is absent or unconfigured.
    try:
        from agentic_rag.config.settings import configure_langfuse_environment
        if configure_langfuse_environment():
            from langfuse import get_client
            get_client().flush()
    except Exception as e:
        print(f"[{settings.app_name}] ⚠ Langfuse flush failed: {e}")
    print(f"[{settings.app_name}] Shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Agentic RAG — Multi-modal, ReAct-powered, MCP-enabled RAG System",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Register routes
    from agentic_rag.entrypoints.rest.routes import chat, health, mcp, rag, session, settings
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])
    app.include_router(session.router, prefix="/api/v1", tags=["Session"])
    app.include_router(mcp.router, prefix="/api/v1", tags=["MCP"])
    app.include_router(settings.router, prefix="/api/v1", tags=["Settings"])

    # Gateway routes (messaging platform webhooks)
    if get_settings().gateway.enabled:
        from agentic_rag.entrypoints.gateway.router import get_gateway_router
        gateway_router = get_gateway_router()
        if gateway_router.routes:
            app.include_router(gateway_router)

    # WebSocket routes
    try:
        from agentic_rag.entrypoints.websocket.handler import router as ws_router
        app.include_router(ws_router, tags=["WebSocket"])
    except ImportError:
        pass

    # Static files & SPA fallback
    static_dir = Path(__file__).resolve().parent.parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def spa_root():
            return FileResponse(str(static_dir / "index.html"))

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": type(exc).__name__,
                "request_id": getattr(request.state, "request_id", "unknown"),
                "timestamp": time.time(),
            },
        )

    return app


app = create_app()
