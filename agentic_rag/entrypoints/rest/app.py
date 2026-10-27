"""FastAPI application — the main REST entry point."""

import hmac
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag.config.settings import get_settings
from agentic_rag.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    """Best-effort request identifier for error responses."""
    return getattr(request.state, "request_id", "") or "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    settings = get_settings()
    logger.info("[%s] Starting on %s:%s", settings.app_name, settings.api.host, settings.api.port)

    # Build process-wide services (DB, MCP, knowledge pipeline, orchestrator)
    # through the composition root instead of wiring them ad hoc here.
    from agentic_rag.runtime.composition import (
        build_application_services,
        shutdown_application_services,
    )
    services = await build_application_services(settings)
    app.state.services = services

    # Start QQ Bot (WebSocket client — not a webhook)
    if settings.gateway.qqbot.enabled:
        from agentic_rag.entrypoints.gateway.qqbot import start_qqbot
        await start_qqbot()

    yield

    # Shutdown: stop QQ Bot, then release services in reverse order.
    if settings.gateway.qqbot.enabled:
        from agentic_rag.entrypoints.gateway.qqbot import stop_qqbot
        await stop_qqbot()

    await shutdown_application_services(services)
    logger.info("[%s] Shutting down", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Agentic RAG — Multi-modal, ReAct-powered, MCP-enabled RAG System",
        lifespan=lifespan,
    )

    # Optional API-key gate for /api/*.  Empty ``api_key`` keeps the local
    # embedded deployment friction-free; set it when exposing the service.
    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        expected = settings.api.api_key
        if not expected or not request.url.path.startswith("/api/"):
            return await call_next(request)
        supplied = request.headers.get("X-API-Key", "")
        if not supplied:
            authorization = request.headers.get("Authorization", "")
            if authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()
        # Constant-time comparison avoids leaking the key through timing.
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "request_id": _request_id(request)},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # CORS.  Browsers reject credentialed wildcard responses, so credentials
    # are only enabled for an explicit origin allowlist.
    origins = settings.api.cors_origins or []
    wildcard = "*" in origins
    if wildcard:
        logger.warning(
            "CORS allows every origin; set API__CORS_ORIGINS to restrict it"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from agentic_rag.entrypoints.rest.routes import (
        attachments, chat, health, mcp, rag, session,
        settings as settings_routes,
    )
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(attachments.router, prefix="/api/v1", tags=["Attachments"])
    app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])
    app.include_router(session.router, prefix="/api/v1", tags=["Session"])
    app.include_router(mcp.router, prefix="/api/v1", tags=["MCP"])
    app.include_router(settings_routes.router, prefix="/api/v1", tags=["Settings"])

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

    # Global exception handler.  Log the full traceback server-side but never
    # return exception details or class names to the client.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = _request_id(request)
        logger.exception(
            "Unhandled error on %s %s (request_id=%s)",
            request.method, request.url.path, request_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

    return app


app = create_app()
