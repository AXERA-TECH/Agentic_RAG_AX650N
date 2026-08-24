"""Aggregated FastAPI router that mounts all enabled platform webhook routers."""

from fastapi import APIRouter


def get_gateway_router() -> APIRouter:
    """Build a single APIRouter that mounts all enabled platform webhook routers."""
    from agentic_rag.config.settings import get_settings

    settings = get_settings()
    gw = settings.gateway

    if not gw.enabled:
        return APIRouter()

    router = APIRouter(prefix="", tags=["Gateway"])

    if gw.wechat_work.enabled:
        from agentic_rag.entrypoints.gateway.wechat_work import router as wx_router
        router.include_router(wx_router)

    if gw.dingtalk.enabled:
        from agentic_rag.entrypoints.gateway.dingtalk import router as dd_router
        router.include_router(dd_router)

    if gw.qqbot.enabled:
        from agentic_rag.entrypoints.gateway.qqbot import router as qq_router
        router.include_router(qq_router)

    return router
