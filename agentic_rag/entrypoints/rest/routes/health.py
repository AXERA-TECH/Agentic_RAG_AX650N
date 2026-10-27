"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    from agentic_rag.agent.single_pass_engine import ENGINE_REVISION
    return {
        "status": "ok",
        "service": "agentic_rag",
        "engine_revision": ENGINE_REVISION,
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check — verifies LLM connectivity."""
    from agentic_rag.services.llm.factory import get_llm
    try:
        llm = get_llm()
        provider_info = {"provider": llm.provider_name, "model": llm.model_name}
        return {"status": "ready", "llm": provider_info}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}
