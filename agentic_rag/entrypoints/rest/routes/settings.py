"""Full .env configuration management endpoints.

GET  /api/v1/settings       → all config sections (keys masked)
PUT  /api/v1/settings       → save one or more sections
POST /api/v1/settings/test  → test LLM provider connection
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from agentic_rag.services.config.settings_service import (
    ConnectionFailed,
    SettingsService,
)

router = APIRouter()


class LLMProviderFields(BaseModel):
    name: str
    api_base: str = ""
    api_key: str | None = None  # None = keep existing
    model: str = ""
    vision_model: str = ""
    max_tokens: int = Field(default=4096, ge=1, le=1_000_000)
    temperature: float = Field(default=0.7, ge=0, le=2)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"^[a-z0-9]+$", v):
            raise ValueError(
                "Provider name may only contain lowercase letters and numbers"
            )
        return v


class EmbeddingFields(BaseModel):
    provider: str = ""
    model: str = ""
    dim: int = Field(default=1536, ge=1)
    api_base: str = ""
    api_key: str | None = None
    batch_size: int = Field(default=100, ge=1)


class MilvusFields(BaseModel):
    host: str = "localhost"
    port: int = 19530
    dim: int = 1536


class APIFields(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class VoiceFields(BaseModel):
    stt_provider: str = "sensevoice"
    stt_model: str = "sensevoice-v1"
    stt_api_base: str = ""
    stt_language: str = "auto"
    tts_provider: str = "qwen"
    tts_model: str = ""
    tts_api_base: str = ""
    tts_language: str = "Chinese"
    tts_voice: str = ""
    tts_speed: float = 1.0


class OCRFields(BaseModel):
    enabled: bool = True
    mode: str = ""
    api_base: str = ""
    model: str = ""
    max_pages: int = 50


class WeChatWorkFields(BaseModel):
    enabled: bool = False
    corp_id: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    agent_id: str = ""
    secret: str = ""


class DingTalkFields(BaseModel):
    enabled: bool = False
    app_key: str = ""
    app_secret: str = ""


class QQBotFields(BaseModel):
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    sandbox: bool = True


class GatewayFields(BaseModel):
    enabled: bool = False
    response_mode: str = "sync"
    max_reply_length: int = 2000
    wechat_work: WeChatWorkFields = Field(default_factory=WeChatWorkFields)
    dingtalk: DingTalkFields = Field(default_factory=DingTalkFields)
    qqbot: QQBotFields = Field(default_factory=QQBotFields)


class ConnectionFields(BaseModel):
    api_base: str = ""


class SectionPayload(BaseModel):
    """A partial update for one config section."""

    section: str  # "llm_provider" | "embedding" | "milvus" | "api" | "voice" | "ocr" | "gateway" | "connection"
    data: dict[str, Any]


class SettingsUpdateBody(BaseModel):
    sections: list[SectionPayload]

    @field_validator("sections")
    @classmethod
    def _non_empty(cls, v: list[SectionPayload]) -> list[SectionPayload]:
        if not v:
            raise ValueError("At least one section is required")
        return v


class TestPayload(BaseModel):
    api_base: str
    api_key: str | None = None
    model: str = ""
    provider_type: str = "openai"  # "openai" | "claude"


def _service() -> SettingsService:
    return SettingsService()


@router.get("/settings")
async def get_settings_all(request: Request):  # noqa: ARG001
    """Return every .env-managed config section (keys masked)."""
    return _service().snapshot()


@router.put("/settings")
async def update_settings(body: SettingsUpdateBody):
    """Persist one or more config sections to .env and runtime memory."""
    try:
        return _service().update(body.sections)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/settings/test")
async def test_connection(body: TestPayload):
    """Test LLM provider connectivity by listing models."""
    try:
        detected = await _service().test_connection(
            api_base=body.api_base,
            api_key=body.api_key or "",
            model=body.model,
            provider_type=body.provider_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConnectionFailed as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "connected", "detected_model": detected}
