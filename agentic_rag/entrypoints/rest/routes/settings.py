"""Full .env configuration management endpoints.

GET  /api/v1/settings       → all config sections (keys masked)
PUT  /api/v1/settings       → save one or more sections
POST /api/v1/settings/test  → test LLM provider connection
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from agentic_rag.config.settings import get_settings

router = APIRouter()

ENV_PATH = Path(".env")
MCP_JSON_PATH = Path("mcp_servers.json")

_KEY_NAMES = re.compile(r"^[A-Z0-9_]+$")

# ── helpers ──────────────────────────────────────────────────────


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••••••"
    return f"{value[:3]}••••{value[-4:]}"


def _bool_str(value: Any) -> str:
    return "true" if str(value).lower() in ("true", "1", "yes") else "false"


def _parse_env() -> dict[str, str]:
    """Parse .env into a flat key→value dict (comments stripped)."""
    result: dict[str, str] = {}
    if not ENV_PATH.exists():
        return result
    for line in ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            k, v = stripped.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _write_env(updates: dict[str, str | None]) -> None:
    """Apply key→value updates to .env, preserving untouched lines & comments.

    A value of ``None`` removes the key.
    """
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    pending = {**updates}
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in pending:
                seen.add(key)
                val = pending.pop(key)
                if val is not None:
                    output.append(f"{key}={val}")
                # else skip (delete key)
                continue
        output.append(line)

    # Append new keys that weren't found
    new_entries = [(k, v) for k, v in pending.items() if k not in seen and v is not None]
    if new_entries:
        if output and output[-1].strip():
            output.append("")
        output.append("# Added via Web Settings UI")
        for k, v in new_entries:
            output.append(f"{k}={v}")

    ENV_PATH.write_text("\n".join(output).rstrip() + "\n")


# ── request models ───────────────────────────────────────────────


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
            raise ValueError("Provider name may only contain lowercase letters and numbers")
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
    host: str = "0.0.0.0"
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


# ── GET /api/v1/settings ────────────────────────────────────────


@router.get("/settings")
async def get_settings_all(request: Request):  # noqa: ARG001
    """Return every .env-managed config section (keys masked)."""
    s = get_settings()
    env = _parse_env()

    # LLM providers
    providers = []
    for name, cfg in s.llm_providers.items():
        providers.append(
            {
                "name": name,
                "api_base": cfg.api_base,
                "model": cfg.model,
                "vision_model": cfg.vision_model,
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "has_api_key": bool(cfg.api_key),
                "api_key_preview": _mask(cfg.api_key),
            }
        )

    # Embedding
    emb = s.embedding
    embedding = {
        "provider": emb.provider,
        "model": emb.model,
        "dim": emb.dim,
        "api_base": emb.api_base,
        "batch_size": emb.batch_size,
        "has_api_key": bool(emb.api_key),
        "api_key_preview": _mask(emb.api_key),
    }

    # Milvus
    m = s.milvus
    milvus = {"host": m.host, "port": m.port, "dim": m.dim}

    # API
    a = s.api
    api = {"host": a.host, "port": a.port}

    # Voice
    v = s.voice
    voice = {
        "stt_model": v.stt_model,
        "stt_api_base": v.stt_api_base,
        "stt_language": v.stt_language,
        "tts_api_base": v.tts_api_base,
        "tts_language": v.tts_language,
        "tts_voice": v.tts_voice,
        "tts_speed": v.tts_speed,
    }

    # OCR
    o = s.ocr
    ocr = {"enabled": o.enabled, "mode": o.mode, "api_base": o.api_base, "model": o.model, "max_pages": o.max_pages}

    # Gateway
    gw = s.gateway
    ww = gw.wechat_work
    dt = gw.dingtalk
    qq = gw.qqbot
    gateway = {
        "enabled": gw.enabled,
        "response_mode": gw.response_mode,
        "max_reply_length": gw.max_reply_length,
        "wechat_work": {
            "enabled": ww.enabled,
            "corp_id": ww.corp_id,
            "has_token": bool(ww.token),
            "has_encoding_aes_key": bool(ww.encoding_aes_key),
            "agent_id": ww.agent_id,
            "has_secret": bool(ww.secret),
        },
        "dingtalk": {"enabled": dt.enabled, "app_key": dt.app_key, "has_app_secret": bool(dt.app_secret)},
        "qqbot": {"enabled": qq.enabled, "app_id": qq.app_id, "has_app_secret": bool(qq.app_secret), "sandbox": qq.sandbox},
    }

    # MCP via dedicated file
    mcp = {}
    if MCP_JSON_PATH.exists():
        try:
            mcp = json.loads(MCP_JSON_PATH.read_text()).get("mcpServers", {})
        except json.JSONDecodeError:
            pass

    return {
        "default_provider": s.default_provider,
        "llm_providers": providers,
        "embedding": embedding,
        "milvus": milvus,
        "api": api,
        "voice": voice,
        "ocr": ocr,
        "gateway": gateway,
        "mcp_servers": mcp,
    }


# ── PUT /api/v1/settings ────────────────────────────────────────


def _apply_llm_provider(provider: dict[str, Any]) -> dict[str, str]:
    name = str(provider["name"]).strip().lower()
    pfx = f"LLM_PROVIDERS__{name.upper()}__"
    env_updates = {"DEFAULT_PROVIDER": name}
    env_updates[f"{pfx}API_BASE"] = str(provider.get("api_base", "")).strip()
    env_updates[f"{pfx}MODEL"] = str(provider.get("model", "")).strip()
    env_updates[f"{pfx}VISION_MODEL"] = str(provider.get("vision_model", "")).strip()
    env_updates[f"{pfx}MAX_TOKENS"] = str(provider.get("max_tokens", 4096))
    env_updates[f"{pfx}TEMPERATURE"] = str(provider.get("temperature", "0.7"))
    if provider.get("api_key") is not None:
        api_key = str(provider["api_key"]).strip()
        if api_key:
            env_updates[f"{pfx}API_KEY"] = api_key

    # apply in-memory
    settings = get_settings()
    existing = settings.llm_providers.get(name)
    from agentic_rag.config.settings import LLMProviderConfig

    api_key = (
        provider.get("api_key") or (existing.api_key if existing else "")
        if provider.get("api_key") is not None
        else (existing.api_key if existing else "")
    )
    settings.default_provider = name
    settings.llm_providers[name] = LLMProviderConfig(
        api_key=str(api_key).strip() if api_key else "",
        api_base=str(provider.get("api_base", "")).strip(),
        model=str(provider.get("model", "")).strip(),
        vision_model=str(provider.get("vision_model", "")).strip(),
        max_tokens=int(provider.get("max_tokens", 4096)),
        temperature=float(provider.get("temperature", 0.7)),
        frequency_penalty=existing.frequency_penalty if existing else 0.3,
        presence_penalty=existing.presence_penalty if existing else 0.3,
    )
    from agentic_rag.services.llm.factory import LLMFactory

    LLMFactory.clear_cache()
    return env_updates


def _apply_embedding(data: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for field, env_key in [
        ("provider", "EMBEDDING__PROVIDER"),
        ("model", "EMBEDDING__MODEL"),
        ("dim", "EMBEDDING__DIM"),
        ("api_base", "EMBEDDING__API_BASE"),
        ("batch_size", "EMBEDDING__BATCH_SIZE"),
    ]:
        if field in data:
            updates[env_key] = str(data[field])
    if data.get("api_key") is not None and str(data["api_key"]).strip():
        updates["EMBEDDING__API_KEY"] = str(data["api_key"]).strip()

    # in-memory
    s = get_settings()
    if "provider" in data:
        s.embedding.provider = str(data["provider"])
    if "model" in data:
        s.embedding.model = str(data["model"])
    if "dim" in data:
        s.embedding.dim = int(data["dim"])
    if "api_base" in data:
        s.embedding.api_base = str(data["api_base"]).strip()
    if "batch_size" in data:
        s.embedding.batch_size = int(data["batch_size"])
    if data.get("api_key") is not None and str(data["api_key"]).strip():
        s.embedding.api_key = str(data["api_key"]).strip()
    from agentic_rag.services.llm.factory import LLMFactory

    LLMFactory.clear_cache()
    return updates


def _apply_milvus(data: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for field, env_key in [("host", "MILVUS__HOST"), ("port", "MILVUS__PORT"), ("dim", "MILVUS__DIM")]:
        if field in data:
            updates[env_key] = str(data[field])
    s = get_settings()
    if "host" in data:
        s.milvus.host = str(data["host"])
    if "port" in data:
        s.milvus.port = int(data["port"])
    if "dim" in data:
        s.milvus.dim = int(data["dim"])
    return updates


def _apply_api(data: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    if "host" in data:
        updates["API__HOST"] = str(data["host"])
        get_settings().api.host = str(data["host"])
    if "port" in data:
        updates["API__PORT"] = str(data["port"])
        get_settings().api.port = int(data["port"])
    return updates


def _apply_voice(data: dict[str, Any]) -> dict[str, str]:
    field_map = {
        "stt_provider": "VOICE__STT_PROVIDER",
        "stt_model": "VOICE__STT_MODEL",
        "stt_api_base": "VOICE__STT_API_BASE",
        "stt_language": "VOICE__STT_LANGUAGE",
        "tts_provider": "VOICE__TTS_PROVIDER",
        "tts_model": "VOICE__TTS_MODEL",
        "tts_api_base": "VOICE__TTS_API_BASE",
        "tts_language": "VOICE__TTS_LANGUAGE",
        "tts_voice": "VOICE__TTS_VOICE",
        "tts_speed": "VOICE__TTS_SPEED",
    }
    updates: dict[str, str] = {}
    for field, env_key in field_map.items():
        if field in data:
            updates[env_key] = str(data[field])
    v = get_settings().voice
    if "stt_provider" in data:
        v.stt_provider = str(data["stt_provider"])
    if "stt_model" in data:
        v.stt_model = str(data["stt_model"])
    if "stt_api_base" in data:
        v.stt_api_base = str(data["stt_api_base"])
    if "stt_language" in data:
        v.stt_language = str(data["stt_language"])
    if "tts_provider" in data:
        v.tts_provider = str(data["tts_provider"])
    if "tts_model" in data:
        v.tts_model = str(data["tts_model"])
    if "tts_api_base" in data:
        v.tts_api_base = str(data["tts_api_base"])
    if "tts_language" in data:
        v.tts_language = str(data["tts_language"])
    if "tts_voice" in data:
        v.tts_voice = str(data["tts_voice"])
    if "tts_speed" in data:
        v.tts_speed = float(data["tts_speed"])
    return updates


def _apply_ocr(data: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    if "enabled" in data:
        updates["OCR__ENABLED"] = _bool_str(data["enabled"])
        get_settings().ocr.enabled = bool(data["enabled"])
    if "mode" in data:
        updates["OCR__MODE"] = str(data["mode"])
        get_settings().ocr.mode = str(data["mode"])
    if "api_base" in data:
        updates["OCR__API_BASE"] = str(data["api_base"])
        get_settings().ocr.api_base = str(data["api_base"])
    if "model" in data:
        updates["OCR__MODEL"] = str(data["model"])
        get_settings().ocr.model = str(data["model"])
    if "max_pages" in data:
        updates["OCR__MAX_PAGES"] = str(data["max_pages"])
        get_settings().ocr.max_pages = int(data["max_pages"])
    return updates


def _apply_gateway(data: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    gw = get_settings().gateway
    if "enabled" in data:
        updates["GATEWAY__ENABLED"] = _bool_str(data["enabled"])
        gw.enabled = bool(data["enabled"])
    if "response_mode" in data:
        updates["GATEWAY__RESPONSE_MODE"] = str(data["response_mode"])
        gw.response_mode = str(data["response_mode"])
    if "max_reply_length" in data:
        updates["GATEWAY__MAX_REPLY_LENGTH"] = str(data["max_reply_length"])
        gw.max_reply_length = int(data["max_reply_length"])

    for platform, prefix, target in [
        ("wechat_work", "GATEWAY__WECHAT_WORK", gw.wechat_work),
        ("dingtalk", "GATEWAY__DINGTALK", gw.dingtalk),
        ("qqbot", "GATEWAY__QQBOT", gw.qqbot),
    ]:
        sub = data.get(platform, {})
        if not isinstance(sub, dict):
            continue
        field_to_env = {
            "enabled": f"{prefix}__ENABLED",
            "app_key": f"{prefix}__APP_KEY",
            "app_secret": f"{prefix}__APP_SECRET",
            "sandbox": f"{prefix}__SANDBOX",
            "app_id": f"{prefix}__APP_ID",
            "corp_id": f"{prefix}__CORP_ID",
            "token": f"{prefix}__TOKEN",
            "encoding_aes_key": f"{prefix}__ENCODING_AES_KEY",
            "agent_id": f"{prefix}__AGENT_ID",
            "secret": f"{prefix}__SECRET",
        }
        for field, env_key in field_to_env.items():
            if field in sub:
                val = sub[field]
                if isinstance(val, bool):
                    val = _bool_str(val)
                updates[env_key] = str(val)
                if hasattr(target, field):
                    setattr(target, field, bool(val) if isinstance(sub[field], bool) else str(val))
    return updates


_SECTION_HANDLERS = {
    "llm_provider": _apply_llm_provider,
    "embedding": _apply_embedding,
    "milvus": _apply_milvus,
    "api": _apply_api,
    "voice": _apply_voice,
    "ocr": _apply_ocr,
    "gateway": _apply_gateway,
}


@router.put("/settings")
async def update_settings(body: SettingsUpdateBody):
    """Persist one or more config sections to .env and runtime memory."""
    all_updates: dict[str, str] = {}
    applied: list[str] = []

    for section in body.sections:
        handler = _SECTION_HANDLERS.get(section.section)
        if handler is None:
            raise HTTPException(status_code=400, detail=f"Unknown section: {section.section}")
        if not isinstance(section.data, dict):
            raise HTTPException(status_code=400, detail=f"data must be a dict, got {type(section.data).__name__}")
        env_part = handler(section.data)
        all_updates.update(env_part)
        applied.append(section.section)

    if all_updates:
        _write_env(all_updates)

    return {"status": "saved", "sections": applied, "keys_written": len(all_updates)}


# ── POST /api/v1/settings/test ──────────────────────────────────


@router.post("/settings/test")
async def test_connection(body: TestPayload):
    """Test LLM provider connectivity by listing models."""
    if not body.api_base.strip():
        raise HTTPException(status_code=400, detail="API Base URL is required")

    api_base = body.api_base.strip()
    api_key = (body.api_key or "").strip()
    provider_type = (body.provider_type or "openai").strip().lower()

    try:
        if provider_type == "claude":
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key, base_url=api_base or None, timeout=20.0, max_retries=0)
            resp = await client.models.list(limit=1)
            detected = resp.data[0].id if resp.data else body.model
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=api_base, timeout=20.0, max_retries=0)
            resp = await client.models.list()
            detected = resp.data[0].id if resp.data else body.model
    except Exception as exc:
        msg = str(exc)
        # truncate overly long API error messages
        if len(msg) > 500:
            msg = msg[:497] + "..."
        raise HTTPException(status_code=400, detail=f"Connection failed: {msg}") from exc

    return {"status": "connected", "detected_model": detected}
