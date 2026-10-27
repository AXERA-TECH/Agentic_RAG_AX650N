"""Settings service — .env persistence and runtime configuration updates.

The REST layer owns request/response models; this module owns file parsing,
section application, and provider connectivity checks.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from agentic_rag.config.settings import get_settings
from agentic_rag.services.config.mcp_config import get_mcp_config_store
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)

_MASK_THRESHOLD = 8


class ConnectionFailed(Exception):
    """Provider connectivity check failed."""


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= _MASK_THRESHOLD:
        return "••••••••"
    return f"{value[:3]}••••{value[-4:]}"


def _bool_str(value: Any) -> str:
    return "true" if str(value).lower() in ("true", "1", "yes") else "false"


class EnvFileStore:
    """Read/update ``.env`` while preserving comments and untouched lines."""

    def __init__(self, path_provider: Callable[[], Path] | None = None) -> None:
        self._path_provider = path_provider or (lambda: Path(".env"))
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path_provider()

    def read(self) -> dict[str, str]:
        """Parse .env into a flat key→value dict (comments stripped)."""
        path = self.path
        result: dict[str, str] = {}
        if not path.exists():
            return result
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.exception("Failed to read %s", path)
            return result
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                result[key.strip()] = value.strip()
        return result

    def write(self, updates: dict[str, str | None]) -> None:
        """Apply key→value updates; ``None`` removes a key."""
        path = self.path
        with self._lock:
            lines = (
                path.read_text(encoding="utf-8").splitlines()
                if path.exists() else []
            )
            pending = {**updates}
            seen: set[str] = set()
            output: list[str] = []

            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in pending:
                        seen.add(key)
                        value = pending.pop(key)
                        if value is not None:
                            output.append(f"{key}={value}")
                        continue
                output.append(line)

            new_entries = [
                (key, value)
                for key, value in pending.items()
                if key not in seen and value is not None
            ]
            if new_entries:
                if output and output[-1].strip():
                    output.append("")
                output.append("# Added via Web Settings UI")
                for key, value in new_entries:
                    output.append(f"{key}={value}")

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


class SettingsService:
    """Read and mutate runtime settings plus their .env representation."""

    def __init__(
        self,
        env_store: EnvFileStore | None = None,
        settings_provider: Callable[[], Any] | None = None,
        mcp_store=None,
    ) -> None:
        self._env = env_store or EnvFileStore()
        self._settings_provider = settings_provider or get_settings
        self._mcp = mcp_store or get_mcp_config_store()

    def _settings(self):
        return self._settings_provider()

    # ── GET ──────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return every .env-managed config section (secrets masked)."""
        settings = self._settings()
        providers = [
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
            for name, cfg in settings.llm_providers.items()
        ]

        embedding_cfg = settings.embedding
        embedding = {
            "provider": embedding_cfg.provider,
            "model": embedding_cfg.model,
            "dim": embedding_cfg.dim,
            "api_base": embedding_cfg.api_base,
            "batch_size": embedding_cfg.batch_size,
            "has_api_key": bool(embedding_cfg.api_key),
            "api_key_preview": _mask(embedding_cfg.api_key),
        }

        milvus_cfg = settings.milvus
        milvus = {
            "host": milvus_cfg.host,
            "port": milvus_cfg.port,
            "dim": milvus_cfg.dim,
        }
        api = {"host": settings.api.host, "port": settings.api.port}

        voice_cfg = settings.voice
        voice = {
            "stt_model": voice_cfg.stt_model,
            "stt_api_base": voice_cfg.stt_api_base,
            "stt_language": voice_cfg.stt_language,
            "tts_api_base": voice_cfg.tts_api_base,
            "tts_language": voice_cfg.tts_language,
            "tts_voice": voice_cfg.tts_voice,
            "tts_speed": voice_cfg.tts_speed,
        }

        ocr_cfg = settings.ocr
        ocr = {
            "enabled": ocr_cfg.enabled,
            "mode": ocr_cfg.mode,
            "api_base": ocr_cfg.api_base,
            "model": ocr_cfg.model,
            "max_pages": ocr_cfg.max_pages,
        }

        gateway_cfg = settings.gateway
        wechat = gateway_cfg.wechat_work
        dingtalk = gateway_cfg.dingtalk
        qqbot = gateway_cfg.qqbot
        gateway = {
            "enabled": gateway_cfg.enabled,
            "response_mode": gateway_cfg.response_mode,
            "max_reply_length": gateway_cfg.max_reply_length,
            "wechat_work": {
                "enabled": wechat.enabled,
                "corp_id": wechat.corp_id,
                "has_token": bool(wechat.token),
                "has_encoding_aes_key": bool(wechat.encoding_aes_key),
                "agent_id": wechat.agent_id,
                "has_secret": bool(wechat.secret),
            },
            "dingtalk": {
                "enabled": dingtalk.enabled,
                "app_key": dingtalk.app_key,
                "has_app_secret": bool(dingtalk.app_secret),
            },
            "qqbot": {
                "enabled": qqbot.enabled,
                "app_id": qqbot.app_id,
                "has_app_secret": bool(qqbot.app_secret),
                "sandbox": qqbot.sandbox,
            },
        }

        return {
            "default_provider": settings.default_provider,
            "llm_providers": providers,
            "embedding": embedding,
            "milvus": milvus,
            "api": api,
            "voice": voice,
            "ocr": ocr,
            "gateway": gateway,
            "mcp_servers": self._mcp.read_servers(),
        }

    # ── PUT ──────────────────────────────────────────────────

    def update(self, sections: list[dict]) -> dict:
        """Persist validated sections to .env and runtime memory."""
        all_updates: dict[str, str] = {}
        applied: list[str] = []
        for section in sections:
            name = section.section
            handler = self._SECTION_HANDLERS.get(name)
            if handler is None:
                raise ValueError(f"Unknown section: {name}")
            if not isinstance(section.data, dict):
                raise ValueError(
                    f"data must be a dict, got {type(section.data).__name__}"
                )
            all_updates.update(handler(self, section.data))
            applied.append(name)

        if all_updates:
            self._env.write(all_updates)
        return {
            "status": "saved",
            "sections": applied,
            "keys_written": len(all_updates),
        }

    def _apply_llm_provider(self, provider: dict[str, Any]) -> dict[str, str]:
        name = str(provider["name"]).strip().lower()
        prefix = f"LLM_PROVIDERS__{name.upper()}__"
        updates = {"DEFAULT_PROVIDER": name}
        updates[f"{prefix}API_BASE"] = str(provider.get("api_base", "")).strip()
        updates[f"{prefix}MODEL"] = str(provider.get("model", "")).strip()
        updates[f"{prefix}VISION_MODEL"] = str(
            provider.get("vision_model", "")
        ).strip()
        updates[f"{prefix}MAX_TOKENS"] = str(provider.get("max_tokens", 4096))
        updates[f"{prefix}TEMPERATURE"] = str(provider.get("temperature", "0.7"))
        if provider.get("api_key") is not None:
            api_key = str(provider["api_key"]).strip()
            if api_key:
                updates[f"{prefix}API_KEY"] = api_key

        settings = self._settings()
        existing = settings.llm_providers.get(name)
        from agentic_rag.config.settings import LLMProviderConfig

        if provider.get("api_key") is not None:
            api_key = provider.get("api_key") or (
                existing.api_key if existing else ""
            )
        else:
            api_key = existing.api_key if existing else ""
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
        return updates

    def _apply_embedding(self, data: dict[str, Any]) -> dict[str, str]:
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

        settings = self._settings()
        if "provider" in data:
            settings.embedding.provider = str(data["provider"])
        if "model" in data:
            settings.embedding.model = str(data["model"])
        if "dim" in data:
            settings.embedding.dim = int(data["dim"])
        if "api_base" in data:
            settings.embedding.api_base = str(data["api_base"]).strip()
        if "batch_size" in data:
            settings.embedding.batch_size = int(data["batch_size"])
        if data.get("api_key") is not None and str(data["api_key"]).strip():
            settings.embedding.api_key = str(data["api_key"]).strip()
        from agentic_rag.services.llm.factory import LLMFactory
        LLMFactory.clear_cache()
        return updates

    def _apply_milvus(self, data: dict[str, Any]) -> dict[str, str]:
        updates: dict[str, str] = {}
        for field, env_key in [
            ("host", "MILVUS__HOST"),
            ("port", "MILVUS__PORT"),
            ("dim", "MILVUS__DIM"),
        ]:
            if field in data:
                updates[env_key] = str(data[field])
        settings = self._settings()
        if "host" in data:
            settings.milvus.host = str(data["host"])
        if "port" in data:
            settings.milvus.port = int(data["port"])
        if "dim" in data:
            settings.milvus.dim = int(data["dim"])
        return updates

    def _apply_api(self, data: dict[str, Any]) -> dict[str, str]:
        updates: dict[str, str] = {}
        settings = self._settings()
        if "host" in data:
            updates["API__HOST"] = str(data["host"])
            settings.api.host = str(data["host"])
        if "port" in data:
            updates["API__PORT"] = str(data["port"])
            settings.api.port = int(data["port"])
        return updates

    def _apply_voice(self, data: dict[str, Any]) -> dict[str, str]:
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
        updates = {
            env_key: str(data[field])
            for field, env_key in field_map.items()
            if field in data
        }
        voice = self._settings().voice
        for field in field_map:
            if field in data:
                current = getattr(voice, field)
                value = data[field]
                setattr(
                    voice, field,
                    float(value) if isinstance(current, float) else type(current)(value),
                )
        return updates

    def _apply_ocr(self, data: dict[str, Any]) -> dict[str, str]:
        updates: dict[str, str] = {}
        ocr = self._settings().ocr
        if "enabled" in data:
            updates["OCR__ENABLED"] = _bool_str(data["enabled"])
            ocr.enabled = bool(data["enabled"])
        if "mode" in data:
            updates["OCR__MODE"] = str(data["mode"])
            ocr.mode = str(data["mode"])
        if "api_base" in data:
            updates["OCR__API_BASE"] = str(data["api_base"])
            ocr.api_base = str(data["api_base"])
        if "model" in data:
            updates["OCR__MODEL"] = str(data["model"])
            ocr.model = str(data["model"])
        if "max_pages" in data:
            updates["OCR__MAX_PAGES"] = str(data["max_pages"])
            ocr.max_pages = int(data["max_pages"])
        return updates

    def _apply_gateway(self, data: dict[str, Any]) -> dict[str, str]:
        updates: dict[str, str] = {}
        gateway = self._settings().gateway
        if "enabled" in data:
            updates["GATEWAY__ENABLED"] = _bool_str(data["enabled"])
            gateway.enabled = bool(data["enabled"])
        if "response_mode" in data:
            updates["GATEWAY__RESPONSE_MODE"] = str(data["response_mode"])
            gateway.response_mode = str(data["response_mode"])
        if "max_reply_length" in data:
            updates["GATEWAY__MAX_REPLY_LENGTH"] = str(data["max_reply_length"])
            gateway.max_reply_length = int(data["max_reply_length"])

        for platform, prefix, target in [
            ("wechat_work", "GATEWAY__WECHAT_WORK", gateway.wechat_work),
            ("dingtalk", "GATEWAY__DINGTALK", gateway.dingtalk),
            ("qqbot", "GATEWAY__QQBOT", gateway.qqbot),
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
                    value = sub[field]
                    if isinstance(value, bool):
                        value = _bool_str(value)
                    updates[env_key] = str(value)
                    if hasattr(target, field):
                        setattr(
                            target, field,
                            bool(sub[field])
                            if isinstance(sub[field], bool)
                            else str(value),
                        )
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

    # ── POST /settings/test ──────────────────────────────────

    async def test_connection(
        self,
        api_base: str,
        api_key: str = "",
        model: str = "",
        provider_type: str = "openai",
    ) -> str:
        """Return the first model reported by a provider, or raise."""
        api_base = api_base.strip()
        if not api_base:
            raise ValueError("API Base URL is required")
        api_key = (api_key or "").strip()
        provider_type = (provider_type or "openai").strip().lower()
        try:
            if provider_type == "claude":
                import anthropic

                client = anthropic.AsyncAnthropic(
                    api_key=api_key, base_url=api_base or None,
                    timeout=20.0, max_retries=0,
                )
                response = await client.models.list(limit=1)
                return response.data[0].id if response.data else model

            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=api_key or "not-needed", base_url=api_base,
                timeout=20.0, max_retries=0,
            )
            response = await client.models.list()
            return response.data[0].id if response.data else model
        except Exception as exc:
            message = str(exc)
            if len(message) > 500:
                message = message[:497] + "..."
            raise ConnectionFailed(f"Connection failed: {message}") from exc
