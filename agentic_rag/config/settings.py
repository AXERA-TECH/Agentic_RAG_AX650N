"""Configuration management with Pydantic Settings."""

import os
import re
from pathlib import Path

# Mute gRPC "too_many_pings" noise from Milvus Lite.
# Must be set BEFORE any pymilvus import — settings.py is loaded first.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "none")
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "60000")
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_PREFIX = ""


class LLMProviderConfig(BaseSettings):
    """Configuration for a single LLM provider."""
    api_key: str = ""
    api_base: str = ""
    model: str = ""
    max_tokens: int = 4096  # output limit per LLM call
    # Total context window (prompt + output). Set this to the server's
    # max_model_len; it drives the history budget in ContextBuilder.
    context_window: int = 16384
    temperature: float = 0.7
    frequency_penalty: float = 0.3   # discourage token repetition (0-2)
    presence_penalty: float = 0.3    # discourage topic looping (0-2)
    vision_model: str = ""  # Model for vision tasks
    enable_native_tool_calls: bool = True  # False → pure ReAct text mode


class EmbeddingConfig(BaseModel):
    """Dedicated embedding model configuration.

    Separated from LLM because:
    - Embedding may use a different provider (e.g., Claude for LLM + OpenAI for embeddings)
    - Self-hosted embedding services have their own API endpoint
    - The dimension must match the vector store schema
    """
    provider: str = "openai"                # Which LLM provider config to reuse, or "custom"
    model: str = "text-embedding-3-small"   # Embedding model name
    model_type: str = "text"                # "text" | "clip" | "multimodal"
    dim: int = 1536                          # Vector dimension (1536 for text-embedding-3-small)
    api_key: str = ""                        # Override API key (uses LLM provider's key if empty)
    api_base: str = ""                       # Override API base (uses LLM provider's base if empty)
    batch_size: int = 100                    # Max texts per embedding request


class MilvusConfig(BaseSettings):
    """Milvus vector database configuration."""
    host: str = "localhost"
    port: int = 19530
    collection_prefix: str = "agentic_rag"
    dim: int = 1536
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"


class MemoryConfig(BaseSettings):
    """Context-assembly configuration.

    Persisted session history is assembled by
    ``services.memory.context_builder.ContextBuilder`` using the block
    eviction / rolling summary settings below.
    """
    # ── Context assembly ───────────────────────────────────────────
    # history_budget = context_window - output_reserve - system_reserve
    #                  - evidence_reserve - safety_margin
    output_reserve: int = 2048       # tokens kept for the answer
    system_reserve: int = 1600       # system prompt + tool descriptions
    evidence_reserve: int = 3072     # tool observations during the ReAct loop
    safety_margin: int = 512
    block_turns: int = 4             # turns per eviction block
    min_recent_turns: int = 2        # never summarize the last N turns
    high_watermark: float = 0.90     # rebuild when usage exceeds this fraction
    low_watermark: float = 0.55      # evict until usage falls below this fraction
    summary_max_tokens: int = 600
    summary_timeout_seconds: float = 20.0


class MediaConfig(BaseSettings):
    """Attachment handling for chat (not knowledge-base ingestion).

    Attachments are uploaded once, stored under ``workspace/attachments`` and
    referenced by opaque IDs.  The limits below are deliberately enforced at
    the API boundary instead of trusting client-provided paths or MIME types.
    """
    max_attachment_bytes: int = 20 * 1024 * 1024
    max_digest_chars: int = 4000
    max_video_frames: int = 4
    # Optional dedicated vision provider/model used for captions, OCR and
    # media-tool observations.  Empty values fall back to the active LLM.
    vision_provider: str = ""
    vision_model: str = ""


class AgentConfig(BaseSettings):
    """Routing and iterative-retrieval limits.

    The deterministic router is always the default.  The planner is only
    invoked for explicitly complex/multi-hop requests, and every follow-up
    retrieval is capped so a weak model cannot loop on the knowledge base.
    """
    planner_enabled: bool = True
    planner_timeout_seconds: float = 12.0
    # Routing profile for requests without a strong evidence signal.
    # strict_kb: always search the KB and fail closed when it is empty;
    # balanced: search the KB first, allow a model answer as a last resort;
    # general: answer from the model without retrieval.
    routing_profile: str = "balanced"
    # Simple knowledge/live requests use the deterministic one-tool path.
    # Complex, multi-hop, and media turns still use ReAct.
    simple_kb_single_pass: bool = True
    # Total rag_search rounds per turn (the deterministic seed counts as 1).
    max_retrieval_rounds: int = 2
    sufficiency_check_enabled: bool = True
    # MCP tools are filtered per request only when the server exposes more
    # tools than this.  Small tool sets are passed through unchanged.
    tool_selection_enabled: bool = True
    max_mcp_tools: int = 5


class APIConfig(BaseSettings):
    """API server configuration."""
    # Bind loopback by default; containers/edge deployments opt in explicitly.
    host: str = "127.0.0.1"
    port: int = 8000
    # Kept for browser clients served from another origin.  Credentials are
    # disabled automatically when the wildcard is used (see ``create_app``).
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 60
    # When set, every /api/* request must send ``X-API-Key`` (or a bearer
    # token).  Empty keeps the local/embedded default (no auth).
    api_key: str = ""
    max_upload_bytes: int = 50 * 1024 * 1024


class OCRConfig(BaseSettings):
    """Optional OCR configuration; empty ``mode`` means Docling-first parsing."""
    enabled: bool = True
    mode: str = ""
    api_base: str = "http://localhost:8000/v1"
    model: str = "PaddlePaddle/PaddleOCR-VL"
    api_key: str = "not-needed"
    max_pages: int = 50  # max pages to OCR per document


class WeChatWorkGatewayConfig(BaseSettings):
    """企业微信自建应用 gateway configuration.

    Reference: https://developer.work.weixin.qq.com/document/path/90238
    """
    enabled: bool = False
    corp_id: str = ""           # 企业ID (myCorpId)
    token: str = ""             # 回调 Token
    encoding_aes_key: str = ""  # 回调 EncodingAESKey (43 chars)
    agent_id: str = ""          # 应用 AgentId
    secret: str = ""            # 应用 Secret (用于获取 access_token 推送消息)
    webhook_path: str = "/gateway/wechat_work"


class QQBotGatewayConfig(BaseSettings):
    """QQ Bot (官方) gateway configuration.

    沙箱模式: wss://sandbox.api.sgroup.qq.com/websocket
    正式环境: wss://api.sgroup.qq.com/websocket
    """
    enabled: bool = False
    app_id: str = ""            # BotAppID
    app_secret: str = ""        # BotSecret (用于获取 access_token)
    sandbox: bool = True        # True=沙箱环境, False=正式环境
    webhook_path: str = "/gateway/qqbot"  # 用于查看 Bot 状态


class DingTalkGatewayConfig(BaseSettings):
    """钉钉 bot gateway configuration."""
    enabled: bool = False
    app_key: str = ""
    app_secret: str = ""
    webhook_path: str = "/gateway/dingtalk"


class GatewayConfig(BaseSettings):
    """Messaging platform gateway configuration."""
    enabled: bool = False
    response_mode: str = "sync"    # "sync" = reply in webhook response; "async" = push via API
    max_reply_length: int = 2000
    wechat_work: WeChatWorkGatewayConfig = Field(default_factory=WeChatWorkGatewayConfig)
    dingtalk: DingTalkGatewayConfig = Field(default_factory=DingTalkGatewayConfig)
    qqbot: QQBotGatewayConfig = Field(default_factory=QQBotGatewayConfig)


class VoiceConfig(BaseSettings):
    """Voice/STT/TTS configuration."""
    # STT (Speech-to-Text)
    stt_provider: str = "sensevoice"   # "sensevoice" | "whisper" | "openai"
    stt_model: str = "sensevoice"      # sensevoice | base | small | whisper-1
    stt_api_base: str = "http://localhost:8000"  # ASR server URL (POST /asr)
    stt_api_key: str = ""
    stt_language: str = "auto"         # auto | zh | en | ja | ko | yue
    # TTS (Text-to-Speech)
    tts_provider: str = "qwen"         # "qwen" | "kokoro" | "edge" | "openai"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    tts_api_base: str = "http://localhost:8091"
    tts_api_key: str = "EMPTY"
    tts_task_type: str = "VoiceDesign"  # VoiceDesign | CustomVoice | Base (qwen)
    tts_instructions: str = "A clear, professional voice in Chinese"  # qwen
    tts_language: str = "Chinese"       # qwen language / kokoro lang code (zh/en/ja)
    tts_speaker: str = ""               # qwen CustomVoice speaker
    tts_voice: str = "zh-CN-XiaoxiaoNeural"  # Edge-TTS fallback / kokoro voice (zf_xiaoyi)
    tts_speed: float = 1.0             # kokoro playback speed
    tts_response_format: str = "wav"    # wav | mp3 | flac | pcm
    sample_rate: int = 16000


def _parse_llm_providers_from_env() -> dict[str, LLMProviderConfig]:
    """Manually extract LLM provider configs from environment + .env file.

    pydantic-settings cannot auto-populate ``dict[str, Model]`` from env vars
    because the dict keys are dynamic.  We scan for the pattern::

        RAG__LLM_PROVIDERS__<NAME>__<FIELD>=<value>

    in both ``os.environ`` AND the ``.env`` file (since pydantic-settings reads
    ``.env`` internally but does NOT export into ``os.environ``).
    """
    providers: dict[str, dict] = {}
    pattern = re.compile(rf"^{re.escape(_PREFIX)}_?LLM_PROVIDERS__([A-Z0-9]+)__([A-Z_]+)$")

    def _collect(source: dict[str, str]) -> None:
        for key, value in source.items():
            m = pattern.match(key)
            if not m:
                continue
            provider_name = m.group(1).lower()
            field_name = m.group(2).lower()
            providers.setdefault(provider_name, {})[field_name] = value

    # 1. os.environ (exported vars + python-dotenv if loaded externally)
    _collect(dict(os.environ))

    # 2. .env file (pydantic-settings reads it internally; we must too)
    env_path = Path(".env")
    if env_path.exists():
        env_vars = _parse_dotenv(env_path)
        _collect(env_vars)

    return {name: LLMProviderConfig(**fields) for name, fields in providers.items()}


def _parse_mcp_servers_from_env() -> dict[str, dict]:
    """Extract MCP server configs.

    Priority:
    1. ``mcp_servers.json`` — standard MCP config (like Claude Code)
    2. ``mcp_servers.yaml`` — YAML alternative
    3. Environment variables: ``RAG__MCP_SERVERS__<NAME>__<FIELD>=<value>``
    """
    # 1. Try JSON config (standard MCP format)
    json_path = Path("mcp_servers.json")
    if json_path.exists():
        try:
            import json as _json
            with open(json_path) as f:
                data = _json.load(f)
            servers_raw = data.get("mcpServers", {})
            result: dict[str, dict] = {}
            for name, cfg in servers_raw.items():
                if cfg.get("disabled", False):
                    continue
                args = cfg.get("args", [])
                # args can be a list or string
                if isinstance(args, list):
                    args = " ".join(args)
                result[name.lower()] = {
                    "command": cfg.get("command", ""),
                    "args": args,
                }
                # Preserve the standard MCP nested env mapping. The startup
                # code merges it with the process environment before spawning
                # the server; flattening these keys loses credentials such as
                # TAVILY_API_KEY because startup only reads config["env"].
                env = cfg.get("env", {})
                if isinstance(env, dict) and env:
                    result[name.lower()]["env"] = {
                        str(k): str(v) for k, v in env.items() if v
                    }
            if result:
                return result
        except Exception:
            pass

    # 2. Try YAML config
    yaml_path = Path("mcp_servers.yaml")
    if yaml_path.exists():
        try:
            import yaml as _yaml
            with open(yaml_path) as f:
                data = _yaml.safe_load(f) or {}
            if isinstance(data, dict) and "servers" in data:
                return {k.lower(): v for k, v in data["servers"].items()}
        except Exception:
            pass

    # 3. Fallback: env vars
    servers: dict[str, dict] = {}
    pattern = re.compile(rf"^{re.escape(_PREFIX)}_?MCP_SERVERS__([A-Z0-9]+)__([A-Z_]+)$")

    def _collect(source: dict[str, str]) -> None:
        for key, value in source.items():
            m = pattern.match(key)
            if not m:
                continue
            server_name = m.group(1).lower()
            field_name = m.group(2).lower()
            servers.setdefault(server_name, {})[field_name] = value

    _collect(dict(os.environ))
    env_path = Path(".env")
    if env_path.exists():
        _collect(_parse_dotenv(env_path))

    return servers


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict (without touching os.environ)."""
    result: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                value = v.strip()
                # Match dotenv behavior for quoted values such as
                # LANGFUSE_BASE_URL="https://cloud.langfuse.com".
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                result[k.strip()] = value
    return result


def configure_langfuse_environment() -> bool:
    """Expose Langfuse settings from ``.env`` to its SDK.

    Pydantic reads ``.env`` without exporting its values into ``os.environ``,
    while the Langfuse OpenAI integration reads its configuration from the
    process environment. Only Langfuse-specific values are exported here.

    Returns ``True`` when both API keys are configured, so callers can safely
    fall back to the native OpenAI client when tracing is not enabled.
    """
    values = dict(os.environ)
    env_path = Path(".env")
    if env_path.exists():
        for key, value in _parse_dotenv(env_path).items():
            if key.startswith("LANGFUSE_"):
                values.setdefault(key, value)

    for key, value in values.items():
        if key.startswith("LANGFUSE_") and value:
            os.environ.setdefault(key, value)

    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


class Settings(BaseSettings):
    """Root settings for Agentic RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="",
        extra="ignore",
    )

    # App
    app_name: str = "agentic_rag"
    debug: bool = False
    llm_debug: bool = False
    log_level: str = "INFO"

    # LLM
    default_provider: str = "openai"
    llm_providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)

    # Embedding (dedicated config — may differ from LLM provider)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Services
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    # Database
    db_path: str = "data/agentic_rag.db"

    # Gateway (messaging platforms)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)

    # MCP
    mcp_servers: dict[str, dict] = Field(default_factory=dict)

    # Workspace
    workspace_dir: str = "workspace"

    @model_validator(mode="after")
    def _inject_dict_fields(self):
        """Populate dict fields from env vars (pydantic-settings can't do dynamic keys)."""
        if not self.llm_providers:
            self.llm_providers = _parse_llm_providers_from_env()
        if not self.mcp_servers:
            self.mcp_servers = _parse_mcp_servers_from_env()
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Settings":
        """Load settings from a YAML file, then overlay env vars."""
        path = Path(yaml_path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        return cls(**data)


# Global settings instance (initialized at startup)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def init_settings(**kwargs) -> Settings:
    """Initialize settings (called at app startup)."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
