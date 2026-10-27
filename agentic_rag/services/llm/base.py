"""Base LLM Provider abstract class."""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from agentic_rag.data.models import LLMChunk, LLMResponse, Message, ToolDefinition
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def estimate_tokens(text: str) -> int:
    """Conservative token estimate without a model-specific tokenizer.

    English averages ~4 characters per token, while CJK text is close to
    1 token per character (sometimes more).  The old ``len(text) // 4``
    estimate undercounted Chinese by roughly 4x, which made every context
    budget far too optimistic.  Treat every non-ASCII character as one token
    and ASCII as 4 chars/token; this slightly over-reserves, which is the
    safe direction for a context limit.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii)


def llm_debug_enabled() -> bool:
    """Return whether complete LLM responses should be printed to the backend."""
    if _env_flag("LLM_DEBUG") or _env_flag("DEBUG"):
        return True
    try:
        from agentic_rag.config.settings import get_settings
        settings = get_settings()
        return bool(settings.llm_debug or settings.debug)
    except Exception:
        return False


def log_llm_result(
    provider: str,
    model: str,
    content: str,
    tool_calls=None,
    *,
    streaming: bool = False,
    native_tools_enabled: bool = False,
) -> None:
    """Print one complete model result without exposing credentials or prompts."""
    if not llm_debug_enabled():
        return
    calls = []
    for call in tool_calls or []:
        if hasattr(call, "model_dump"):
            calls.append(call.model_dump())
        elif isinstance(call, dict):
            calls.append(call)
        else:
            calls.append(str(call))
    mode = "stream" if streaming else "generate"
    logger.debug(
        "[LLM] %s/%s (%s, native_tools=%s) content=%r tool_calls=%s",
        provider,
        model,
        mode,
        "on" if native_tools_enabled else "off",
        content or "",
        json.dumps(calls, ensure_ascii=False, default=str) if calls else "[]",
    )


def strip_reasoning(text: str) -> str:
    """Remove model reasoning from generated text.

    Handles both reasoning conventions seen in OpenAI-compatible servers:
    - Explicit ``<think>...</think>`` blocks (possibly multiple, possibly
      split anywhere in the text).
    - A lone ``</think>`` when the chat template pre-fills the opening tag
      (e.g. Qwen3 served via vLLM/SGLang): everything before the first
      ``</think>`` is reasoning and is dropped.
    """
    if not text:
        return text
    while True:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if cleaned == text:
            break
        text = cleaned
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1]
    # Strip leftover unclosed/partial tags (e.g. truncated streams)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    provider_name: str
    model_name: str
    supports_vision: bool = False

    def __init__(self, model_name: str, api_key: str, api_base: str,
                 max_tokens: int = 4096, temperature: float = 0.7,
                 vision_model: str = "", **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.vision_model = vision_model or model_name
        # Total prompt + output window for the *active* provider.  Context
        # budgeting must use this instance rather than settings.default_provider.
        self.context_window = int(kwargs.get("context_window", 0) or 16384)
        # Copied from the configured provider entry by LLMFactory.create().
        # It lives on the instance (not looked up via settings.default_provider)
        # so a per-request provider switch uses that provider's own setting.
        self.enable_native_tool_calls = True

    @abstractmethod
    async def agenerate(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate a response (non-streaming)."""
        ...

    @abstractmethod
    async def agenerate_stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming response."""
        ...

    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    def count_tokens(self, text: str) -> int:
        """Estimate token count. Override for provider-specific counting."""
        return estimate_tokens(text)

    def supports_tool_calling(self) -> bool:
        """Check if this provider supports native tool/function calling."""
        return True

    def supports_native_tools_on_wire(self) -> bool:
        """Return whether the ``tools`` parameter may be sent to the server.

        Some OpenAI-compatible local servers (e.g. Axera-hosted Qwen) cannot
        handle the ``tools`` parameter at all: they drop streamed tool-call
        deltas and, in non-streaming mode, leak chat-template tokens
        (``<|im_start|>user``) or serialize the call as a corrupted
        ``<tool_call>{...}</tool_call>`` text blob instead of populating
        ``message.tool_calls``.  For such servers we must not send ``tools``
        on the wire at all — the agent falls back to the text ReAct format,
        and its Action lines are resolved against the real tool list by
        ``parse_react_output`` (which tolerates corrupted MCP names like
        ``mcp_._tavily_search``).

        Ephemeral local/LAN inference servers — hosted on bare IPs — are the
        ones observed to mishandle native tool calls; hosted HTTPS APIs
        (OpenAI, DashScope, …) implement function calling correctly.

        The decision uses this instance's own configuration.  Reading
        ``settings.default_provider`` here was wrong for per-request provider
        switches, where the active LLM is not the default one.
        """
        if not getattr(self, "enable_native_tool_calls", True):
            return False
        api_base = (getattr(self, "api_base", "") or "").strip()
        if not api_base or not api_base.startswith("http://"):
            return True
        host = api_base.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        return any(c.isalpha() for c in host)  # hostname, not a bare IP

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal Message format to provider-specific format."""
        converted = []
        for msg in messages:
            entry: dict = {"role": msg.role.value}

            if isinstance(msg.content, str):
                entry["content"] = msg.content
            else:
                entry["content"] = msg.content  # multimodal content list

            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

            converted.append(entry)
        return converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert internal ToolDefinition to provider-specific format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]


class ReasoningStreamFilter:
    """Incrementally strip reasoning (``<think>`` blocks) from a chunk stream.

    Tags may be split across chunk boundaries, so a small carry-over buffer
    is kept to detect partial ``<think>`` / ``</think>`` tokens.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._in_think = False
        self._carry = ""  # pending text that may be part of a tag

    def feed(self, delta: str) -> str:
        """Consume a content delta, return the non-reasoning portion."""
        if not delta:
            return ""
        buf = self._carry + delta
        self._carry = ""
        out: list[str] = []
        i = 0
        while i < len(buf):
            if self._in_think:
                idx = buf.find(self._CLOSE, i)
                if idx >= 0:
                    self._in_think = False
                    i = idx + len(self._CLOSE)
                else:
                    # No close tag yet — check for a partial close tag at the end
                    for k in range(len(self._CLOSE) - 1, 0, -1):
                        if buf.endswith(self._CLOSE[:k]):
                            self._carry = buf[-k:]
                            buf = buf[:-k]
                            break
                    return "".join(out)
            else:
                idx = buf.find(self._OPEN, i)
                if idx >= 0:
                    out.append(buf[i:idx])
                    self._in_think = True
                    i = idx + len(self._OPEN)
                else:
                    # Check for a partial open tag at the end
                    carry_at = len(buf)
                    for k in range(len(self._OPEN) - 1, 0, -1):
                        if buf.endswith(self._OPEN[:k]):
                            carry_at = len(buf) - k
                            self._carry = buf[carry_at:]
                            break
                    out.append(buf[i:carry_at])
                    return "".join(out)
        return "".join(out)

    def flush(self) -> str:
        """Return any remaining buffered text (call at end of stream)."""
        remainder = "" if self._in_think else self._carry
        self._carry = ""
        return remainder
