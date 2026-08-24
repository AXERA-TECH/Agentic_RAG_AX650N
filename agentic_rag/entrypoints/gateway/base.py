"""Base adaptor interface and shared models for messaging platform gateways."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import Request, Response

if TYPE_CHECKING:
    from agentic_rag.data.models import AgentOutput


# ═══════════════════════════════════════════════════════════════════
# Normalized Message / Response Models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PlatformMessage:
    """Normalized inbound message from any messaging platform."""

    platform: str                     # "wechat_work" | "dingtalk" | "feishu"
    sender_id: str                    # Platform-specific user ID
    sender_name: str = ""             # Display name (optional)
    chat_id: str = ""                 # Group chat ID or individual chat ID
    chat_type: str = "single"         # "single" | "group"
    text: str = ""                    # Extracted text content
    msg_type: str = "text"            # "text" | "image" | "voice" | "event"
    raw_payload: dict[str, Any] = field(default_factory=dict)  # Original platform payload
    reply_token: str = ""             # Token/URL needed to send a reply


@dataclass
class PlatformResponse:
    """Normalized outbound response to be sent back to a platform."""

    content: str                      # Formatted reply text
    msg_type: str = "text"            # "text" | "markdown" | "news" | "image"
    status_code: int = 200
    extra: dict[str, Any] = field(default_factory=dict)  # Platform-specific extras (at_list, buttons, etc.)


# ═══════════════════════════════════════════════════════════════════
# Base Adaptor (Template Method)
# ═══════════════════════════════════════════════════════════════════

class BasePlatformAdaptor(ABC):
    """Template-method adaptor for a messaging platform.

    Subclasses override the platform-specific steps; the pipeline
    (verify → parse → route → format) is shared.
    """

    platform_name: str = ""

    def __init__(self, config: Any) -> None:
        self.config = config

    # ── Template method ──────────────────────────────────────

    async def process(self, request: Request) -> Response:
        """Full pipeline: verify → parse → run agent → format → respond."""
        # 1. Verify
        if not await self.verify_request(request):
            return Response(status_code=403, content="Signature verification failed")

        # 2. Parse
        msg = await self.parse_message(request)

        # Skip non-text messages gracefully
        if msg.msg_type not in ("text",):
            return Response(status_code=200, content=self._empty_ack())

        if not msg.text.strip():
            return Response(status_code=200, content=self._empty_ack())

        # 3. Quick ACK if async mode
        try:
            from agentic_rag.config.settings import get_settings
            response_mode = get_settings().gateway.response_mode
        except Exception:
            response_mode = "sync"

        # 4. Run agent (may be long)
        if response_mode == "async":
            # Fire-and-forget: return 200 immediately, push result later
            import asyncio
            asyncio.create_task(self._process_async(msg))
            return Response(status_code=200, content=self._empty_ack())

        # 5. Sync: run agent inline and return result
        output = await self._run_agent(msg)
        presp = await self.format_response(output, msg)
        return await self._build_http_response(presp)

    # ── Steps subclasses must implement ──────────────────────

    @abstractmethod
    async def verify_request(self, request: Request) -> bool:
        """Verify the incoming webhook signature/token."""
        ...

    @abstractmethod
    async def parse_message(self, request: Request) -> PlatformMessage:
        """Parse the platform-specific payload into a PlatformMessage."""
        ...

    @abstractmethod
    async def format_response(
        self, output: "AgentOutput", msg: PlatformMessage
    ) -> PlatformResponse:
        """Convert AgentOutput to a platform-compatible response."""
        ...

    @abstractmethod
    async def _build_http_response(self, presp: PlatformResponse) -> Response:
        """Build the HTTP response object for this platform."""
        ...

    @abstractmethod
    async def push_message(self, msg: PlatformMessage, text: str) -> None:
        """Send a message to the platform's push API (used in async mode)."""
        ...

    # ── Shared agent invocation ──────────────────────────────

    async def _run_agent(self, msg: PlatformMessage) -> "AgentOutput":
        """Run the message through the shared chat orchestrator."""
        from agentic_rag.runtime.orchestrator import get_orchestrator
        from agentic_rag.runtime.unified_context import UnifiedContext

        # Resolve session
        sid = await self._resolve_session(msg)

        return await get_orchestrator().process(
            query=msg.text,
            session_id=sid,
            context=UnifiedContext.create(session_id=sid),
        )

    async def _resolve_session(self, msg: PlatformMessage) -> str:
        """Map (platform, sender_id, chat_id) → internal session_id."""
        from agentic_rag.entrypoints.gateway.session import get_platform_session_map
        session_map = get_platform_session_map()
        return session_map.get_or_create(
            platform=msg.platform,
            user_id=msg.sender_id,
            chat_id=msg.chat_id,
        )

    async def _process_async(self, msg: PlatformMessage) -> None:
        """Background: run agent and push result to platform."""
        try:
            output = await self._run_agent(msg)
            presp = await self.format_response(output, msg)
            await self.push_message(msg, presp.content)
        except Exception:
            import sys
            print(f"  [Gateway/{self.platform_name}] ⚠ Async process failed", flush=True)
            sys.stdout.flush()

    def _empty_ack(self) -> str:
        return ""

    # ── Content helpers ──────────────────────────────────────

    def _chunk_text(self, text: str, max_len: int | None = None) -> list[str]:
        """Split long text into platform-friendly chunks."""
        if max_len is None:
            try:
                from agentic_rag.config.settings import get_settings
                max_len = get_settings().gateway.max_reply_length
            except Exception:
                max_len = 2000
        chunks = []
        while len(text) > max_len:
            split_at = text.rfind("\n", 0, max_len)
            if split_at < max_len // 2:
                split_at = text.rfind("。", 0, max_len)
            if split_at < max_len // 2:
                split_at = text.rfind(". ", 0, max_len)
            if split_at < max_len // 2:
                split_at = max_len
            chunks.append(text[: split_at + 1])
            text = text[split_at + 1 :].lstrip()
        if text.strip():
            chunks.append(text)
        return chunks
