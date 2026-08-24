"""钉钉 (DingTalk) bot gateway adaptor."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Response

from agentic_rag.entrypoints.gateway.base import (
    BasePlatformAdaptor,
    PlatformMessage,
    PlatformResponse,
)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
# DingTalk Adaptor
# ═══════════════════════════════════════════════════════════════

class DingTalkAdaptor(BasePlatformAdaptor):
    """钉钉 bot webhook adaptor.

    Reference: https://open.dingtalk.com/document/orgapp/receive-messages
    """

    platform_name = "dingtalk"

    # ── Verify ────────────────────────────────────────────────

    async def verify_request(self, request: Request) -> bool:
        """Verify DingTalk HMAC-SHA256 signature."""
        timestamp = request.headers.get("timestamp", "")
        sign = request.headers.get("sign", "")
        if not timestamp or not sign:
            return False

        secret = getattr(self.config, "app_secret", "")
        if not secret:
            return True  # No secret configured — skip verification

        expected = self._hmac_sign(timestamp, secret)
        return sign == expected

    @staticmethod
    def _hmac_sign(timestamp: str, secret: str) -> str:
        """Compute DingTalk HMAC-SHA256 signature."""
        raw = f"{timestamp}\n{secret}"
        mac = hmac.new(
            secret.encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode()

    # ── Parse ─────────────────────────────────────────────────

    async def parse_message(self, request: Request) -> PlatformMessage:
        """Parse DingTalk robot callback JSON body."""
        try:
            body = await request.json()
        except Exception:
            return PlatformMessage(platform="dingtalk", sender_id="", msg_type="unknown")

        msg_type = body.get("msgtype", "unknown")
        sender_id = body.get("senderStaffId", body.get("senderId", ""))
        sender_name = body.get("senderNick", "")
        conversation_id = body.get("conversationId", "")
        chat_type = body.get("conversationType", "1")  # 1=single, 2=group
        session_webhook = body.get("sessionWebhook", "")

        # Extract text
        text = ""
        if msg_type == "text":
            text = body.get("text", {}).get("content", "")
        elif msg_type == "image":
            text = "[图片消息]"

        return PlatformMessage(
            platform="dingtalk",
            sender_id=str(sender_id),
            sender_name=sender_name,
            chat_id=conversation_id,
            chat_type="group" if str(chat_type) == "2" else "single",
            text=text,
            msg_type=msg_type,
            raw_payload=body,
            reply_token=session_webhook,
        )

    # ── Format Response ───────────────────────────────────────

    async def format_response(
        self, output, msg: PlatformMessage
    ) -> PlatformResponse:
        """Convert agent output to DingTalk markdown."""
        from agentic_rag.config.settings import get_settings
        max_len = get_settings().gateway.max_reply_length

        answer = output.final_answer or "抱歉，我暂时无法回答这个问题。"

        # Truncate if too long
        if len(answer) > max_len:
            chunks = self._chunk_text(answer, max_len)
            # Return first chunk; rest sent via push if needed
            answer = chunks[0]
            if len(chunks) > 1:
                answer += f"\n\n...（共 {len(chunks)} 段，第 1 段）"

        return PlatformResponse(
            content=answer,
            msg_type="text",
        )

    # ── HTTP Response ─────────────────────────────────────────

    async def _build_http_response(self, presp: PlatformResponse) -> Response:
        """Build DingTalk-compatible JSON response."""
        return Response(
            content=json.dumps(
                {"msgtype": "text", "text": {"content": presp.content}},
                ensure_ascii=False,
            ),
            media_type="application/json",
            status_code=presp.status_code,
        )

    # ── Push (async mode) ─────────────────────────────────────

    async def push_message(self, msg: PlatformMessage, text: str) -> None:
        """Send message back via DingTalk sessionWebhook."""
        webhook_url = msg.reply_token
        if not webhook_url:
            return

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    webhook_url,
                    json={"msgtype": "text", "text": {"content": text}},
                )
        except Exception:
            import sys
            print(f"  [DingTalk] ⚠ Push to webhook failed", flush=True)
            sys.stdout.flush()


# ── Global adaptor instance (lazy) ──────────────────────────────

_adaptor: Optional[DingTalkAdaptor] = None


def _get_adaptor() -> DingTalkAdaptor:
    global _adaptor
    if _adaptor is None:
        from agentic_rag.config.settings import get_settings
        _adaptor = DingTalkAdaptor(get_settings().gateway.dingtalk)
    return _adaptor


# ═══════════════════════════════════════════════════════════════
# Webhook Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/gateway/dingtalk")
async def dingtalk_callback(request: Request):
    """Receive DingTalk robot callback messages.

    DingTalk POSTs JSON with headers:
    - timestamp: Unix timestamp string
    - sign: HMAC-SHA256(timestamp + "\\n" + app_secret)

    Body (text message):
    {
        "conversationId": "...",
        "senderId": "...",
        "senderNick": "张三",
        "msgtype": "text",
        "text": {"content": "hello"},
        "sessionWebhook": "https://oapi.dingtalk.com/robot/..."
    }
    """
    adaptor = _get_adaptor()
    return await adaptor.process(request)
