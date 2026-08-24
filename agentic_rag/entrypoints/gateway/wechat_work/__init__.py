"""企业微信 (WeChat Work) 自建应用 gateway adaptor.

Reference: https://developer.work.weixin.qq.com/document/path/90238
"""

from __future__ import annotations

import json
import time as time_module
import xml.etree.ElementTree as ET
from typing import Optional

import httpx
from fastapi import APIRouter, Query, Request, Response

from agentic_rag.entrypoints.gateway.base import (
    BasePlatformAdaptor,
    PlatformMessage,
    PlatformResponse,
)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
# Token Manager (for active push)
# ═══════════════════════════════════════════════════════════════

class WeChatTokenManager:
    """Cache and refresh WeChat Work access_token.

    Token expires in 7200s; we refresh at half-life.
    Ref: https://developer.work.weixin.qq.com/document/path/91039
    """

    def __init__(self, corp_id: str, secret: str) -> None:
        self._corp_id = corp_id
        self._secret = secret
        self._token: str = ""
        self._expires_at: float = 0.0

    async def get_token(self) -> str:
        """Get a valid access_token, refreshing if needed."""
        if self._token and time_module.time() < self._expires_at - 300:
            return self._token

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
                    params={
                        "corpid": self._corp_id,
                        "corpsecret": self._secret,
                    },
                )
                data = resp.json()
                if data.get("errcode") == 0:
                    self._token = data["access_token"]
                    self._expires_at = time_module.time() + data.get("expires_in", 7200)
                    return self._token
        except Exception:
            pass
        return self._token


# ═══════════════════════════════════════════════════════════════
# Adaptor
# ═══════════════════════════════════════════════════════════════

class WeChatWorkAdaptor(BasePlatformAdaptor):
    """企业微信自建应用 adaptor."""

    platform_name = "wechat_work"

    def __init__(self, config) -> None:
        super().__init__(config)
        from agentic_rag.entrypoints.gateway.wechat_work.crypto import WeChatCrypto
        self._crypto = WeChatCrypto(
            token=config.token,
            encoding_aes_key=config.encoding_aes_key,
            corp_id=config.corp_id,
        )
        self._token_mgr: Optional[WeChatTokenManager] = None
        if config.secret:
            self._token_mgr = WeChatTokenManager(config.corp_id, config.secret)

    # ── URL Verification (GET) ────────────────────────────────

    async def verify_url(
        self, msg_signature: str, timestamp: str, nonce: str, echostr: str
    ) -> str:
        """Handle URL verification challenge from WeChat Work.

        Returns the decrypted echostr string (plain text), or raises ValueError.
        """
        if not self._crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
            raise ValueError("Signature verification failed")
        return self._crypto.decrypt(echostr)

    # ── Verify (POST) ─────────────────────────────────────────

    async def verify_request(self, request: Request) -> bool:
        """Verify WeChat Work callback signature."""
        msg_signature = request.query_params.get("msg_signature", "")
        timestamp = request.query_params.get("timestamp", "")
        nonce = request.query_params.get("nonce", "")

        if not msg_signature or not timestamp or not nonce:
            return False

        # Read body and verify; body is encrypted XML
        try:
            body = (await request.body()).decode("utf-8")
        except Exception:
            return False

        # Extract <Encrypt> from XML for signature
        try:
            root = ET.fromstring(body)
            encrypt_text = root.findtext("Encrypt", "")
        except Exception:
            encrypt_text = body

        return self._crypto.verify_signature(
            msg_signature, timestamp, nonce, encrypt_text
        )

    # ── Parse ─────────────────────────────────────────────────

    async def parse_message(self, request: Request) -> PlatformMessage:
        """Decrypt and parse WeChat Work callback XML."""
        msg_signature = request.query_params.get("msg_signature", "")
        timestamp = request.query_params.get("timestamp", "")
        nonce = request.query_params.get("nonce", "")

        body = (await request.body()).decode("utf-8")

        # Extract and decrypt
        try:
            root = ET.fromstring(body)
            encrypt_text = root.findtext("Encrypt", "")
        except ET.ParseError:
            return PlatformMessage(platform="wechat_work", sender_id="", msg_type="unknown")

        if not encrypt_text:
            return PlatformMessage(platform="wechat_work", sender_id="", msg_type="unknown")

        plain_xml = self._crypto.decrypt(encrypt_text)

        # Parse decrypted XML
        try:
            msg_root = ET.fromstring(plain_xml)
        except ET.ParseError:
            return PlatformMessage(platform="wechat_work", sender_id="", msg_type="unknown")

        msg_type = msg_root.findtext("MsgType", "unknown")
        sender_id = msg_root.findtext("FromUserName", "")
        agent_id = msg_root.findtext("AgentID", "")
        create_time = msg_root.findtext("CreateTime", "")

        text = ""
        if msg_type == "text":
            text = msg_root.findtext("Content", "")
        elif msg_type == "image":
            text = "[图片消息]"
        elif msg_type == "voice":
            recognition = msg_root.findtext("Recognition", "")  # 语音识别结果
            text = recognition or "[语音消息]"
        elif msg_type == "event":
            event_type = msg_root.findtext("Event", "")
            if event_type == "click":
                event_key = msg_root.findtext("EventKey", "")
                text = event_key  # 菜单点击
            elif event_type == "subscribe":
                text = "hello"  # 关注事件 → 触发欢迎语

        return PlatformMessage(
            platform="wechat_work",
            sender_id=sender_id,
            sender_name="",
            chat_id=sender_id,  # WeChat Work 单聊; 群聊场景有 ChatId
            chat_type="single",
            text=text,
            msg_type=msg_type,
            raw_payload={
                "agent_id": agent_id,
                "create_time": create_time,
                "msg_type": msg_type,
                "plain_xml": plain_xml,
            },
            reply_token=json.dumps({
                "to_user": sender_id,
                "agent_id": agent_id,
            }),
        )

    # ── Format ────────────────────────────────────────────────

    async def format_response(
        self, output, msg: PlatformMessage
    ) -> PlatformResponse:
        """Convert agent output to WeChat Work text response."""
        from agentic_rag.config.settings import get_settings
        max_len = get_settings().gateway.max_reply_length

        answer = output.final_answer or "抱歉，我暂时无法回答这个问题。"

        # WeChat Work text limit is 2048 chars
        effective_max = min(max_len, 2048)
        if len(answer) > effective_max:
            chunks = self._chunk_text(answer, effective_max)
            answer = chunks[0]
            if len(chunks) > 1:
                answer += f"\n\n...（共 {len(chunks)} 段，第 1 段）"

        # Remove markdown that WeChat Work doesn't support
        answer = self._strip_markdown(answer)

        return PlatformResponse(
            content=answer,
            msg_type="text",
            extra=msg.raw_payload,
        )

    # ── Build HTTP Response (encrypted XML) ──────────────────

    async def _build_http_response(self, presp: PlatformResponse) -> Response:
        """Build encrypted XML response for WeChat Work."""
        to_user = presp.extra.get("to_user", "")
        agent_id = presp.extra.get("agent_id", "")
        agent_id_from_cfg = getattr(self.config, "agent_id", "")

        # Build plain text reply XML
        create_time = str(int(time_module.time()))
        reply_xml = (
            "<xml>"
            f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
            f"<FromUserName><![CDATA[{agent_id or agent_id_from_cfg}]]></FromUserName>"
            f"<CreateTime>{create_time}</CreateTime>"
            "<MsgType><![CDATA[text]]></MsgType>"
            f"<Content><![CDATA[{presp.content}]]></Content>"
            "</xml>"
        )

        # Encrypt
        encrypted = self._crypto.encrypt(reply_xml)
        sig, ts, nonce = self._crypto.build_response_signature(encrypted)

        # Wrap in encrypted XML envelope
        response_xml = (
            "<xml>"
            f"<Encrypt><![CDATA[{encrypted}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{sig}]]></MsgSignature>"
            f"<TimeStamp>{ts}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            "</xml>"
        )

        return Response(content=response_xml, media_type="application/xml")

    # ── Push (async mode via WeChat API) ──────────────────────

    async def push_message(self, msg: PlatformMessage, text: str) -> None:
        """Send message via WeChat Work API (POST /cgi-bin/message/send)."""
        if not self._token_mgr:
            return

        to_user = msg.sender_id
        agent_id = getattr(self.config, "agent_id", "")
        token = await self._token_mgr.get_token()
        if not token:
            return

        text = self._strip_markdown(text)

        body = {
            "touser": to_user,
            "msgtype": "text",
            "agentid": int(agent_id) if agent_id.isdigit() else agent_id,
            "text": {"content": text},
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"https://qyapi.weixin.qq.com/cgi-bin/message/send"
                    f"?access_token={token}",
                    json=body,
                )
        except Exception:
            import sys
            print(f"  [WeChatWork] ⚠ Push message failed", flush=True)
            sys.stdout.flush()

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove unsupported markdown for WeChat Work text messages."""
        import re
        # Bold → plain
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        # Code blocks → plain
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Headers → plain
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Horizontal rules → remove
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
        # Links → text only
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        return text


# ── Global adaptor instance ────────────────────────────────────

_adaptor: Optional[WeChatWorkAdaptor] = None


def _get_adaptor() -> WeChatWorkAdaptor:
    global _adaptor
    if _adaptor is None:
        from agentic_rag.config.settings import get_settings
        _adaptor = WeChatWorkAdaptor(get_settings().gateway.wechat_work)
    return _adaptor


# ═══════════════════════════════════════════════════════════════
# Webhook Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/gateway/wechat_work")
async def wechat_work_verify(
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...),
):
    """URL verification — WeChat Work server sends a GET with echostr challenge.

    Must return the decrypted echostr as plain text (not JSON, not XML).
    """
    adaptor = _get_adaptor()
    try:
        decrypted = await adaptor.verify_url(msg_signature, timestamp, nonce, echostr)
        return Response(content=decrypted, media_type="text/plain")
    except ValueError:
        return Response(status_code=403, content="Verification failed")


@router.post("/gateway/wechat_work")
async def wechat_work_callback(request: Request):
    """Receive WeChat Work callback messages (encrypted XML).

    Query params: msg_signature, timestamp, nonce
    Body: encrypted XML with <Encrypt>...</Encrypt>
    """
    adaptor = _get_adaptor()
    return await adaptor.process(request)
