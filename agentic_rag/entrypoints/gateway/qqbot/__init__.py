"""QQ Bot (官方) gateway adaptor.

QQ Bot 使用 WebSocket 接收事件 + REST API 发送消息，不需要公网 webhook。

沙箱注册: https://q.qq.com
文档: https://bot.q.qq.com/wiki
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import httpx
import websockets
from fastapi import APIRouter

from agentic_rag.entrypoints.gateway.session import get_platform_session_map

router = APIRouter()

# QQ Bot 事件类型
EVENT_C2C_MESSAGE = "C2C_MESSAGE_CREATE"        # 单聊消息
EVENT_GROUP_AT = "GROUP_AT_MESSAGE_CREATE"       # 群聊 @消息
EVENT_READY = "READY"                             # 连接就绪
OPCODE_DISPATCH = 0                               # 服务端推送事件
OPCODE_HEARTBEAT = 1                              # 心跳
OPCODE_IDENTIFY = 2                               # 鉴权
OPCODE_RECONNECT = 7                              # 服务端要求重连
OPCODE_HELLO = 10                                 # 服务端下发心跳周期


# ═══════════════════════════════════════════════════════════════
# QQ Bot Client
# ═══════════════════════════════════════════════════════════════

class QQBotClient:
    """QQ Bot WebSocket 客户端。

    在后台运行，维护与 QQ 服务器的长连接，接收消息并调用 Agent。

    使用方式:
        client = QQBotClient(config)
        asyncio.create_task(client.run())
    """

    def __init__(self, config) -> None:
        self.config = config
        self._ws_url = (
            "wss://sandbox.api.sgroup.qq.com/websocket"
            if config.sandbox
            else "wss://api.sgroup.qq.com/websocket"
        )
        self._http_base = (
            "https://sandbox.api.sgroup.qq.com"
            if config.sandbox
            else "https://api.sgroup.qq.com"
        )
        self._app_id = config.app_id
        self._app_secret = config.app_secret
        self._access_token: str = ""
        self._session_id: str = ""
        self._seq: int = 0
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────

    async def run(self) -> None:
        """主循环: 连接 → 处理事件 → 断线重连"""
        self._running = True
        backoff = 1

        while self._running:
            try:
                await self._connect()
                backoff = 1  # 成功连接后重置退避
            except Exception as e:
                import sys
                print(f"  [QQBot] ⚠ Connection lost: {e}  (retry in {backoff}s)", flush=True)
                sys.stdout.flush()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def stop(self) -> None:
        """停止客户端"""
        self._running = False

    # ── Connection ─────────────────────────────────────────────

    async def _connect(self) -> None:
        """建立 WebSocket 连接并进入事件循环"""
        if not await self._get_token():
            return

        print(f"  [QQBot] Connecting to {self._ws_url}...", flush=True)

        async with websockets.connect(self._ws_url, ping_interval=None) as ws:
            # 等待服务端 Hello
            hello = json.loads(await ws.recv())
            if hello.get("op") != OPCODE_HELLO:
                raise RuntimeError(f"Expected Hello, got op={hello.get('op')}")

            heartbeat_interval = hello["d"]["heartbeat_interval"]  # ms
            print(f"  [QQBot] Connected, heartbeat={heartbeat_interval}ms", flush=True)

            # 发送鉴权
            await self._identify(ws)

            # 等待 Ready
            await self._wait_ready(ws)

            # 启动心跳
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ws, heartbeat_interval)
            )

            try:
                # 事件循环
                await self._event_loop(ws)
            finally:
                heartbeat_task.cancel()

    async def _identify(self, ws) -> None:
        """发送鉴权帧"""
        payload = {
            "op": OPCODE_IDENTIFY,
            "d": {
                "token": f"QQBot {self._access_token}",
                "intents": (1 << 25) | (1 << 0),  # C2C | GUILDS (群@)
                "shard": [0, 1],
                "properties": {},
            },
        }
        await ws.send(json.dumps(payload))
        print("  [QQBot] Identify sent", flush=True)

    async def _wait_ready(self, ws) -> None:
        """等待 Ready 事件（含 session_id）"""
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("t") == EVENT_READY:
                self._session_id = msg["d"].get("session_id", "")
                user = msg["d"].get("user", {})
                print(
                    f"  [QQBot] READY — bot: {user.get('username', '?')}#{user.get('id', '?')}",
                    flush=True,
                )
                return

    # ── Event Loop ─────────────────────────────────────────────

    async def _event_loop(self, ws) -> None:
        """接收并处理事件"""
        async for raw in ws:
            msg = json.loads(raw)
            op = msg.get("op")
            self._seq = msg.get("s", self._seq)

            if op == OPCODE_RECONNECT:
                print("  [QQBot] Server requested reconnect", flush=True)
                return

            if op == OPCODE_DISPATCH:
                event_type = msg.get("t", "")
                event_data = msg.get("d", {})
                asyncio.create_task(self._handle_event(event_type, event_data))

    async def _heartbeat_loop(self, ws, interval_ms: int) -> None:
        """定时发送心跳"""
        interval_s = interval_ms / 1000.0 * 0.8  # 留点余量
        while True:
            await asyncio.sleep(interval_s)
            try:
                await ws.send(json.dumps({"op": OPCODE_HEARTBEAT, "d": self._seq}))
            except Exception:
                break

    # ── Token ──────────────────────────────────────────────────

    async def _get_token(self) -> bool:
        """获取 access_token"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://bots.qq.com/app/getAppAccessToken",
                    json={
                        "appId": self._app_id,
                        "clientSecret": self._app_secret,
                    },
                )
                data = resp.json()
                self._access_token = data.get("access_token", "")
                if self._access_token:
                    print(f"  [QQBot] Token OK ({len(self._access_token)} chars)", flush=True)
                    return True
                print(f"  [QQBot] Token failed: {data}", flush=True)
                return False
        except Exception as e:
            print(f"  [QQBot] Token error: {e}", flush=True)
            return False

    # ── Event Handling ─────────────────────────────────────────

    async def _handle_event(self, event_type: str, data: dict) -> None:
        """处理 QQ 事件"""
        if event_type == EVENT_C2C_MESSAGE:
            await self._on_private_message(data)
        elif event_type == EVENT_GROUP_AT:
            await self._on_group_at(data)

    async def _on_private_message(self, data: dict) -> None:
        """处理单聊消息"""
        author = data.get("author", {})
        user_id = author.get("id", "")
        content = data.get("content", "").strip()
        msg_id = data.get("id", "")

        if not content or not user_id:
            return

        # 去掉命令前缀 "/" 和 @ 的干扰
        content = content.lstrip("/").strip()

        print(f"  [QQBot] 📩 C2C: {user_id} → {content[:50]}", flush=True)
        await self._process_message(user_id, f"qq:{user_id}", content, msg_id)

    async def _on_group_at(self, data: dict) -> None:
        """处理群聊 @消息"""
        author = data.get("author", {})
        user_id = author.get("id", "")
        group_id = data.get("group_openid", data.get("group_id", ""))
        content = data.get("content", "").strip()
        msg_id = data.get("id", "")

        if not content or not user_id:
            return

        # 内容中可能包含 @机器人 的 mention，去掉
        # QQ 消息格式可能是: "<@!bot_id> 内容" 或直接是纯文本
        import re
        content = re.sub(r"<@!\d+>", "", content).strip()

        if not content:
            return

        print(f"  [QQBot] 📩 Group@: {user_id} in {group_id} → {content[:50]}", flush=True)
        await self._process_message(user_id, f"qq:group:{group_id}:{user_id}", content, msg_id)

    async def _process_message(
        self, user_id: str, session_key: str, text: str, msg_id: str = ""
    ) -> None:
        """调用 Agent Engine 并回复"""
        try:
            # 解析 session
            session_map = get_platform_session_map()
            sid = session_map.get_or_create(platform="qqbot", user_id=session_key)

            # 调用 Agent
            output = await self._run_agent(sid, text)

            # 回复
            answer = output.final_answer or "抱歉，我暂时无法回答。"
            # 限制长度
            if len(answer) > 2000:
                answer = answer[:1950] + "\n\n...（内容过长已截断）"

            await self._send_reply(user_id, answer, msg_id)
        except Exception as e:
            import sys
            print(f"  [QQBot] ⚠ Process error: {e}", flush=True)
            sys.stdout.flush()
            await self._send_reply(user_id, "处理消息时出错了，请稍后重试。", msg_id)

    async def _run_agent(self, session_id: str, query: str):
        """通过共享 ChatOrchestrator 调用 Agent。"""
        from agentic_rag.runtime.orchestrator import get_orchestrator
        from agentic_rag.runtime.unified_context import UnifiedContext

        return await get_orchestrator().process(
            query=query,
            session_id=session_id,
            context=UnifiedContext.create(session_id=session_id),
        )

    async def _send_reply(self, user_id: str, text: str, msg_id: str = "") -> None:
        """通过 REST API 发送回复"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self._http_base}/v2/users/{user_id}/messages",
                    headers={
                        "Authorization": f"QQBot {self._access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "content": text,
                        "msg_type": 0,  # 文本消息
                        "msg_id": msg_id,
                    },
                )
                if resp.status_code != 200:
                    print(f"  [QQBot] ⚠ Reply failed ({resp.status_code}): {resp.text[:100]}", flush=True)
        except Exception as e:
            print(f"  [QQBot] ⚠ Reply error: {e}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Global client instance
# ═══════════════════════════════════════════════════════════════

_client: Optional[QQBotClient] = None


async def start_qqbot() -> None:
    """在 app 启动时启动 QQ Bot 客户端"""
    from agentic_rag.config.settings import get_settings

    settings = get_settings()
    cfg = settings.gateway.qqbot
    if not cfg.enabled:
        return

    global _client
    if _client is not None:
        return

    _client = QQBotClient(cfg)
    asyncio.create_task(_client.run())
    print(f"  [QQBot] Started {'(sandbox)' if cfg.sandbox else '(production)'}", flush=True)


async def stop_qqbot() -> None:
    """在 app 关闭时停止 QQ Bot"""
    global _client
    if _client:
        await _client.stop()
        _client = None


# ═══════════════════════════════════════════════════════════════
# Status endpoint
# ═══════════════════════════════════════════════════════════════

@router.get("/gateway/qqbot/status")
async def qqbot_status():
    """查看 QQ Bot 连接状态"""
    global _client
    if _client is None:
        return {"status": "disabled"}
    return {
        "status": "running" if _client._running else "stopped",
        "session_id": _client._session_id,
        "sandbox": _client.config.sandbox,
    }
