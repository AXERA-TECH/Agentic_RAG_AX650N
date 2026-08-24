"""WebSocket handler for bidirectional streaming communication.

Supports:
- Real-time chat with streaming responses
- Voice input streaming (audio chunks → STT → agent → TTS → audio chunks)
- Progress events during agent execution
"""

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self._connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        self._connections.pop(session_id, None)

    def get(self, session_id: str) -> Optional[WebSocket]:
        return self._connections.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for real-time agent interaction.

    Message format (client → server):
    {
        "type": "chat" | "voice" | "control",
        "payload": {
            "message": "...",       // For chat type
            "audio": "<base64>",   // For voice type
            "action": "cancel"     // For control type
        }
    }

    Message format (server → client):
    {
        "type": "text_delta" | "thought" | "tool_call" | "audio" | "error" | "done",
        "data": {...}
    }
    """
    await manager.connect(websocket, session_id)

    try:
        from agentic_rag.runtime.unified_context import UnifiedContext
        from agentic_rag.orchestration.l2_capabilities.chat import ChatCapability

        context = UnifiedContext.create(session_id=session_id)
        chat = ChatCapability()

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "chat")
            payload = msg.get("payload", {})

            if msg_type == "control":
                action = payload.get("action", "")
                if action == "cancel":
                    await websocket.send_json({"type": "done", "data": {"cancelled": True}})
                continue

            elif msg_type == "voice":
                # Voice input: transcribe → agent → TTS → audio
                audio_b64 = payload.get("audio", "")
                text = await _process_voice_input(audio_b64)
                if not text.strip():
                    await websocket.send_json({"type": "error", "data": {"error": "No speech detected"}})
                    continue

                # Send transcribed text back so client can display it
                await websocket.send_json({"type": "transcript", "data": {"text": text}})

                full_answer = ""
                async for event in chat.chat_stream(
                    message=text,
                    session_id=session_id,
                    context=context,
                ):
                    await _send_event(websocket, event)
                    if event.event_type.value == "text_delta":
                        full_answer += event.data.get("content", "") or event.data.get("delta", "")
                    elif event.event_type.value == "done":
                        ans = event.data.get("final_answer", "")
                        if ans:
                            full_answer = ans

                # Synthesize TTS audio from the final answer
                if full_answer.strip():
                    tts = _create_tts()
                    audio_bytes = await tts.synthesize(full_answer)
                    if audio_bytes:
                        import base64
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        await websocket.send_json({
                            "type": "audio",
                            "data": {"audio": audio_b64, "format": tts.response_format},
                        })

                await websocket.send_json({"type": "done", "data": {}})

            else:
                # Text chat
                message = payload.get("message", "")

                async for event in chat.chat_stream(
                    message=message,
                    session_id=session_id,
                    context=context,
                ):
                    await _send_event(websocket, event)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)


async def _send_event(websocket: WebSocket, event) -> None:
    """Send an AgentEvent to the WebSocket client."""
    try:
        await websocket.send_json({
            "type": event.event_type.value,
            "data": event.model_dump_json() if hasattr(event, 'model_dump_json') else event.data,
        })
    except Exception:
        pass


def _create_stt():
    """Create STT service from settings."""
    from agentic_rag.config.settings import get_settings
    from agentic_rag.core.voice.stt import STTService
    s = get_settings().voice
    return STTService(
        provider=s.stt_provider,
        model=s.stt_model,
        api_base=s.stt_api_base,
        api_key=s.stt_api_key,
        language=s.stt_language,
        sample_rate=s.sample_rate,
    )


def _create_tts():
    """Create TTS service from settings."""
    from agentic_rag.config.settings import get_settings
    from agentic_rag.core.voice.tts import TTSService
    s = get_settings().voice
    return TTSService(
        provider=s.tts_provider,
        model=s.tts_model,
        api_base=s.tts_api_base,
        api_key=s.tts_api_key,
        task_type=s.tts_task_type,
        instructions=s.tts_instructions,
        language=s.tts_language,
        speaker=s.tts_speaker,
        voice=s.tts_voice,
        speed=s.tts_speed,
        response_format=s.tts_response_format,
    )


async def _process_voice_input(audio_b64: str) -> str:
    """Process voice input: base64 → STT → text."""
    import base64
    try:
        audio_bytes = base64.b64decode(audio_b64)
        stt = _create_stt()
        return await stt.transcribe_bytes(audio_bytes)
    except Exception:
        return audio_b64  # Fallback: treat as text
