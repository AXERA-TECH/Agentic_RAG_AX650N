"""Chat endpoints — the primary agent interaction interface."""

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = None

from agentic_rag.services.llm.factory import get_llm
from agentic_rag.runtime.session_input import build_chat_input
from agentic_rag.services.media.store import AttachmentError
from agentic_rag.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    """Chat request body.

    Media must be uploaded through ``POST /attachments`` first; the chat
    request only carries opaque IDs.  ``images`` is kept for schema
    compatibility and is rejected when non-empty.
    """
    message: str
    session_id: Optional[str] = None
    stream: bool = False
    attachment_ids: list[str] = Field(default_factory=list)
    # Deprecated: retained so old clients receive a clear 400 instead of
    # silently sending local paths that the agent must never read.
    images: list[str] = Field(
        default_factory=list,
        json_schema_extra={"deprecated": True},
    )


def _reject_inline_images(req: ChatRequest) -> None:
    if req.images:
        raise HTTPException(
            status_code=400,
            detail=(
                "Inline image paths are no longer accepted. Upload files via "
                "POST /api/v1/attachments and pass their IDs in attachment_ids."
            ),
        )


async def _build_input(req: ChatRequest, sid: str, llm, repo=None):
    _reject_inline_images(req)
    try:
        return await build_chat_input(
            sid,
            req.message,
            req.attachment_ids,
            repo=repo,
            llm=llm,
        )
    except AttachmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class ChatResponse(BaseModel):
    """Chat response body."""
    answer: str
    session_id: str
    tool_calls: list = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Send a message and get a response (non-streaming)."""
    try:
        llm = get_llm()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sid = req.session_id or uuid.uuid4().hex
    input_data = await _build_input(req, sid, llm)

    from agentic_rag.runtime.orchestrator import get_orchestrator
    from agentic_rag.runtime.unified_context import UnifiedContext
    output = await get_orchestrator().process(
        query=input_data.query,
        session_id=sid,
        has_media=bool(req.attachment_ids),
        input_data=input_data,
        context=UnifiedContext.create(session_id=sid),
    )

    return ChatResponse(
        answer=output.final_answer,
        session_id=sid,
        tool_calls=[tc.model_dump() for tc in output.tool_calls_made],
        diagnostics=output.diagnostics,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Send a message and stream the response via SSE."""
    try:
        llm = get_llm()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sid = req.session_id or uuid.uuid4().hex
    from agentic_rag.data.db.session_repo import SessionRepo
    repo = SessionRepo()
    input_data = await _build_input(req, sid, llm, repo=repo)

    from agentic_rag.runtime.orchestrator import get_orchestrator
    from agentic_rag.runtime.unified_context import UnifiedContext
    orchestrator = get_orchestrator()
    context = UnifiedContext.create(session_id=sid)

    async def event_generator():
        _done_emitted = False     # track if engine already emitted done

        try:
            async for event in orchestrator.process_stream(
                query=input_data.query,
                session_id=sid,
                has_media=bool(req.attachment_ids),
                input_data=input_data,
                context=context,
            ):
                if event.event_type.value == "done":
                    _done_emitted = True
                yield {
                    "event": event.event_type.value,
                    "data": event.model_dump_json() if hasattr(event, 'model_dump_json') else "{}",
                }
        except Exception:
            request_id = getattr(request.state, "request_id", "")
            logger.exception("Chat stream failed (request_id=%s)", request_id)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": "Stream failed", "request_id": request_id},
                    ensure_ascii=False,
                ),
            }

        # Assistant persistence is handled once by ChatOrchestrator, after the
        # stream loop, so REST / voice / gateway / WebSocket all share it.
        if not _done_emitted:
            yield {"event": "done", "data": "{}"}

    if EventSourceResponse is None:
        raise HTTPException(status_code=501, detail="SSE streaming not available. Install sse-starlette.")
    return EventSourceResponse(event_generator())


class VoiceRequest(BaseModel):
    """Voice chat request — audio in, audio out (base64, kept for WS compat)."""
    audio: str = ""
    sid: str = ""
    tts: bool = True


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile | None = File(None),
    sid: str = Form(""),
    session_id: str = Form(""),
    tts: bool = Form(True),
):
    """Voice chat with SSE streaming: audio → STT → Agent stream → TTS.

    SSE events: transcript | text_delta | tool_* | final_answer | audio | done
    """
    import base64, sys, json as _json

    if audio is None:
        raise HTTPException(status_code=400, detail="No audio file provided")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    from agentic_rag.config.settings import get_settings
    vs = get_settings().voice
    from agentic_rag.services.media.voice.stt import STTService
    stt_svc = STTService(
        provider=vs.stt_provider, model=vs.stt_model,
        api_base=vs.stt_api_base, api_key=vs.stt_api_key,
        language=vs.stt_language, sample_rate=vs.sample_rate,
    )

    def _sse(event: str, data: dict) -> dict:
        return {"event": event, "data": _json.dumps(data, ensure_ascii=False)}

    async def _generator():
        # Step 1: STT — yield transcript immediately
        transcript = await stt_svc.transcribe_bytes(audio_bytes)
        if not transcript.strip():
            yield _sse("error", {"error": "No speech detected"})
            yield _sse("done", {})
            return
        logger.info("[Voice] STT: %s...", transcript[:80])
        yield _sse("transcript", {"text": transcript})

        # Step 2: Agent — stream response
        sid_val = session_id or sid or uuid.uuid4().hex
        try:
            llm = get_llm()
        except ValueError as e:
            yield _sse("error", {"error": str(e)})
            yield _sse("done", {})
            return

        from agentic_rag.data.db.session_repo import SessionRepo
        repo = SessionRepo()
        # Same context construction as /chat: session ensured and prior
        # history loaded. Persistence happens in the orchestrator afterwards.
        input_data = await build_chat_input(
            sid_val, transcript, attachment_ids=[], repo=repo, llm=llm,
        )
        from agentic_rag.runtime.orchestrator import get_orchestrator
        from agentic_rag.runtime.unified_context import UnifiedContext

        full_answer = ""
        async for event in get_orchestrator().process_stream(
            query=transcript,
            session_id=sid_val,
            input_data=input_data,
            context=UnifiedContext.create(session_id=sid_val),
        ):
            evt_type = event.event_type.value
            payload = event.data if hasattr(event, 'data') else {}
            if evt_type == "text_delta":
                # Engine-side gating already strips reasoning and Thought
                # text; these deltas are visible answer text only.
                chunk = payload.get("content", "") or payload.get("delta", "")
                if chunk.strip():
                    full_answer += chunk
                    yield _sse("text_delta", {"content": chunk})
            elif evt_type == "done":
                ans = payload.get("final_answer", "")
                if ans:
                    full_answer = ans
            elif evt_type in ("tool_call_start", "tool_call_result", "error"):
                yield _sse(evt_type, payload)

        logger.info("[Voice] Answer: %s...", full_answer[:80])

        # Persistence is centralised in ChatOrchestrator.process_stream(), so
        # this route no longer writes the assistant message itself.

        # Text must not wait for a potentially slow or unavailable TTS service.
        yield _sse("final_answer", {
            "final_answer": full_answer,
            "session_id": sid_val,
        })

        # Step 3: TTS
        if tts and full_answer.strip():
            logger.info("[Voice] Synthesizing TTS")
            from agentic_rag.services.media.voice.tts import TTSService
            tts_svc = TTSService(
                provider=vs.tts_provider, model=vs.tts_model,
                api_base=vs.tts_api_base, api_key=vs.tts_api_key,
                task_type=vs.tts_task_type, instructions=vs.tts_instructions,
                language=vs.tts_language, speaker=vs.tts_speaker,
                voice=vs.tts_voice, speed=vs.tts_speed,
                response_format=vs.tts_response_format,
            )
            try:
                audio_out = await asyncio.wait_for(
                    tts_svc.synthesize(full_answer),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                logger.warning("[Voice] TTS timed out after 45 seconds")
                audio_out = b""
            except Exception:
                logger.exception("[Voice] TTS failed")
                audio_out = b""
            if audio_out:
                audio_b64 = base64.b64encode(audio_out).decode()
                logger.info("[Voice] TTS: %s bytes", len(audio_out))
                yield _sse("audio", {"audio": audio_b64, "format": vs.tts_response_format})
                yield _sse("done", {"final_answer": full_answer, "session_id": sid_val})
                return
        yield _sse("audio", {})
        yield _sse("done", {"final_answer": full_answer, "session_id": sid_val})

    if EventSourceResponse is None:
        raise HTTPException(status_code=501, detail="SSE streaming not available. Install sse-starlette.")
    return EventSourceResponse(_generator())
