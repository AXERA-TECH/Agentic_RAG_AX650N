"""Chat endpoints — the primary agent interaction interface."""

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = None

from agentic_rag.data.models import AgentInput
from agentic_rag.services.llm.factory import get_llm

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request body."""
    message: str
    session_id: Optional[str] = None
    stream: bool = False
    images: list[str] = Field(default_factory=list)  # image paths for vision


class ChatResponse(BaseModel):
    """Chat response body."""
    answer: str
    session_id: str
    tool_calls: list = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


def _build_input(query: str, images: list[str]) -> AgentInput:
    """Build agent input — embeds images directly if LLM supports vision.

    ``images`` can be:
    - File paths (``/tmp/photo.jpg``) — read from disk and base64-encode
    - Data URIs (``data:image/jpeg;base64,...``) — used as-is
    """
    from agentic_rag.data.models import AgentInput, Message, MessageRole
    import base64
    from pathlib import Path as _Path

    if not images:
        return AgentInput(query=query)

    # Check if current LLM supports vision
    try:
        from agentic_rag.services.llm.factory import get_llm
        llm = get_llm()
    except Exception:
        llm = None

    if not getattr(llm, 'supports_vision', False):
        return AgentInput(query=f"{query}\n\n[Attached: {len(images)} image(s)]")

    # Vision LLM — embed images directly in multimodal message
    content = [{"type": "text", "text": query}]
    for img in images:
        if img.startswith("data:"):
            # Already a data URI — use directly
            content.append({
                "type": "image_url",
                "image_url": {"url": img},
            })
        else:
            # File path — read and encode
            p = _Path(img)
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                ext = p.suffix.lower()
                mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".png": "image/png", ".webp": "image/webp",
                            ".gif": "image/gif", ".bmp": "image/bmp"}
                mime = mime_map.get(ext, "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

    msg = Message(role=MessageRole.USER, content=content)
    return AgentInput(query=query, messages=[msg])


def _load_session_history(repo, sid: str) -> list:
    """Load conversation history from SQLite, trimmed to ~8000 chars (~5K tokens).

    The caller appends the current query itself, so a trailing user message
    (the one persisted by ``_save_user_message`` moments ago) is dropped here.
    """
    from agentic_rag.data.models import Message as MsgModel, MessageRole

    history_messages = []
    db_msgs = repo.get_messages(sid, limit=10)
    if db_msgs and db_msgs[-1].get("role") == "user":
        db_msgs = db_msgs[:-1]  # drop current query, passed separately

    total_chars = 0
    for m in reversed(db_msgs):  # newest first, accumulate until limit
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            continue
        if len(content) > 2000:
            content = content[:2000] + "..."
        if total_chars + len(content) > 8000:
            break  # stop adding older messages
        total_chars += len(content)
        try:
            r = MessageRole(role)
        except ValueError:
            r = MessageRole.USER
        history_messages.insert(0, MsgModel(role=r, content=content))
    return history_messages


def build_chat_input(sid: str, query: str, images: list[str], repo=None):
    """Build agent input and persist the user message — shared by all entries.

    ``/chat``, ``/chat/stream`` and ``/chat/voice`` must construct context
    identically: the session is ensured, the user message is written to
    SQLite, and prior history is loaded into the input. Without this,
    non-streaming turns neither saw history nor survived a restart.
    """
    from agentic_rag.data.db.session_repo import SessionRepo

    if repo is None:
        repo = SessionRepo()
    history_messages = []
    try:
        if not repo.get(sid):
            # Session was deleted (e.g., user cleared messages) — re-create it
            repo.create_with_id(sid, user_id="web_ui")
            print(f"  [chat] Session {sid[:12]} re-created (was deleted)", flush=True)
        repo.add_message(sid, role="user", content=query)
        history_messages = _load_session_history(repo, sid)
    except Exception as e:
        print(f"  [chat] ⚠ History load failed: {e}", flush=True)

    if images:
        # Multimodal input already embeds its own user message; history is
        # prepended so the loop still sees the conversation.
        input_data = _build_input(query, images)
        input_data.messages = history_messages + input_data.messages
    else:
        input_data = AgentInput(query=query, messages=history_messages)
    input_data.parameters["session_id"] = sid
    input_data.parameters["sid"] = sid
    return input_data


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Send a message and get a response (non-streaming)."""
    try:
        llm = get_llm()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sid = req.session_id or uuid.uuid4().hex
    input_data = build_chat_input(sid, req.message, req.images)

    from agentic_rag.runtime.orchestrator import get_orchestrator
    from agentic_rag.runtime.unified_context import UnifiedContext
    output = await get_orchestrator().process(
        query=input_data.query,
        session_id=sid,
        has_media=bool(req.images),
        input_data=input_data,
        context=UnifiedContext.create(session_id=sid),
    )

    # Persist the assistant reply — stream/voice do the same in their loops.
    if output.final_answer.strip():
        try:
            from agentic_rag.data.db.session_repo import SessionRepo
            SessionRepo().add_message(sid, role="assistant", content=output.final_answer)
        except Exception as e:
            print(f"  [chat] ⚠ Failed to save assistant message: {e}", flush=True)

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
    # Load history from SQLite & save the user message (shared with /chat)
    from agentic_rag.data.db.session_repo import SessionRepo
    repo = SessionRepo()
    input_data = build_chat_input(sid, req.message, req.images, repo=repo)

    from agentic_rag.runtime.orchestrator import get_orchestrator
    from agentic_rag.runtime.unified_context import UnifiedContext
    orchestrator = get_orchestrator()
    context = UnifiedContext.create(session_id=sid)

    async def event_generator():
        from agentic_rag.agent.react_parser import clean_user_answer
        full_text_parts = []
        final_answer = ""
        _done_emitted = False     # track if engine already emitted done

        try:
            async for event in orchestrator.process_stream(
                query=input_data.query,
                session_id=sid,
                has_media=bool(req.images),
                input_data=input_data,
                context=context,
            ):
                payload = event.data if hasattr(event, 'data') else {}
                if event.event_type.value == "done":
                    final_answer = payload.get("final_answer", "")
                    _done_emitted = True
                elif event.event_type.value == "text_delta":
                    # The engine only emits TEXT_DELTA for the visible final
                    # answer — reasoning, Thought lines, and repetition are
                    # already filtered upstream. Forward as-is.
                    chunk = payload.get("content", "") or payload.get("delta", "")
                    if chunk:
                        full_text_parts.append(chunk)
                yield {
                    "event": event.event_type.value,
                    "data": event.model_dump_json() if hasattr(event, 'model_dump_json') else "{}",
                }
        except Exception as e:
            import sys
            print(f"  [chat] ⚠ Event generator crashed: {e}", flush=True)
            sys.stdout.flush()
            yield {
                "event": "error",
                "data": f'{{"error": "Stream crashed: {e}"}}',
            }

        # Use final_answer (already clean) if available, else concatenated deltas
        response_text = clean_user_answer(final_answer.strip()) or clean_user_answer("".join(full_text_parts).strip())
        if response_text:
            import sys
            print(f"  [chat] Saving assistant message ({len(response_text)} chars) to session {sid[:12]}...", flush=True)
            try:
                repo.add_message(sid, role="assistant", content=response_text)
                print(f"  [chat] ✓ Assistant message saved", flush=True)
            except Exception as e:
                print(f"  [chat] ⚠ Failed to save assistant message: {e}", flush=True)
        else:
            print(f"  [chat] ⚠ No assistant text collected (final_answer={bool(final_answer)}, deltas={len(full_text_parts)})", flush=True)

        # Only emit done if engine didn't already emit one
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
    from agentic_rag.core.voice.stt import STTService
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
        print(f"  [Voice] STT: \"{transcript[:80]}...\"", flush=True)
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
        # Same context construction as /chat: session ensured, transcript
        # persisted, prior history loaded.
        input_data = build_chat_input(sid_val, transcript, [], repo=repo)
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

        print(f"  [Voice] Answer: \"{full_answer[:80]}...\"", flush=True)

        # Save assistant message
        if full_answer.strip():
            try:
                repo.add_message(sid_val, role="assistant", content=full_answer)
            except Exception:
                pass

        # Text must not wait for a potentially slow or unavailable TTS service.
        yield _sse("final_answer", {
            "final_answer": full_answer,
            "session_id": sid_val,
        })

        # Step 3: TTS
        if tts and full_answer.strip():
            print("  [Voice] Synthesizing TTS...", flush=True)
            from agentic_rag.core.voice.tts import TTSService
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
                print("  [Voice] ⚠ TTS timed out after 45 seconds", flush=True)
                audio_out = b""
            except Exception as exc:
                print(f"  [Voice] ⚠ TTS failed: {exc}", flush=True)
                audio_out = b""
            if audio_out:
                audio_b64 = base64.b64encode(audio_out).decode()
                print(f"  [Voice] TTS: {len(audio_out)} bytes", flush=True)
                yield _sse("audio", {"audio": audio_b64, "format": vs.tts_response_format})
                yield _sse("done", {"final_answer": full_answer, "session_id": sid_val})
                return
        yield _sse("audio", {})
        yield _sse("done", {"final_answer": full_answer, "session_id": sid_val})

    if EventSourceResponse is None:
        raise HTTPException(status_code=501, detail="SSE streaming not available. Install sse-starlette.")
    return EventSourceResponse(_generator())
