"""Chat attachment upload and lookup.

Media enters the system only here.  The chat endpoint accepts opaque IDs, so
clients cannot make the agent read arbitrary local paths.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agentic_rag.services.media.store import (
    AttachmentError,
    AttachmentNotFound,
    AttachmentStore,
    AttachmentTooLarge,
    UnsupportedAttachment,
)

router = APIRouter()


def _store() -> AttachmentStore:
    return AttachmentStore()


@router.post("/attachments")
async def upload_attachment(
    file: UploadFile = File(...),
    session_id: str = Form(""),
    user_id: str = Form("web_ui"),
):
    """Upload one image/audio/video file and return its attachment ID."""
    store = _store()
    # Read at most one byte past the limit: enough to reject without buffering
    # an arbitrarily large upload in memory.
    data = await file.read(store.max_bytes + 1)
    try:
        attachment = store.save_bytes(
            data,
            file.filename or "attachment",
            file.content_type or "",
            user_id=user_id,
            session_id=session_id,
        )
    except AttachmentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except UnsupportedAttachment as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except AttachmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "attachment_id": attachment.id,
        "kind": attachment.kind.value,
        "filename": attachment.filename,
        "mime": attachment.mime,
        "size": attachment.size,
    }


@router.get("/attachments/{attachment_id}")
async def get_attachment(
    attachment_id: str,
    session_id: str = "",
    user_id: str = "",
):
    """Return attachment metadata (raw bytes are not exposed)."""
    try:
        attachment = _store().get(
            attachment_id, user_id=user_id, session_id=session_id,
        )
    except AttachmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "attachment_id": attachment.id,
        "kind": attachment.kind.value,
        "filename": attachment.filename,
        "mime": attachment.mime,
        "size": attachment.size,
        "digest_generated": bool(attachment.digest),
        "created_at": attachment.created_at,
    }
