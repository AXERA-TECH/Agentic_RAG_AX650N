"""Session management endpoints — backed by SQLite via SessionRepo."""

from fastapi import APIRouter, HTTPException

from agentic_rag.data.db.session_repo import SessionRepo

router = APIRouter()


def _repo() -> SessionRepo:
    return SessionRepo()


@router.post("/session")
async def create_session(user_id: str = "default"):
    """Create a new session."""
    session = _repo().create(user_id=user_id)
    return {"session_id": session["id"], "user_id": session["user_id"]}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    session = _repo().get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.put("/session/{session_id}")
async def update_session(session_id: str, title: str = None):
    """Update session metadata (e.g., title)."""
    fields = {}
    if title is not None:
        fields["title"] = title
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    ok = _repo().update(session_id, **fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "updated"}


@router.get("/sessions")
async def list_sessions(user_id: str = "default"):
    """List sessions for a user."""
    return {"sessions": _repo().list(user_id=user_id)}


@router.delete("/sessions")
async def delete_all_sessions(user_id: str = "default"):
    """Delete ALL sessions and messages for a user."""
    count = _repo().delete_all(user_id=user_id)
    return {"status": "deleted", "count": count}


@router.delete("/session/{session_id}/messages")
async def clear_session_messages(session_id: str):
    """Clear all messages in a session (keep the session itself)."""
    session = _repo().get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    count = _repo().clear_messages(session_id)
    return {"status": "cleared", "count": count}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its messages."""
    ok = _repo().delete(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.get("/session/{session_id}/messages")
async def get_messages(session_id: str):
    """Get all messages for a session."""
    session = _repo().get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": _repo().get_messages(session_id)}
