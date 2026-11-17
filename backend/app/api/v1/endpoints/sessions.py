from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.v1.endpoints.auth import get_current_user
from app.services.session_store import session_store

router = APIRouter()


def get_session_owner_id(
    current_user: dict = Depends(get_current_user),
    x_client_id: Optional[str] = Header(None, alias="X-Client-Id"),
) -> Optional[str]:
    """Private ownership comes exclusively from a verified server-side account."""
    if current_user and current_user.get("id"):
        return str(current_user["id"])
    raise HTTPException(401, "Sign in to access private data.")


class SessionRenameRequest(BaseModel):
    """Request to rename a session."""
    title: str = Field(..., min_length=1, max_length=200)


class SessionListItem(BaseModel):
    """Single session in the list response."""
    id: str
    userId: Optional[str] = None
    title: str
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class SessionListResponse(BaseModel):
    """Response containing all user sessions."""
    sessions: list[SessionListItem]


class SessionMessagesResponse(BaseModel):
    """Response containing full message history for a session."""
    session_id: str
    messages: list[dict]


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List conversation sessions",
    description="Returns sessions ordered by most recently updated. Scoped strictly to authenticated user or anonymous client device.",
)
async def list_sessions(owner_id: Optional[str] = Depends(get_session_owner_id)):
    """List conversation sessions with strict user or device isolation."""
    if not owner_id:
        return SessionListResponse(sessions=[])
    raw_sessions = session_store.list_sessions(user_id=owner_id, limit=50)
    return SessionListResponse(
        sessions=[SessionListItem(**s) for s in raw_sessions]
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
    summary="Get full message history for a session",
    description="Returns all messages in chronological order, scoped to session owner.",
)
async def get_session_messages(
    session_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Retrieve full message history for a session with ownership validation."""
    try:
        sid = UUID(session_id)
    except ValueError:
        return SessionMessagesResponse(session_id=session_id, messages=[])

    session_meta = session_store.get_session(sid)
    if not session_meta:
        raise HTTPException(status_code=404, detail="Session not found")

    # Privacy & Security Protection: Only the session owner can read messages
    sess_user = session_meta.get("userId")
    if sess_user != owner_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to access this conversation session."
        )

    messages = session_store.get_full_history(sid)
    return SessionMessagesResponse(session_id=session_id, messages=messages)


@router.put(
    "/sessions/{session_id}",
    summary="Rename a session",
    description="Update the title of an existing session.",
)
@router.patch(
    "/sessions/{session_id}",
    summary="Rename a session (partial update)",
    description="Update the title of an existing session.",
)
async def rename_session(
    session_id: str,
    payload: SessionRenameRequest,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Rename a session's title after verifying ownership."""
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session_meta = session_store.get_session(sid)
    if not session_meta:
        raise HTTPException(status_code=404, detail="Session not found")

    sess_user = session_meta.get("userId")
    if sess_user != owner_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to modify this conversation session."
        )

    success = session_store.rename_session(sid, payload.title)
    return {"success": success, "title": payload.title}


@router.delete(
    "/sessions/{session_id}",
    summary="Delete a session",
    description="Permanently delete a session and all its messages.",
)
async def delete_session(
    session_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Delete a session and all its messages, enforcing owner ownership."""
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    session_meta = session_store.get_session(sid)
    if not session_meta:
        raise HTTPException(status_code=404, detail="Session not found")

    sess_user = session_meta.get("userId")
    if sess_user != owner_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to delete this conversation session."
        )

    success = session_store.delete_session(sid, user_id=sess_user)
    return {"success": success, "status": "deleted" if success else "not_found"}


