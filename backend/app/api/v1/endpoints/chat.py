from typing import Optional, Union
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.api.v1.endpoints.auth import get_current_user_optional, get_current_user
from app.core.config import settings
from app.core.security import limiter
from app.schemas.chat import ChatRequest, ChatResponse, FeedbackRequest
from app.schemas.error import ProblemDetails
from app.services.chat_service import ChatService
from app.services.conversation_log_service import conversation_log_service

router = APIRouter()



@router.post(
    "/chat",
    response_model=Union[ChatResponse, None],
    responses={
        200: {
            "description": "SSE Event Stream (when stream=true) or ChatResponse JSON (when stream=false)",
            "content": {
                "text/event-stream": {
                    "example": "event: message\ndata: {\"type\": \"token\", \"content\": \"Hello!\"}\n\n"
                },
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatResponse"}
                },
            },
        },
        429: {"model": ProblemDetails, "description": "Rate limit exceeded"},
        502: {"model": ProblemDetails, "description": "AI provider error"},
    },
    summary="Submit user prompt and receive AI response",
    description="Supports real-time token streaming via Server-Sent Events (SSE) or synchronous JSON generation.",
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: Optional[dict] = Depends(get_current_user_optional),
    x_client_id: Optional[str] = Header(None, alias="X-Client-Id"),
):
    # Never retain ownership supplied in the request body, including when a
    # guest client header is absent or invalid.
    payload.user_id = None
    # Authenticated identity takes precedence over anonymous device partitioning.
    if current_user and current_user.get("id"):
        payload.user_id = str(current_user["id"])
    # Client identifiers and conversation IDs are never ownership credentials.
    payload.conversation_id = str(uuid4())

    # 2. Enforce session ownership: Verify that caller owns the session before appending turns
    if payload.session_id:
        try:
            from uuid import UUID
            from app.services.session_store import session_store
            sid = UUID(str(payload.session_id))
            existing_sess = session_store.get_session(sid)
            if existing_sess:
                sess_owner = existing_sess.get("userId")
                if not payload.user_id or sess_owner != payload.user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not have permission to append messages to this conversation session."
                    )
        except (ValueError, AttributeError):
            pass
    if not payload.user_id:
        if payload.attachments:
            raise HTTPException(401, "Sign in to process private attachments.")
        payload.session_id = None  # Anonymous chat is stateless; never load supplied history.

    if payload.stream:
        # Stream response chunks via Server-Sent Events
        return StreamingResponse(
            chat_service.stream_chat_response(payload, client_disconnect_checker=request.is_disconnected),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disables proxy buffering on Render/Nginx/Cloudflare
                "Content-Type": "text/event-stream; charset=utf-8",
            },
        )

    # Return standard synchronous response
    return await chat_service.generate_chat_response(payload)


@router.post(
    "/feedback",
    summary="Record user feedback for a conversation turn",
    description="Updates feedback signal ('up' | 'down' | 'regenerated') on the conversation log entry asynchronously.",
)
async def feedback_endpoint(payload: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    """Lightweight feedback endpoint to update satisfaction signal."""
    success = conversation_log_service.update_feedback(
        conversation_id=payload.conversation_id,
        signal=payload.signal,
        owner_id=current_user["id"],
    )
    if not success:
        raise HTTPException(404, "Feedback target not found for this account.")
    return {
        "status": "success" if success else "ignored",
        "conversation_id": payload.conversation_id,
        "feedback": payload.signal,
        "updated": success,
    }

