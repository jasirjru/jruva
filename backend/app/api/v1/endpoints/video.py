"""Video lecture summarization endpoint."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.api.v1.endpoints.auth import get_current_user_optional
from app.core.security import EXPENSIVE_ENDPOINT_LIMIT, limiter
from app.services.video.video_summary_service import video_summary_service

router = APIRouter()


class VideoSummaryRequest(BaseModel):
    url: str
    focus_topic: Optional[str] = None


class ChapterItem(BaseModel):
    title: str
    timestamp: str


class VideoSummaryResponse(BaseModel):
    video_id: str
    title: str
    author: str
    summary: str
    chapters: List[ChapterItem]
    provider: str
    status: str


@router.post(
    "/summarize",
    response_model=VideoSummaryResponse,
    summary="Summarize YouTube lecture video",
    description="Analyzes lecture video from YouTube with SSRF protection behind ENABLE_VIDEO_UNDERSTANDING feature flag.",
)
@limiter.limit(EXPENSIVE_ENDPOINT_LIMIT)
async def summarize_video_endpoint(
    request: Request,
    payload: VideoSummaryRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Handle video lecture summary request."""
    user_id = str(current_user["id"]) if current_user and current_user.get("id") else "guest"
    result = await video_summary_service.summarize_video(
        url=payload.url,
        user_id=user_id,
        focus_topic=payload.focus_topic,
    )
    return VideoSummaryResponse(**result)
