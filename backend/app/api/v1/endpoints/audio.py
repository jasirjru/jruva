"""Audio transcription endpoint using Groq Whisper."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.api.v1.endpoints.auth import get_current_user_optional
from app.core.security import EXPENSIVE_ENDPOINT_LIMIT, limiter
from app.services.audio.whisper_service import (
    MAX_AUDIO_BYTES,
    whisper_service,
)

router = APIRouter()


class TranscriptionResponse(BaseModel):
    """Transcription response payload."""
    text: str
    duration_seconds: float
    language: str
    model: str
    provider: str


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe audio recording via Groq Whisper",
    description="Transcribes audio files (mp3, wav, m4a, webm, ogg, flac) up to 10MB using whisper-large-v3-turbo.",
)
@limiter.limit(EXPENSIVE_ENDPOINT_LIMIT)
async def transcribe_audio_endpoint(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Handle audio upload and transcription."""
    user_id = str(current_user["id"]) if current_user and current_user.get("id") else "guest"

    # Read at most limit+1 bytes so an oversized upload is never fully buffered in memory.
    file_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    if len(file_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file exceeds the 10MB limit.",
        )

    res = await whisper_service.transcribe_audio(
        filename=file.filename or "recording.webm",
        file_bytes=file_bytes,
        mime_type=file.content_type or "audio/webm",
        user_id=user_id,
        language=language,
        prompt=prompt,
    )

    return TranscriptionResponse(**res)
