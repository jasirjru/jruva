"""Groq Whisper Voice Input Service.

Implements cloud-hosted audio transcription using Groq's whisper-large-v3-turbo model.
Protected by feature flag `ENABLE_VOICE_TRANSCRIPTION=False` by default.
Enforces:
1. Feature flag guard (fails closed when False)
2. Strict size limits (maximum 10 MB audio payload)
3. Audio MIME type and extension validation
4. Daily user audio quota (maximum 120s / user / UTC day)
5. Zero local GPU or Ollama dependencies
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple

import httpx
from app.core.config import settings
from app.core.exceptions import BaseAppException
from app.services.quota.daily_usage_service import (
    DailyQuotaExceededError,
    async_check_and_consume_audio_seconds,
)

logger = logging.getLogger("chatbot.audio")

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3-turbo"
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}
ALLOWED_AUDIO_MIMES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/m4a",
    "audio/x-m4a",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "video/mp4",
    "audio/mp4",
}


class VoiceTranscriptionDisabledError(BaseAppException):
    """Raised when voice transcription endpoint is accessed while feature-flagged OFF."""

    def __init__(self, detail: str = "Voice transcription is currently disabled on this deployment."):
        super().__init__(
            detail=detail,
            status_code=503,
            error_code="VOICE_TRANSCRIPTION_DISABLED",
            title="Voice Transcription Disabled",
        )


class AudioPayloadTooLargeError(BaseAppException):
    """Raised when uploaded audio file exceeds maximum allowable byte size."""

    def __init__(self, size_bytes: int, max_bytes: int = MAX_AUDIO_BYTES):
        super().__init__(
            detail=f"Audio file size ({size_bytes} bytes) exceeds maximum allowable limit of {max_bytes} bytes (10MB).",
            status_code=413,
            error_code="AUDIO_PAYLOAD_TOO_LARGE",
            title="Audio Payload Too Large",
        )


class UnsupportedAudioFormatError(BaseAppException):
    """Raised when audio extension or MIME type is not supported."""

    def __init__(self, filename: str, mime_type: str = ""):
        super().__init__(
            detail=f"Unsupported audio format for file '{filename}' (MIME: {mime_type}). Allowed formats: mp3, wav, m4a, webm, ogg, flac.",
            status_code=415,
            error_code="UNSUPPORTED_AUDIO_FORMAT",
            title="Unsupported Audio Format",
        )


class GroqWhisperService:
    """Service wrapping Groq cloud-hosted Whisper transcription API."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or getattr(settings, "GROQ_API_KEY", "")

    @property
    def is_enabled(self) -> bool:
        """Check whether voice transcription feature flag is currently active."""
        return bool(getattr(settings, "ENABLE_VOICE_TRANSCRIPTION", False))

    def validate_audio_file(self, filename: str, file_bytes: bytes, mime_type: str = "") -> None:
        """Validate file size, extension, and content headers."""
        if not self.is_enabled:
            raise VoiceTranscriptionDisabledError()

        if len(file_bytes) > MAX_AUDIO_BYTES:
            raise AudioPayloadTooLargeError(len(file_bytes), MAX_AUDIO_BYTES)

        if len(file_bytes) == 0:
            raise ValueError("Audio payload is empty (0 bytes).")

        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_AUDIO_EXTENSIONS and mime_type not in ALLOWED_AUDIO_MIMES:
            raise UnsupportedAudioFormatError(filename, mime_type)

    async def transcribe_audio(
        self,
        filename: str,
        file_bytes: bytes,
        mime_type: str = "audio/webm",
        user_id: str = "guest",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate, check daily quota, and transcribe audio via Groq Whisper API."""
        self.validate_audio_file(filename, file_bytes, mime_type)

        api_key = self._api_key or getattr(settings, "GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured for voice transcription.")

        # Estimate duration conservatively (e.g. 1MB audio is roughly 60 seconds compressed)
        # We perform an initial admission reservation of at least 10 seconds or file-size heuristic
        estimated_duration = max(10, min(120, int(len(file_bytes) / 16000)))
        guest_pool = settings.GUEST_DAILY_AUDIO_SECONDS if user_id == "guest" else None
        await async_check_and_consume_audio_seconds(user_id, estimated_duration, limit=guest_pool)

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        data_fields: Dict[str, Any] = {
            "model": WHISPER_MODEL,
            "response_format": "verbose_json",
            "temperature": "0.0",
        }
        if language:
            data_fields["language"] = language
        if prompt:
            data_fields["prompt"] = prompt

        files = {
            "file": (filename, io.BytesIO(file_bytes), mime_type or "audio/webm"),
        }

        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                resp = await client.post(
                    GROQ_TRANSCRIPTION_URL,
                    headers=headers,
                    data=data_fields,
                    files=files,
                )

            if resp.status_code != 200:
                logger.error(f"Groq Whisper API returned {resp.status_code}: {resp.text}")
                raise RuntimeError(f"Groq Whisper transcription upstream error ({resp.status_code}): {resp.text}")

            result = resp.json()
            transcription_text = result.get("text", "").strip()
            actual_duration = float(result.get("duration", estimated_duration))

            return {
                "text": transcription_text,
                "duration_seconds": round(actual_duration, 2),
                "language": result.get("language", language or "en"),
                "model": WHISPER_MODEL,
                "provider": "groq",
            }
        except httpx.TimeoutException as te:
            logger.error(f"Groq Whisper API timeout: {te}")
            raise RuntimeError("Voice transcription request timed out.")
        except Exception as exc:
            if isinstance(exc, BaseAppException):
                raise
            logger.exception(f"Unexpected error during audio transcription: {exc}")
            raise RuntimeError(f"Audio transcription failed: {str(exc)}")


whisper_service = GroqWhisperService()
