"""Test suite for Groq Whisper voice input and transcription service."""

import io
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from app.core.config import settings
from app.services.audio.whisper_service import (
    AudioPayloadTooLargeError,
    GroqWhisperService,
    UnsupportedAudioFormatError,
    VoiceTranscriptionDisabledError,
)
from app.services.quota.daily_usage_service import (
    DailyQuotaExceededError,
    async_check_and_consume_audio_seconds,
)


@pytest.fixture
def whisper_service():
    """Create service instance with test API key."""
    return GroqWhisperService(api_key="gsk-test-mock-key")


def test_whisper_fails_closed_when_flag_disabled(whisper_service):
    """When ENABLE_VOICE_TRANSCRIPTION is False (default), all attempts must fail closed."""
    with patch.object(settings, "ENABLE_VOICE_TRANSCRIPTION", False):
        assert whisper_service.is_enabled is False
        with pytest.raises(VoiceTranscriptionDisabledError) as exc:
            whisper_service.validate_audio_file("test.mp3", b"fake audio content", "audio/mpeg")
        assert exc.value.status_code == 503
        assert exc.value.error_code == "VOICE_TRANSCRIPTION_DISABLED"


def test_whisper_rejects_payload_exceeding_10mb(whisper_service):
    """Payloads exceeding 10 MB must be rejected with 413 error."""
    with patch.object(settings, "ENABLE_VOICE_TRANSCRIPTION", True):
        oversized = b"0" * (10 * 1024 * 1024 + 1)
        with pytest.raises(AudioPayloadTooLargeError) as exc:
            whisper_service.validate_audio_file("recording.mp3", oversized, "audio/mpeg")
        assert exc.value.status_code == 413


def test_whisper_rejects_unsupported_formats(whisper_service):
    """Executables, PDFs, or unapproved file extensions must be rejected with 415."""
    with patch.object(settings, "ENABLE_VOICE_TRANSCRIPTION", True):
        with pytest.raises(UnsupportedAudioFormatError) as exc:
            whisper_service.validate_audio_file("malicious.exe", b"binary", "application/x-msdownload")
        assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_whisper_enforces_daily_audio_seconds_quota(whisper_service):
    """Verify daily audio duration cap is enforced before upstream request."""
    uid = f"audio-quota-user-{uuid.uuid4().hex[:8]}"

    with patch.object(settings, "ENABLE_VOICE_TRANSCRIPTION", True):
        with patch.object(settings, "DAILY_AUDIO_SECONDS_PER_USER", 30):
            # 1st: consume 25 seconds
            await async_check_and_consume_audio_seconds(uid, 25)

            # 2nd: attempting another 10s exceeds 30s limit
            with pytest.raises(DailyQuotaExceededError) as exc:
                await whisper_service.transcribe_audio(
                    filename="audio.webm",
                    file_bytes=b"fake-audio-bytes" * 100,
                    mime_type="audio/webm",
                    user_id=uid,
                )
            assert exc.value.status_code == 429
            assert exc.value.quota_type == "audio_seconds"


@pytest.mark.asyncio
async def test_whisper_successful_transcription(whisper_service):
    """Mock Groq Whisper HTTP response and verify transcription result contract."""
    uid = f"audio-success-user-{uuid.uuid4().hex[:8]}"
    mock_response_payload = {
        "text": "Hello world, this is a test audio transcription.",
        "duration": 4.5,
        "language": "en",
    }

    mock_resp = httpx.Response(
        status_code=200,
        json=mock_response_payload,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )

    with patch.object(settings, "ENABLE_VOICE_TRANSCRIPTION", True):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await whisper_service.transcribe_audio(
                filename="meeting.mp3",
                file_bytes=b"ID3" + b"\x00" * 500,
                mime_type="audio/mpeg",
                user_id=uid,
            )

            assert result["text"] == "Hello world, this is a test audio transcription."
            assert result["duration_seconds"] == 4.5
            assert result["language"] == "en"
            assert result["model"] == "whisper-large-v3-turbo"
            assert result["provider"] == "groq"
            assert mock_post.called


@pytest.mark.asyncio
async def test_guests_draw_from_their_own_shared_audio_pool(whisper_service):
    """All guests share one quota row, so they get a site-wide pool sized to Groq's free tier, not one person's 120 s."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"text": "What is the tallest mountain?", "duration": 3.0, "language": "en"},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/audio/transcriptions"),
    )
    with patch.object(settings, "ENABLE_VOICE_TRANSCRIPTION", True), \
            patch.object(settings, "DAILY_AUDIO_SECONDS_PER_USER", 0), \
            patch.object(settings, "GUEST_DAILY_AUDIO_SECONDS", 10**6), \
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        result = await whisper_service.transcribe_audio(
            filename="recording.webm", file_bytes=b"\x1aE\xdf\xa3" + b"\x00" * 500, mime_type="audio/webm", user_id="guest"
        )
    assert result["text"] == "What is the tallest mountain?"


def test_guest_audio_pool_fits_groq_free_tier():
    # Groq Whisper free tier: 28,800 audio seconds/day; leave headroom for signed-in users.
    assert 120 < settings.GUEST_DAILY_AUDIO_SECONDS <= 28_800 // 2
