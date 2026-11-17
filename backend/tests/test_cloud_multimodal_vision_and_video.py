"""Test suite for Phase 6: Cloud Multimodal Vision (Gemini 3.6 Flash) and Video Understanding."""

import uuid
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from app.core.config import settings
from app.schemas.chat import AttachmentItem, ChatRequest
from app.services.chat_service import ChatService
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.router import LLMRouter
from app.services.video.video_summary_service import (
    VideoSSRFSecurityError,
    VideoUnderstandingDisabledError,
    VideoUnderstandingService,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cloud Multimodal Vision Tests (Gemini 3.6 Flash)
# ─────────────────────────────────────────────────────────────────────────────

def test_router_selects_gemini_flash_for_visual_attachments():
    """When Groq vision is unavailable, visual attachments must go to Gemini 3.6 Flash."""
    router = LLMRouter()
    groq_off = patch.object(type(router.groq_provider), "is_available", new_callable=PropertyMock, return_value=False)
    with patch.object(router.gemini_provider, "_api_key", "mock-gemini-key"), groq_off:
        img_att = AttachmentItem(
            name="architecture_diagram.png",
            mime_type="image/png",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        )
        primary, backup, model_id = router._resolve_providers(
            requested_model="auto",
            attachments=[img_att],
        )
        assert primary.provider_name == "gemini"
        assert "gemini" in model_id.lower()


@pytest.mark.asyncio
async def test_gemini_multimodal_vision_chat_flow():
    """Verify end-to-end chat flow with image attachment routes through Gemini."""
    service = ChatService()
    uid = f"vision-user-{uuid.uuid4().hex[:8]}"

    mock_vision_text = "The uploaded chart displays a bar graph with revenue growing from $10M in Q1 to $18M in Q3."
    service.router.generate_response = AsyncMock(
        return_value=(mock_vision_text, "gemini", "gemini-3.6-flash", False)
    )

    req = ChatRequest(
        message="What are the key trends shown in this revenue chart?",
        model="auto",
        stream=False,
        attachments=[
            AttachmentItem(
                name="chart.png",
                mime_type="image/png",
                data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            )
        ],
    )
    req.user_id = uid

    response = await service.generate_chat_response(req)
    assert response.provider == "gemini"
    assert response.model == "gemini-3.6-flash"
    assert "revenue growing from $10M" in response.content


# ─────────────────────────────────────────────────────────────────────────────
# 2. Server-Side Video Understanding & SSRF Sanitization Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_video_fails_closed_when_feature_flag_disabled():
    """Video service must reject requests with 503 when ENABLE_VIDEO_UNDERSTANDING is False."""
    service = VideoUnderstandingService()
    with patch.object(settings, "ENABLE_VIDEO_UNDERSTANDING", False):
        assert service.is_enabled is False
        with pytest.raises(VideoUnderstandingDisabledError) as exc:
            service.validate_and_extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert exc.value.status_code == 503


def test_video_ssrf_rejects_private_ip_literals():
    """SSRF: Direct IP literals and private address ranges must be blocked."""
    service = VideoUnderstandingService()
    with patch.object(settings, "ENABLE_VIDEO_UNDERSTANDING", True):
        # AWS metadata service IP
        with pytest.raises(VideoSSRFSecurityError):
            service.validate_and_extract_video_id("http://169.254.169.254/latest/meta-data")

        # Localhost / loopback
        with pytest.raises(VideoSSRFSecurityError):
            service.validate_and_extract_video_id("http://127.0.0.1:8000/secret")

        # Private RFC1918 range
        with pytest.raises(VideoSSRFSecurityError):
            service.validate_and_extract_video_id("http://10.0.0.1/admin")


def test_video_ssrf_rejects_non_youtube_domains():
    """SSRF: Arbitrary external hosts must be rejected."""
    service = VideoUnderstandingService()
    with patch.object(settings, "ENABLE_VIDEO_UNDERSTANDING", True):
        with pytest.raises(VideoSSRFSecurityError) as exc:
            service.validate_and_extract_video_id("https://malicious-external-site.com/exploit")
        assert "not a permitted video source" in str(exc.value)


def test_video_ssrf_rejects_non_http_schemes():
    """SSRF: File, ftp, gopher protocols must be rejected."""
    service = VideoUnderstandingService()
    with patch.object(settings, "ENABLE_VIDEO_UNDERSTANDING", True):
        with pytest.raises(VideoSSRFSecurityError):
            service.validate_and_extract_video_id("file:///etc/passwd")


def test_video_id_extraction_valid_youtube_formats():
    """Extract valid 11-char video ID across canonical YouTube URL variations."""
    service = VideoUnderstandingService()
    with patch.object(settings, "ENABLE_VIDEO_UNDERSTANDING", True):
        # 1. Standard watch URL
        vid1 = service.validate_and_extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vid1 == "dQw4w9WgXcQ"

        # 2. youtu.be short URL
        vid2 = service.validate_and_extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42")
        assert vid2 == "dQw4w9WgXcQ"

        # 3. Shorts URL
        vid3 = service.validate_and_extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ")
        assert vid3 == "dQw4w9WgXcQ"

        # 4. Embed URL
        vid4 = service.validate_and_extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert vid4 == "dQw4w9WgXcQ"


@pytest.mark.asyncio
async def test_video_summarization_workflow():
    """Mock oEmbed API and verify structured lecture summary output."""
    service = VideoUnderstandingService()
    uid = f"video-user-{uuid.uuid4().hex[:8]}"

    mock_oembed = {
        "title": "Introduction to Deep Learning - Lecture 1",
        "author": "MIT OpenCourseWare",
        "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg",
    }

    with patch.object(settings, "ENABLE_VIDEO_UNDERSTANDING", True):
        with patch.object(service, "fetch_video_metadata", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_oembed

            result = await service.summarize_video(
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                user_id=uid,
                focus_topic="Backpropagation",
            )

            assert result["video_id"] == "dQw4w9WgXcQ"
            assert result["title"] == "Introduction to Deep Learning - Lecture 1"
            assert result["author"] == "MIT OpenCourseWare"
            assert "Backpropagation" in result["summary"]
            assert len(result["chapters"]) >= 4
            assert result["provider"] == "gemini-3.6-flash"
