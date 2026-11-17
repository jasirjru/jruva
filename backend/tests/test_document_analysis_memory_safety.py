import base64
import io
from unittest.mock import patch

import numpy as np
import pytest
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.image.ocr_service import get_ocr_engine, ocr_service
from app.services.image.visual_analysis_service import VisualAnalysisService
from app.services.llm.gemini_provider import GeminiProvider
from PIL import Image


def test_production_mode_guards_rapidocr_initialization():
    """Verify that in production mode, get_ocr_engine returns None to protect 512MB RAM ceiling."""
    with patch("app.core.config.settings.ENVIRONMENT", "production"), \
         patch("app.core.config.settings.ENABLE_LOCAL_OCR", False):
        # Reset cached engine
        import app.services.image.ocr_service as ocr_mod
        ocr_mod._OCR_ENGINE = None

        engine = get_ocr_engine()
        assert engine is None, "RapidOCR must not be initialized in production when ENABLE_LOCAL_OCR is False"

        # extract_text_from_image_bytes must return empty string safely
        fake_bytes = b"fake_jpeg_bytes_header_1234567890_abcdefghij"
        extracted = ocr_service.extract_text_from_image_bytes(fake_bytes)
        assert extracted == ""


def test_visual_analysis_service_downscales_large_images():
    """Verify that _downscale_for_cv caps dimensions to max_dim without error."""
    large_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    downscaled = VisualAnalysisService._downscale_for_cv(large_img, max_dim=1280)
    assert max(downscaled.shape[:2]) == 1280
    assert downscaled.shape[1] == 1280  # width was 4000, scaled to 1280
    assert downscaled.shape[0] == 960   # height 3000 scaled proportionally to 960


def test_gemini_provider_normalizes_mime_types():
    """Verify gemini_provider normalizes image/jpg and cleans base64 data."""
    provider = GeminiProvider(api_key="mock_key", model="gemini-3.6-flash")
    raw_bytes = b"sample_png_bytes"
    b64_str = base64.b64encode(raw_bytes).decode("ascii") + "\n\r"

    attachment = {
        "name": "photo.jpg",
        "mime_type": "image/jpg",
        "data": f"data:image/jpeg;base64,{b64_str}",
    }

    payload = provider._build_payload(
        prompt="Analyze attached file(s)",
        attachments=[attachment],
    )

    parts = payload["contents"][0]["parts"]
    assert len(parts) == 2
    inline_part = parts[0]["inline_data"]
    assert inline_part["mime_type"] == "image/jpeg", "image/jpg must be normalized to image/jpeg"
    assert "\n" not in inline_part["data"]
    assert "\r" not in inline_part["data"]


@pytest.mark.asyncio
async def test_stream_chat_response_with_image_attachment():
    """Verify stream_chat_response handles image attachment without OCR OOM in production."""
    img = Image.new("RGB", (400, 300), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

    request = ChatRequest(
        message="Analyze attached file(s)",
        user_id="test-user-id",
        attachments=[{
            "name": "WhatsApp Image 2026.jpeg",
            "mime_type": "image/jpeg",
            "data": b64_img,
        }],
        stream=True,
    )

    with patch("app.core.config.settings.ENVIRONMENT", "production"), \
         patch("app.core.config.settings.ENABLE_LOCAL_OCR", False):
        service = ChatService()
        events = []
        async for evt_str in service.stream_chat_response(request):
            events.append(evt_str)

        assert len(events) > 0
        # Verify status event was emitted
        assert any("Analyzing visual input" in e for e in events)
        # Verify no unhandled crash occurred
        assert not any("502" in e for e in events)
