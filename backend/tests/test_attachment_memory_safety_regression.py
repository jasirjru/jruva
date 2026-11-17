import io
from unittest.mock import MagicMock, patch

import cv2
import pytest
from app.schemas.chat import AttachmentItem
from app.services.image.ocr_service import OCRService
from app.services.image.visual_analysis_service import (
    MAX_IMAGE_DIM,
    MAX_IMAGE_PIXELS,
    VisualAnalysisService,
)
from app.services.llm.router import LLMProviderException, LLMRouter
from PIL import Image


def test_ocr_service_avoids_pdf_rasterization_when_ocr_disabled():
    """Verify that when local OCR is disabled (e.g. in production), PDF page rasterization is skipped."""
    service = OCRService()

    with patch("app.services.image.ocr_service.get_ocr_engine", return_value=None) as mock_get_engine, \
         patch("pypdfium2.PdfDocument", create=True) as mock_pdfium:

        result = service.extract_text_from_scanned_pdf_page(b"%PDF-1.4 dummy content", page_index=0)

        # Must return empty string
        assert result == ""
        # Must have checked the engine first
        mock_get_engine.assert_called_once()
        # MUST NOT have called pdfium to rasterize pages
        mock_pdfium.assert_not_called()


def test_visual_analysis_normal_jpeg_and_png():
    """Verify that safe decoding succeeds with normal valid JPEG and PNG images."""
    service = VisualAnalysisService()

    # Normal JPEG
    img_jpg = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf_jpg = io.BytesIO()
    img_jpg.save(buf_jpg, format="JPEG", quality=90)
    decoded_jpg = service._safe_decode_and_downscale(buf_jpg.getvalue(), max_dim=1280)
    assert decoded_jpg is not None
    assert decoded_jpg.shape == (480, 640, 3)

    # Normal PNG
    img_png = Image.new("RGB", (400, 300), color=(50, 200, 50))
    buf_png = io.BytesIO()
    img_png.save(buf_png, format="PNG")
    decoded_png = service._safe_decode_and_downscale(buf_png.getvalue(), max_dim=1280)
    assert decoded_png is not None
    assert decoded_png.shape == (300, 400, 3)

    # Grayscale decoding
    decoded_gray = service._safe_decode_and_downscale(buf_jpg.getvalue(), max_dim=1280, grayscale=True)
    assert decoded_gray is not None
    assert decoded_gray.shape == (480, 640)


def test_visual_analysis_safe_decode_and_downscaling():
    """Verify that high-resolution images within bounds are safely downscaled to target max_dim."""
    service = VisualAnalysisService()

    # 4000x3000 is 12 MP (fits under 20 MP budget)
    large_img = Image.new("RGB", (4000, 3000), color=(128, 128, 128))
    buf = io.BytesIO()
    large_img.save(buf, format="JPEG", quality=80)
    large_bytes = buf.getvalue()

    decoded = service._safe_decode_and_downscale(large_bytes, max_dim=1280)
    assert decoded is not None
    # Maximum dimension must be clamped to <= 1280
    assert max(decoded.shape[0], decoded.shape[1]) <= 1280


def test_visual_analysis_rejects_oversized_dimensions():
    """Verify that images exceeding MAX_IMAGE_DIM on any edge are rejected upfront."""
    service = VisualAnalysisService()

    # Create mock image whose header reports dimensions above MAX_IMAGE_DIM
    oversized_w = MAX_IMAGE_DIM + 1
    oversized_h = 1000

    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    with patch.object(Image, "open") as mock_open:
        mock_pil = MagicMock()
        mock_pil.format = "JPEG"
        mock_pil.size = (oversized_w, oversized_h)
        mock_open.return_value.__enter__.return_value = mock_pil

        result = service._safe_decode_and_downscale(raw_bytes, max_dim=1280)
        assert result is None, "Should reject images exceeding MAX_IMAGE_DIM"


def test_visual_analysis_rejects_excessive_pixel_count():
    """Verify that images exceeding MAX_IMAGE_PIXELS are rejected even if edges are below MAX_IMAGE_DIM."""
    service = VisualAnalysisService()

    # 4800 x 4500 = 21,600,000 pixels (> MAX_IMAGE_PIXELS of 20,000,000)
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    with patch.object(Image, "open") as mock_open:
        mock_pil = MagicMock()
        mock_pil.format = "JPEG"
        mock_pil.size = (4800, 4500)
        mock_open.return_value.__enter__.return_value = mock_pil

        result = service._safe_decode_and_downscale(raw_bytes, max_dim=1280)
        assert result is None, f"Should reject images exceeding MAX_IMAGE_PIXELS ({MAX_IMAGE_PIXELS})"


def test_visual_analysis_rejects_malicious_dimensions():
    """Verify that zero or negative dimensions are safely rejected."""
    service = VisualAnalysisService()

    raw_bytes = b"\xff\xd8\xff\xe0"  # Minimal JPEG header stub

    with patch.object(Image, "open") as mock_open:
        mock_pil = MagicMock()
        mock_pil.format = "JPEG"
        mock_pil.size = (0, 1000)
        mock_open.return_value.__enter__.return_value = mock_pil

        result = service._safe_decode_and_downscale(raw_bytes, max_dim=1280)
        assert result is None


def test_visual_analysis_rejects_decompression_bomb():
    """Verify that DecompressionBombError is caught safely without leaking memory or crashing."""
    service = VisualAnalysisService()

    raw_bytes = b"fake-decompression-bomb"

    with patch.object(Image, "open", side_effect=Image.DecompressionBombError("Too many pixels")):
        result = service._safe_decode_and_downscale(raw_bytes, max_dim=1280)
        assert result is None


def test_visual_analysis_rejects_corrupted_files():
    """Verify that corrupted data, random bytes, and empty inputs return None gracefully."""
    service = VisualAnalysisService()

    # Empty payload
    assert service._safe_decode_and_downscale(b"") is None

    # Random noise
    assert service._safe_decode_and_downscale(b"not-an-image-data-payload") is None

    # Truncated header
    assert service._safe_decode_and_downscale(b"\x89PNG\r\n\x1a\n\x00\x00\x00") is None


def test_visual_analysis_rejects_unsupported_formats():
    """Verify that unsupported image formats are rejected."""
    service = VisualAnalysisService()

    raw_bytes = b"fake-bytes"

    with patch.object(Image, "open") as mock_open:
        mock_pil = MagicMock()
        mock_pil.format = "PSD"  # Unsupported format
        mock_pil.size = (500, 500)
        mock_open.return_value.__enter__.return_value = mock_pil

        result = service._safe_decode_and_downscale(raw_bytes, max_dim=1280)
        assert result is None


def test_visual_analysis_zero_cv2_imdecode_fallback():
    """Verify that cv2.imdecode is NEVER called when Pillow fails, eliminating unsafe fallback spikes."""
    service = VisualAnalysisService()

    with patch.object(cv2, "imdecode") as mock_imdecode:
        # Corrupted input
        res1 = service._safe_decode_and_downscale(b"corrupted bytes")
        assert res1 is None

        # Decompression bomb input
        with patch.object(Image, "open", side_effect=Image.DecompressionBombError("Bomb")):
            res2 = service._safe_decode_and_downscale(b"bomb bytes")
            assert res2 is None

        # cv2.imdecode must never be invoked
        mock_imdecode.assert_not_called()


def test_model_router_strict_multimodal_isolation():
    """Verify that visual requests strictly route to Gemini and never silently fallback to text-only models."""
    mock_gemini = MagicMock()
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True

    mock_groq = MagicMock()
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True

    router = LLMRouter(gemini_provider=mock_gemini, groq_provider=mock_groq)

    # Create a visual attachment mock
    attachment = AttachmentItem(
        name="test.jpg",
        mime_type="image/jpeg",
        data="ZmFrZQ==",
    )

    # When Gemini is ready
    primary, backup, target_model = router._resolve_providers(
        requested_model="auto",
        attachments=[attachment],
    )
    assert primary == mock_gemini
    assert backup is None, "Backup MUST be None for visual requests to prevent silent text-only fallback"
    assert target_model == "gemini-3.6-flash"

    # When Gemini is unavailable
    mock_gemini.is_available = False
    with pytest.raises(LLMProviderException) as exc_info:
        router._resolve_providers(
            requested_model="auto",
            attachments=[attachment],
        )
    assert "vision-capable AI model" in str(exc_info.value)
