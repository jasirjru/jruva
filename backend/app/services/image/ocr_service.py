"""Local High-Performance Optical Character Recognition (OCR) Service.

Uses ONNX-accelerated RapidOCR (PaddleOCR v4 architecture) and pypdfium2
for zero-latency local document and image text extraction.
Supports:
- Scanned PDF pages and document photos (marks sheets, notices, fee receipts)
- PNG, JPG, JPEG, WEBP, BMP, TIFF image text recognition
- Automatic image orientation and multi-column document parsing
- Thread-safe lazy initialization and LRU byte hash caching
"""
import hashlib
import io
import logging
import time

import numpy as np
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

_OCR_ENGINE = None


def get_ocr_engine():
    """Lazily instantiate RapidOCR engine with ONNX runtime if safe and enabled."""
    global _OCR_ENGINE
    from app.core.config import settings

    # The production policy applies even if an engine has already been cached.
    is_prod = (settings.ENVIRONMENT or "").lower().strip() == "production"
    if is_prod and not settings.ENABLE_LOCAL_OCR:
        return None
    if _OCR_ENGINE is None:
        try:
            from app.core.config import settings

            # 1. In production (e.g. Render Free Tier 512MB RAM limit), loading ONNX
            # models into RAM triggers immediate Linux kernel OOM kills.
            # Visual analysis is safely handled via multimodal cloud vision (Gemini Flash).
            is_prod = (getattr(settings, "ENVIRONMENT", "") or "").lower().strip() == "production"
            enable_local = getattr(settings, "ENABLE_LOCAL_OCR", False)
            if is_prod:
                if not enable_local:
                    logger.info(
                        "[OCRService] Local RapidOCR disabled in production to protect 512MB RAM ceiling. "
                        "Delegating visual analysis to multimodal cloud vision."
                    )
                    return None

                # 2. Dynamic RSS memory safety ceiling check (380 MiB safeguard) if enabled in prod
                try:
                    import os

                    import psutil
                    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    max_rss = getattr(settings, "MAX_PROCESS_RSS_MB", 380.0)
                    if rss_mb > max_rss:
                        logger.warning(
                            f"[OCRService] Current RSS ({rss_mb:.1f} MiB) exceeds safety ceiling ({max_rss:.1f} MiB). "
                            "Skipping RapidOCR initialization to prevent OOM."
                        )
                        return None
                except Exception:
                    pass

            from rapidocr_onnxruntime import RapidOCR
            logger.info("Initializing ONNX-accelerated RapidOCR engine...")
            _OCR_ENGINE = RapidOCR()
        except Exception as exc:
            logger.error(f"Failed to initialize RapidOCR: {exc}")
            _OCR_ENGINE = None
    return _OCR_ENGINE


# In-memory LRU cache for OCR results: sha256 -> (text, confidence)
_OCR_CACHE: dict[str, str] = {}
_MAX_CACHE_SIZE = 256


class OCRService:
    """Enterprise OCR extraction engine for images and scanned PDF documents."""

    @staticmethod
    def _preprocess_image_bytes(image_bytes: bytes) -> bytes | None:
        """Enhance low-contrast, skewed, or dim images using OpenCV CLAHE & PIL."""
        try:
            from app.services.image.visual_analysis_service import visual_analysis_service
            enhanced_bytes, meta = visual_analysis_service.enhance_image(image_bytes)
            if enhanced_bytes and len(enhanced_bytes) > 32:
                return enhanced_bytes

            from PIL import Image, ImageEnhance, ImageOps
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = ImageOps.autocontrast(img, cutoff=1)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            logger.warning(
                f"[SilentFailureGuard] Image preprocessing failed: {exc}",
                exc_info=True,
            )
            telemetry_service.record_silent_failure("visual_analysis", str(exc))
            return None

    @staticmethod
    def extract_text_from_image_bytes(image_bytes: bytes, min_confidence: float = 0.5) -> str:
        """Extract textual content from raw image bytes (PNG, JPG, WEBP, BMP, etc.)."""
        if not image_bytes or len(image_bytes) < 32:
            return ""

        content_hash = hashlib.sha256(image_bytes).hexdigest()
        if content_hash in _OCR_CACHE:
            return _OCR_CACHE[content_hash]

        engine = get_ocr_engine()
        if not engine:
            return ""

        try:
            t0 = time.perf_counter()
            # RapidOCR accepts raw bytes directly
            results, elapse = engine(image_bytes)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # If no text detected, try with enhanced preprocessed image
            if not results:
                preprocessed = OCRService._preprocess_image_bytes(image_bytes)
                if preprocessed:
                    results, elapse = engine(preprocessed)

            if not results:
                logger.info(f"RapidOCR completed in {elapsed_ms:.1f}ms: 0 text lines detected.")
                return ""

            # Sort items spatially: primary by y (vertical), secondary by x (horizontal)
            # Each item format: [box_coordinates, text, confidence_score]
            valid_items = []
            for item in results:
                text = item[1].strip() if len(item) > 1 else ""
                try:
                    conf = float(item[2]) if len(item) > 2 else 1.0
                except (ValueError, TypeError):
                    conf = 1.0

                if text and conf >= min_confidence:
                    box = item[0] if len(item) > 0 else None
                    # Calculate center y and left x for spatial layout
                    if box and len(box) >= 4:
                        y_center = sum(pt[1] for pt in box) / 4.0
                        x_left = min(pt[0] for pt in box)
                        height = max(pt[1] for pt in box) - min(pt[1] for pt in box)
                    else:
                        y_center, x_left, height = 0.0, 0.0, 10.0
                    valid_items.append({"text": text, "conf": conf, "y": y_center, "x": x_left, "h": height})

            if not valid_items:
                return ""

            # Cluster items into horizontal rows based on vertical line proximity
            avg_height = sum(it["h"] for it in valid_items) / len(valid_items)
            row_tolerance = max(8.0, avg_height * 0.6)

            valid_items.sort(key=lambda it: it["y"])
            rows = []
            current_row = []
            current_row_y = None

            for it in valid_items:
                if current_row_y is None:
                    current_row = [it]
                    current_row_y = it["y"]
                elif abs(it["y"] - current_row_y) <= row_tolerance:
                    current_row.append(it)
                    # update rolling average of row y
                    current_row_y = sum(x["y"] for x in current_row) / len(current_row)
                else:
                    rows.append(current_row)
                    current_row = [it]
                    current_row_y = it["y"]
            if current_row:
                rows.append(current_row)

            # Sort items in each row from left to right and join with tab or spaces
            formatted_lines = []
            for row in rows:
                row.sort(key=lambda it: it["x"])
                # If multiple items in a row, format as tabular row
                if len(row) > 1:
                    row_text = "  |  ".join(it["text"] for it in row)
                else:
                    row_text = row[0]["text"]
                formatted_lines.append(row_text)

            extracted = "\n".join(formatted_lines)
            logger.info(
                f"RapidOCR extracted {len(formatted_lines)} structured lines ({len(extracted)} chars) "
                f"in {elapsed_ms:.1f}ms"
            )

            if len(_OCR_CACHE) >= _MAX_CACHE_SIZE:
                _OCR_CACHE.pop(next(iter(_OCR_CACHE)))
            _OCR_CACHE[content_hash] = extracted
            return extracted

        except Exception as exc:
            logger.warning(f"OCR extraction failed: {exc}")
            return ""

    @staticmethod
    def extract_text_from_scanned_pdf_page(
        pdf_bytes: bytes,
        page_index: int = 0,
        render_scale: float = 1.8
    ) -> str:
        """Render a specific PDF page to an image and perform OCR extraction if enabled."""
        # Check OCR engine availability upfront before expensive page rasterization
        engine = get_ocr_engine()
        if not engine:
            return ""

        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_bytes)
            if page_index < 0 or page_index >= len(pdf):
                return ""

            page = pdf[page_index]
            # Render page to PIL image at 1.8x scale (approx 130 DPI, high OCR fidelity)
            pil_image = page.render(scale=render_scale).to_pil()

            # Convert PIL to numpy RGB array
            img_arr = np.array(pil_image.convert("RGB"))

            results, elapse = engine(img_arr)
            if not results:
                return ""

            lines = [item[1].strip() for item in results if len(item) > 1 and item[1].strip()]
            return "\n".join(lines)

        except Exception as exc:
            logger.warning(f"Failed to OCR PDF page {page_index}: {exc}")
            return ""


ocr_service = OCRService()
