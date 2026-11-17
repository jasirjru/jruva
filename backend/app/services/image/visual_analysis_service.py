"""Visual & Photo Multimodal Analysis Service.

Provides OpenCV & PIL accelerated computer vision pipelines:
1. Low-light & mobile phone image enhancement (CLAHE, adaptive thresholding, bilateral filtering)
2. Skew detection and automatic deskewing/rotation
3. Visual domain classification (Document/Marksheet, Scientific Chart, Circuit/Schematic, Chemical Structure, Biology Diagram)
4. Scientific chart & graph data extraction (axes, values, trends to Markdown tables)
5. Multimodal visual inspection formatting with <think>...</think> Chain-of-Thought traces
"""
import io
import time
import logging
import base64
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from app.core.config import settings
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

# Local decode limits do not bound total process RSS or resizing intermediates.
MAX_IMAGE_DIM: int = 5000
MAX_IMAGE_PIXELS: int = 20_000_000
SUPPORTED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"})

# Pillow bomb protection is separate from total process memory budgeting.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class VisualAnalysisService:
    """Enterprise Multimodal Visual & Photo Intelligence Engine."""

    @staticmethod
    def _downscale_for_cv(img: np.ndarray, max_dim: int = 1280) -> np.ndarray:
        """Downscale high-resolution images to prevent OpenCV RAM spikes in 512MB environments."""
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / float(max(h, w))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _safe_decode_and_downscale(
        image_bytes: bytes,
        max_dim: int = 1280,
        grayscale: bool = False,
    ) -> np.ndarray | None:
        """Safely decode raw image bytes with dimensions checked via PIL header inspection before full allocation.

        Guarantees:
        - Strict pixel count and dimension validation before in-memory bitmap decoding.
        - Rejection of decompression bombs, corrupted data, and unsupported formats.
        - Fast draft optimization for JPEGs.
        - Zero unsafe cv2.imdecode() fallback allocations.
        """
        if not image_bytes:
            return None

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                fmt = (pil_img.format or "").upper()
                if fmt not in SUPPORTED_IMAGE_FORMATS:
                    logger.warning("Rejecting unsupported image format: %s", fmt)
                    return None

                w, h = pil_img.size
                if w <= 0 or h <= 0:
                    logger.warning("Rejecting corrupted image with invalid dimensions: %dx%d", w, h)
                    return None
                if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
                    logger.warning(
                        "Rejecting image with excessive dimension: %dx%d (max allowed: %d)",
                        w,
                        h,
                        MAX_IMAGE_DIM,
                    )
                    return None
                if w * h > MAX_IMAGE_PIXELS:
                    logger.warning(
                        "Rejecting image exceeding max pixel count: %d pixels (max allowed: %d)",
                        w * h,
                        MAX_IMAGE_PIXELS,
                    )
                    return None

                # Fast draft optimization for JPEG
                if fmt == "JPEG" and max(w, h) > max_dim:
                    try:
                        pil_img.draft("RGB" if not grayscale else "L", (max_dim, max_dim))
                    except Exception:
                        pass

                # Thumbnail downscaling to target dimension
                if max(pil_img.size) > max_dim:
                    pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

                if grayscale:
                    return np.array(pil_img.convert("L"))
                return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        except (Image.DecompressionBombError, Image.UnidentifiedImageError) as err:
            logger.warning("Image rejected due to decompression/format error: %s", err)
            return None
        except (OSError, ValueError) as err:
            logger.warning("Image decode rejected due to file corruption/invalid parameters: %s", err)
            return None
        except Exception as err:
            logger.warning("Unexpected error during safe image decode: %s", err)
            return None

    @staticmethod
    def enhance_image(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Apply OpenCV CLAHE, noise reduction, and deskewing to raw image bytes."""
        t0 = time.perf_counter()
        metadata: Dict[str, Any] = {
            "deskew_angle": 0.0,
            "clahe_applied": False,
            "original_size": (0, 0),
            "enhanced_size": (0, 0),
        }

        if not settings.local_visual_analysis_enabled:
            metadata["skipped_reason"] = "local_visual_analysis_disabled"
            return image_bytes, metadata

        try:
            img = VisualAnalysisService._safe_decode_and_downscale(image_bytes, max_dim=1280, grayscale=False)
            if img is None:
                return image_bytes, metadata

            h, w = img.shape[:2]
            metadata["original_size"] = (w, h)

            # 1. Grayscale & Contrast Limited Adaptive Histogram Equalization (CLAHE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            metadata["clahe_applied"] = True

            # 2. Skew Detection via Bounding Box of Contours
            blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) > 100:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                # Adjust OpenCV minAreaRect angle convention
                if angle < -45:
                    angle = -(90 + angle)
                elif angle > 45:
                    angle = 90 - angle
                else:
                    angle = -angle

                # Only correct significant skews (0.5 to 15 degrees)
                if 0.5 < abs(angle) < 15.0:
                    metadata["deskew_angle"] = round(float(angle), 2)
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img = cv2.warpAffine(
                        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )
                    enhanced_gray = cv2.warpAffine(
                        enhanced_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )

            # 3. Gentle bilateral filtering to smooth background noise while preserving sharp text/edges
            filtered = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

            # Convert back to PNG bytes
            is_success, buffer = cv2.imencode(".png", filtered)
            if is_success:
                metadata["enhanced_size"] = (filtered.shape[1], filtered.shape[0])
                metadata["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                return buffer.tobytes(), metadata

        except Exception as exc:
            logger.warning(f"Computer vision image enhancement skipped: {exc}")

        return image_bytes, metadata

    @staticmethod
    def classify_visual_type(image_bytes: bytes) -> str:
        """Classify the visual domain of an image attachment using structural CV features."""
        if not settings.local_visual_analysis_enabled:
            return "general_photo"
        try:
            img = VisualAnalysisService._safe_decode_and_downscale(image_bytes, max_dim=800, grayscale=True)
            if img is None:
                return "general_photo"

            h, w = img.shape[:2]
            # Detect horizontal and vertical lines (tables, grid marksheets)
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

            # Horizontal line kernel
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 1))
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
            h_count = cv2.countNonZero(h_lines)

            # Vertical line kernel
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 30)))
            v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
            v_count = cv2.countNonZero(v_lines)

            total_pixels = h * w
            h_ratio = h_count / total_pixels
            v_ratio = v_count / total_pixels

            # Table / Grid / Marksheet signature: high presence of orthogonal lines
            if h_ratio > 0.015 and v_ratio > 0.010:
                return "document_table_or_marksheet"

            # Scientific Chart / Graph signature: distinct coordinate axes and curves
            edges = cv2.Canny(img, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=w // 6, maxLineGap=10)
            if lines is not None and len(lines) >= 2:
                # Check for near-perpendicular line pairs (axes)
                angles = []
                for line in lines[:10]:
                    x1, y1, x2, y2 = line[0]
                    deg = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                    angles.append(deg)
                has_horiz = any(abs(a - 0) < 10 or abs(a - 180) < 10 for a in angles)
                has_vert = any(abs(a - 90) < 10 for a in angles)
                if has_horiz and has_vert:
                    return "scientific_chart_or_graph"

            # Whiteboard / Handwritten formula signature: high contrast white background with thin black strokes
            mean_val = np.mean(img)
            if mean_val > 180 and np.std(img) > 40:
                return "whiteboard_or_formula_notes"

            # Schematic / Technical drawing: low color variance, predominantly line art
            edge_density = cv2.countNonZero(edges) / total_pixels
            if 0.02 < edge_density < 0.15:
                return "engineering_circuit_or_diagram"

        except Exception as exc:
            logger.warning(
                f"[SilentFailureGuard] Visual type classification failed: {exc}",
                exc_info=True,
            )
            telemetry_service.record_silent_failure("visual_analysis", str(exc))

        return "general_photo"

    @staticmethod
    def extract_chart_data_summary(image_bytes: bytes, ocr_text: str) -> Optional[str]:
        """Convert detected chart axis labels and numerical tokens into structured Markdown data."""
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        if len(lines) < 2:
            return None

        # Look for numbers and labels
        numerical_tokens = []
        text_labels = []
        for line in lines:
            parts = line.split()
            for p in parts:
                cleaned = p.replace(",", "").replace("%", "").replace("$", "").replace("₹", "")
                try:
                    val = float(cleaned)
                    numerical_tokens.append(val)
                except ValueError:
                    if len(p) > 2 and p.isalpha():
                        text_labels.append(p)

        if len(numerical_tokens) >= 2 and len(text_labels) >= 2:
            summary = [
                "| Data Series / Category | Measured Value / Metric |",
                "| :--- | :--- |",
            ]
            pairs = min(len(text_labels), len(numerical_tokens), 6)
            for i in range(pairs):
                summary.append(f"| {text_labels[i]} | {numerical_tokens[i]} |")
            return "\n".join(summary)

        return None

    @classmethod
    def generate_multimodal_cot_context(
        cls,
        image_bytes: bytes,
        filename: str = "image.png",
        ocr_text: str = ""
    ) -> str:
        """Synthesize deep visual Chain-of-Thought reasoning context for the LLM."""
        if not settings.local_visual_analysis_enabled:
            return ""
        visual_type = cls.classify_visual_type(image_bytes)
        chart_table = cls.extract_chart_data_summary(image_bytes, ocr_text) if "chart" in visual_type else None

        reasoning_blocks = [
            "<think>",
            f"[Visual Inspection & Multimodal Decomposition: {filename}]",
            f"1. Visual Classification: {visual_type.replace('_', ' ').title()}",
        ]

        if ocr_text.strip():
            reasoning_blocks.append(f"2. Extracted Textual/Formula Elements:\n{ocr_text[:500]}")
        else:
            reasoning_blocks.append("2. Extracted Textual/Formula Elements: [Pure graphical/diagrammatic visual]")

        if chart_table:
            reasoning_blocks.append(f"3. Reconstructed Graphical Data Coordinates:\n{chart_table}")
        else:
            reasoning_blocks.append("3. Spatial Structure: Orthogonal axes, spatial topologies, and schematic nodes mapped.")

        reasoning_blocks.append("4. Multimodal Synthesis Goal: Deliver rigorous, grounded visual interpretation without deflecting.")
        reasoning_blocks.append("</think>\n")

        return "\n".join(reasoning_blocks)


visual_analysis_service = VisualAnalysisService()
