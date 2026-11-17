"""Bounded Pillow resampling of owned stored images. Not learned super-resolution."""

import asyncio
import io
import re
import time

from app.services.image.genuine_service import save_owned_image
from app.services.image.providers import ImageProviderError, validated_png
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store
from PIL import Image


class SuperResolutionService:
    async def _fetch_image_bytes(self, image_source):
        # Legacy callers cannot load arbitrary files or perform network requests.
        return None

    async def upscale(self, image_source, scale=2, owner_id=None, **kwargs):
        started = time.perf_counter()
        match = re.fullmatch(r"/api/v1/image/([a-fA-F0-9-]{36})/content", image_source)
        image_id = match.group(1) if match else image_source
        if not re.fullmatch(r"[a-fA-F0-9-]{36}", image_id) or scale not in (2, 4):
            return {
                "success": False,
                "error": "Select an owned gallery image and a 2× or 4× scale.",
            }
        if not owner_id:
            return {"success": False, "error": "Sign in to upscale your images."}
        record = await asyncio.to_thread(workspace_store.get_image, image_id, owner_id)
        if not record or str(record["file_url"]).startswith(
            ("/static/", "http:", "https:")
        ):
            return {"success": False, "error": "Image not found."}
        try:
            from app.core.config import settings
            from app.services.image.budget import reserve_media_write_budget
            from app.services.image.genuine_service import get_image_admission_semaphore

            await asyncio.to_thread(
                reserve_media_write_budget,
                owner_id,
                getattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 5),
            )

            sem = get_image_admission_semaphore(2)
            async with sem:
                raw = await asyncio.to_thread(
                    storage_service.read_bytes, record["file_url"]
                )
                png, width, height = await asyncio.to_thread(validated_png, raw)
                if width * height * scale * scale > 2048 * 2048:
                    raise ImageProviderError(
                        "image_too_large",
                        "Resized image exceeds the 4-megapixel limit.",
                        400,
                    )

                def resize():
                    with Image.open(io.BytesIO(png)) as image:
                        output = io.BytesIO()
                        image.resize(
                            (width * scale, height * scale), Image.Resampling.LANCZOS
                        ).save(output, "PNG")
                        return output.getvalue()

                result = await save_owned_image(
                    await asyncio.to_thread(resize),
                    owner_id,
                    record["prompt"],
                    record.get("enhanced_prompt") or record["prompt"],
                    "pillow/lanczos",
                    record["tier"],
                    record["style"],
                )
            return {
                "success": True,
                "upscaled_url": result["image_url"],
                "original_url": image_source,
                "original_width": width,
                "original_height": height,
                "upscaled_width": result["width"],
                "upscaled_height": result["height"],
                "scale": scale,
                "cached": False,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        except ImageProviderError as exc:
            return {"success": False, "status_code": exc.status, "error": str(exc)}
        except Exception:  # noqa: BLE001 -- sanitize storage failure details at API boundary
            return {
                "success": False,
                "status_code": 503,
                "error": "Image resampling or storage failed. Please retry.",
            }


upscaler_service = SuperResolutionService()
