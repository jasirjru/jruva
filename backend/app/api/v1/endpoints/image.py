"""Image Generation API endpoint for FLUX & SDXL synthesis across reliability tiers."""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.endpoints.sessions import get_session_owner_id
from app.core.config import settings
from app.core.security import limiter
from app.services.image.genuine_service import private_image_url
from app.services.image.image_service import image_service
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store

router = APIRouter()
IDEMPOTENCY_HEARTBEAT_SECONDS = 30


class ImageGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=2, max_length=2000, description="Image description prompt")
    model: str = Field(default="flux", max_length=100, description="Model engine: flux, turbo, sdxl")
    tier: str = Field(default="B", max_length=10, description="Reliability tier: D, C, B, A, S, S++, S+++")
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int | None = Field(default=None)
    enhance: bool = Field(default=True)
    style: str = Field(default="photorealistic", max_length=50, description="Style preset: photorealistic, cinematic, architecture, cyberpunk, anime")


class ImageGenerateResponse(BaseModel):
    success: bool = True
    image_url: str
    original_prompt: str
    enhanced_prompt: str
    model: str
    seed: int
    width: int
    height: int
    tier: str = "B"
    cached: bool = False
    failover: bool = False
    title: str | None = None
    description: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    image_id: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
    operation_id: str | None = None


class ImageUpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_url: str = Field(..., max_length=500, description="Image URL or static path to upscale")
    scale: int = Field(default=2, ge=2, le=4, description="Upscaling multiplier: 2 or 4")


class ImageUpscaleResponse(BaseModel):
    success: bool
    upscaled_url: str | None = None
    original_url: str | None = None
    original_width: int | None = None
    original_height: int | None = None
    upscaled_width: int | None = None
    upscaled_height: int | None = None
    scale: int | None = None
    latency_ms: int | None = None
    cached: bool = False
    error: str | None = None


@router.post("/generate", response_model=ImageGenerateResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def generate_image(request: Request, payload: ImageGenerateRequest, owner_id: str = Depends(get_session_owner_id)):
    """Synthesize high-fidelity image with active validation, idempotency, and multi-tier reliability."""
    import hashlib
    import json

    from app.services.image.budget import (
        active_in_flight_generation,
        claim_idempotency_lease,
        complete_idempotency,
        fail_idempotency,
        generation_operation,
        get_idempotency_record,
        heartbeat_idempotency_lease,
    )
    from app.services.image.providers import ImageProviderError

    payload_dict = payload.model_dump()
    payload_bytes = json.dumps(payload_dict, sort_keys=True).encode()
    request_hash = hashlib.sha256(payload_bytes).hexdigest()
    # Legacy clients omit a key. Give identical owned requests a durable
    # identity so losing the first response cannot trigger fresh inference.
    # A caller supplies a fresh key when it explicitly intends a new image.
    idempotency_key = request.headers.get("Idempotency-Key") or f"implicit:{request_hash}"
    if len(idempotency_key) > 128:
        raise HTTPException(400, "Idempotency key must be at most 128 characters.")
    claim_fence: int | None = None
    claim_worker: str | None = None
    if idempotency_key:
        try:
            cached_result, claim_fence, claim_worker = await asyncio.to_thread(
                claim_idempotency_lease, owner_id, idempotency_key, request_hash
            )
            if cached_result is not None:
                return ImageGenerateResponse(**cached_result)
        except ImageProviderError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc))

    async def _execute_generation():
        try:
            result = await image_service.generate_and_validate(
                prompt=payload.prompt,
                model=payload.model,
                width=payload.width,
                height=payload.height,
                seed=payload.seed,
                enhance=payload.enhance,
                tier=payload.tier,
                style=payload.style,
                owner_id=owner_id,
            )
            if not result.get("success"):
                if idempotency_key:
                    await asyncio.to_thread(
                        fail_idempotency, owner_id, idempotency_key, claim_worker, claim_fence
                    )
                return JSONResponse(status_code=result.get("status_code", 503), content=result)

            if idempotency_key and result.get("success"):
                completed = await asyncio.to_thread(
                    complete_idempotency, owner_id, idempotency_key, result, claim_worker, claim_fence
                )
                if not completed:
                    # Fence the response and compensate only this attempt's
                    # owned image. This is not atomic with object storage.
                    image_id = result.get("image_id")
                    record = await asyncio.to_thread(workspace_store.get_image, image_id, owner_id) if image_id else None
                    if record:
                        await asyncio.to_thread(storage_service.delete_verified, record["file_url"])
                        await asyncio.to_thread(workspace_store.delete_image, image_id)
                    raise HTTPException(409, "Generation lease ownership was lost. Retry with the same idempotency key.")

            return ImageGenerateResponse(**result)
        except (Exception, asyncio.CancelledError):
            if idempotency_key:
                try:
                    await asyncio.to_thread(
                        fail_idempotency, owner_id, idempotency_key, claim_worker, claim_fence
                    )
                except Exception:
                    pass
            raise

    try:
        if idempotency_key:
            async def _renew_lease():
                while True:
                    await asyncio.sleep(IDEMPOTENCY_HEARTBEAT_SECONDS)
                    renewed = await asyncio.to_thread(
                        heartbeat_idempotency_lease, owner_id, idempotency_key, claim_worker, claim_fence
                    )
                    if not renewed:
                        # A successful completion may commit before its thread
                        # returns to the event loop. That is not ownership loss.
                        record = await asyncio.to_thread(get_idempotency_record, owner_id, idempotency_key)
                        if record and record["status"] == "completed" and record["worker_id"] == claim_worker and record["fencing_token"] == claim_fence:
                            return
                        raise HTTPException(409, "Generation lease ownership was lost. Retry with the same idempotency key.")

            with active_in_flight_generation(owner_id, idempotency_key), generation_operation(owner_id, idempotency_key, claim_worker, claim_fence):
                generation = asyncio.create_task(_execute_generation())
                heartbeat = asyncio.create_task(_renew_lease())
                try:
                    done, _ = await asyncio.wait((generation, heartbeat), return_when=asyncio.FIRST_COMPLETED)
                    if generation in done:
                        return await generation
                    if heartbeat in done:
                        await heartbeat
                    return await generation
                finally:
                    for task in (generation, heartbeat):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(generation, heartbeat, return_exceptions=True)
        else:
            return await _execute_generation()
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Image generation or storage failed. Please retry.") from None


@router.get("/gallery", summary="List historical generated images")
async def get_gallery(limit: int = Query(50,ge=1,le=100), owner_id: str = Depends(get_session_owner_id)):
    """Retrieve user image gallery with prompt and style metadata."""
    from app.services.workspace_store import workspace_store
    try:
        rows = await asyncio.to_thread(workspace_store.list_images, user_id=owner_id, limit=limit)
    except Exception:
        raise HTTPException(status_code=503, detail="Image gallery is temporarily unavailable. Please retry.") from None
    return {"images": [{**row,'file_url':private_image_url(row['id'])} for row in rows if not str(row['file_url']).startswith(('/static/','http:','https:'))]}


@router.get('/{image_id}/content',summary='Read an owned generated image')
async def get_image_content(image_id: str, owner_id: str = Depends(get_session_owner_id)):
    row = await asyncio.to_thread(workspace_store.get_image,image_id,owner_id)
    if not row or str(row['file_url']).startswith(('/static/','http:','https:')):
        raise HTTPException(404,'Image not found')
    try:
        data = await asyncio.to_thread(storage_service.read_bytes,row['file_url'])
    except FileNotFoundError:
        raise HTTPException(404,'Image not found') from None
    except Exception:
        raise HTTPException(503,'Image storage is unavailable.') from None
    return Response(data,media_type='image/png',headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff','Content-Disposition':'inline; filename="generated.png"'})


@router.delete("/{image_id}", summary="Delete an image record")
async def delete_image(image_id: str, owner_id: str = Depends(get_session_owner_id)):
    """Delete image from user gallery."""
    from app.services.workspace_store import workspace_store
    record = await asyncio.to_thread(workspace_store.get_image,image_id,owner_id)
    if not record:
        raise HTTPException(404, "Image not found for this account")
    if not str(record['file_url']).startswith(('/static/','http:','https:')):
        try:
            await asyncio.to_thread(storage_service.delete_verified,record['file_url'])
        except Exception:
            raise HTTPException(503,'Image deletion could not be completed. Please retry.') from None
    success = await asyncio.to_thread(workspace_store.delete_image,image_id)
    if not success:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"success": True}


@router.post("/upscale", response_model=ImageUpscaleResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def upscale_image(request: Request, payload: ImageUpscaleRequest, owner_id: str = Depends(get_session_owner_id)):
    """Super-resolve and upscale image to True 4K resolution with micro-detail enhancement."""
    from app.services.image.upscaler import upscaler_service
    res = await upscaler_service.upscale(payload.image_url, scale=payload.scale, owner_id=owner_id)
    if not res.get("success"):
        status_code = res.get("status_code", 400)
        raise HTTPException(status_code=status_code, detail=res.get("error", "Image upscaling failed"))
    return ImageUpscaleResponse(**res)


@router.get("/styles")
async def get_styles():
    """Retrieve catalog of supported visual style directives."""
    from app.services.image.prompt_enhancer import STYLE_DIRECTIVES
    return {"styles": list(STYLE_DIRECTIVES.keys())}


@router.get("/health")
async def get_image_health():
    """Retrieve health status of image providers."""
    return {'enabled':settings.IMAGE_GENERATION_ENABLED,'free_plan_verified':settings.IMAGE_CF_FREE_PLAN_VERIFIED,
            'klein_enabled':settings.IMAGE_CF_KLEIN_ENABLED and settings.IMAGE_CF_KLEIN_FREE_VERIFIED,
            'live_inference_verified':False}

