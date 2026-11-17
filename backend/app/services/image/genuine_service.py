"""One generation path shared by chat and Vision Studio; private owned media only."""

import asyncio
import hashlib
import json
import os
import random
import time
import urllib.parse

from app.core.config import settings
from app.services.image.budget import (
    begin_external_dispatch,
    claim_idempotency_lease,
    complete_idempotency,
    current_generation_lease,
    fail_idempotency,
    record_external_outcome,
    reserve_image_budget,
)
from app.services.image.providers import (
    KLEIN,
    CloudflareImageProvider,
    ImageProviderError,
    validated_png,
)
from app.services.storage import storage_service
from app.services.tenant_access import active_account
from app.services.workspace_store import workspace_store

_IMAGE_SEMAPHORE: asyncio.Semaphore | None = None
_IMAGE_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


def get_image_admission_semaphore(limit: int = 2) -> asyncio.Semaphore:
    """Enforce a strict concurrency admission ceiling for image processing (512MB RAM safe)."""
    global _IMAGE_SEMAPHORE, _IMAGE_SEMAPHORE_LOOP
    try:
        curr_loop = asyncio.get_running_loop()
    except RuntimeError:
        curr_loop = None
    if _IMAGE_SEMAPHORE is None or _IMAGE_SEMAPHORE_LOOP is not curr_loop:
        _IMAGE_SEMAPHORE = asyncio.Semaphore(limit)
        _IMAGE_SEMAPHORE_LOOP = curr_loop
    return _IMAGE_SEMAPHORE


def _is_local_storage() -> bool:
    val = (os.getenv("STORAGE_BACKEND") or getattr(settings, "STORAGE_BACKEND", "local")).lower().strip()
    return val == "local"


def private_image_url(image_id):
    return f"/api/v1/image/{image_id}/content"


async def _observe_mutation(task):
    """Drain a real thread result even when cancellation is requested repeatedly."""
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.done():
                return task.result(), True
            cancelled = True


async def save_owned_image(data, owner, prompt, enhanced, model, tier, style, operation_id=None):
    if not active_account(owner):
        raise ImageProviderError("authentication_required", "Sign in to generate or access images.", 401)
    png, width, height = await asyncio.to_thread(validated_png, data)
    if settings.ENVIRONMENT == "production" and _is_local_storage():
        raise ImageProviderError("storage_unavailable", "Durable image storage is not configured.")
    from app.services.image.storage_quota import (
        begin_storage_write,
        clear_reservation_context,
        mark_storage_cleanup,
        release_storage_quota,
        reserve_storage_quota,
    )
    from app.services.storage.base import image_write_scope

    reservation, cancelled = await _observe_mutation(
        asyncio.create_task(
            asyncio.to_thread(
                reserve_storage_quota, owner, len(png), operation_id=operation_id, require_active_owner=True
            )
        )
    )
    if cancelled:
        await _observe_mutation(
            asyncio.create_task(asyncio.to_thread(release_storage_quota, owner, len(png), reservation))
        )
        raise asyncio.CancelledError()
    path, record = None, None
    writing = False
    write_attempted, write_confirmed = False, False

    def write():
        with image_write_scope(reservation):
            return storage_service.save_bytes("generated.png", png)

    try:
        path = await asyncio.to_thread(storage_service.reserved_image_path, reservation)
        _, cancelled = await _observe_mutation(
            asyncio.create_task(
                asyncio.to_thread(begin_storage_write, owner, reservation, path, require_active_owner=True)
            )
        )
        writing = True
        if cancelled:
            raise asyncio.CancelledError()
        write_attempted = True
        write_task = asyncio.create_task(asyncio.to_thread(write))
        saved_path, cancelled = await _observe_mutation(write_task)
        if saved_path != path:
            raise RuntimeError("Unexpected reserved object path")
        write_confirmed = True
        if cancelled:
            raise asyncio.CancelledError()
        record_task = asyncio.create_task(
            asyncio.to_thread(
                workspace_store.record_image,
                prompt=prompt,
                file_url=path,
                model=model,
                enhanced_prompt=enhanced,
                tier=tier,
                style=style,
                width=width,
                height=height,
                user_id=owner,
                bytes=len(png),
                reservation_id=reservation,
                require_active_owner=True,
            )
        )
        try:
            record, cancelled = await _observe_mutation(record_task)
        except Exception as write_error:
            from app.services.image.storage_recovery import resolve_image_metadata_write

            try:
                record, cancelled = await _observe_mutation(asyncio.create_task(asyncio.to_thread(
                    resolve_image_metadata_write, owner, reservation, path, len(png)
                )))
            except Exception:
                raise ImageProviderError(
                    "storage_metadata_indeterminate",
                    "Image metadata outcome is unknown. The object and reservation were retained for recovery.",
                ) from None
            if record is None:
                raise write_error
            cancelled = cancelled or bool(asyncio.current_task().cancelling())
        if cancelled:
            raise asyncio.CancelledError()
        if not active_account(owner):
            raise ImageProviderError("authentication_required", "Your account is no longer available.", 401)
    except (Exception, asyncio.CancelledError) as error:
        from app.services.storage.local import LocalStorageBackend

        if isinstance(error, ImageProviderError) and error.kind == "storage_metadata_indeterminate":
            raise
        if writing and write_attempted and not write_confirmed and not isinstance(storage_service, LocalStorageBackend):
            await _observe_mutation(asyncio.create_task(asyncio.to_thread(mark_storage_cleanup, owner, reservation)))
            raise
        # Observed thread completion precedes cleanup. A failed absence check
        # leaves its receipt counted, rather than falsely refunding capacity.
        if writing:
            await _observe_mutation(asyncio.create_task(asyncio.to_thread(storage_service.delete_verified, path)))
        if record:
            await _observe_mutation(asyncio.create_task(asyncio.to_thread(workspace_store.delete_image, record["id"])))
        else:
            await _observe_mutation(
                asyncio.create_task(
                    asyncio.to_thread(release_storage_quota, owner, len(png), reservation, verified_absent=writing)
                )
            )
        raise
    finally:
        clear_reservation_context()
    return {"image_id": record["id"], "image_url": private_image_url(record["id"]), "width": width, "height": height}


async def generate(
    prompt,
    owner=None,
    model="flux",
    width=1024,
    height=1024,
    seed=None,
    enhance=True,
    tier="B",
    style="photorealistic",
    idempotency_key=None,
):
    from app.services.image.image_service import check_image_safety
    from app.services.image.prompt_enhancer import STYLE_DIRECTIVES

    identity = hashlib.sha256(
        json.dumps(
            {
                "prompt": prompt,
                "model": model,
                "width": width,
                "height": height,
                "seed": seed,
                "enhance": enhance,
                "tier": tier,
                "style": style,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    lease, internal = None, False
    base = {
        "success": False,
        "image_url": "",
        "original_prompt": prompt,
        "enhanced_prompt": prompt,
        "model": "none",
        "seed": seed,
        "width": width,
        "height": height,
        "tier": tier,
        "cached": False,
        "failover": False,
    }
    started = time.perf_counter()
    try:
        safe, reason = check_image_safety(prompt)
        if not safe:
            raise ImageProviderError(
                "content_filter",
                reason or "Image request violates the content policy.",
                400,
            )
        if not active_account(owner):
            raise ImageProviderError("authentication_required", "Sign in to generate images.", 401)
        if settings.ENVIRONMENT == "production" and _is_local_storage():
            raise ImageProviderError("storage_unavailable", "Durable image storage is not configured.")
        if model not in (
            "flux",
            "auto",
            "schnell",
            "klein",
            KLEIN,
            "@cf/black-forest-labs/flux-1-schnell",
        ):
            raise ImageProviderError("unsupported_model", "Choose an approved Cloudflare image model.", 400)
        provider = CloudflareImageProvider(settings)
        candidates = provider.available_models()
        if model in ("schnell", "@cf/black-forest-labs/flux-1-schnell"):
            candidates = [candidates[-1]]
        if model in ("klein", KLEIN) and KLEIN not in candidates:
            raise ImageProviderError(
                "provider_not_configured",
                "Klein free eligibility and schema have not been verified.",
            )
        enhanced = (prompt + ". " + STYLE_DIRECTIVES.get(style, ""))[:2048] if enhance else prompt
        base["enhanced_prompt"] = enhanced
        last_error = None
        sem = get_image_admission_semaphore(2)
        async with sem:
            lease = current_generation_lease()
            if lease is None:
                internal = True
                key = idempotency_key or f"shared:{identity}"
                cached, fence, worker = await asyncio.to_thread(claim_idempotency_lease, owner, key, identity)
                if cached is not None:
                    return cached
                lease = (owner, key, worker, fence)
            for index, candidate in enumerate(candidates):
                # 1024 square: Klein output 4*26.05 neurons; Schnell 4*4.8+4*9.6.
                # Round conservatively; reserve every attempt, including timeout/fallback.
                await asyncio.to_thread(
                    reserve_image_budget,
                    settings.CLOUDFLARE_ACCOUNT_ID,
                    owner,
                    106 if candidate == KLEIN else 120,
                    settings.IMAGE_DAILY_NEURON_LIMIT,
                    settings.IMAGE_USER_DAILY_ATTEMPTS,
                )
                operation_id = await asyncio.to_thread(begin_external_dispatch, *lease, candidate)
                base["operation_id"] = operation_id
                try:
                    result = await provider.generate(enhanced, seed, candidate)
                except ImageProviderError as exc:
                    await asyncio.to_thread(record_external_outcome, *lease, exc.outcome)
                    if exc.outcome != "confirmed_failure" or not exc.retryable:
                        raise
                    last_error = exc
                    continue
                except (Exception, asyncio.CancelledError):
                    # The committed dispatch marker survives even if this update
                    # cannot execute during shutdown or a database outage.
                    await asyncio.to_thread(record_external_outcome, *lease, "indeterminate")
                    raise
                if not await asyncio.to_thread(record_external_outcome, *lease, "succeeded"):
                    raise ImageProviderError(
                        "operation_indeterminate",
                        "Provider completed but ownership was lost. No generation was repeated.",
                    )
                try:
                    saved = await save_owned_image(
                        result.data, owner, prompt, enhanced, result.model, tier, style, operation_id=operation_id
                    )
                    response = (
                        base
                        | saved
                        | {
                            "success": True,
                            "model": result.model,
                            "provider": result.provider,
                            "failover": index > 0,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                        }
                    )
                    if internal and not await asyncio.to_thread(
                        complete_idempotency, lease[0], lease[1], response, lease[2], lease[3]
                    ):
                        raise ImageProviderError(
                            "operation_indeterminate",
                            "Stored result awaits ownership reconciliation. No generation was repeated.",
                        )
                    return response
                except ImageProviderError as exc:
                    last_error = exc
                    # A provider already succeeded; storage errors cannot make
                    # another external inference safe, even if marked retryable.
                    raise
        raise last_error or ImageProviderError("provider_unavailable", "No authorized image provider succeeded.")
    except ImageProviderError as exc:
        if internal and lease:
            await asyncio.to_thread(fail_idempotency, *lease)
        return base | {
            "image_url": f"error:{exc.kind}:{urllib.parse.quote(str(exc))}",
            "error_type": exc.kind,
            "error_message": str(exc),
            "status_code": exc.status,
        }
    except Exception:  # noqa: BLE001 -- never disclose storage/provider exception details
        return base | {
            "image_url": "error:storage_unavailable:Image%20storage%20failed",
            "error_type": "storage_unavailable",
            "error_message": "Image storage failed. No image was published. Please retry.",
            "status_code": 503,
        }
