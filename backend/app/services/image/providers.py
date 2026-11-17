"""Explicitly enabled image providers. No billing gateway or paid fallback."""

import base64
import binascii
import io
import re
from dataclasses import dataclass

import httpx
from PIL import Image, UnidentifiedImageError

SCHNELL = "@cf/black-forest-labs/flux-1-schnell"
KLEIN = "@cf/black-forest-labs/flux-2-klein-4b"
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 2048 * 2048


class ImageProviderError(Exception):
    def __init__(self, kind, message, status=503, retryable=False, diagnostic="", outcome=None):
        super().__init__(message)
        self.kind, self.status, self.retryable, self.diagnostic = (
            kind,
            status,
            retryable,
            diagnostic,
        )
        self.message = message
        confirmed_rejections = {"provider_not_configured", "free_plan_required", "provider_authentication",
                                "provider_schema", "quota_exhausted", "invalid_prompt", "unsupported_model",
                                "content_filter", "authentication_required", "quota_configuration"}
        self.outcome = outcome or ("confirmed_failure" if kind in confirmed_rejections else "indeterminate")


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    model: str
    provider: str = "cloudflare"


def validated_png(data: bytes) -> tuple[bytes, int, int]:
    """Decode bounded raster bytes and strip metadata; never trust MIME alone."""
    if not data or len(data) > MAX_BYTES:
        raise ImageProviderError(
            "invalid_image", "Provider returned an invalid or oversized image."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format not in {"PNG", "JPEG", "WEBP"}
                or image.width * image.height > MAX_PIXELS
            ):
                raise ValueError("unsupported image")
            width, height = image.size
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, "PNG")
        if len(output.getvalue()) > MAX_BYTES:
            raise ValueError("oversized image")
        return output.getvalue(), width, height
    except (
        ValueError,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageProviderError(
            "invalid_image", "Provider returned undecodable image bytes."
        ) from exc


class CloudflareImageProvider:
    def __init__(self, config, transport=None):
        self.config, self.transport = config, transport

    def available_models(self):
        cfg = self.config
        if not cfg.IMAGE_GENERATION_ENABLED:
            raise ImageProviderError(
                "provider_not_configured",
                "Image generation is not enabled. Configure an approved free provider.",
            )
        if not cfg.IMAGE_CF_FREE_PLAN_VERIFIED:
            raise ImageProviderError(
                "free_plan_required",
                "A verified Workers Free plan is required; paid accounts are not authorized.",
            )
        if not cfg.CLOUDFLARE_API_TOKEN or not re.fullmatch(
            r"[a-fA-F0-9]{32}", cfg.CLOUDFLARE_ACCOUNT_ID
        ):
            raise ImageProviderError(
                "provider_not_configured",
                "Cloudflare image credentials are not configured.",
            )
        models = [SCHNELL]
        if cfg.IMAGE_CF_KLEIN_ENABLED and cfg.IMAGE_CF_KLEIN_FREE_VERIFIED:
            models.insert(0, KLEIN)
        return models

    async def generate(self, prompt, seed, model):
        if model not in self.available_models():
            raise ImageProviderError(
                "provider_not_configured",
                "This image model has not been approved for free use.",
            )
        if not prompt.strip() or len(prompt) > 2048:
            raise ImageProviderError(
                "invalid_prompt", "Image prompt must contain 1–2048 characters.", 400
            )
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.config.CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}"
        headers = {"Authorization": f"Bearer {self.config.CLOUDFLARE_API_TOKEN}"}
        if model == SCHNELL:
            arguments = {"json": {"prompt": prompt, "steps": 4}}
        elif model == KLEIN:
            payload = {"prompt": prompt, "width": 1024, "height": 1024}
            if seed is not None:
                payload["seed"] = seed
            arguments = {"files": {k: (None, str(v)) for k, v in payload.items()}}
        else:
            payload = {"prompt": prompt}
            if seed is not None:
                payload["seed"] = seed
            arguments = {"json": payload | {"steps": 4}}
        dispatched = False
        try:
            async with httpx.AsyncClient(
                timeout=45,
                follow_redirects=False,
                transport=self.transport,
                trust_env=False,
            ) as client:
                # Recheck immediately before inference; an operator flag alone is insufficient.
                # Conservative policy: any account subscription (paid or unknown) blocks.
                # Missing read permission, malformed response or uncertainty also blocks.
                billing_url = f"https://api.cloudflare.com/client/v4/accounts/{self.config.CLOUDFLARE_ACCOUNT_ID}/subscriptions"
                billing = await client.get(billing_url, headers=headers)
                try:
                    subscriptions = billing.json()
                except ValueError:
                    subscriptions = {}
                if (
                    billing.status_code != 200
                    or not isinstance(subscriptions, dict)
                    or subscriptions.get("success") is not True
                    or subscriptions.get("result") != []
                ):
                    raise ImageProviderError(
                        "free_plan_required",
                        "Free-plan verification failed or account subscriptions exist. No image inference was requested.",
                    )
                dispatched = True
                async with client.stream(
                    "POST", url, headers=headers, **arguments
                ) as response:
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_BYTES * 2:
                            raise ImageProviderError(
                                "invalid_image",
                                "Provider response exceeded the size limit.",
                            )
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    try:
                        import json

                        body = json.loads(raw)
                    except (ValueError, UnicodeError):
                        body = {}
                    if not isinstance(body, dict):
                        body = {}
                    errors = body.get("errors") or []
                    codes = (
                        {e.get("code") for e in errors if isinstance(e, dict)}
                        if isinstance(errors, list)
                        else set()
                    )
                    error_details = [
                        str(e.get("message", ""))
                        for e in errors
                        if isinstance(e, dict) and e.get("message")
                    ]
                    status = response.status_code
                    diagnostic = (
                        f"status={status}; codes={sorted(codes)}; "
                        f"details={'; '.join(error_details)}"
                    )
                    api_token = getattr(self.config, "CLOUDFLARE_API_TOKEN", "")
                    account_id = getattr(self.config, "CLOUDFLARE_ACCOUNT_ID", "")
                    if api_token:
                        diagnostic = diagnostic.replace(api_token, "[REDACTED]")
                    if account_id:
                        diagnostic = diagnostic.replace(account_id, "[REDACTED]")

                    if status == 429 or 3036 in codes:
                        raise ImageProviderError(
                            "quota_exhausted",
                            "Image quota is exhausted. Daily Cloudflare quota resets at 00:00 UTC.",
                            429,
                            diagnostic=diagnostic,
                        )
                    if status == 402 or 5035 in codes:
                        raise ImageProviderError(
                            "free_plan_required",
                            "Provider requires paid access; JRUVA will not upgrade or charge.",
                            503,
                            diagnostic=diagnostic,
                        )
                    if status in (401, 403):
                        raise ImageProviderError(
                            "provider_authentication",
                            "Image provider credentials or permissions are invalid.",
                            diagnostic=diagnostic,
                        )
                    if status == 400:
                        raise ImageProviderError(
                            "provider_schema",
                            "Image provider rejected the request format.",
                            diagnostic=diagnostic,
                        )
                    if status >= 500:
                        raise ImageProviderError(
                            "provider_unavailable",
                            "Image provider is temporarily unavailable. Please retry.",
                            retryable=True,
                            diagnostic=diagnostic,
                        )
                    if status != 200 or body.get("success") is False:
                        raise ImageProviderError(
                            "provider_unavailable",
                            "Image provider did not return a successful image.",
                            diagnostic=diagnostic,
                        )
                    if (
                        response.headers.get("content-type", "").split(";")[0]
                        != "application/json"
                    ):
                        raise ImageProviderError(
                            "invalid_image",
                            "Image provider returned an unexpected MIME type.",
                            diagnostic=f"content-type={response.headers.get('content-type')}",
                        )
                    result = body.get("result", body)
                    encoded = result.get("image") if isinstance(result, dict) else None
                    if not isinstance(encoded, str):
                        raise ImageProviderError(
                            "invalid_image",
                            "Image provider returned no image bytes.",
                            diagnostic=f"keys={list(result.keys()) if isinstance(result, dict) else type(result)}",
                        )
                    try:
                        data = base64.b64decode(encoded, validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise ImageProviderError(
                            "invalid_image",
                            "Image provider returned malformed image bytes.",
                            diagnostic=str(exc),
                        ) from None
                    validated_png(data)
                    return GeneratedImage(data, model)
        except httpx.TimeoutException:
            raise ImageProviderError(
                "provider_timeout",
                "Provider response timed out; its outcome may be unknown.",
                retryable=not dispatched,
                outcome="indeterminate" if dispatched else "confirmed_failure",
            ) from None
        except httpx.HTTPError:
            raise ImageProviderError(
                "provider_unavailable",
                "Provider connection failed; its outcome may be unknown.",
                retryable=not dispatched,
                outcome="indeterminate" if dispatched else "confirmed_failure",
            ) from None
