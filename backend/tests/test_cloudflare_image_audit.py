"""Test suite auditing Cloudflare Workers AI FLUX.1 Schnell image provider fail-closed safety."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from app.core.config import settings
from app.services.image.providers import (
    SCHNELL,
    CloudflareImageProvider,
    ImageProviderError,
    validated_png,
)


@pytest.fixture
def test_config():
    """Mock config object with Cloudflare settings."""
    cfg = MagicMock()
    cfg.IMAGE_GENERATION_ENABLED = False
    cfg.IMAGE_CF_FREE_PLAN_VERIFIED = False
    cfg.CLOUDFLARE_API_TOKEN = "test-token"
    cfg.CLOUDFLARE_ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
    cfg.IMAGE_CF_KLEIN_ENABLED = False
    cfg.IMAGE_CF_KLEIN_FREE_VERIFIED = False
    return cfg


def test_image_provider_fails_closed_when_generation_disabled(test_config):
    """When IMAGE_GENERATION_ENABLED is False, available_models must raise immediately."""
    provider = CloudflareImageProvider(test_config)
    with pytest.raises(ImageProviderError) as exc:
        provider.available_models()
    assert exc.value.kind == "provider_not_configured"
    assert "Image generation is not enabled" in exc.value.message


def test_image_provider_requires_verified_free_plan(test_config):
    """Even if enabled, IMAGE_CF_FREE_PLAN_VERIFIED must be True to authorize inference."""
    test_config.IMAGE_GENERATION_ENABLED = True
    test_config.IMAGE_CF_FREE_PLAN_VERIFIED = False
    provider = CloudflareImageProvider(test_config)
    with pytest.raises(ImageProviderError) as exc:
        provider.available_models()
    assert exc.value.kind == "free_plan_required"


def test_image_provider_validates_account_id_format(test_config):
    """Cloudflare account ID must be exactly 32 hex characters."""
    test_config.IMAGE_GENERATION_ENABLED = True
    test_config.IMAGE_CF_FREE_PLAN_VERIFIED = True
    test_config.CLOUDFLARE_ACCOUNT_ID = "invalid-account-id"
    provider = CloudflareImageProvider(test_config)
    with pytest.raises(ImageProviderError) as exc:
        provider.available_models()
    assert exc.value.kind == "provider_not_configured"


def test_image_provider_available_models_returns_schnell_when_authorized(test_config):
    """When all safety criteria pass, available_models includes FLUX.1 Schnell."""
    test_config.IMAGE_GENERATION_ENABLED = True
    test_config.IMAGE_CF_FREE_PLAN_VERIFIED = True
    test_config.CLOUDFLARE_ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
    provider = CloudflareImageProvider(test_config)
    models = provider.available_models()
    assert SCHNELL in models


@pytest.mark.asyncio
async def test_image_provider_blocks_on_paid_subscription_preflight(test_config):
    """Preflight check to /subscriptions must block if any paid subscription exists."""
    test_config.IMAGE_GENERATION_ENABLED = True
    test_config.IMAGE_CF_FREE_PLAN_VERIFIED = True

    provider = CloudflareImageProvider(test_config)

    # Mock response with 1 paid subscription
    mock_sub_resp = httpx.Response(
        status_code=200,
        json={"success": True, "result": [{"id": "sub_paid_plan"}]},
        request=httpx.Request("GET", "https://api.cloudflare.com/client/v4/accounts/test/subscriptions"),
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_sub_resp
        with pytest.raises(ImageProviderError) as exc:
            await provider.generate("a snowy landscape", seed=42, model=SCHNELL)
        assert exc.value.kind == "free_plan_required"
        assert "account subscriptions exist" in exc.value.message


def test_validated_png_rejects_empty_or_corrupt_bytes():
    """Raster byte decoding must reject corrupt or empty data."""
    with pytest.raises(ImageProviderError):
        validated_png(b"")

    with pytest.raises(ImageProviderError):
        validated_png(b"not-an-image-data-stream")
