"""Offline HTTP contracts; fixtures are test images, never inference evidence."""

import base64
import io
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image


def png_bytes():
    out = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(out, "PNG")
    return out.getvalue()


def config(**overrides):
    values = {
        "IMAGE_GENERATION_ENABLED": True,
        "CLOUDFLARE_ACCOUNT_ID": "a" * 32,
        "CLOUDFLARE_API_TOKEN": "synthetic-token",
        "IMAGE_CF_FREE_PLAN_VERIFIED": True,
        "IMAGE_CF_KLEIN_ENABLED": False,
        "IMAGE_CF_KLEIN_FREE_VERIFIED": False,
    }
    return SimpleNamespace(**(values | overrides))


@pytest.mark.asyncio
async def test_schnell_json_and_actual_model():
    from app.services.image.providers import SCHNELL, CloudflareImageProvider

    async def handler(request):
        if request.url.path.endswith("/subscriptions"):
            return httpx.Response(200, json={"success": True, "result": []})
        assert str(request.url).endswith("/ai/run/" + SCHNELL)
        assert b'"steps":4' in request.content
        assert "cf-aig-gateway-id" not in request.headers
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {"image": base64.b64encode(png_bytes()).decode()},
            },
        )

    provider = CloudflareImageProvider(config(), transport=httpx.MockTransport(handler))
    output = await provider.generate("a boat", 42, SCHNELL)
    assert output.model == SCHNELL
    assert output.data == png_bytes()


@pytest.mark.asyncio
async def test_unverified_plan_makes_no_inference_call():
    from app.services.image.providers import (
        SCHNELL,
        CloudflareImageProvider,
        ImageProviderError,
    )

    def handler(request):
        pytest.fail("Unverified billing must never make a network call")

    provider = CloudflareImageProvider(
        config(IMAGE_CF_FREE_PLAN_VERIFIED=False),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ImageProviderError, match="Free plan"):
        await provider.generate("boat", 42, SCHNELL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code,kind",
    [
        (429, 3036, "quota_exhausted"),
        (403, 5035, "free_plan_required"),
        (401, 10000, "provider_authentication"),
        (400, 5006, "provider_schema"),
    ],
)
async def test_provider_error_is_specific_and_sanitized(status, code, kind):
    from app.services.image.providers import (
        SCHNELL,
        CloudflareImageProvider,
        ImageProviderError,
    )

    def handler(request):
        if request.url.path.endswith("/subscriptions"):
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(
            status,
            json={
                "success": False,
                "errors": [{"code": code, "message": "SECRET synthetic-token"}],
            },
        )

    provider = CloudflareImageProvider(config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ImageProviderError) as error:
        await provider.generate("boat", 42, SCHNELL)
    assert error.value.kind == kind
    assert "SECRET" not in str(error.value)


@pytest.mark.asyncio
async def test_klein_uses_multipart_only_after_separate_verification():
    from app.services.image.providers import KLEIN, CloudflareImageProvider

    def handler(request):
        if request.url.path.endswith("/subscriptions"):
            return httpx.Response(200, json={"success": True, "result": []})
        assert request.headers["content-type"].startswith("multipart/form-data;")
        assert b'name="prompt"' in request.content
        assert b'name="width"' in request.content
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {"image": base64.b64encode(png_bytes()).decode()},
            },
        )

    provider = CloudflareImageProvider(
        config(IMAGE_CF_KLEIN_ENABLED=True, IMAGE_CF_KLEIN_FREE_VERIFIED=True),
        transport=httpx.MockTransport(handler),
    )
    assert (await provider.generate("boat", 42, KLEIN)).model == KLEIN


@pytest.mark.parametrize("data", [b"<html>error</html>", b"not an image"])
def test_invalid_image_bytes_are_rejected(data):
    from app.services.image.providers import ImageProviderError, validated_png

    with pytest.raises(ImageProviderError):
        validated_png(data)


def test_daily_budget_is_atomic_and_persistent(tmp_path, monkeypatch):
    import app.services.image.budget as module
    from app.services.image.budget import reserve_image_budget
    from app.services.image.providers import ImageProviderError
    from app.services.workspace_store import WorkspaceStore

    monkeypatch.setattr(
        module, "workspace_store", WorkspaceStore(str(tmp_path / "budget.db"))
    )
    reserve_image_budget("synthetic-account", "alice", 60, 100, 5)
    with pytest.raises(ImageProviderError) as error:
        reserve_image_budget("synthetic-account", "bob", 60, 100, 5)
    assert error.value.kind == "quota_exhausted"


@pytest.mark.asyncio
async def test_timeout_is_retryable():
    from app.services.image.providers import (
        SCHNELL,
        CloudflareImageProvider,
        ImageProviderError,
    )

    def handler(request):
        raise httpx.ReadTimeout("SECRET", request=request)

    with pytest.raises(ImageProviderError) as error:
        await CloudflareImageProvider(
            config(), transport=httpx.MockTransport(handler)
        ).generate("boat", 42, SCHNELL)
    assert error.value.kind == "provider_timeout"
    assert error.value.retryable is True


@pytest.mark.asyncio
async def test_paid_subscription_blocks_inference_even_with_operator_attestation():
    from app.services.image.providers import (
        SCHNELL,
        CloudflareImageProvider,
        ImageProviderError,
    )

    def handler(request):
        assert request.method == "GET", "Paid account must never POST inference"
        return httpx.Response(
            200,
            json={"success": True, "result": [{"rate_plan": {"id": "workers_paid"}}]},
        )

    with pytest.raises(ImageProviderError) as error:
        await CloudflareImageProvider(
            config(), transport=httpx.MockTransport(handler)
        ).generate("boat", 42, SCHNELL)
    assert error.value.kind == "free_plan_required"


def test_concurrent_budget_reservations_cannot_overspend(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import app.services.image.budget as module
    from app.services.image.budget import reserve_image_budget
    from app.services.image.providers import ImageProviderError
    from app.services.workspace_store import WorkspaceStore

    monkeypatch.setattr(
        module, "workspace_store", WorkspaceStore(str(tmp_path / "concurrent.db"))
    )

    def reserve(index):
        try:
            reserve_image_budget("shared-account", "owner-" + str(index), 60, 120, 5)
            return True
        except ImageProviderError:
            return False

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(reserve, range(12))) == 2


def test_owner_limit_does_not_spend_another_owners_budget(tmp_path, monkeypatch):
    import app.services.image.budget as module
    from app.services.image.budget import reserve_image_budget
    from app.services.image.providers import ImageProviderError
    from app.services.workspace_store import WorkspaceStore

    monkeypatch.setattr(
        module, "workspace_store", WorkspaceStore(str(tmp_path / "owner-limit.db"))
    )
    reserve_image_budget("shared-account", "alice", 60, 120, 1)
    with pytest.raises(ImageProviderError):
        reserve_image_budget("shared-account", "alice", 60, 120, 1)
    reserve_image_budget("shared-account", "bob", 60, 120, 1)


@pytest.mark.asyncio
async def test_schnell_payload_construction_strictly_prompt_and_steps():
    """FLUX.1 Schnell schema regression: only prompt and steps: 4, no seed or extras."""
    import json
    from app.services.image.providers import SCHNELL, CloudflareImageProvider

    captured_request = {}

    async def handler(request):
        if request.url.path.endswith("/subscriptions"):
            return httpx.Response(200, json={"success": True, "result": []})
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {"image": base64.b64encode(png_bytes()).decode()},
            },
        )

    provider = CloudflareImageProvider(config(), transport=httpx.MockTransport(handler))
    output = await provider.generate(prompt="A beautiful wooden sailboat", seed=999, model=SCHNELL)
    assert output.model == SCHNELL

    body = captured_request["body"]
    assert body == {"prompt": "A beautiful wooden sailboat", "steps": 4}
    assert "seed" not in body
    assert "width" not in body
    assert "height" not in body


@pytest.mark.asyncio
async def test_prompt_validation_bounds():
    """Invalid prompt lengths are rejected before network requests."""
    from app.services.image.providers import SCHNELL, CloudflareImageProvider, ImageProviderError

    provider = CloudflareImageProvider(config())
    with pytest.raises(ImageProviderError, match="Image prompt must contain 1–2048 characters.") as exc:
        await provider.generate(prompt="   ", seed=None, model=SCHNELL)
    assert exc.value.status == 400

    with pytest.raises(ImageProviderError, match="Image prompt must contain 1–2048 characters.") as exc:
        await provider.generate(prompt="x" * 2049, seed=None, model=SCHNELL)
    assert exc.value.status == 400


@pytest.mark.asyncio
async def test_error_handling_sanitizes_internal_diagnostics():
    """Diagnostics capture error codes/messages but redact tokens and account IDs."""
    from app.services.image.providers import SCHNELL, CloudflareImageProvider, ImageProviderError

    test_token = "cf-secret-token-12345"
    test_account = "11223344556677889900aabbccddeeff"

    def handler(request):
        if request.url.path.endswith("/subscriptions"):
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(
            400,
            json={
                "success": False,
                "errors": [
                    {
                        "code": 10005,
                        "message": f"Validation failed for account {test_account} with token {test_token}: unexpected param 'seed'",
                    }
                ],
            },
        )

    cfg = config(CLOUDFLARE_API_TOKEN=test_token, CLOUDFLARE_ACCOUNT_ID=test_account)
    provider = CloudflareImageProvider(cfg, transport=httpx.MockTransport(handler))

    with pytest.raises(ImageProviderError) as exc_info:
        await provider.generate("boat", 42, SCHNELL)

    err = exc_info.value
    assert err.kind == "provider_schema"
    # User-facing message must NEVER leak secrets or raw provider strings
    assert test_token not in str(err)
    assert test_account not in str(err)
    assert "unexpected param" not in str(err)
    assert str(err) == "Image provider rejected the request format."

    # Internal diagnostics must redact tokens and account ID
    assert test_token not in err.diagnostic
    assert test_account not in err.diagnostic
    assert "[REDACTED]" in err.diagnostic
    assert "10005" in err.diagnostic


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_result,expected_msg",
    [
        ({"image": "not-a-valid-base64!!!"}, "malformed image bytes"),
        ({"image": base64.b64encode(b"not-an-image-payload").decode()}, "undecodable image bytes"),
        ({}, "returned no image bytes"),
        ({"image": 12345}, "returned no image bytes"),
    ],
)
async def test_response_decoding_error_handling(corrupt_result, expected_msg):
    """Regression test: invalid or corrupt image payloads raise invalid_image."""
    from app.services.image.providers import SCHNELL, CloudflareImageProvider, ImageProviderError

    def handler(request):
        if request.url.path.endswith("/subscriptions"):
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(200, json={"success": True, "result": corrupt_result})

    provider = CloudflareImageProvider(config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ImageProviderError) as exc_info:
        await provider.generate("boat", None, SCHNELL)

    assert exc_info.value.kind == "invalid_image"
    assert expected_msg in str(exc_info.value)


@pytest.mark.asyncio
async def test_jpeg_and_webp_decoding():
    """Provider supports decoding JPEG and WEBP into validated raster bytes."""
    from app.services.image.providers import SCHNELL, CloudflareImageProvider

    for fmt in ["JPEG", "WEBP"]:
        out = io.BytesIO()
        Image.new("RGB", (64, 64), "red").save(out, fmt)
        img_bytes = out.getvalue()

        def make_handler(b64):
            def handler(request):
                if request.url.path.endswith("/subscriptions"):
                    return httpx.Response(200, json={"success": True, "result": []})
                return httpx.Response(200, json={"success": True, "result": {"image": b64}})
            return handler

        b64_val = base64.b64encode(img_bytes).decode()
        provider = CloudflareImageProvider(config(), transport=httpx.MockTransport(make_handler(b64_val)))
        output = await provider.generate("boat", None, SCHNELL)
        assert output.model == SCHNELL
        assert output.data == img_bytes

