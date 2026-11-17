"""Real API/storage ownership contracts with explicitly synthetic provider bytes."""

import io

import pytest
from app.core.config import settings
from app.main import app
from app.services.auth_service import auth_service
from app.services.image.providers import (
    KLEIN,
    SCHNELL,
    CloudflareImageProvider,
    GeneratedImage,
    ImageProviderError,
)
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store
from fastapi.testclient import TestClient
from PIL import Image

from tests.private_accounts import private_account_headers


@pytest.fixture
def image_context(monkeypatch, tmp_path):
    # Persisted budgets intentionally survive process restarts. Every test gets
    # its own database/storage so repeated test runs cannot exhaust one another.
    monkeypatch.setattr(workspace_store, "db_path", str(tmp_path / "owned-images.db"))
    monkeypatch.setattr(auth_service, "db_path", workspace_store.db_path)
    workspace_store._init_sqlite_db()
    auth_service._init_db()
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(storage_service, "base_dir", str(media))
    monkeypatch.setattr(settings, "IMAGE_GENERATION_ENABLED", True)
    monkeypatch.setattr(settings, "IMAGE_CF_FREE_PLAN_VERIFIED", True)
    monkeypatch.setattr(settings, "CLOUDFLARE_ACCOUNT_ID", "b" * 32)
    monkeypatch.setattr(settings, "CLOUDFLARE_API_TOKEN", "synthetic-never-network")
    monkeypatch.setattr(settings, "IMAGE_DAILY_NEURON_LIMIT", 10000)
    monkeypatch.setattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 100)

    async def generate(self, prompt, seed, model):
        out = io.BytesIO()
        Image.new("RGB", (64, 48), "orange").save(out, "PNG")
        return GeneratedImage(out.getvalue(), model)

    monkeypatch.setattr(CloudflareImageProvider, "generate", generate)
    return (
        TestClient(app),
        private_account_headers("image-owner"),
        private_account_headers("image-attacker"),
    )


@pytest.mark.parametrize(
    "prompt,style",
    [
        ("a boat", "photorealistic"),
        ("a realistic sunset landscape", "photorealistic"),
        ("an anime-style illustration", "anime"),
    ],
)
def test_owned_image_roundtrip_and_delete(image_context, prompt, style):
    client, alice, bob = image_context
    response = client.post(
        "/api/v1/image/generate", headers=alice, json={"prompt": prompt, "style": style}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model"] == SCHNELL
    assert (result["width"], result["height"]) == (
        64,
        48,
    )  # Actual fixture dimensions, not requested 1024.
    url = result["image_url"]
    assert url.startswith("/api/v1/image/")
    assert client.get(url, headers=bob).status_code == 404
    assert client.get(url).status_code == 401
    raw = client.get(url, headers=alice)
    assert raw.status_code == 200
    assert raw.headers["content-type"] == "image/png"
    assert raw.headers["cache-control"] == "private, no-store"
    with Image.open(io.BytesIO(raw.content)) as image:
        assert image.size == (64, 48)
    image_id = result["image_id"]
    gallery = client.get("/api/v1/image/gallery", headers=alice).json()["images"]
    assert any(row["id"] == image_id and row["file_url"] == url for row in gallery)
    assert not any(
        row["id"] == image_id
        for row in client.get("/api/v1/image/gallery", headers=bob).json()["images"]
    )
    assert client.delete("/api/v1/image/" + image_id, headers=bob).status_code == 404
    record = workspace_store.get_image(
        image_id, next(row["user_id"] for row in gallery if row["id"] == image_id)
    )
    assert client.delete("/api/v1/image/" + image_id, headers=alice).status_code == 200
    assert not storage_service.file_exists(record["file_url"])
    assert client.get(url, headers=alice).status_code == 404


def test_quota_message_without_catalog_replacement(image_context, monkeypatch):
    client, alice, _ = image_context

    async def exhausted(*args, **kwargs):
        raise ImageProviderError(
            "quota_exhausted", "Daily image quota is exhausted.", 429
        )

    monkeypatch.setattr(CloudflareImageProvider, "generate", exhausted)
    result = client.post(
        "/api/v1/image/generate", headers=alice, json={"prompt": "boat"}
    )
    assert result.status_code == 429
    assert result.json()["error_type"] == "quota_exhausted"
    assert "university" not in result.text


def test_confirmed_rejection_can_fallback_and_reports_actual_model(image_context, monkeypatch):
    client, alice, _ = image_context
    monkeypatch.setattr(settings, "IMAGE_CF_KLEIN_ENABLED", True)
    monkeypatch.setattr(settings, "IMAGE_CF_KLEIN_FREE_VERIFIED", True)
    original = CloudflareImageProvider.generate

    async def timeout(self, prompt, seed, model):
        if model == KLEIN:
            raise ImageProviderError("provider_unavailable", "Pre-dispatch failure.", retryable=True, outcome="confirmed_failure")
        return await original(self, prompt, seed, model)

    monkeypatch.setattr(CloudflareImageProvider, "generate", timeout)
    result = client.post(
        "/api/v1/image/generate", headers=alice, json={"prompt": "boat"}
    ).json()
    assert result["success"] is True
    assert result["failover"] is True
    assert result["model"] == SCHNELL


@pytest.mark.parametrize(
    "source",
    [
        "http://127.0.0.1/secret",
        "http://169.254.169.254/metadata",
        "/static/../secret",
        "C:/private/file.png",
    ],
)
def test_upscale_rejects_arbitrary_sources(image_context, source):
    client, alice, _ = image_context
    response = client.post(
        "/api/v1/image/upscale", headers=alice, json={"image_url": source}
    )
    assert response.status_code == 400


def test_upscale_requires_owner_and_preserves_privacy(image_context):
    client, alice, bob = image_context
    image = client.post(
        "/api/v1/image/generate", headers=alice, json={"prompt": "boat"}
    ).json()
    payload = {"image_url": image["image_url"], "scale": 2}
    assert client.post("/api/v1/image/upscale", json=payload).status_code == 401
    assert (
        client.post("/api/v1/image/upscale", headers=bob, json=payload).status_code
        == 400
    )
    result = client.post("/api/v1/image/upscale", headers=alice, json=payload)
    assert result.status_code == 200, result.text
    assert result.json()["upscaled_width"] == 128
    assert client.get(result.json()["upscaled_url"], headers=bob).status_code == 404


def test_legacy_public_generated_urls_are_blocked():
    assert TestClient(app).get("/static/generated/anything.png").status_code == 404


def test_chat_image_uses_the_owned_generation_path(image_context):
    client, alice, _bob = image_context
    response = client.post(
        "/api/v1/chat", headers=alice, json={"message": "generate image of boat"}
    )
    assert response.status_code == 200, response.text
    assert SCHNELL in response.text
    assert "/api/v1/image/" in response.text
    assert "university repository" not in response.text
    images = client.get("/api/v1/image/gallery", headers=alice).json()["images"]
    assert any(row["prompt"] == "boat" for row in images)


def test_streaming_chat_reports_actual_image_model(image_context):
    client, alice, _bob = image_context
    response = client.post(
        "/api/v1/chat",
        headers=alice,
        json={"message": "generate image of boat", "stream": True},
    )
    assert response.status_code == 200, response.text
    assert SCHNELL in response.text
    assert "/api/v1/image/" in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/static//generated/probe.png",
        "/static/generated/probe.png",
        "/static/Generated/probe.png",
    ],
)
def test_static_generated_path_normalization_cannot_bypass_private_downloads(
    path, tmp_path, monkeypatch
):
    from pathlib import Path

    import app.main as module

    target = Path(module.__file__).parent / "static" / "generated" / "probe.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic private content")
    try:
        response = TestClient(app).get(path)
        assert response.status_code == 404
    finally:
        target.unlink()
