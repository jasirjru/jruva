"""Generation must not masquerade catalog photos or accept arbitrary upscale input."""

import pytest
from app.services.image.image_service import ImageGenerationService
from app.services.image.upscaler import SuperResolutionService


@pytest.mark.asyncio
async def test_generate_campus_never_returns_a_static_showcase(monkeypatch):
    async def unavailable(*args, **kwargs):
        return None

    monkeypatch.setattr(
        ImageGenerationService, "_generate_via_huggingface", unavailable
    )
    result = await ImageGenerationService().generate_and_validate(
        "generate image of campus", enhance=False
    )
    assert result["success"] is False
    assert not result["image_url"].startswith("/static/demo/")


@pytest.mark.asyncio
async def test_boat_error_preserves_generation_cause(monkeypatch):
    async def unavailable(*args, **kwargs):
        return None

    monkeypatch.setattr(
        ImageGenerationService, "_generate_via_huggingface", unavailable
    )
    result = await ImageGenerationService().generate_and_validate("boat", enhance=False)
    assert "university" not in result["error_message"].lower()
    assert result["error_type"] in {
        "authentication_required",
        "provider_not_configured",
    }


@pytest.mark.asyncio
async def test_upscale_never_reads_arbitrary_local_file(tmp_path):
    path = tmp_path / "private-data.txt"
    path.write_bytes(b"private bytes")
    assert await SuperResolutionService()._fetch_image_bytes(str(path)) is None
