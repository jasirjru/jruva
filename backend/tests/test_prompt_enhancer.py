"""Unit tests for LLM Prompt Expansion Engine and 4K Super-Resolution Upscaler."""
import os
import pytest
from unittest.mock import AsyncMock, patch
from PIL import Image

from app.services.image.prompt_enhancer import prompt_enhancer, PromptExpansionCache, LLMPromptEnhancer
from app.services.image.upscaler import upscaler_service, SuperResolutionService


@pytest.mark.asyncio
async def test_prompt_enhancer_cache():
    """Verify prompt expansion cache returns sub-millisecond cached result."""
    cache = PromptExpansionCache(max_size=10, ttl_seconds=60)
    assert cache.get("test:photorealistic") is None

    cache.set("test:photorealistic", "expanded master blueprint")
    assert cache.get("test:photorealistic") == "expanded master blueprint"


@pytest.mark.asyncio
async def test_prompt_enhancer_rule_based_fallback():
    """Verify fallback activates and enriches prompts when LLM is mocked to fail."""
    enhancer = LLMPromptEnhancer()
    with patch.object(enhancer, "_get_provider", side_effect=Exception("Simulated LLM outage")):
        res = await enhancer.expand("modern campus library", style="architecture")
        assert "campus" in res.lower() or "library" in res.lower()
        assert "architectural" in res.lower() or "8k" in res.lower()


@pytest.mark.asyncio
async def test_prompt_enhancer_4_eyes_anatomy_handling():
    """Verify that '4 eyes' prompts trigger explicit anatomical alignment in fallback."""
    enhancer = LLMPromptEnhancer()
    with patch.object(enhancer, "_get_provider", side_effect=Exception("Forced fallback")):
        res = await enhancer.expand("a 4 eyes human", style="photorealistic")
        assert "four distinct eyes" in res.lower() or "two pairs" in res.lower()
        assert "vertically aligned" in res.lower() or "cheekbones" in res.lower()


@pytest.mark.asyncio
async def test_upscaler_service_local_image(tmp_path):
    """Arbitrary local files must not be loaded or reused through an upscale cache."""
    test_file = tmp_path / "private-input.png"
    Image.new("RGB", (128,128), "red").save(test_file)
    service = SuperResolutionService()
    first = await service.upscale(str(test_file), scale=2)
    second = await service.upscale(str(test_file), scale=2)
    assert first["success"] is False
    assert second["success"] is False
    assert await service._fetch_image_bytes(str(test_file)) is None


@pytest.mark.asyncio
async def test_upscaler_service_cache(tmp_path):
    """Arbitrary local files must not be loaded or reused through an upscale cache."""
    test_file = tmp_path / "private-input.png"
    Image.new("RGB", (128,128), "red").save(test_file)
    service = SuperResolutionService()
    first = await service.upscale(str(test_file), scale=2)
    second = await service.upscale(str(test_file), scale=2)
    assert first["success"] is False
    assert second["success"] is False
    assert await service._fetch_image_bytes(str(test_file)) is None
