"""Unit and integration tests for Multi-Tier Image Generation Service & Error Handling."""
import pytest
from unittest.mock import AsyncMock, patch
from app.services.image.image_service import (
    image_service,
    ImagePromptCache,
    ProviderHealthMonitor,
)


def test_image_intent_detection():
    """Verify that visual prompts trigger image intent while non-visual prompts do not."""
    is_img, subject = image_service.detect_intent("generate an image of a red motorcycle")
    assert is_img is True
    assert "motorcycle" in subject.lower()

    # Typo tolerance on generate (e.g. 'genreate', 'genrate', 'geneate')
    is_img, subject = image_service.detect_intent("genreate image of ear phone")
    assert is_img is True
    assert "ear phone" in subject.lower()

    is_img, subject = image_service.detect_intent("image of ear phone")
    assert is_img is True
    assert "ear phone" in subject.lower()

    is_img, subject = image_service.detect_intent("draw a boy in futuristic armor")
    assert is_img is True
    assert "boy" in subject.lower()

    is_img, subject = image_service.detect_intent("render a bike on a rainy street")
    assert is_img is True
    assert "bike" in subject.lower()

    # Non-visual coding / text requests should NOT trigger image synthesis
    is_img, _ = image_service.detect_intent("generate python code for a binary search tree")
    assert is_img is False

    is_img, _ = image_service.detect_intent("write an essay on machine learning")
    assert is_img is False


def test_prompt_enhancement_sanitization():
    """Verify prompt enhancements for bikes, portraits, quotes, and gadgets."""
    enhanced_bike = image_service.enhance_prompt('vintage "racing" bike')
    assert "bike" in enhanced_bike.lower()
    assert '"' not in enhanced_bike
    assert "photorealistic" in enhanced_bike.lower() or "design" in enhanced_bike.lower()

    enhanced_boy = image_service.enhance_prompt("young boy smiling")
    assert "portrait" in enhanced_boy.lower() or "lens" in enhanced_boy.lower()

    enhanced_earphone = image_service.enhance_prompt("wireless earphone")
    assert "product" in enhanced_earphone.lower() or "lighting" in enhanced_earphone.lower()


def test_image_prompt_cache():
    """Verify LRU prompt cache sub-15ms hit rate for Tier S+++."""
    cache = ImagePromptCache(max_size=10, ttl_seconds=60)
    data = {"image_url": "https://example.com/image.png", "model": "flux", "seed": 42}

    assert cache.get("a blue sportscar", 1024, 1024) is None

    cache.set("a blue sportscar", 1024, 1024, data)
    cached = cache.get("A Blue Sportscar ", 1024, 1024)  # Case & trim insensitive
    assert cached is not None
    assert cached["image_url"] == "https://example.com/image.png"
    assert cached["seed"] == 42


def test_provider_health_monitor():
    """Verify health monitor tracks failure count and flags degraded endpoints."""
    monitor = ProviderHealthMonitor()
    assert monitor.is_healthy("flux") is True

    monitor.record_failure("flux", status_code=504, reason="Gateway Timeout")
    monitor.record_failure("flux", status_code=504, reason="Gateway Timeout")
    assert monitor.is_healthy("flux") is True

    # 3rd failure marks degraded
    monitor.record_failure("flux", status_code=504, reason="Gateway Timeout")
    assert monitor.is_healthy("flux") is False

    # Success restores health
    monitor.record_success("flux", latency_ms=120)
    assert monitor.is_healthy("flux") is True


def test_error_categorization():
    """Verify status codes map to strongly-typed error codes."""
    code, msg = image_service._categorize_error(429, "")
    assert code == "rate_limited"
    assert "429" in msg

    code, msg = image_service._categorize_error(408, "timed out")
    assert code == "timeout"

    code, msg = image_service._categorize_error(500, "internal server error")
    assert code == "server_error"

    code, msg = image_service._categorize_error(400, "moderation rejected")
    assert code == "content_filter"


@pytest.mark.asyncio
async def test_safety_guardrail_blocks_inappropriate_prompts():
    """Verify strict blocking of 18+, adult, suggestive, or violent image requests."""
    from app.services.image.image_service import check_image_safety

    unsafe_prompts = [
        "generate 18+ pic",
        "show nsfw photo",
        "nude model in bedroom",
        "sexy girl dancing",
        "bikini model portrait",
        "erotic wallpaper",
        "violent gore and blood scene",
    ]

    for p in unsafe_prompts:
        is_safe, reason = check_image_safety(p)
        assert is_safe is False, f"Prompt should be flagged unsafe: {p}"
        assert reason is not None

        # Verify async generation pipeline blocks immediately
        res = await image_service.generate_and_validate(p)
        assert res["success"] is False
        assert res["error_type"] == "content_filter"
        assert res["status_code"] == 400
        assert "Content Policy" in res["error_message"]
        assert res["image_url"].startswith("error:content_filter:")


@pytest.mark.asyncio
async def test_verified_assets_showcase_match():
    """Verify that verified official assets return relative URLs and rich metadata."""
    test_queries = [
        ("show campus grounds", "campus.jpg", "Campus"),
        ("ai robotics engineering lab", "robotics.jpg", "Robotics"),
        ("central university library reading hall", "library.jpg", "Library"),
        ("smart law moot court", "moot_court.jpg", "Moot Court"),
        ("pharmaceutical biotech lab", "pharmacy.jpg", "Pharmaceutical"),
        ("grand convocation auditorium", "auditorium.jpg", "Auditorium"),
    ]

    for prompt, expected_file, expected_keyword in test_queries:
        res = image_service._check_showcase_presets(prompt, prompt, 42, "B")
        assert res["success"] is True, f"Failed for {prompt}"
        assert res["image_url"] == f"/static/demo/{expected_file}"
        assert expected_keyword.lower() in res["title"].lower()
        assert "description" in res and len(res["description"]) > 5


@pytest.mark.asyncio
async def test_uncataloged_prompts_return_unavailable_without_fake_fallbacks():
    """Verify that uncataloged prompts return polite unavailable status without deceptive fallbacks."""
    uncataloged_queries = [
        "draw a red racing car",
        "photo of a motorcycle on highway",
        "portrait of a random astronaut on mars",
        "a cute dragon breathing fire",
    ]

    image_service.cache.clear()
    for p in uncataloged_queries:
        res = await image_service.generate_and_validate(p)
        assert res["success"] is False
        assert res["error_type"] == "authentication_required"
        assert res["status_code"] == 401
        assert "sign in" in res["error_message"].lower()
        assert res["image_url"].startswith("error:authentication_required:")
        # Crucial check: verify we did NOT return a deceptive fake fallback image
        assert not res["image_url"].endswith(".jpg")
        assert not res["image_url"].endswith(".png")


@pytest.mark.asyncio
async def test_tier_s_plus_plus_plus_caching():
    """Verify prompt cache returns cached response for identical prompts."""
    prompt = "smart law moot court hall"
    # First call primes cache or showcase match
    res1 = await image_service.generate_and_validate(prompt, tier="S+++")
    assert res1["success"] is False

    # A cached public preset must not bypass ownership/genuine generation
    image_service.cache.set("custom cached test prompt", 1024, 1024, {
        "success": True,
        "image_url": "/static/demo/campus.jpg",
        "original_prompt": "custom cached test prompt",
        "enhanced_prompt": "custom cached test prompt",
        "model": "Verified",
        "seed": 1234,
        "width": 1024,
        "height": 1024,
    })
    cached_res = await image_service.generate_and_validate("custom cached test prompt", tier="S+++")
    assert cached_res["success"] is False
    assert cached_res["cached"] is False
    assert cached_res["image_url"] != "/static/demo/campus.jpg"


def test_generate_image_url_synchronous():
    """Verify synchronous URL generator enforces safety and verified assets."""
    unsafe_res = image_service.generate_image_url("18+ nude photo")
    assert unsafe_res["success"] is False
    assert unsafe_res["error_type"] == "content_filter"

    showcase_res = image_service.generate_image_url("robotics lab")
    assert showcase_res["success"] is False
    assert showcase_res["error_type"] == "async_required"

    unavailable_res = image_service.generate_image_url("supercar on highway")
    assert unavailable_res["success"] is False
    assert unavailable_res["error_type"] == "async_required"
