"""Comprehensive verification tests for Image Generation Product Flow and Intent Routing.

Validates Phase 1 success criteria:
- Real product flow: "Generate a photorealistic red Ferrari parked beside a beach at sunset."
- Image request -> correct capability/endpoint and intent detection.
- Image success -> image markdown / card rendered.
- Provider unavailable -> clear error without deceptive fallback.
- Not signed in -> clear authentication required notice.
- Quota reached -> clear daily limit notice.
- Ambiguous result -> no unsafe automatic retry.
- Non-image chat -> behavior completely unchanged.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.image.image_service import image_service
from app.services.llm.router import detect_task_type, LLMRouter


def test_ferrari_real_user_prompt_intent_detection():
    """Verify the exact failing prompt from user issue is properly classified."""
    prompt = "Generate a photorealistic red Ferrari parked beside a beach at sunset."
    is_img, subject = image_service.detect_intent(prompt)
    assert is_img is True, f"Failed to detect image intent for: {prompt}"
    assert "ferrari" in subject.lower(), f"Extracted subject should contain ferrari, got: {subject}"
    assert "sunset" in subject.lower() or "beach" in subject.lower()

    # Router classification should flag as 'image', never 'daily'
    task_type = detect_task_type(prompt)
    assert task_type == "image", f"Router should classify task as image, got: {task_type}"


def test_various_image_generation_phrasings():
    """Verify various natural language styles for image generation."""
    test_cases = [
        ("Generate a photorealistic red Ferrari parked beside a beach at sunset.", True, "ferrari"),
        ("Can you generate a realistic sunset over the ocean?", True, "sunset"),
        ("generate a 3d render of an iPhone on a wooden table", True, "iphone"),
        ("Create a cinematic shot of an astronaut on Mars", True, "astronaut"),
        ("Draw a futuristic cybernetic tiger", True, "tiger"),
        ("Render a cozy cafe interior in Paris during winter", True, "cafe"),
        ("Create an oil painting of a cottage in the woods", True, "cottage"),
        ("Generate a 4k wallpaper of Tokyo skyline at night", True, "tokyo"),
        ("/imagine a majestic dragon flying over snowy mountains", True, "dragon"),
        ("photo of a cute golden retriever puppy", True, "retriever"),
    ]

    for query, expected_is_img, expected_kw in test_cases:
        is_img, subject = image_service.detect_intent(query)
        assert is_img is expected_is_img, f"Mismatch for '{query}': expected {expected_is_img}, got {is_img}"
        assert expected_kw.lower() in subject.lower(), f"Expected '{expected_kw}' in subject '{subject}'"


def test_non_image_requests_are_not_falsely_detected():
    """Verify coding, reasoning, doc, and standard chat prompts are never hijacked."""
    non_image_queries = [
        "generate a python script to parse a csv file",
        "create a react component with typescript for a login form",
        "write an essay on the effects of climate change",
        "what is the capital of France?",
        "how do I solve this integral: integral of x^2 dx",
        "summarize the main points of the Indian Constitution",
        "generate a sql query to select all active users from database",
        "generate unit tests for the auth service",
        "can you write a function to calculate fibonacci numbers?",
        "explain the difference between process and thread",
    ]

    for query in non_image_queries:
        is_img, _ = image_service.detect_intent(query)
        assert is_img is False, f"Prompt falsely detected as image: '{query}'"
        task_type = detect_task_type(query)
        assert task_type != "image", f"Router falsely classified '{query}' as image: {task_type}"


@pytest.mark.asyncio
async def test_chat_service_image_unauthenticated_flow():
    """Verify that unauthenticated image generation returns clean auth-required message."""
    service = ChatService()
    import uuid
    session_id = str(uuid.uuid4())
    req = ChatRequest(
        message="Generate a photorealistic red Ferrari parked beside a beach at sunset.",
        session_id=session_id,
        user_id=None,  # Not signed in
    )

    # In genuine_service, unauthenticated calls return 401 authentication_required
    with patch.object(image_service, "generate_and_validate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "success": False,
            "error_type": "authentication_required",
            "error_message": "Sign in to generate images.",
            "status_code": 401,
            "image_url": "error:authentication_required:Sign%20in%20to%20generate%20images.",
            "model": "none",
        }

        resp = await service.generate_chat_response(req)
        assert "Authentication Required" in resp.content or "Sign in" in resp.content
        assert "I cannot assist with that request" not in resp.content
        assert "Auto unavailable" not in resp.content


@pytest.mark.asyncio
async def test_chat_service_image_quota_exhausted_flow():
    """Verify that image generation when quota is reached returns clear quota notice."""
    service = ChatService()
    import uuid
    req = ChatRequest(
        message="Generate a photorealistic red Ferrari parked beside a beach at sunset.",
        session_id=str(uuid.uuid4()),
        user_id="user-123",
    )

    with patch.object(image_service, "generate_and_validate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "success": False,
            "error_type": "quota_exhausted",
            "error_message": "Daily image attempt quota is exhausted. Retry after 00:00 UTC.",
            "status_code": 429,
            "image_url": "error:quota_exhausted:Daily%20image%20quota%20exhausted",
            "model": "none",
        }

        resp = await service.generate_chat_response(req)
        assert "Daily Image Quota Reached" in resp.content or "quota" in resp.content.lower()
        assert "I cannot assist with that request" not in resp.content


@pytest.mark.asyncio
async def test_chat_service_image_provider_unavailable_flow():
    """Verify that upstream provider errors return clean error without silent refusal."""
    service = ChatService()
    import uuid
    req = ChatRequest(
        message="Generate a photorealistic red Ferrari parked beside a beach at sunset.",
        session_id=str(uuid.uuid4()),
        user_id="user-123",
    )

    with patch.object(image_service, "generate_and_validate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "success": False,
            "error_type": "provider_unavailable",
            "error_message": "No authorized image provider succeeded.",
            "status_code": 503,
            "image_url": "error:provider_unavailable:No%20provider",
            "model": "none",
        }

        resp = await service.generate_chat_response(req)
        assert "Image Service Unavailable" in resp.content or "unavailable" in resp.content.lower()
        assert "I cannot assist with that request" not in resp.content


@pytest.mark.asyncio
async def test_chat_service_image_success_flow():
    """Verify that successful image generation renders markdown image card."""
    service = ChatService()
    import uuid
    req = ChatRequest(
        message="Generate a photorealistic red Ferrari parked beside a beach at sunset.",
        session_id=str(uuid.uuid4()),
        user_id="user-123",
    )

    with patch.object(image_service, "generate_and_validate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "success": True,
            "image_url": "/api/v1/image/img-uuid-456/content",
            "model": "flux-schnell",
            "provider": "cloudflare",
            "original_prompt": "red Ferrari parked beside a beach at sunset",
            "enhanced_prompt": "red Ferrari parked beside a beach at sunset. 8k, photorealistic",
            "cached": False,
            "failover": False,
        }

        resp = await service.generate_chat_response(req)
        assert "![Generated image](/api/v1/image/img-uuid-456/content)" in resp.content
        assert "flux-schnell" in resp.content
        assert "I cannot assist with that request" not in resp.content


@pytest.mark.asyncio
async def test_chat_service_streaming_image_success_flow():
    """Verify streaming generation produces status, image markdown, and finishes cleanly."""
    service = ChatService()
    import uuid
    req = ChatRequest(
        message="Generate a photorealistic red Ferrari parked beside a beach at sunset.",
        session_id=str(uuid.uuid4()),
        user_id="user-123",
    )

    with patch.object(image_service, "generate_and_validate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "success": True,
            "image_url": "/api/v1/image/img-uuid-stream/content",
            "model": "flux-schnell",
            "provider": "cloudflare",
            "original_prompt": "red Ferrari parked beside a beach at sunset",
            "enhanced_prompt": "red Ferrari parked beside a beach at sunset. 8k",
            "cached": False,
            "failover": False,
        }

        events = []
        async for chunk in service.stream_chat_response(req):
            events.append(chunk)

        combined = "".join(events)
        assert "/api/v1/image/img-uuid-stream/content" in combined
        assert "event: done" in combined or "event: message" in combined


def test_router_never_selects_gpt_oss_120b_for_image_task():
    """Verify router _resolve_providers never selects text-only gpt-oss-120b for image tasks."""
    router = LLMRouter()
    primary, backup, model_id = router._resolve_providers(
        requested_model="auto",
        prompt="Generate a photorealistic red Ferrari parked beside a beach at sunset.",
    )
    # The primary model must NOT be text-only gpt-oss-120b or qwen-coder
    assert "gpt-oss-120b" not in model_id.lower()
    assert "coder" not in model_id.lower()
    # It must select a vision/multimodal capable provider like Gemini
    assert "gemini" in model_id.lower() or "ling" in model_id.lower()


def test_phase2_ferrari_variants_and_style_preservation():
    """Verify Phase 2 requirements: full visual request preservation and non-image rejection."""
    # Requirement A: Preserve "photorealistic red"
    ferrari_prompt = "Generate a photorealistic red Ferrari parked beside a beach at sunset."
    is_img, subject = image_service.detect_intent(ferrari_prompt)
    assert is_img is True
    assert "photorealistic" in subject.lower(), f"Expected 'photorealistic' in subject: {subject}"
    assert "red" in subject.lower(), f"Expected 'red' in subject: {subject}"
    assert "ferrari" in subject.lower(), f"Expected 'ferrari' in subject: {subject}"

    # Phase 2 test variants:
    cases = [
        ("Generate a cinematic photo of a sports car.", True, ["cinematic", "sports car"]),
        ("Draw a mountain landscape.", True, ["mountain landscape"]),
        ("Create a watercolor painting of a house.", True, ["watercolor", "house"]),
        ("Generate a Python image-processing script.", False, []),
        ("Explain how diffusion models work.", False, []),
    ]
    for prompt, expected_is_img, expected_tokens in cases:
        detected, subj = image_service.detect_intent(prompt)
        assert detected is expected_is_img, f"Mismatch for '{prompt}': expected {expected_is_img}, got {detected}"
        if expected_is_img:
            for tok in expected_tokens:
                assert tok.lower() in subj.lower(), f"Expected '{tok}' in subject '{subj}' for prompt: {prompt}"

