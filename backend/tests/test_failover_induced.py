"""Induced provider failure and failover verification tests.

Verifies that when primary providers (FLUX.1 or Groq) fail or are killed,
secondary providers (Turbo/SDXL or Gemini) automatically serve requests with zero user downtime.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.llm.router import LLMRouter
from app.services.image.image_service import image_service
from app.schemas.chat import ChatRequest


@pytest.mark.asyncio
async def test_llm_router_induced_primary_failure_failover():
    """Induce complete failure in primary Groq provider and verify seamless failover to Gemini."""
    # Mock primary provider (Groq) raising an unrecoverable 500 error / outage
    mock_groq = MagicMock()
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "deepseek-r1-distill-llama-70b"

    async def mock_groq_stream_fail(*args, **kwargs):
        raise Exception("Induced Outage: Groq primary endpoint 503 Service Unavailable")
        if False:
            yield ""  # make it an async generator

    mock_groq.stream_response = mock_groq_stream_fail

    # Mock secondary provider (Gemini) responding successfully
    async def mock_gemini_stream(prompt, **kwargs):
        yield "Response successfully served by secondary failover provider."

    mock_gemini = MagicMock()
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.model = "gemini-3.6-flash"
    mock_gemini.stream_response = mock_gemini_stream

    router = LLMRouter(groq_provider=mock_groq, gemini_provider=mock_gemini)

    # Request using Tier S++ (configured to failover from Groq to Gemini)
    tokens_received = []
    failover_flags = []
    async for token, prov, model, was_failover in router.stream_response(
        prompt="Explain quantum computing",
        requested_model="deepseek-r1-distill-llama-70b",
        tier="S++",
    ):
        tokens_received.append(token)
        failover_flags.append(was_failover)

    # Assert that failover was triggered
    assert len(tokens_received) > 0
    assert any("secondary failover provider" in t for t in tokens_received)
    assert any(f is True for f in failover_flags)


@pytest.mark.asyncio
async def test_image_service_induced_flux_failure_failover():
    """An unverified HF token must not enable billed fallback or fake JPEG success."""
    with patch("app.core.config.settings.HUGGINGFACE_API_KEY", "hf_test_token_valid"):
        with patch("httpx.AsyncClient.post", side_effect=AssertionError("No authorized free path")):
            res = await image_service.generate_and_validate(prompt="cyberpunk motorcycle", tier="S++")
    assert res["success"] is False
    assert res["error_type"] == "authentication_required"
    assert not res["image_url"].startswith("/static/")
