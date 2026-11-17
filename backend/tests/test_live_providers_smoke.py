"""Optional Live Provider Smoke Tests for Jruva AI Platform.

GATED BY ENVIRONMENT VARIABLE:
These tests ONLY run when explicitly enabled via:
  RUN_LIVE_PROVIDER_TESTS=1

In normal CI or local test execution, these tests are automatically skipped to avoid
incurring upstream API costs, rate-limit consumption, or failures due to offline environments.

Verifies:
  1. Gemini live endpoint with active model (gemini-2.0-flash)
  2. Groq live endpoint with active model (llama-3.3-70b-versatile)
  3. OpenRouter live endpoint with active model (liquid/lfm-2.5-2.6b:free)
"""

import os
import pytest

RUN_LIVE = os.getenv("RUN_LIVE_PROVIDER_TESTS", "").strip() == "1"


@pytest.mark.skipif(not RUN_LIVE, reason="Live provider tests require RUN_LIVE_PROVIDER_TESTS=1")
@pytest.mark.asyncio
async def test_live_gemini_smoke():
    """Verify live Gemini API connectivity and non-empty token generation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not configured in environment")

    from app.services.llm.gemini_provider import GeminiProvider
    provider = GeminiProvider(api_key=api_key)
    response = await provider.generate_response(
        prompt="Respond with exactly one word: 'READY'.",
        model="gemini-2.0-flash",
    )
    assert response and len(response.strip()) > 0
    assert "READY" in response.upper()


@pytest.mark.skipif(not RUN_LIVE, reason="Live provider tests require RUN_LIVE_PROVIDER_TESTS=1")
@pytest.mark.asyncio
async def test_live_groq_smoke():
    """Verify live Groq Cloud API connectivity and fast inference."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not configured in environment")

    from app.services.llm.groq_provider import GroqProvider
    provider = GroqProvider(api_key=api_key)
    response = await provider.generate_response(
        prompt="Respond with exactly one word: 'CONFIRMED'.",
        model="llama-3.3-70b-versatile",
    )
    assert response and len(response.strip()) > 0
    assert "CONFIRMED" in response.upper()


@pytest.mark.skipif(not RUN_LIVE, reason="Live provider tests require RUN_LIVE_PROVIDER_TESTS=1")
@pytest.mark.asyncio
async def test_live_openrouter_smoke():
    """Verify live OpenRouter API connectivity with active model mapping."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not configured in environment")

    from app.services.llm.openrouter_provider import OpenRouterProvider
    provider = OpenRouterProvider(api_key=api_key)
    response = await provider.generate_response(
        prompt="Respond with: 'ONLINE'.",
        model="liquid/lfm-2.5-2.6b:free",
    )
    assert response and len(response.strip()) > 0
    assert "ONLINE" in response.upper()
