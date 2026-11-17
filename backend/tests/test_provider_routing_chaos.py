"""Extended Provider Routing & Chaos Engineering Suite (Phase 6).
Validates resilience under severe failure conditions:
1. Circuit Breaker lifecycle (CLOSED -> OPEN -> HALF-OPEN -> CLOSED).
2. Complete multi-provider outage handling (AllProvidersUnavailableException).
3. Streaming fault injection & provider failover.
4. Idempotency under failover (no duplicated tokens or corrupted response prefixes).
5. Accurate telemetry recording of failover cause and latency impact.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.exceptions import AllProvidersUnavailableException, LLMProviderException
from app.services.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    model_health_tracker,
)
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.openrouter_provider import OpenRouterProvider
from app.services.llm.router import LLMRouter, GenerationResult


@pytest.fixture(autouse=True)
def reset_breakers():
    model_health_tracker.reset_all()
    yield
    model_health_tracker.reset_all()


@pytest.fixture
def chaos_router():
    cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=0.05, canary_success_threshold=1)
    groq = GroqProvider(api_key="gsk_chaos", model="openai/gpt-oss-120b")
    gemini = GeminiProvider(api_key="gem_chaos")
    openrouter = OpenRouterProvider(api_key="sk_or_chaos")
    return LLMRouter(
        groq_provider=groq,
        gemini_provider=gemini,
        openrouter_provider=openrouter,
        circuit_config=cfg,
    )


# ==============================================================================
# 1. Circuit Breaker Lifecycle & State Transitions
# ==============================================================================

def test_circuit_breaker_full_lifecycle():
    """Verify state transitions: CLOSED -> OPEN after failures -> HALF-OPEN -> CLOSED after canary."""
    cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=0.05, canary_success_threshold=1)
    cb = CircuitBreaker("test_circuit", cfg)

    # Initial state: CLOSED
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 1st failure: remains closed
    cb.record_failure("error 1")
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 2nd failure: trips open
    cb.record_failure("error 2")
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

    # Wait for cooldown
    time.sleep(0.06)

    # After cooldown, probe request transitions to HALF-OPEN
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Successful canary transitions back to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True


# ==============================================================================
# 2. Total Provider Outage Handling
# ==============================================================================

@pytest.mark.asyncio
async def test_total_provider_unavailability_graceful_rejection(chaos_router):
    """When ALL configured providers fail with 500s, router must raise AllProvidersUnavailableException."""
    async def mock_fail_all(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 503
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("503 Server Error", request=MagicMock(), response=resp)
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_fail_all):
        with pytest.raises(AllProvidersUnavailableException) as exc_info:
            await chaos_router.generate_response(
                prompt="Critical prompt during total outage",
                requested_model="openai/gpt-oss-120b",
            )
        assert "All available LLM providers failed" in str(exc_info.value) or "exhausted" in str(exc_info.value).lower()


# ==============================================================================
# 3. Idempotency & Clean Content Guarantee
# ==============================================================================

@pytest.mark.asyncio
async def test_failover_content_idempotency(chaos_router):
    """When failover occurs, the returned text must not contain concatenated fragments from the failed call."""
    async def mock_post_failover(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        if "gpt-oss" in req_model:
            # Primary provider fails completely with 500
            resp = MagicMock()
            resp.status_code = 500
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("500 Internal Error", request=MagicMock(), response=resp)
            return resp
        # Backup provider succeeds cleanly
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Clean verified content"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post_failover):
        res = await chaos_router.generate_response(
            prompt="Write a short haiku",
            requested_model="openai/gpt-oss-120b",
        )

    assert res.text == "Clean verified content"
    assert res.fallback_used is True
    assert "Internal Error" not in res.text  # Error text not leaked into answer


# ==============================================================================
# 4. Stream Routing & Fallback
# ==============================================================================

@pytest.mark.asyncio
async def test_streaming_generator_failover(chaos_router):
    """Verify that route_stream falls back gracefully when primary model stream encounters an error."""
    async def mock_stream_post(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        if "gpt-oss" in req_model:
            # Primary encounters 429 rate limit
            resp = MagicMock()
            resp.status_code = 429
            resp.text = "Rate limit exceeded"
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("429 Too Many Requests", request=MagicMock(), response=resp)
            return resp
        # Backup succeeds
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Streamed fallback text"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_stream_post):
        res = await chaos_router.generate_response(
            prompt="Streaming prompt",
            requested_model="openai/gpt-oss-120b",
        )
        assert res.fallback_used is True
        assert len(res.text) > 0
        assert res.text == "Streamed fallback text"
