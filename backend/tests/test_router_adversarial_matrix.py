"""Adversarial Chaos & Multi-Provider Invariant Test Matrix for Model Router."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.services.llm.circuit_breaker import model_health_tracker
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.openrouter_provider import OpenRouterProvider
from app.services.llm.router import LLMRouter


@pytest.fixture(autouse=True)
def reset_breakers():
    model_health_tracker.reset_all()
    yield
    model_health_tracker.reset_all()


@pytest.fixture
def test_router():
    groq = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    gemini = GeminiProvider(api_key="gem_test")
    openrouter = OpenRouterProvider(api_key="sk_or_test")
    return LLMRouter(groq_provider=groq, gemini_provider=gemini, openrouter_provider=openrouter)


@pytest.mark.asyncio
async def test_adversarial_429_rate_limit_failover(test_router):
    """When primary model encounters HTTP 429, router must fail over and set fallback_used=True."""
    call_count = 0

    async def mock_post(url, json=None, **kwargs):
        nonlocal call_count
        call_count += 1
        req_model = (json or {}).get("model", "")
        if "gpt-oss" in req_model:
            resp = MagicMock()
            resp.status_code = 429
            resp.text = "Rate limit exceeded"
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("429 Rate Limit", request=MagicMock(), response=resp)
            return resp
        # Fallback succeeds
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Fallback answer from backup"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await test_router.generate_response(
            prompt="Explain quantum entanglement",
            requested_model="openai/gpt-oss-120b",
        )

    assert res.fallback_used is True
    assert res.was_failover is True
    # Invariant: requested_model != actual_model => fallback_used is True
    if res.requested_model != res.actual_model:
        assert res.fallback_used is True


@pytest.mark.asyncio
async def test_adversarial_500_503_server_error_failover(test_router):
    """HTTP 500 / 503 from upstream provider triggers fallback with clean telemetry."""
    async def mock_post(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        if "qwen" in req_model:
            resp = MagicMock()
            resp.status_code = 503
            resp.text = "Service Temporarily Unavailable"
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("503 Unavailable", request=MagicMock(), response=resp)
            return resp
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Fallback code"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await test_router.generate_response(
            prompt="Write quicksort in Python",
            requested_model="qwen/qwen3.8-27b",
        )

    assert res.fallback_used is True
    assert res.requested_model == "qwen/qwen3.8-27b"
    assert res.actual_model != "qwen/qwen3.8-27b"


@pytest.mark.asyncio
async def test_adversarial_timeout_failover(test_router):
    """Upstream read/connect timeout triggers immediate failover to backup model."""
    async def mock_post(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        if "gpt-oss" in req_model:
            raise httpx.ReadTimeout("Connection timed out after 30s")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Backup answer after timeout"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await test_router.generate_response(
            prompt="Summarize relativity",
            requested_model="openai/gpt-oss-120b",
        )

    assert res.fallback_used is True
    assert res.was_failover is True
    assert res.fallback_reason is not None



@pytest.mark.asyncio
async def test_adversarial_malformed_json_failover(test_router):
    """Upstream provider returning invalid/corrupted JSON payload must trigger fallback."""
    async def mock_post(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        if "gpt-oss" in req_model:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.side_effect = ValueError("Invalid JSON: line 1 column 1")
            return resp
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Recovered from JSON error"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await test_router.generate_response(
            prompt="Hello world",
            requested_model="openai/gpt-oss-120b",
        )

    assert res.fallback_used is True


@pytest.mark.asyncio
async def test_adversarial_empty_choices_failover(test_router):
    """Empty choices list [] from provider must be treated as degradation and fail over."""
    async def mock_post(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        if "gpt-oss" in req_model:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": []}
            return resp
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "Recovered from empty choices"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await test_router.generate_response(
            prompt="Hello world",
            requested_model="openai/gpt-oss-120b",
        )

    assert res.fallback_used is True


@pytest.mark.asyncio
async def test_concurrent_requests_zero_cross_model_leakage(test_router):
    """Concurrent requests requesting different models must never cross-contaminate model identities."""
    async def mock_post(url, json=None, **kwargs):
        req_model = (json or {}).get("model", "")
        # Yield control to encourage concurrency interleaving
        await asyncio.sleep(0.01)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": f"Response generated by {req_model}"}}]}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        tasks = [
            test_router.generate_response("Req 1", requested_model="qwen/qwen3.8-27b"),
            test_router.generate_response("Req 2", requested_model="openai/gpt-oss-120b"),
            test_router.generate_response("Req 3", requested_model="qwen/qwen3.8-27b"),
            test_router.generate_response("Req 4", requested_model="openai/gpt-oss-120b"),
        ]
        results = await asyncio.gather(*tasks)

    assert results[0].requested_model == "qwen/qwen3.8-27b"
    assert results[0].actual_model == "qwen/qwen3.8-27b"
    assert results[1].requested_model == "openai/gpt-oss-120b"
    assert results[1].actual_model == "openai/gpt-oss-120b"
    assert results[2].requested_model == "qwen/qwen3.8-27b"
    assert results[2].actual_model == "qwen/qwen3.8-27b"
    assert results[3].requested_model == "openai/gpt-oss-120b"
    assert results[3].actual_model == "openai/gpt-oss-120b"

    for r in results:
        # Strict Invariant: requested_model != actual_model => fallback_used must be True
        if r.requested_model != r.actual_model:
            assert r.fallback_used is True
        else:
            assert r.fallback_used is False
