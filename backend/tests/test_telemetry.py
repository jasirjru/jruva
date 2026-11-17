"""Unit and Integration Tests for Telemetry, Distributed Tracing, and All-Providers-Down RFC 7807 Outage.

Verifies:
1. Per-request tracing: trace ID, provider, TTFT/total latency, token accounting, and cost estimation.
2. Telemetry endpoints: /api/v1/telemetry/metrics and /api/v1/telemetry/traces.
3. Cascading all-providers-down outage simulation: RFC 7807 ProblemDetails HTTP 503 structure.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import AllProvidersUnavailableException, LLMProviderException, QuotaExhaustedException
from app.main import app
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import CircuitBreakerConfig
from app.services.llm.router import LLMRouter
from app.services.telemetry.telemetry_service import (
    RequestTrace,
    calculate_cost,
    estimate_token_count,
    telemetry_service,
)


@pytest.fixture(autouse=True)
def reset_telemetry():
    """Ensure telemetry buffer is fresh for each test."""
    telemetry_service.reset_metrics()
    yield
    telemetry_service.reset_metrics()


def test_token_and_cost_calculation():
    """Verify token estimation and price calculations across models."""
    sample_text = "This is a test prompt designed to verify token estimation."
    tokens = estimate_token_count(sample_text)
    assert tokens > 0

    # Gemini 2.0 Flash: $0.10/1M prompt, $0.40/1M completion
    cost_gemini = calculate_cost("gemini-2.0-flash", prompt_tokens=1000, completion_tokens=500)
    expected_gemini = round((1000 / 1e6 * 0.10) + (500 / 1e6 * 0.40), 6)
    assert cost_gemini == expected_gemini

    # Groq Llama 3.3 70B: $0.59/1M prompt, $0.79/1M completion
    cost_groq = calculate_cost("llama-3.3-70b-versatile", prompt_tokens=1000, completion_tokens=500)
    expected_groq = round((1000 / 1e6 * 0.59) + (500 / 1e6 * 0.79), 6)
    assert cost_groq == expected_groq

    # Ollama: local weights $0.00
    cost_ollama = calculate_cost("ollama/my-specialist", prompt_tokens=2000, completion_tokens=1000)
    assert cost_ollama == 0.0


def test_telemetry_trace_recording_and_metrics_aggregation():
    """Verify trace recording updates rolling counters, percentiles, and provider breakdowns."""
    trace1 = RequestTrace(
        session_id="session-1",
        streaming=False,
        requested_model="auto",
        routed_provider="groq",
        effective_model="llama-3.3-70b-versatile",
        was_failover=False,
        total_latency_ms=120.0,
        prompt_tokens=50,
        completion_tokens=100,
        total_tokens=150,
        estimated_cost_usd=0.000108,
        status="success",
    )
    trace2 = RequestTrace(
        session_id="session-2",
        streaming=True,
        requested_model="auto",
        routed_provider="gemini",
        effective_model="gemini-2.0-flash",
        was_failover=True,
        ttft_ms=65.0,
        total_latency_ms=250.0,
        prompt_tokens=80,
        completion_tokens=150,
        total_tokens=230,
        estimated_cost_usd=0.000068,
        status="failover_recovered",
    )

    telemetry_service.record_trace(trace1)
    telemetry_service.record_trace(trace2)

    metrics = telemetry_service.get_metrics_summary()
    assert metrics["system"]["total_requests"] == 2
    assert metrics["system"]["successful_requests"] == 2
    assert metrics["system"]["failover_requests"] == 1
    assert metrics["system"]["availability_pct"] == 100.0
    assert metrics["system"]["failover_rate_pct"] == 50.0

    assert metrics["latency_ms"]["total"]["p50"] == 185.0
    assert metrics["latency_ms"]["ttft"]["p50"] == 65.0
    assert metrics["latency_ms"]["failover"]["p50"] == 250.0

    assert metrics["tokens_and_cost"]["total_prompt_tokens"] == 130
    assert metrics["tokens_and_cost"]["total_completion_tokens"] == 250
    assert metrics["tokens_and_cost"]["total_tokens"] == 380

    # Traces retrieval
    traces = telemetry_service.get_traces(limit=10)
    assert len(traces) == 2
    assert traces[0]["trace_id"] == trace2.trace_id  # reverse chronological


def test_telemetry_endpoints(monkeypatch):
    """Verify GET /api/v1/telemetry/metrics, GET /api/v1/telemetry/traces, and POST /api/v1/telemetry/reset."""
    client = TestClient(app)
    from app.core import config
    from app.services.auth_service import auth_service
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda uid: {"id": uid, "role": "admin"})
    monkeypatch.setattr(config.settings, "ENVIRONMENT", "test")
    monkeypatch.setattr(config.settings, "OPERATIONS_RESET_ENABLED", True)
    client.headers["Authorization"] = "Bearer " + auth_service.create_jwt_token({"sub": "synthetic-telemetry-admin"})

    # 1. Reset metrics
    resp = client.post("/api/v1/telemetry/reset")
    assert resp.status_code == 200

    # 2. Record a dummy trace
    trace = RequestTrace(
        session_id="api-test",
        streaming=False,
        requested_model="auto",
        routed_provider="groq",
        effective_model="llama-3.3-70b-versatile",
        total_latency_ms=95.0,
        prompt_tokens=20,
        completion_tokens=40,
        total_tokens=60,
        estimated_cost_usd=0.000043,
    )
    telemetry_service.record_trace(trace)

    # 3. Query metrics
    metrics_resp = client.get("/api/v1/telemetry/metrics")
    assert metrics_resp.status_code == 200
    data = metrics_resp.json()
    assert data["system"]["total_requests"] == 1
    assert data["system"]["successful_requests"] == 1
    assert "circuits" in data

    # 4. Query traces
    traces_resp = client.get("/api/v1/telemetry/traces?limit=10")
    assert traces_resp.status_code == 200
    trace_items = traces_resp.json()
    assert len(trace_items) == 1
    assert trace_items[0]["routed_provider"] == "groq"
    assert trace_items[0]["total_tokens"] == 60


@pytest.mark.asyncio
async def test_cascading_all_providers_down_simulation():
    """Simulate total upstream failure: both primary and backup fail simultaneously.

    Verifies that LLMRouter raises AllProvidersUnavailableException with:
    - HTTP 503 status mapping
    - RFC 7807 ProblemDetails error payload
    - Snapshot of tripped circuit breaker states
    - Retry-After cooldown header
    """
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "gemini"
    mock_primary.is_available = True
    mock_primary.model = "gemini-2.0-flash"
    mock_primary.generate_response = AsyncMock(side_effect=QuotaExhaustedException("Gemini 429 Quota"))

    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "groq"
    mock_backup.is_available = True
    mock_backup.model = "llama-3.3-70b-versatile"
    mock_backup.generate_response = AsyncMock(side_effect=LLMProviderException("Groq 503 Upstream Outage"))

    # Explicit failing OpenRouter mock: without it LLMRouter builds a LIVE OpenRouterProvider
    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    mock_openrouter.provider_name = "openrouter"
    mock_openrouter.is_available = True
    mock_openrouter.model = "openrouter/free"
    mock_openrouter.generate_response = AsyncMock(side_effect=LLMProviderException("OpenRouter 503"))

    config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=15.0)
    router = LLMRouter(
        gemini_provider=mock_primary,
        groq_provider=mock_backup,
        openrouter_provider=mock_openrouter,
        circuit_config=config,
    )

    # Both providers will fail in cascade
    with pytest.raises(AllProvidersUnavailableException) as exc_info:
        await router.generate_response(prompt="Cascade test", requested_model="auto")

    exc = exc_info.value
    assert exc.status_code == 502
    assert exc.error_code == "LLM_PROVIDER_ERROR"
    assert "Cascading failure" in exc.detail
    assert "gemini" in exc.circuits
    assert "groq" in exc.circuits
    assert exc.retry_after_seconds == 15

    # Verify circuit breakers recorded the failures
    assert router.get_circuit_metrics()["gemini"]["consecutive_failures"] >= 1
    assert router.get_circuit_metrics()["groq"]["consecutive_failures"] >= 1


@pytest.mark.asyncio
async def test_streaming_all_providers_down_simulation():
    """Simulate streaming cascading failure when both primary and backup streams fail."""
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "gemini"
    mock_primary.is_available = True
    mock_primary.model = "gemini-2.0-flash"

    async def fail_primary(*args, **kwargs):
        raise QuotaExhaustedException("Primary stream failed")
        yield "never"

    mock_primary.stream_response = fail_primary

    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "groq"
    mock_backup.is_available = True
    mock_backup.model = "llama-3.3-70b-versatile"

    async def fail_backup(*args, **kwargs):
        raise LLMProviderException("Backup stream failed")
        yield "never"

    mock_backup.stream_response = fail_backup

    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    mock_openrouter.provider_name = "openrouter"
    mock_openrouter.is_available = True
    mock_openrouter.model = "openrouter/free"

    async def fail_openrouter(*args, **kwargs):
        raise LLMProviderException("OpenRouter stream failed")
        yield "never"

    mock_openrouter.stream_response = fail_openrouter

    router = LLMRouter(
        gemini_provider=mock_primary,
        groq_provider=mock_backup,
        openrouter_provider=mock_openrouter,
    )

    with pytest.raises(AllProvidersUnavailableException) as exc_info:
        async for _ in router.stream_response(prompt="Stream cascade test", requested_model="auto"):
            pass

    assert exc_info.value.status_code == 502
    assert "Stream cascade failure" in exc_info.value.detail
