"""Comprehensive Unit & Integration Tests for Circuit Breaker and LLM Router Resilience."""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from app.services.llm.router import LLMRouter


def test_circuit_breaker_fsm_state_transitions():
    """Verify finite state machine transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        cooldown_seconds=0.2,  # 200ms for fast testing
        canary_success_threshold=2,
    )
    cb = CircuitBreaker(provider_name="groq", config=config)

    # 1. Initial state is CLOSED
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 2. Record 2 failures: still CLOSED (threshold is 3)
    cb.record_failure("HTTP 429 Rate Limit")
    cb.record_failure("HTTP 429 Rate Limit")
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 3. 3rd failure: trips to OPEN
    cb.record_failure("HTTP 503 Service Unavailable")
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False
    assert cb.metrics.total_trips == 1

    # 4. Wait for cooldown period (200ms)
    time.sleep(0.25)

    # 5. Cooldown elapsed: state should transition to HALF_OPEN on query
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request() is True

    # 6. Canary probe 1 succeeds
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN

    # 7. Canary probe 2 succeeds: resets to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True
    assert cb.metrics.consecutive_failures == 0
    assert len(cb.metrics.recovery_times_sec) == 1
    assert cb.metrics.recovery_times_sec[0] > 0.2


def test_circuit_breaker_canary_failure_reverts_to_open():
    """Verify that a failing canary probe immediately trips HALF_OPEN back to OPEN."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        cooldown_seconds=0.1,
        canary_success_threshold=2,
    )
    cb = CircuitBreaker(provider_name="gemini", config=config)

    # Trip immediately with 1 failure
    cb.record_failure("HTTP 500")
    assert cb.state == CircuitState.OPEN

    # Wait for cooldown
    time.sleep(0.12)
    assert cb.state == CircuitState.HALF_OPEN

    # Canary fails
    cb.record_failure("HTTP 503 on canary")
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False


@pytest.mark.asyncio
async def test_router_fast_path_failover_when_circuit_open():
    """Verify router immediately bypasses a tripped primary provider without calling it."""
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "groq"
    mock_primary.is_available = True
    mock_primary.model = "openai/gpt-oss-120b"
    mock_primary.generate_response = AsyncMock(side_effect=QuotaExhaustedException("Groq 429"))

    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "gemini"
    mock_backup.is_available = True
    mock_backup.model = "gemini-3.7-flash"
    mock_backup.generate_response = AsyncMock(return_value="Recovered via Gemini")

    config = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=10.0)
    router = LLMRouter(gemini_provider=mock_backup, groq_provider=mock_primary, circuit_config=config)

    # Request 1: Primary fails, fails over to backup
    text, provider, _, was_failover = await router.generate_response(prompt="Test 1", requested_model="groq")
    assert was_failover is True
    assert provider == "gemini"
    assert mock_primary.generate_response.call_count == 1

    # Request 2: Primary fails again -> circuit trips to OPEN!
    text, provider, _, was_failover = await router.generate_response(prompt="Test 2", requested_model="groq")
    assert was_failover is True
    assert mock_primary.generate_response.call_count == 2
    assert router.get_breaker("groq").state == CircuitState.OPEN

    # Request 3: Primary circuit is OPEN!
    # Primary generate_response must NOT be called; fast-path straight to backup!
    mock_primary.generate_response.reset_mock()
    text, provider, _, was_failover = await router.generate_response(prompt="Test 3", requested_model="groq")
    assert text == "Recovered via Gemini"
    assert provider == "gemini"
    assert was_failover is True
    # Crucial assertion: primary was bypassed with zero wasted network attempts!
    assert mock_primary.generate_response.call_count == 0


@pytest.mark.asyncio
async def test_router_cascade_failure_records_drop():
    """Verify that when both primary and backup fail, a dropped request is logged."""
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "groq"
    mock_primary.is_available = True
    mock_primary.model = "openai/gpt-oss-120b"
    mock_primary.generate_response = AsyncMock(side_effect=LLMProviderException("Primary 500"))

    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "gemini"
    mock_backup.is_available = True
    mock_backup.model = "gemini-3.7-flash"
    mock_backup.generate_response = AsyncMock(side_effect=LLMProviderException("Backup 500"))

    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    mock_openrouter.provider_name = "openrouter"
    mock_openrouter.is_available = True
    mock_openrouter.model = "liquid/lfm-2.5-2.6b:free"
    mock_openrouter.generate_response = AsyncMock(side_effect=LLMProviderException("OpenRouter 500"))

    router = LLMRouter(
        gemini_provider=mock_backup,
        groq_provider=mock_primary,
        openrouter_provider=mock_openrouter,
    )

    with pytest.raises(LLMProviderException):
        await router.generate_response(prompt="Failing", requested_model="groq")

    metrics = router.get_circuit_metrics()
    assert metrics["groq"]["dropped_requests"] >= 1


@pytest.mark.asyncio
async def test_router_streaming_circuit_failover():
    """Verify streaming responses measure failover latency to first token."""
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "groq"
    mock_primary.is_available = True
    mock_primary.model = "openai/gpt-oss-120b"

    async def failing_stream(*args, **kwargs):
        raise QuotaExhaustedException("Groq rate limit 429")
        yield "token"

    mock_primary.stream_response = failing_stream

    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "gemini"
    mock_backup.is_available = True
    mock_backup.model = "gemini-3.7-flash"

    async def backup_stream(*args, **kwargs):
        await asyncio.sleep(0.05)  # 50ms latency simulation
        yield "Hello "
        yield "from "
        yield "Gemini"

    mock_backup.stream_response = backup_stream

    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    mock_openrouter.provider_name = "openrouter"
    mock_openrouter.is_available = True
    mock_openrouter.model = "liquid/lfm-2.5-2.6b:free"

    router = LLMRouter(
        gemini_provider=mock_backup,
        groq_provider=mock_primary,
        openrouter_provider=mock_openrouter,
    )

    tokens = []
    async for token, provider, _, was_failover in router.stream_response(prompt="Hi", requested_model="groq"):
        tokens.append(token)

    assert "".join(tokens) == "Hello from Gemini"
    summary = router.get_breaker("groq").get_summary()
    assert summary["failover_events"] == 1
    # Latency to first token was at least 50ms
    assert summary["latency_p50_ms"] >= 40.0
