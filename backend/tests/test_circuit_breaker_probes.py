"""HALF_OPEN must admit only a few canary probes, and MISCONFIGURED must expire."""
import time

from app.services.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    HealthState,
    ModelHealthTracker,
)


def _half_open_breaker() -> CircuitBreaker:
    cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.0, canary_success_threshold=2))
    cb.record_failure("boom")  # CLOSED -> OPEN; cooldown 0 -> HALF_OPEN on next read
    assert cb.state == CircuitState.HALF_OPEN
    return cb


def test_half_open_admits_only_canary_threshold_probes() -> None:
    cb = _half_open_breaker()
    assert cb.try_acquire() is True
    assert cb.try_acquire() is True
    assert cb.try_acquire() is False  # third concurrent request is diverted
    assert cb.allow_request() is False


def test_release_frees_a_probe_slot() -> None:
    cb = _half_open_breaker()
    assert cb.try_acquire() and cb.try_acquire()
    cb.release()
    assert cb.allow_request() is True
    assert cb.try_acquire() is True


def test_allow_request_does_not_consume_slots() -> None:
    cb = _half_open_breaker()
    for _ in range(10):
        assert cb.allow_request() is True
    assert cb.try_acquire() is True


def test_two_canary_successes_close_the_circuit() -> None:
    cb = _half_open_breaker()
    for _ in range(2):
        assert cb.try_acquire()
        cb.record_success()
        cb.release()
    assert cb.state == CircuitState.CLOSED


def test_failed_canary_counts_as_new_trip() -> None:
    cb = _half_open_breaker()
    trips_before = cb.metrics.total_trips
    assert cb.try_acquire()
    cb.record_failure("still down")
    cb.release()
    assert cb.metrics.state == CircuitState.OPEN
    assert cb.metrics.total_trips == trips_before + 1


def test_misconfigured_model_recovers_after_cooldown() -> None:
    tracker = ModelHealthTracker()
    tracker.record_failure("groq", "llama-test", "unauthorized", status_code=401)
    assert tracker.get_health("groq", "llama-test") == HealthState.MISCONFIGURED

    tracker.get_record("groq", "llama-test").last_state_change = time.time() - 301
    assert tracker.get_health("groq", "llama-test") == HealthState.DEGRADED
    assert tracker.is_eligible("groq", "llama-test") is True


def test_http_400_does_not_mark_model_misconfigured() -> None:
    tracker = ModelHealthTracker()
    tracker.record_failure("groq", "llama-test", "context length exceeded", status_code=400)
    assert tracker.get_health("groq", "llama-test") != HealthState.MISCONFIGURED
