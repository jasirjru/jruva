"""Stateful Circuit Breaker for multi-provider LLM routing and resilience tracking.

Implements a Finite State Machine:
  CLOSED -> [failure_threshold reached] -> OPEN
  OPEN -> [cooldown_seconds elapsed] -> HALF_OPEN
  HALF_OPEN -> [canary_success_threshold reached] -> CLOSED
  HALF_OPEN -> [any failure] -> OPEN
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Possible operational states of a provider circuit."""
    CLOSED = "CLOSED"         # Normal operation: all traffic sent to this provider
    OPEN = "OPEN"             # Provider unhealthy: fail-fast / divert to backup
    HALF_OPEN = "HALF_OPEN"   # Cooldown elapsed: dispatching canary probe requests


class HealthState(str, Enum):
    """Granular operational health states for individual models and providers."""
    HEALTHY = "HEALTHY"             # Normal operation; successful recent inference
    DEGRADED = "DEGRADED"           # Slower response or intermittent non-terminal errors
    RATE_LIMITED = "RATE_LIMITED"   # HTTP 429 quota reached; fast-skip during cooldown
    DOWN = "DOWN"                   # Server unreachable, socket connection refused, or 5xx outage
    MISCONFIGURED = "MISCONFIGURED" # HTTP 404, 401, 403, or decommissioned upstream model ID


@dataclass
class ModelHealthRecord:
    """State tracking record for an individual model within a provider."""
    provider: str
    model_id: str
    state: HealthState = HealthState.HEALTHY
    consecutive_failures: int = 0
    last_state_change: float = field(default_factory=time.time)
    last_failure_reason: Optional[str] = None
    cooldown_seconds: float = 60.0


class ModelHealthTracker:
    """Central state machine tracking operational health per (provider, model_id) tuple."""

    def __init__(self) -> None:
        self._records: Dict[str, ModelHealthRecord] = {}

    def _key(self, provider: str, model_id: str) -> str:
        return f"{provider.lower().strip()}:{model_id.lower().strip()}"

    def get_record(self, provider: str, model_id: str) -> ModelHealthRecord:
        key = self._key(provider, model_id)
        if key not in self._records:
            self._records[key] = ModelHealthRecord(provider=provider, model_id=model_id)
        return self._records[key]

    def get_health(self, provider: str, model_id: str) -> HealthState:
        """Evaluate current health state, automatically expiring cooldown periods."""
        from app.core.model_registry import is_decommissioned
        if is_decommissioned(model_id):
            return HealthState.MISCONFIGURED

        rec = self.get_record(provider, model_id)
        # MISCONFIGURED also expires: a single 401/403/404 must not disable a model
        # for every user until the process restarts.
        if rec.state in (
            HealthState.RATE_LIMITED,
            HealthState.DEGRADED,
            HealthState.DOWN,
            HealthState.MISCONFIGURED,
        ):
            elapsed = time.time() - rec.last_state_change
            if elapsed >= rec.cooldown_seconds:
                # Probational canary eligibility: transition to DEGRADED
                rec.state = HealthState.DEGRADED
        return rec.state

    def is_eligible(self, provider: str, model_id: str) -> bool:
        """Check whether a model should be attempted (HEALTHY or probe DEGRADED)."""
        current = self.get_health(provider, model_id)
        return current in (HealthState.HEALTHY, HealthState.DEGRADED)

    def has_any_eligible_model(self, provider: str, candidate_models: list[str]) -> bool:
        """Check whether at least one candidate model for this provider is eligible."""
        if not candidate_models:
            return True
        return any(self.is_eligible(provider, m) for m in candidate_models)

    def reset(self) -> None:
        """Reset all tracked health records (used in test isolation)."""
        self._records.clear()

    def reset_all(self) -> None:
        """Alias for reset."""
        self.reset()

    def record_success(self, provider: str, model_id: str) -> None:
        """Record successful inference on model."""
        rec = self.get_record(provider, model_id)
        rec.state = HealthState.HEALTHY
        rec.consecutive_failures = 0
        rec.last_state_change = time.time()
        rec.last_failure_reason = None

    def record_failure(
        self,
        provider: str,
        model_id: str,
        reason: str,
        status_code: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Record model-specific failure and update health classification."""
        rec = self.get_record(provider, model_id)
        rec.consecutive_failures += 1
        rec.last_failure_reason = reason
        rec.last_state_change = time.time()

        reason_lower = reason.lower()
        if status_code == 429 or "429" in reason or "quota" in reason_lower or "rate limit" in reason_lower:
            rec.state = HealthState.RATE_LIMITED
            rec.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else 60.0
            logger.warning(f"[HealthTracker] {provider}:{model_id} marked RATE_LIMITED (cooldown {rec.cooldown_seconds:.1f}s): {reason}")
        elif status_code in (401, 403) or "unauthorized" in reason_lower or "forbidden" in reason_lower:
            rec.state = HealthState.MISCONFIGURED
            rec.cooldown_seconds = 300.0
            logger.warning(f"[HealthTracker] {provider}:{model_id} marked MISCONFIGURED (auth): {reason}")
        elif (
            # 400 is usually a bad prompt (too long, filtered), not a broken model.
            status_code == 404
            or "not found" in reason_lower
            or "decommissioned" in reason_lower
            or "does not exist" in reason_lower
        ):
            rec.state = HealthState.MISCONFIGURED
            rec.cooldown_seconds = 300.0
            logger.warning(f"[HealthTracker] {provider}:{model_id} marked MISCONFIGURED (decommissioned/404): {reason}")
        elif "connecterror" in reason_lower or "offline" in reason_lower or "unreachable" in reason_lower:
            rec.state = HealthState.DOWN
            rec.cooldown_seconds = 60.0
            logger.warning(f"[HealthTracker] {provider}:{model_id} marked DOWN: {reason}")
        elif status_code in (500, 502, 503, 504) or "server error" in reason_lower or "timeout" in reason_lower:
            rec.state = HealthState.DOWN if rec.consecutive_failures >= 3 else HealthState.DEGRADED
            rec.cooldown_seconds = 30.0
            logger.warning(f"[HealthTracker] {provider}:{model_id} marked {rec.state.value}: {reason}")
        else:
            rec.state = HealthState.DEGRADED
            rec.cooldown_seconds = 30.0

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic health summary across all tracked models."""
        return {
            k: {
                "state": self.get_health(r.provider, r.model_id).value,
                "consecutive_failures": r.consecutive_failures,
                "last_failure_reason": r.last_failure_reason,
                "cooldown_remaining_sec": max(0.0, round(r.cooldown_seconds - (time.time() - r.last_state_change), 1)),
            }
            for k, r in self._records.items()
        }


# Global model health state tracker singleton
model_health_tracker = ModelHealthTracker()


@dataclass
class CircuitBreakerConfig:
    """Configuration parameters for circuit trip and recovery thresholds."""
    failure_threshold: int = 3           # Consecutive failures to trip CLOSED -> OPEN
    cooldown_seconds: float = 15.0       # Duration in OPEN before testing HALF_OPEN
    canary_success_threshold: int = 2    # Consecutive successes to reset HALF_OPEN -> CLOSED


@dataclass
class CircuitBreakerMetrics:
    """Telemetry counters and latency logs for circuit breaker monitoring."""
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_trips: int = 0
    dropped_requests: int = 0
    failover_events: int = 0
    last_state_change: float = field(default_factory=time.time)
    last_trip_time: Optional[float] = None
    last_failure_reason: Optional[str] = None
    failover_latencies_ms: List[float] = field(default_factory=list)
    recovery_times_sec: List[float] = field(default_factory=list)


class CircuitBreaker:
    """Stateful circuit breaker guarding an individual LLM provider."""

    def __init__(self, provider_name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.provider_name = provider_name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        # Number of HALF_OPEN canary probes currently running. Capped at
        # canary_success_threshold so a still-broken provider is not flooded.
        self._probes_in_flight = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, automatically evaluating cooldown expiry."""
        if self.metrics.state == CircuitState.OPEN:
            elapsed = time.time() - self.metrics.last_state_change
            if elapsed >= self.config.cooldown_seconds:
                self._transition_to(CircuitState.HALF_OPEN, reason=f"Cooldown of {self.config.cooldown_seconds}s elapsed")
        return self.metrics.state

    def allow_request(self) -> bool:
        """Read-only check: could a request be attempted against this provider now?

        Returns:
            True if CLOSED, or HALF_OPEN with a free canary probe slot.
            False if OPEN, or HALF_OPEN with all probe slots taken.
        Use try_acquire() when actually dispatching a request.
        """
        current_state = self.state
        if current_state == CircuitState.OPEN:
            return False
        if current_state == CircuitState.HALF_OPEN:
            return self._probes_in_flight < self.config.canary_success_threshold
        return True

    def try_acquire(self) -> bool:
        """Reserve permission to dispatch one request. Pair with release()."""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            return False
        if current_state == CircuitState.HALF_OPEN:
            if self._probes_in_flight >= self.config.canary_success_threshold:
                return False
            self._probes_in_flight += 1
        return True

    def release(self) -> None:
        """Free a HALF_OPEN probe slot reserved by try_acquire (safe to call in finally)."""
        if self._probes_in_flight > 0:
            self._probes_in_flight -= 1

    def record_success(self) -> None:
        """Record a successful request execution."""
        self.metrics.total_requests += 1
        self.metrics.total_successes += 1

        if self.metrics.state == CircuitState.HALF_OPEN:
            self.metrics.consecutive_successes += 1
            logger.info(
                f"[CircuitBreaker:{self.provider_name}] Canary probe success "
                f"({self.metrics.consecutive_successes}/{self.config.canary_success_threshold})"
            )
            if self.metrics.consecutive_successes >= self.config.canary_success_threshold:
                recovery_time = (
                    time.time() - self.metrics.last_trip_time
                    if self.metrics.last_trip_time else 0.0
                )
                self.metrics.recovery_times_sec.append(recovery_time)
                self._transition_to(
                    CircuitState.CLOSED,
                    reason=f"Canary threshold reached. Recovered in {recovery_time:.2f}s"
                )
                self.metrics.consecutive_failures = 0
                self.metrics.consecutive_successes = 0

        elif self.metrics.state == CircuitState.CLOSED:
            self.metrics.consecutive_failures = 0

    def record_failure(self, reason: str) -> None:
        """Record a request failure and evaluate state transitions."""
        self.metrics.total_requests += 1
        self.metrics.total_failures += 1
        self.metrics.last_failure_reason = reason
        self.metrics.consecutive_failures += 1

        logger.warning(
            f"[CircuitBreaker:{self.provider_name}] Failure recorded: {reason} "
            f"(Consecutive: {self.metrics.consecutive_failures}/{self.config.failure_threshold})"
        )

        if self.metrics.state == CircuitState.HALF_OPEN:
            # Canary failed; immediately trip back to OPEN
            self._transition_to(CircuitState.OPEN, reason=f"Canary probe failed: {reason}")
            self.metrics.consecutive_successes = 0

        elif self.metrics.state == CircuitState.CLOSED:
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                self.metrics.total_trips += 1
                self.metrics.last_trip_time = time.time()
                self._transition_to(
                    CircuitState.OPEN,
                    reason=f"Threshold reached ({self.metrics.consecutive_failures} failures): {reason}"
                )

    def record_failover(self, latency_ms: float) -> None:
        """Log a successful failover event and its transition latency with sub-millisecond precision."""
        self.metrics.failover_events += 1
        self.metrics.failover_latencies_ms.append(latency_ms)
        if latency_ms < 0.1:
            latency_str = f"{latency_ms * 1000.0:.1f}us (in-memory transition)"
        elif latency_ms < 1.0:
            latency_str = f"{latency_ms:.3f}ms"
        else:
            latency_str = f"{latency_ms:.1f}ms"

        logger.info(
            f"[CircuitBreaker:{self.provider_name}] Failover dispatch completed in {latency_str} "
            f"(Total failovers: {self.metrics.failover_events})"
        )

    def record_drop(self) -> None:
        """Log an unrecoverable dropped request where all providers failed."""
        self.metrics.dropped_requests += 1
        logger.error(
            f"[CircuitBreaker:{self.provider_name}] Request DROPPED! "
            f"(Total dropped: {self.metrics.dropped_requests})"
        )

    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        """Perform atomic state transition and log transition telemetry."""
        old_state = self.metrics.state
        self.metrics.state = new_state
        self.metrics.last_state_change = time.time()
        if new_state == CircuitState.HALF_OPEN:
            self._probes_in_flight = 0
        if new_state == CircuitState.OPEN and old_state == CircuitState.HALF_OPEN:
            # A failed canary is a new trip; keep recovery-time stats honest.
            self.metrics.total_trips += 1
            self.metrics.last_trip_time = time.time()
        logger.warning(
            f"[CircuitBreaker:{self.provider_name}] State Transition: "
            f"{old_state.value} -> {new_state.value} | Reason: {reason}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Generate structured metrics summary including latency percentiles."""
        latencies = sorted(self.metrics.failover_latencies_ms)
        n = len(latencies)

        def percentile(p: float) -> float:
            if not latencies:
                return 0.0
            if n == 1:
                return round(latencies[0], 2)
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            if f == c:
                return round(latencies[f], 2)
            d0 = latencies[f] * (c - k)
            d1 = latencies[c] * (k - f)
            return round(d0 + d1, 2)

        return {
            "provider": self.provider_name,
            "current_state": self.state.value,
            "total_requests": self.metrics.total_requests,
            "total_successes": self.metrics.total_successes,
            "total_failures": self.metrics.total_failures,
            "consecutive_failures": self.metrics.consecutive_failures,
            "total_trips": self.metrics.total_trips,
            "failover_events": self.metrics.failover_events,
            "dropped_requests": self.metrics.dropped_requests,
            "latency_p50_ms": percentile(0.50),
            "latency_p95_ms": percentile(0.95),
            "latency_p99_ms": percentile(0.99),
            "mean_recovery_time_sec": (
                round(sum(self.metrics.recovery_times_sec) / len(self.metrics.recovery_times_sec), 2)
                if self.metrics.recovery_times_sec else 0.0
            ),
            "last_failure_reason": self.metrics.last_failure_reason,
        }
