"""Concurrent Load-Testing & Fault-Injection Harness for Multi-Provider LLM Resilience.

Simulates N concurrent user sessions with probabilistic fault injection (429 rate-limiting
and 5xx upstream server errors) to empirically evaluate circuit breaker failover latencies,
state machine transitions, and request drop rates.
"""
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import CircuitBreakerConfig, CircuitState
from app.services.llm.router import LLMRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_test_harness")


class MockFaultyProvider(BaseLLMProvider):
    """Simulated LLM Provider with configurable latency and probabilistic failure injection."""

    def __init__(
        self,
        provider_name: str,
        base_latency_s: float = 0.05,
        failure_rate: float = 0.0,
        rate_limit_ratio: float = 0.7,
        cascade_fail: bool = False,
    ) -> None:
        self._provider_name = provider_name
        self.model = f"{provider_name}-model"
        self.base_latency_s = base_latency_s
        self.failure_rate = failure_rate
        self.rate_limit_ratio = rate_limit_ratio
        self.cascade_fail = cascade_fail
        self.call_count = 0
        self.failure_count = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: Optional[List[Any]] = None,
    ) -> str:
        self.call_count += 1
        # Simulate network latency
        latency = self.base_latency_s + random.uniform(0.01, 0.04)
        await asyncio.sleep(latency)

        if self.cascade_fail:
            self.failure_count += 1
            raise LLMProviderException(f"Simulated fatal cascade failure on {self.provider_name}")

        if random.random() < self.failure_rate:
            self.failure_count += 1
            if random.random() < self.rate_limit_ratio:
                raise QuotaExhaustedException(f"HTTP 429: {self.provider_name} free-tier rate limit exceeded")
            else:
                raise LLMProviderException(f"HTTP 502: {self.provider_name} upstream gateway error")

        return f"Response from {self.provider_name} for '{prompt[:15]}...'"

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: Optional[List[Any]] = None,
    ) -> AsyncGenerator[str, None]:
        self.call_count += 1
        latency = self.base_latency_s + random.uniform(0.01, 0.04)
        await asyncio.sleep(latency)

        if self.cascade_fail:
            self.failure_count += 1
            raise LLMProviderException(f"Simulated fatal cascade failure on stream {self.provider_name}")

        if random.random() < self.failure_rate:
            self.failure_count += 1
            if random.random() < self.rate_limit_ratio:
                raise QuotaExhaustedException(f"HTTP 429: {self.provider_name} rate limit on stream")
            else:
                raise LLMProviderException(f"HTTP 503: {self.provider_name} service unavailable on stream")

        tokens = ["Simulated ", "streamed ", "tokens ", f"from {self.provider_name}"]
        for token in tokens:
            await asyncio.sleep(0.01)
            yield token


@dataclass
class RequestResult:
    session_id: int
    prompt: str
    success: bool
    provider_used: str
    was_failover: bool
    total_latency_ms: float
    error: Optional[str] = None


async def run_session_worker(
    worker_id: int,
    request_count: int,
    router: LLMRouter,
    results: List[RequestResult],
    semaphore: asyncio.Semaphore,
) -> None:
    """Simulate a single concurrent client session issuing sequential prompts."""
    for req_idx in range(request_count):
        async with semaphore:
            prompt = f"Session-{worker_id} request #{req_idx}"
            t0 = time.perf_counter()
            try:
                text, provider, model, was_failover = await router.generate_response(
                    prompt=prompt,
                    requested_model="groq",
                )
                elapsed = (time.perf_counter() - t0) * 1000
                results.append(
                    RequestResult(
                        session_id=worker_id,
                        prompt=prompt,
                        success=True,
                        provider_used=provider,
                        was_failover=was_failover,
                        total_latency_ms=round(elapsed, 2),
                    )
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                results.append(
                    RequestResult(
                        session_id=worker_id,
                        prompt=prompt,
                        success=False,
                        provider_used="none",
                        was_failover=False,
                        total_latency_ms=round(elapsed, 2),
                        error=str(exc),
                    )
                )
            # Realistic user think-time between turns (20-60ms)
            await asyncio.sleep(random.uniform(0.02, 0.06))


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate p50, p90, p95, p99 from a list of numbers."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)

    def p(pct: float) -> float:
        if n == 1:
            return round(s[0], 2)
        k = (n - 1) * pct
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return round(s[f], 2)
        return round(s[f] * (c - k) + s[c] * (k - f), 2)

    return {
        "min": round(s[0], 2),
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": round(s[-1], 2),
    }


async def run_load_test(
    concurrency: int = 50,
    total_requests: int = 250,
    primary_failure_rate: float = 0.30,
    allow_cascade_fail: bool = False,
) -> Dict[str, Any]:
    """Execute concurrent load test with fault injection and generate metrics report."""
    requests_per_worker = max(1, total_requests // concurrency)
    total_actual_requests = requests_per_worker * concurrency

    logger.info(
        f"Starting load test: {concurrency} concurrent sessions, {total_actual_requests} total requests, "
        f"{int(primary_failure_rate * 100)}% primary fault injection rate..."
    )

    # Primary (Groq simulation): fast base latency (40ms), 30% failure rate
    mock_primary = MockFaultyProvider(
        provider_name="groq",
        base_latency_s=0.04,
        failure_rate=primary_failure_rate,
        rate_limit_ratio=0.7,
    )

    # Backup (Gemini simulation): healthy fallback (80ms base latency)
    mock_backup = MockFaultyProvider(
        provider_name="gemini",
        base_latency_s=0.08,
        failure_rate=0.05 if allow_cascade_fail else 0.0,
    )

    circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        cooldown_seconds=0.4,  # Fast cooldown for testing cycle
        canary_success_threshold=2,
    )

    router = LLMRouter(
        gemini_provider=mock_backup,
        groq_provider=mock_primary,
        circuit_config=circuit_config,
    )

    results: List[RequestResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    start_wall_time = time.perf_counter()
    tasks = [
        run_session_worker(i, requests_per_worker, router, results, semaphore)
        for i in range(concurrency)
    ]
    await asyncio.gather(*tasks)
    total_duration_s = time.perf_counter() - start_wall_time

    # Analyze results
    success_count = sum(1 for r in results if r.success)
    failover_count = sum(1 for r in results if r.was_failover)
    dropped_count = sum(1 for r in results if not r.success)

    all_latencies = [r.total_latency_ms for r in results]
    failover_latencies = [r.total_latency_ms for r in results if r.was_failover]
    direct_latencies = [r.total_latency_ms for r in results if not r.was_failover and r.success]

    groq_metrics = router.get_breaker("groq").get_summary()
    gemini_metrics = router.get_breaker("gemini").get_summary()

    report = {
        "config": {
            "concurrent_sessions": concurrency,
            "total_requests": len(results),
            "primary_injected_failure_rate": primary_failure_rate,
            "circuit_failure_threshold": circuit_config.failure_threshold,
            "circuit_cooldown_seconds": circuit_config.cooldown_seconds,
        },
        "throughput": {
            "total_duration_seconds": round(total_duration_s, 2),
            "requests_per_second": round(len(results) / total_duration_s, 2),
        },
        "reliability": {
            "successful_requests": success_count,
            "dropped_requests": dropped_count,
            "availability_rate_pct": round((success_count / len(results)) * 100, 2),
            "failover_requests": failover_count,
            "failover_rate_pct": round((failover_count / len(results)) * 100, 2),
        },
        "latency_ms": {
            "overall": calculate_percentiles(all_latencies),
            "failover": calculate_percentiles(failover_latencies),
            "direct_primary": calculate_percentiles(direct_latencies),
        },
        "circuit_breaker_telemetry": {
            "groq": groq_metrics,
            "gemini": gemini_metrics,
        },
    }

    return report


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING SCENARIO A: Single-Provider Degradation (30% Groq 429/502 Faults)")
    print("=" * 70)
    report_a = asyncio.run(
        run_load_test(
            concurrency=50,
            total_requests=250,
            primary_failure_rate=0.30,
            allow_cascade_fail=False,
        )
    )
    print(json.dumps(report_a, indent=2))

    print("\n" + "=" * 70)
    print("RUNNING SCENARIO B: Dual Provider Cascade (40% Groq Faults, 10% Gemini Faults)")
    print("=" * 70)
    report_b = asyncio.run(
        run_load_test(
            concurrency=50,
            total_requests=250,
            primary_failure_rate=0.40,
            allow_cascade_fail=True,
        )
    )
    print(json.dumps(report_b, indent=2))
