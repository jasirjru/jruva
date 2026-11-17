"""Service Level Objectives (SLO) and Service Level Indicators (SLI) definitions.

Replaces marketing tier claims with measurable, monitorable engineering targets.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EndpointSLO:
    endpoint: str
    target_availability_pct: float
    target_error_rate_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    description: str


# Production SLO catalog mapped to real monitored metrics
PRODUCTION_SLOS: Dict[str, EndpointSLO] = {
    "chat_cached": EndpointSLO(
        endpoint="/api/v1/chat (cache hit)",
        target_availability_pct=99.99,
        target_error_rate_pct=0.01,
        latency_p50_ms=5.0,
        latency_p95_ms=15.0,
        latency_p99_ms=30.0,
        description="Sub-15ms semantic in-memory response resolution for identical prompts",
    ),
    "chat_stream_ttft": EndpointSLO(
        endpoint="/api/v1/chat (streaming TTFT)",
        target_availability_pct=99.5,
        target_error_rate_pct=0.5,
        latency_p50_ms=450.0,
        latency_p95_ms=1200.0,
        latency_p99_ms=2500.0,
        description="Time to first token on primary reasoning providers (Groq/Gemini)",
    ),
    "image_cached": EndpointSLO(
        endpoint="/api/v1/image/generate (cache hit)",
        target_availability_pct=99.99,
        target_error_rate_pct=0.01,
        latency_p50_ms=5.0,
        latency_p95_ms=15.0,
        latency_p99_ms=30.0,
        description="Instant prompt resolution for cached 1024x1024 renders",
    ),
    "image_synthesis": EndpointSLO(
        endpoint="/api/v1/image/generate (live synthesis)",
        target_availability_pct=98.5,
        target_error_rate_pct=1.5,
        latency_p50_ms=4500.0,
        latency_p95_ms=12000.0,
        latency_p99_ms=18000.0,
        description="FLUX.1 / SDXL synthesis with multi-provider retry and failover policies",
    ),
    "health_and_meta": EndpointSLO(
        endpoint="/api/v1/health, /api/v1/models",
        target_availability_pct=99.99,
        target_error_rate_pct=0.01,
        latency_p50_ms=3.0,
        latency_p95_ms=10.0,
        latency_p99_ms=25.0,
        description="System health probes and catalog queries",
    ),
}


def get_slo(key: str) -> EndpointSLO:
    """Retrieve codified SLO definition by key."""
    if key not in PRODUCTION_SLOS:
        raise KeyError(f"SLO key '{key}' not registered in PRODUCTION_SLOS")
    return PRODUCTION_SLOS[key]


def is_latency_compliant(key: str, measured_p95_ms: float) -> bool:
    """Check if measured p95 latency meets target."""
    slo = get_slo(key)
    return measured_p95_ms <= slo.latency_p95_ms


def is_availability_compliant(key: str, measured_error_rate_pct: float) -> bool:
    """Check if measured error rate meets target."""
    slo = get_slo(key)
    return measured_error_rate_pct <= slo.target_error_rate_pct
