"""Versioned, bounded operational outputs with no identity or free-text fields."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict

Provider = Literal["gemini", "groq", "openrouter", "ollama", "local_qwen", "unknown"]
PROVIDERS = ("gemini", "groq", "openrouter", "ollama", "local_qwen")


class OperationalTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routed_provider: Provider
    status: Literal["success", "failover_recovered", "error", "unknown"]
    tier: Literal["B", "A", "S", "unknown"]
    streaming: bool
    cached: bool
    was_failover: bool
    rag_used: bool
    rag_reranker_used: bool
    rag_degraded: bool
    total_latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    retrieved_chunk_count: int


class CircuitDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_state: Literal["closed", "open", "half_open", "unknown"]
    total_requests: float
    total_successes: float
    total_failures: float
    consecutive_failures: float
    total_trips: float
    failover_events: float
    dropped_requests: float


class OperationalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    system: dict[str, float]
    latency_ms: dict[str, dict[str, float]]
    tokens_and_cost: dict[str, float | dict[str, dict[str, float]]]
    tiers: dict[str, float | dict[str, float]]
    silent_failures: dict[str, float]
    circuits: dict[str, CircuitDiagnostic]
    cache: dict[str, float]


def safe_number(value: object) -> float:
    """Reject nonnumeric, nonfinite, negative and excessive values without echoing input."""
    if type(value) not in (int, float):
        return 0.0
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and 0 <= number <= 1e12 else 0.0


def numeric_fields(values: dict, fields: tuple[str, ...]) -> dict[str, float]:
    return {field: safe_number(values.get(field)) for field in fields}


def safe_choice(value: object, choices: tuple[str, ...]) -> str:
    return value if type(value) is str and value in choices else "unknown"


def project_trace(trace: dict) -> OperationalTrace:
    """No raw trace/session IDs, model names, endpoints, errors, or reasons escape."""
    numbers = numeric_fields(
        trace,
        (
            "total_latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "retrieved_chunk_count",
        ),
    )
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retrieved_chunk_count",
    ):
        numbers[field] = int(numbers[field])
    return OperationalTrace(
        **numbers,
        ttft_ms=safe_number(trace["ttft_ms"])
        if trace.get("ttft_ms") is not None
        else None,
        routed_provider=safe_choice(trace.get("routed_provider"), PROVIDERS),
        status=safe_choice(
            trace.get("status"), ("success", "failover_recovered", "error")
        ),
        tier=safe_choice(trace.get("tier"), ("B", "A", "S")),
        **{
            field: trace.get(field) is True
            for field in (
                "streaming",
                "cached",
                "was_failover",
                "rag_used",
                "rag_reranker_used",
                "rag_degraded",
            )
        },
    )


def project_metrics(summary: dict, cache: dict) -> OperationalMetrics:
    """Project fixed-size dictionaries, including fixed provider and tier vocabularies."""
    tokens = summary.get("tokens_and_cost", {})
    tiers = summary.get("tiers", {})
    circuit_fields = (
        "total_requests",
        "total_successes",
        "total_failures",
        "consecutive_failures",
        "total_trips",
        "failover_events",
        "dropped_requests",
    )
    circuits = {}
    for provider in PROVIDERS:
        circuit = summary.get("circuits", {}).get(provider)
        if isinstance(circuit, dict):
            circuits[provider] = CircuitDiagnostic(
                **numeric_fields(circuit, circuit_fields),
                current_state=safe_choice(
                    circuit.get("current_state"), ("closed", "open", "half_open")
                ),
            )
    return OperationalMetrics(
        system=numeric_fields(
            summary.get("system", {}),
            (
                "total_requests",
                "successful_requests",
                "dropped_requests",
                "failover_requests",
                "availability_pct",
                "failover_rate_pct",
            ),
        ),
        latency_ms={
            kind: numeric_fields(
                summary.get("latency_ms", {}).get(kind, {}),
                ("p50", "p90", "p95", "p99", "min", "max"),
            )
            for kind in ("total", "ttft", "failover")
        },
        tokens_and_cost={
            **numeric_fields(
                tokens,
                (
                    "total_prompt_tokens",
                    "total_completion_tokens",
                    "total_tokens",
                    "accumulated_cost_usd",
                ),
            ),
            "by_provider": {
                p: numeric_fields(
                    tokens.get("by_provider", {}).get(p, {}),
                    ("requests", "tokens", "cost_usd"),
                )
                for p in PROVIDERS
            },
        },
        tiers={
            "cache_hits": safe_number(tiers.get("cache_hits")),
            **{
                kind: numeric_fields(tiers.get(kind, {}), ("B", "A", "S"))
                for kind in ("requests_by_tier", "cost_by_tier")
            },
        },
        silent_failures={
            "total": safe_number(summary.get("silent_failures", {}).get("total"))
        },
        circuits=circuits,
        cache=numeric_fields(
            cache,
            (
                "total_entries",
                "max_size",
                "similarity_threshold",
                "exact_hits",
                "semantic_hits",
                "total_hits",
                "hits",
                "misses",
                "hit_ratio_pct",
                "saved_tokens",
                "saved_cost_usd",
            ),
        ),
    )
