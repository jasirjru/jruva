"""Prometheus Scrapeable Metrics Endpoint for System Observability."""

import math

from fastapi import APIRouter, Depends, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import Metric

from app.api.operations_authorization import OperationsRoute, require_operations_admin
from app.services.image.image_service import image_service

router = APIRouter(dependencies=[Depends(require_operations_admin)], route_class=OperationsRoute)

# Core SLI Prometheus Metrics
REQUESTS_TOTAL = Counter(
    "jru_requests_total",
    "Total HTTP requests received",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "jru_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.005, 0.015, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
)

CACHE_HITS_TOTAL = Counter(
    "jru_cache_hits_total",
    "Total semantic cache hits",
    ["cache_type"],
)

CACHE_MISSES_TOTAL = Counter(
    "jru_cache_misses_total",
    "Total semantic cache misses",
    ["cache_type"],
)

FAILOVERS_TOTAL = Counter(
    "jru_failovers_total",
    "Total multi-provider failover occurrences",
    ["primary_provider", "fallback_provider"],
)

PROVIDER_HEALTH = Gauge(
    "jru_provider_health_status",
    "Health status of upstream provider (1=Healthy, 0=Degraded)",
    ["provider"],
)

_PROVIDERS = frozenset(
    {
        "gemini",
        "groq",
        "openrouter",
        "ollama",
        "local_qwen",
        "flux",
        "turbo",
        "sdxl",
        "unknown",
    }
)
_ENDPOINTS = frozenset(
    {
        "/api/v1/chat",
        "/api/v1/health",
        "/api/v1/image/generate",
        "/api/v1/coding/analyze",
        "/metrics",
        "/api/v1/metrics",
        "/api/v1/telemetry/metrics",
        "/api/v1/telemetry/traces",
    }
)
_LABEL_VALUES = {
    "endpoint": _ENDPOINTS,
    "method": frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}),
    "status": frozenset(str(code) for code in range(100, 600)),
    "cache_type": frozenset({"exact", "semantic", "response"}),
    "primary_provider": _PROVIDERS,
    "fallback_provider": _PROVIDERS,
    "provider": _PROVIDERS,
    "le": frozenset(
        {
            "0.005",
            "0.015",
            "0.05",
            "0.1",
            "0.25",
            "0.5",
            "1.0",
            "2.5",
            "5.0",
            "10.0",
            "20.0",
            "+Inf",
        }
    ),
}


class OperationalCollector:
    """Export only owned metric families and fixed-vocabulary labels, without exemplars."""

    def collect(self):
        count = 0
        for collector in (
            REQUESTS_TOTAL,
            REQUEST_LATENCY,
            CACHE_HITS_TOTAL,
            CACHE_MISSES_TOTAL,
            FAILOVERS_TOTAL,
            PROVIDER_HEALTH,
        ):
            for family in collector.collect():
                safe_family = Metric(family.name, family.documentation, family.type)
                for sample in family.samples:
                    if any(
                        value not in _LABEL_VALUES.get(key, ())
                        for key, value in sample.labels.items()
                    ):
                        continue
                    if not math.isfinite(sample.value) or not 0 <= sample.value <= 1e12:
                        continue
                    if count >= 2048:
                        break
                    safe_family.add_sample(
                        sample.name, dict(sample.labels), sample.value
                    )
                    count += 1
                yield safe_family


@router.get("/metrics", summary="Prometheus Metrics Endpoint", tags=["Observability"])
async def get_prometheus_metrics():
    """Expose application SLI metrics in standard Prometheus exposition format."""
    try:
        for provider in ("flux", "turbo", "sdxl"):
            PROVIDER_HEALTH.labels(provider=provider).set(
                1 if image_service.health.is_healthy(provider) else 0
            )
        registry = CollectorRegistry()
        registry.register(OperationalCollector())
        payload = generate_latest(registry)
        return Response(
            content=payload,
            media_type=CONTENT_TYPE_LATEST,
            headers={"Cache-Control": "no-store"},
        )
    except Exception:  # noqa: BLE001 -- never serialize internal diagnostic exceptions
        raise HTTPException(
            503,
            "Operational diagnostics unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None
