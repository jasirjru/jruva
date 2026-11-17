"""Telemetry and Tracing API Endpoints.

Exposes real-time system metrics, distributed request traces, token usage,
estimated costs, and circuit breaker diagnostic states.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_llm_router
from app.api.operations_authorization import (
    OperationsRoute,
    require_operations_admin,
    require_operations_reset,
)
from app.schemas.operations import (
    OperationalMetrics,
    OperationalTrace,
    Provider,
    project_metrics,
    project_trace,
)
from app.services.cache.response_cache import response_cache
from app.services.llm.router import LLMRouter
from app.services.telemetry.telemetry_service import telemetry_service

router = APIRouter(
    prefix="/telemetry",
    tags=["Telemetry & Observability"],
    dependencies=[Depends(require_operations_admin)],
    route_class=OperationsRoute,
)


@router.get(
    "/metrics",
    summary="Get aggregated telemetry metrics",
    description="Returns request counts, availability %, latency percentiles (p50/p90/p95/p99), token consumption, estimated USD costs, and circuit breaker states.",
    response_model=OperationalMetrics,
)
async def get_telemetry_metrics(
    router_instance: Annotated[LLMRouter, Depends(get_llm_router)],
) -> OperationalMetrics:
    """Retrieve bounded administrative aggregates without internal error details."""
    try:
        circuits = router_instance.get_circuit_metrics()
        metrics = telemetry_service.get_metrics_summary(circuit_breakers=circuits)
        return project_metrics(metrics, response_cache.get_stats())
    except Exception:  # noqa: BLE001 -- never serialize internal diagnostic exceptions
        raise HTTPException(
            503,
            "Operational diagnostics unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.get(
    "/traces",
    summary="Get recent request traces",
    description="Returns sliding window of recent request traces with per-request latency, token accounting, routing history, and failover status.",
    response_model=list[OperationalTrace],
)
async def get_telemetry_traces(
    limit: Annotated[
        int, Query(ge=1, le=200, description="Max number of traces to return")
    ] = 50,
    provider: Annotated[
        Provider | None, Query(description="Filter by an allowlisted provider")
    ] = None,
    failed_only: bool = Query(
        default=False, description="Filter only failed or dropped requests"
    ),
) -> list[OperationalTrace]:
    """Return at most 200 traces with no caller identity or free-text diagnostics."""
    try:
        return [
            project_trace(trace)
            for trace in telemetry_service.get_traces(
                limit=limit, provider=provider, failed_only=failed_only
            )[:limit]
        ]
    except Exception:  # noqa: BLE001 -- never serialize internal diagnostic exceptions
        raise HTTPException(
            503,
            "Operational diagnostics unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.post(
    "/reset",
    summary="Reset telemetry counters",
    description="Clears all trace history and resets token, request, and cost counters (useful for load testing benchmark isolation).",
)
async def reset_telemetry_metrics(
    _admin: Annotated[dict, Depends(require_operations_reset)],
) -> dict[str, str]:
    """Reset all in-memory telemetry data."""
    try:
        telemetry_service.reset_metrics()
        response_cache.clear()
    except Exception:  # noqa: BLE001 -- never serialize internal diagnostic exceptions
        raise HTTPException(
            503, "Operational reset unavailable.", headers={"Cache-Control": "no-store"}
        ) from None
    return {
        "status": "success",
        "message": "Telemetry metrics and response cache reset successfully",
    }
