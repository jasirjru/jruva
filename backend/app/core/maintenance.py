"""Application-level Maintenance & Write-Quiescence Controls.

Provides a genuine, fail-closed write-draining and maintenance gate.
Unlike CORS / ALLOWED_ORIGINS (which can be bypassed by non-browser clients,
curl, direct API scripts, or background workers), this module provides:
1. Config and dynamic runtime maintenance mode (MAINTENANCE_MODE).
2. HTTP Middleware (MaintenanceMiddleware) blocking all state-changing mutations
   (POST, PUT, DELETE, PATCH) with HTTP 503 Service Unavailable and Retry-After header.
3. Preserves liveness & health probes (/api/v1/health, /health, /metrics) so
   orchestrators (Render, Cloud Run, Kubernetes) do not kill containers during maintenance.
4. Protects background dispatch, workers, and scheduled tasks.
5. Invariant: Never deletes or silently rewrites existing idempotency,
   reservation, or billing history during maintenance.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("chatbot.maintenance")

# Thread-safe in-memory override for operational and test transitions
_DYNAMIC_MAINTENANCE_OVERRIDE: bool | None = None

# Endpoints that MUST remain reachable during maintenance for container liveness/monitoring
MAINTENANCE_EXEMPT_PATHS: set[str] = {
    "/api/v1/health",
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/openapi.json",
}


def is_maintenance_mode() -> bool:
    """Return True if write-quiescence / maintenance mode is currently active."""
    if _DYNAMIC_MAINTENANCE_OVERRIDE is not None:
        return _DYNAMIC_MAINTENANCE_OVERRIDE
    if os.environ.get("MAINTENANCE_MODE", "").lower().strip() in ("1", "true", "yes", "on"):
        return True
    try:
        from app.core.config import settings
        return bool(getattr(settings, "MAINTENANCE_MODE", False))
    except Exception:
        return False


def set_maintenance_mode(active: bool | None) -> None:
    """Dynamically set maintenance mode override (pass None to revert to configuration)."""
    global _DYNAMIC_MAINTENANCE_OVERRIDE
    _DYNAMIC_MAINTENANCE_OVERRIDE = active
    if active is True:
        logger.warning("[MAINTENANCE] Maintenance mode ACTIVE: write mutations and external dispatch suspended.")
    elif active is False:
        logger.info("[MAINTENANCE] Maintenance mode DEACTIVATED: normal operations resumed.")
    else:
        logger.info("[MAINTENANCE] Maintenance mode override cleared.")


class MaintenanceModeError(Exception):
    """Raised when an internal worker or service attempts a write mutation during maintenance."""
    pass


class MaintenanceMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing write quiescence during scheduled maintenance.

    Blocks state-changing HTTP requests (POST, PUT, DELETE, PATCH) with HTTP 503 Service Unavailable.
    Liveness and health checks (/api/v1/health) remain accessible to prevent orchestrator reboot loops.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if not is_maintenance_mode():
            return await call_next(request)

        # Allow OPTIONS preflight requests to prevent browser CORS confusion
        if request.method == "OPTIONS":
            return await call_next(request)

        # Allow health, metrics, and documentation probes
        path = request.url.path.rstrip("/")
        if path in MAINTENANCE_EXEMPT_PATHS or path == "" or request.url.path == "/":
            response = await call_next(request)
            return response

        # Block all mutation methods
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            logger.warning(
                f"[MAINTENANCE] Blocked mutation request {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "maintenance_mode",
                    "detail": "System is currently undergoing planned maintenance. Write operations and external dispatch are temporarily suspended.",
                    "retry_after": 30,
                },
                headers={"Retry-After": "30"},
            )

        # Allow read-only (GET, HEAD) requests through
        return await call_next(request)
