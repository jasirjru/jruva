import asyncio
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import check_database_health
from app.core.security import limiter

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str
    commit: str | None = None


class DatabaseHealthResponse(BaseModel):
    status: str
    database: str
    backend: str
    timestamp: datetime


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check & keep-warm probe",
    description="Endpoint invoked by uptime monitors or cron jobs to keep Render instances warm and verify service health."
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        commit=os.getenv("RENDER_GIT_COMMIT") or os.getenv("RELEASE_COMMIT_SHA") or None,
    )


@router.get(
    "/health/db",
    response_model=DatabaseHealthResponse,
    summary="Database health check & connectivity probe",
    description="Endpoint invoked to verify database liveness and active connection capability.",
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def health_db_check(request: Request) -> DatabaseHealthResponse:
    try:
        db_health = await asyncio.wait_for(check_database_health(), timeout=3.0)
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connectivity probe timed out. Please retry.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connectivity probe failed. Please retry.",
        ) from None

    if db_health.get("status") != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connectivity is currently unavailable. Please retry.",
        )
    return DatabaseHealthResponse(
        status="healthy",
        database="connected",
        backend="operational",
        timestamp=datetime.now(timezone.utc),
    )

