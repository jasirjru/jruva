"""FastAPI Application Factory and Entry Point."""
from contextlib import asynccontextmanager
import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.google_identity import validate_production_auth
from app.core.database import validate_production_database
from app.api.v1.endpoints.auth import get_current_user
from fastapi import Depends
from app.core.exceptions import BaseAppException, app_exception_handler
from app.core.security import limiter
from app.core.csrf import CsrfOriginMiddleware
from app.core.logging_config import configure_structured_logging
from app.schemas.chat import FeedbackRequest
from app.api.v1.endpoints.metrics import (
    router as metrics_router
)

# Configure structured JSON logging
configure_structured_logging(debug=settings.DEBUG)
logger = logging.getLogger("chatbot")


def warm_chat_imports() -> None:
    """Import modules the chat pipeline otherwise loads on the first question (SymPy is the slow one)."""
    try:
        import app.services.math.calculus_engine  # noqa: F401
    except Exception as exc:
        logger.warning(f"Chat import warm-up skipped: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown hooks."""
    logger.info(f"Starting {settings.PROJECT_NAME} in {settings.ENVIRONMENT} mode...")
    logger.info(f"Rate limiting active: {settings.RATE_LIMIT_PER_MINUTE} req/min")
    import asyncio

    # Background thread: the server accepts requests at once; a keep-warm ping gives it time to finish.
    warm_task = asyncio.create_task(asyncio.to_thread(warm_chat_imports))
    yield
    warm_task.cancel()
    logger.info("Shutting down application...")
    from app.services.chat_service import flush_background_usage_writes
    try:
        await flush_background_usage_writes()
    except Exception as exc:
        logger.warning(f"Error flushing background usage writes: {exc}")
    from app.core.database import shutdown_sync_db_worker
    from app.services.storage import close_storage_backend
    try:
        shutdown_sync_db_worker()
    except Exception as exc:
        logger.warning(f"Error shutting down database worker: {exc}")
    try:
        close_storage_backend()
    except Exception as exc:
        logger.warning(f"Error closing storage backend: {exc}")


def validate_production_storage(cfg=None) -> None:
    current_settings = cfg or settings
    if current_settings.ENVIRONMENT.lower() == "production":
        if cfg is not None:
            storage_backend = getattr(current_settings, "STORAGE_BACKEND", "local")
        else:
            storage_backend = os.getenv("STORAGE_BACKEND") or getattr(current_settings, "STORAGE_BACKEND", "local")
        storage_backend = (storage_backend or "local").lower().strip()
        if storage_backend == "supabase":
            if cfg is not None:
                url = getattr(current_settings, "SUPABASE_URL", "")
                key = (
                    getattr(current_settings, "SUPABASE_SERVICE_ROLE_KEY", "")
                    or getattr(current_settings, "SUPABASE_KEY", "")
                )
            else:
                url = os.getenv("SUPABASE_URL") or getattr(current_settings, "SUPABASE_URL", "")
                key = (
                    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                    or getattr(current_settings, "SUPABASE_SERVICE_ROLE_KEY", "")
                    or os.getenv("SUPABASE_KEY")
                )
            if not url or not key:
                raise RuntimeError(
                    "Durable storage configuration fatal: STORAGE_BACKEND is set to 'supabase' "
                    "in production environment, but SUPABASE_URL or service key is missing."
                )
        elif storage_backend in ("gcs", "s3", "cloud"):
            raise RuntimeError(
                f"Durable storage configuration fatal: storage backend '{storage_backend}' "
                "is an unimplemented stub in production."
            )
        elif storage_backend not in ("local", "supabase"):
            raise RuntimeError(
                f"Unsupported storage backend '{storage_backend}' in production environment."
            )


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    validate_production_database(settings)
    validate_production_storage(settings)
    validate_production_auth(settings)
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    from fastapi import Request
    from fastapi.responses import JSONResponse

    @application.middleware("http")
    async def limit_image_payload_size(request: Request, call_next):
        if request.url.path.startswith("/api/v1/image/") and request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            max_bytes = 64 * 1024
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        return JSONResponse(status_code=413, content={"detail": "Payload too large. Maximum size for image requests is 64KB."})
                except ValueError:
                    pass
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > max_bytes:
                    return JSONResponse(status_code=413, content={"detail": "Payload too large. Maximum size for image requests is 64KB."})
            raw_body = bytes(body)
            request._body = raw_body
            request._stream_consumed = False
            async def receive():
                return {"type": "http.request", "body": raw_body, "more_body": False}
            request._receive = receive
        return await call_next(request)

    # Attach SlowAPI rate limiter
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    application.add_exception_handler(BaseAppException, app_exception_handler)

    # CORS configuration
    cors_kwargs = {
        "allow_origins": settings.cors_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    # Any-origin regex only for local development; staging/production must use the allow-list.
    if settings.ENVIRONMENT.strip().lower() == "development":
        cors_kwargs["allow_origin_regex"] = r".*"
    application.add_middleware(CORSMiddleware, **cors_kwargs)

    # CSRF Origin verification for state-changing requests (production only).
    # Must be added AFTER CORSMiddleware so preflight OPTIONS are handled
    # by CORS first and never hit the CSRF check.
    application.add_middleware(
        CsrfOriginMiddleware,
        allowed_origins=settings.cors_origins,
    )

    # Fail-closed write quiescence and planned maintenance gate.
    from app.core.maintenance import MaintenanceMiddleware
    application.add_middleware(MaintenanceMiddleware)

    # Observability: Register /metrics at root and in v1
    application.include_router(metrics_router, tags=["Observability"])

    # Generated media is private. Block legacy public paths before the static mount.
    @application.get('/static/generated/{legacy_path:path}', include_in_schema=False)
    async def deny_public_generated_media(legacy_path: str):
        from fastapi.responses import Response
        return Response(status_code=404)

    # Mount public showcase assets only; generation uses authenticated storage.
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    class PublicStaticFiles(StaticFiles):
        async def get_response(self, path, scope):
            from fastapi.responses import Response
            normalized = os.path.normpath(path).replace('\\', '/').lstrip('/').casefold()
            if normalized == 'generated' or normalized.startswith('generated/'):
                return Response(status_code=404)
            return await super().get_response(path, scope)

    application.mount("/static", PublicStaticFiles(directory=static_dir), name="static")

    # Register API routes
    application.include_router(api_v1_router, prefix=settings.API_V1_STR)

    @application.post("/feedback", tags=["Feedback"], summary="Root feedback endpoint")
    async def root_feedback(payload: FeedbackRequest, current_user: dict = Depends(get_current_user)):
        from app.api.v1.endpoints.chat import feedback_endpoint
        return await feedback_endpoint(payload, current_user)

    @application.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.PROJECT_NAME,
            "status": "online",
            "docs": "/docs",
            "api_version": "v1",
            "metrics": "/metrics",
        }

    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
