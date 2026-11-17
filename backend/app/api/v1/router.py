"""API v1 Router Aggregation."""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth,
    chat,
    coding,
    health,
    image,
    library,
    memory,
    metrics,
    models,
    plugins,
    projects,
    scheduled,
    search,
    sessions,
    storage,
    telemetry,
    audio,
    video,
)

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(auth.router, tags=["Authentication"])
api_v1_router.include_router(memory.router, tags=["Memory Card"])
api_v1_router.include_router(chat.router, tags=["Chat"])
api_v1_router.include_router(coding.router, prefix="/coding", tags=["Coding Intelligence & Sandbox"])
api_v1_router.include_router(image.router, prefix="/image", tags=["Image Generation"])
api_v1_router.include_router(projects.router, tags=["Projects Workspace"])
api_v1_router.include_router(library.router, tags=["Knowledge Library"])
api_v1_router.include_router(storage.router, tags=["Storage & Quota Governance"])
api_v1_router.include_router(audio.router, prefix="/audio", tags=["Voice Transcription"])
api_v1_router.include_router(video.router, prefix="/video", tags=["Video Understanding"])
api_v1_router.include_router(scheduled.router, tags=["Scheduled Automations"])
api_v1_router.include_router(plugins.router, tags=["Plugins & Integrations"])
api_v1_router.include_router(search.router, prefix="/search", tags=["Global Search"])
api_v1_router.include_router(models.router, tags=["Models"])
api_v1_router.include_router(sessions.router, tags=["Sessions"])
api_v1_router.include_router(telemetry.router, tags=["Telemetry & Observability"])
api_v1_router.include_router(metrics.router, tags=["Prometheus Metrics"])



