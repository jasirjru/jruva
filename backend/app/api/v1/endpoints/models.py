"""Model catalog endpoint for client model discovery and capability introspection."""
import asyncio
from fastapi import APIRouter, Depends

from app.api.dependencies import get_llm_router, get_settings
from app.core.config import Settings
from app.schemas.chat import ModelInfoItem, ModelListResponse
from app.services.llm.router import LLMRouter

router = APIRouter()


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available AI models and provider statuses",
    description="Returns the catalog of supported models (Gemini, Groq) with real-time availability based on configured free-tier API keys.",
)
async def list_models(
    router_service: LLMRouter = Depends(get_llm_router),
    settings_obj: Settings = Depends(get_settings),
) -> ModelListResponse:
    """Retrieve catalog of AI models and current runtime availability."""
    catalog = await asyncio.to_thread(router_service.get_catalog, include_hidden=False)
    models = [ModelInfoItem(**item) for item in catalog]
    active_count = sum(1 for m in models if m.available)

    return ModelListResponse(
        models=models,
        default_provider=settings_obj.DEFAULT_PROVIDER,
        active_count=active_count,
    )
