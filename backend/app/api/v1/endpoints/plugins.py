"""Plugins and External Integrations API Endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends
from app.api.v1.endpoints.sessions import get_session_owner_id
from app.api.v1.endpoints.auth import get_current_user_optional
from pydantic import BaseModel

from app.core.config import settings
from app.services.workspace_store import workspace_store

router = APIRouter(prefix="/plugins", tags=["Plugins & Integrations"])

DEFAULT_CATALOG = [
    {
        "id": "web_search",
        "name": "Live Web Search",
        "description": "Live web results with cited sources (Tavily, with DuckDuckGo and Wikipedia as fallbacks).",
        "icon": "🌐",
        "category": "Search & Knowledge",
        "status": "ready",
        "tools": ["web_search"],
    },
    {
        "id": "python_sandbox",
        "name": "Python Code Intelligence",
        "description": "AST static analysis, Big-O complexity estimation, and pattern linting. Execution disabled by default in production (requires container isolation).",
        "icon": "⚡",
        "category": "Developer Tools",
        "status": "static_analysis_active",
        "tools": ["analyze_python", "test_cases", "templates"],
    },
    {
        "id": "document_ocr",
        "name": "Vision OCR & Scanner",
        "description": "High-accuracy tabular document and image text transcription.",
        "icon": "📄",
        "category": "Multimodal",
        "status": "ready",
        "tools": ["ocr_transcribe"],
    },
    {
        "id": "github",
        "name": "GitHub Connector (Draft)",
        "description": "Placeholder connector specification for repository search and PR reviews (OAuth integration in development).",
        "icon": "🐙",
        "category": "Developer Tools",
        "status": "planned",
        "tools": ["search_repo", "read_file", "list_issues"],
    },
    {
        "id": "image_studio",
        "name": "FLUX Image Studio",
        "description": "Multi-tier neural visual synthesis and 4K super-resolution upscaler.",
        "icon": "🎨",
        "category": "Generative Media",
        "status": "ready",
        "tools": ["generate_image", "upscale_image"],
    },
]


class PluginToggleRequest(BaseModel):
    is_enabled: bool


@router.get("", summary="List available plugins and user connection states")
async def list_plugins(user: Optional[dict] = Depends(get_current_user_optional)):
    """Retrieve catalog of tools and integrations with user toggle status."""
    user_integrations = workspace_store.list_integrations(user_id=user["id"]) if user else []
    enabled_map = {i["plugin_id"]: bool(i["is_enabled"]) for i in user_integrations}

    # Features behind deployment switches report "unavailable" when the switch is off.
    feature_flags = {
        "document_ocr": settings.ENABLE_LOCAL_OCR,
        "image_studio": settings.IMAGE_GENERATION_ENABLED,
    }

    catalog = []
    for item in DEFAULT_CATALOG:
        # Default all ready plugins to enabled unless explicitly disabled
        is_enabled = enabled_map.get(item["id"], True)
        status = "unavailable" if feature_flags.get(item["id"]) is False else item["status"]
        catalog.append({**item, "status": status, "is_enabled": is_enabled})

    return {"plugins": catalog}


@router.post("/{plugin_id}/toggle", summary="Toggle a plugin or integration")
async def toggle_plugin(plugin_id: str, payload: PluginToggleRequest, owner_id: str = Depends(get_session_owner_id)):
    """Enable or disable integration for active user."""
    workspace_store.toggle_integration(plugin_id, payload.is_enabled, user_id=owner_id)
    return {"success": True, "plugin_id": plugin_id, "is_enabled": payload.is_enabled}
