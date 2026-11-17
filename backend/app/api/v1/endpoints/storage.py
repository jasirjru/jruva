"""Storage capacity, quota ledger, and daily usage governance API endpoints."""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.v1.endpoints.auth import get_current_user
from app.services.quota.daily_usage_service import (
    async_get_daily_usage,
    get_current_utc_date_str,
    get_seconds_until_utc_midnight,
)
from app.services.storage.document_quota import (
    async_get_document_storage_summary,
)

logger = logging.getLogger("chatbot.storage_api")

router = APIRouter(prefix="/storage", tags=["Storage & Quota Governance"])


@router.get("/usage")
async def get_storage_usage(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve authenticated user and project storage consumption, limits, and capacity status."""
    user_id = current_user.get("id") if current_user else None
    try:
        summary = await async_get_document_storage_summary(user_id=user_id)
        return {
            "status": "success",
            "storage": summary,
        }
    except Exception as exc:
        logger.error(f"Error querying storage usage: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve storage usage summary.")


@router.get("/daily-quotas")
async def get_user_daily_quotas(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve daily free-tier usage metrics and limits for the authenticated user."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required to view daily usage.")

    user_id = current_user["id"]
    from app.core.config import settings

    try:
        usage = await async_get_daily_usage(user_id)
        enable_prompt_limit = getattr(settings, "ENABLE_DAILY_PROMPT_LIMIT", False)
        max_prompts = getattr(settings, "DAILY_PROMPTS_PER_USER", 20) if enable_prompt_limit else None
        max_searches = getattr(settings, "DAILY_SEARCHES_PER_USER", 3)
        max_uploads = getattr(settings, "DAILY_UPLOADS_PER_USER", 5)
        max_audio_seconds = getattr(settings, "DAILY_AUDIO_SECONDS_PER_USER", 120)
        retry_after = get_seconds_until_utc_midnight()

        return {
            "status": "success",
            "user_id": user_id,
            "usage_date_utc": usage["usage_date"],
            "resets_in_seconds": retry_after,
            "quotas": {
                "chat_prompts": {
                    "used": usage["chat_prompts"],
                    "limit": max_prompts,
                    "remaining": None if max_prompts is None else max(0, max_prompts - usage["chat_prompts"]),
                },
                "web_searches": {
                    "used": usage["web_searches"],
                    "limit": max_searches,
                    "remaining": max(0, max_searches - usage["web_searches"]),
                },
                "document_uploads": {
                    "used": usage["document_uploads"],
                    "limit": max_uploads,
                    "remaining": max(0, max_uploads - usage["document_uploads"]),
                },
                "audio_seconds": {
                    "used": usage["audio_seconds"],
                    "limit": max_audio_seconds,
                    "remaining": max(0, max_audio_seconds - usage["audio_seconds"]),
                },
            },
        }
    except Exception as exc:
        logger.error(f"Error querying daily quotas: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve daily quota status.")
