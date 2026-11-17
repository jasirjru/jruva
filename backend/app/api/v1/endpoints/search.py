"""Unified Workspace Global Search Endpoint."""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Depends
from app.api.v1.endpoints.sessions import get_session_owner_id

from app.services.workspace_store import workspace_store

router = APIRouter()


@router.get(
    "/global",
    summary="Global Unified Workspace Search",
    description="Search across conversations, projects, uploaded documents, generated images, and scheduled tasks in real-time.",
)
async def global_search(
    q: str = Query(..., min_length=1, max_length=200, description="Search query string"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results per category"),
    owner_id: str = Depends(get_session_owner_id),
) -> Dict[str, Any]:
    """Execute multi-domain search across all relational workspace stores."""
    return workspace_store.global_search(query=q, user_id=owner_id, limit=limit)
