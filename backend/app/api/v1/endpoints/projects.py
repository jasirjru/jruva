"""Projects Workspace API endpoints."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from app.api.v1.endpoints.sessions import get_session_owner_id
from pydantic import BaseModel, Field

from app.services.workspace_store import workspace_store

router = APIRouter(prefix="/projects", tags=["Projects Workspace"])


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    description: Optional[str] = Field(default=None, max_length=1000)
    instructions: Optional[str] = Field(default=None, max_length=5000)
    preferred_model: str = Field(default="auto")


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    description: Optional[str] = None
    instructions: Optional[str] = None
    preferred_model: Optional[str] = None


@router.get("", summary="List all projects")
async def list_projects(owner_id: str = Depends(get_session_owner_id)):
    """Retrieve all user projects ordered by recency."""
    return {"projects": workspace_store.list_projects(user_id=owner_id)}


@router.post("", summary="Create a new project workspace")
async def create_project(payload: ProjectCreateRequest, owner_id: str = Depends(get_session_owner_id)):
    """Create a persistent workspace with custom instructions."""
    proj = workspace_store.create_project(
        user_id=owner_id,
        name=payload.name,
        description=payload.description,
        instructions=payload.instructions,
        preferred_model=payload.preferred_model,
    )
    return {"success": True, "project": proj}


@router.get("/{project_id}", summary="Get project workspace details")
async def get_project(project_id: str, owner_id: str = Depends(get_session_owner_id)):
    """Retrieve specific project by ID."""
    proj = workspace_store.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    if proj.get("user_id") != owner_id:
        raise HTTPException(403, "Forbidden project ownership")
    return {"project": proj}


@router.put("/{project_id}", summary="Update project instructions and settings")
async def update_project(project_id: str, payload: ProjectUpdateRequest, owner_id: str = Depends(get_session_owner_id)):
    await get_project(project_id, owner_id)
    """Update project metadata or system instructions."""
    success = workspace_store.update_project(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        instructions=payload.instructions,
        preferred_model=payload.preferred_model,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True}


@router.delete("/{project_id}", summary="Delete a project workspace")
async def delete_project(project_id: str, owner_id: str = Depends(get_session_owner_id)):
    await get_project(project_id, owner_id)
    """Delete project permanently."""
    success = workspace_store.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True}
