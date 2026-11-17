"""Endpoints for User Memory Card management and context personalization."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.endpoints.auth import get_current_user
from app.schemas.memory import (
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryUpdateRequest,
    UserMemoryItem,
)
from app.services.memory_service import memory_service

router = APIRouter(prefix="/memory", tags=["Memory Card"])


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List all memories for the authenticated user",
    description="Retrieve all active memories stored in the user's personal Memory Card.",
)
def list_user_memories(current_user: dict = Depends(get_current_user)):
    """Return all memories belonging to the logged-in user."""
    user_id = current_user["id"]
    memories = memory_service.get_user_memories(user_id, limit=50)
    return MemoryListResponse(
        memories=[UserMemoryItem(**m) for m in memories],
        count=len(memories),
        user_id=user_id,
    )


@router.post(
    "",
    response_model=UserMemoryItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new fact to the user's Memory Card",
)
def add_user_memory(
    payload: MemoryCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Add a new memory item for the logged-in user."""
    user_id = current_user["id"]
    result = memory_service.add_memory(
        user_id=user_id,
        fact=payload.fact,
        category=payload.category,
        source="manual",
        confidence=1.0,
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to save memory fact.",
        )
    return UserMemoryItem(**result)


@router.put(
    "/{memory_id}",
    response_model=dict,
    summary="Update an existing memory item",
)
def update_user_memory(
    memory_id: str,
    payload: MemoryUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update a memory item owned by the logged-in user."""
    user_id = current_user["id"]
    success = memory_service.update_memory(
        memory_id=memory_id,
        user_id=user_id,
        fact=payload.fact,
        category=payload.category,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or you do not have permission to modify it.",
        )
    return {"status": "success", "message": "Memory updated successfully."}


@router.delete(
    "/{memory_id}",
    response_model=dict,
    summary="Delete a memory from the Memory Card",
)
def delete_user_memory(
    memory_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a memory item owned by the logged-in user."""
    user_id = current_user["id"]
    success = memory_service.delete_memory(memory_id=memory_id, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or you do not have permission to delete it.",
        )
    return {"status": "success", "message": "Memory deleted successfully."}


@router.delete(
    "",
    response_model=dict,
    summary="Clear all memories from the user's Memory Card",
)
def clear_all_user_memories(current_user: dict = Depends(get_current_user)):
    """Clear all memories for the logged-in user."""
    user_id = current_user["id"]
    memory_service.clear_user_memories(user_id)
    return {"status": "success", "message": "All memories cleared successfully."}
