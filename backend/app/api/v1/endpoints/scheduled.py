"""Scheduled Tasks & Automation API Endpoints.

Stored cron task definitions and manual trigger workflows.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.security import EXPENSIVE_ENDPOINT_LIMIT, limiter
from pydantic import BaseModel, Field

from app.api.v1.endpoints.sessions import get_session_owner_id
from app.services.llm.router import LLMRouter
from app.services.workspace_store import workspace_store

router = APIRouter(prefix="/scheduled", tags=["Scheduled Automations"])
_router_instance = LLMRouter()


class TaskCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    prompt: str = Field(..., min_length=2, max_length=5000)
    cron_expression: str = Field(default="0 8 * * *", description="Cron schedule expression or preset")
    timezone: str = Field(default="UTC")


@router.get("/tasks", summary="List all scheduled AI tasks")
async def list_tasks(owner_id: Optional[str] = Depends(get_session_owner_id)):
    """Retrieve list of scheduled automations scoped to caller."""
    effective_user = owner_id or "guest"
    return {"tasks": workspace_store.list_scheduled_tasks(user_id=effective_user)}


@router.post("/tasks", summary="Schedule a new AI task")
async def create_task(
    payload: TaskCreateRequest,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Schedule recurring or one-time AI task."""
    effective_user = owner_id or "guest"
    task = workspace_store.create_scheduled_task(
        name=payload.name,
        prompt=payload.prompt,
        cron_expression=payload.cron_expression,
        timezone_str=payload.timezone,
        user_id=effective_user,
    )
    return {"success": True, "task": task}


@router.patch("/tasks/{task_id}/toggle", summary="Toggle task active/paused status")
@router.post("/tasks/{task_id}/toggle", summary="Toggle task active/paused status")
async def toggle_task(
    task_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Toggle status between active and paused."""
    effective_user = owner_id or "guest"
    tasks = workspace_store.list_scheduled_tasks(user_id=effective_user)
    matched = next((t for t in tasks if t["id"] == task_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="Task not found")

    new_status = workspace_store.toggle_scheduled_task(task_id)
    if not new_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "status": new_status}


@router.post("/tasks/{task_id}/run", summary="Trigger manual execution of scheduled task")
@limiter.limit(EXPENSIVE_ENDPOINT_LIMIT)
async def run_task(
    request: Request,
    task_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Execute the scheduled task immediately through the LLM engine and save output."""
    effective_user = owner_id or "guest"
    tasks = workspace_store.list_scheduled_tasks(user_id=effective_user)
    matched = next((t for t in tasks if t["id"] == task_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="Task not found")

    prompt = matched["prompt"]
    try:
        output, provider_name, model_name, was_failover = await _router_instance.generate_response(
            prompt=prompt,
            tier="B",
        )

        record = workspace_store.record_execution(
            task_id=task_id,
            status="success",
            output=output,
        )
        return {"success": True, "execution": record}
    except Exception as exc:
        record = workspace_store.record_execution(
            task_id=task_id,
            status="failure",
            error_message=str(exc),
        )
        return {"success": False, "execution": record, "error": str(exc)}


@router.get("/tasks/{task_id}/history", summary="Get task execution history")
@router.get("/tasks/{task_id}/executions", summary="Get task execution history")
async def get_history(
    task_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Retrieve execution log for a specific task."""
    effective_user = owner_id or "guest"
    tasks = workspace_store.list_scheduled_tasks(user_id=effective_user)
    matched = next((t for t in tasks if t["id"] == task_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="Task not found")

    execs = workspace_store.list_executions(task_id)
    return {"history": execs, "executions": execs}


@router.delete("/tasks/{task_id}", summary="Delete a scheduled task")
async def delete_task(
    task_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Remove task and execution history permanently."""
    effective_user = owner_id or "guest"
    tasks = workspace_store.list_scheduled_tasks(user_id=effective_user)
    matched = next((t for t in tasks if t["id"] == task_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="Task not found")

    success = workspace_store.delete_scheduled_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}
