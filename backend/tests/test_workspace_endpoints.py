"""Automated API Tests for AI Workspace Endpoints (Phase 1-9)."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

from tests.private_accounts import private_account_headers

@pytest.fixture(autouse=True)
def authenticated_workspace_client():
    client.headers.update(private_account_headers("workspace-fixture"))
    yield
    client.headers.clear()

client = TestClient(app)


def test_projects_crud_workflow():
    # 1. Create Project
    res = client.post(
        "/api/v1/projects",
        json={
            "name": "Test Neural Workspace",
            "description": "Autonomous R&D workspace",
            "instructions": "Always respond concisely in structured markdown.",
            "preferred_model": "deepseek/deepseek-chat",
        },
    )
    assert res.status_code == 200, res.text
    resp_data = res.json()
    assert "project" in resp_data
    project = resp_data["project"]
    project_id = project["id"]
    assert project["name"] == "Test Neural Workspace"

    # 2. List Projects
    res = client.get("/api/v1/projects")
    assert res.status_code == 200
    projects = res.json()["projects"]
    assert any(p["id"] == project_id for p in projects)

    # 3. Delete Project
    res = client.delete(f"/api/v1/projects/{project_id}")
    assert res.status_code == 200


def test_plugins_toggle_workflow():
    # 1. List Plugins
    res = client.get("/api/v1/plugins")
    assert res.status_code == 200
    plugins = res.json()["plugins"]
    assert len(plugins) >= 5

    # 2. Toggle Plugin
    res = client.post("/api/v1/plugins/web_search/toggle", json={"is_enabled": True})
    assert res.status_code == 200
    assert res.json()["is_enabled"] is True


def test_scheduled_task_workflow():
    # 1. Create Scheduled Task
    res = client.post(
        "/api/v1/scheduled/tasks",
        json={
            "name": "Daily Morning AI Briefing",
            "prompt": "Summarize top 3 AI breakthrough papers from arXiv.",
            "cron_expression": "0 8 * * *",
        },
    )
    assert res.status_code == 200
    task = res.json()["task"]
    task_id = task["id"]
    assert task["status"] == "active"

    # 2. Trigger Task Execution
    res = client.post(f"/api/v1/scheduled/tasks/{task_id}/run")
    assert res.status_code == 200
    exec_data = res.json()["execution"]
    assert exec_data["status"] in ("success", "failure")

    # 3. List Executions
    res = client.get(f"/api/v1/scheduled/tasks/{task_id}/executions")
    assert res.status_code == 200
    executions = res.json()["executions"]
    assert len(executions) >= 1

    # 4. Toggle Status
    res = client.post(f"/api/v1/scheduled/tasks/{task_id}/toggle")
    assert res.status_code == 200
    assert res.json()["status"] == "paused"

    # 5. Delete Task
    res = client.delete(f"/api/v1/scheduled/tasks/{task_id}")
    assert res.status_code == 200


def test_library_upload_json_workflow():
    # Base64 encoded simple text: "Autonomous Agent Specification 2026"
    b64_content = "data:text/plain;base64,QXV0b25vbW91cyBBZ2VudCBTcGVjaWZpY2F0aW9uIDIwMjY="
    res = client.post(
        "/api/v1/library/upload-json",
        json={
            "filename": "autonomous_agent_spec.txt",
            "file_type": "text/plain",
            "data": b64_content,
        },
    )
    assert res.status_code == 200
    file_record = res.json()["file"]
    file_id = file_record["id"]
    assert file_record["chunk_count"] >= 1

    # Delete File
    res = client.delete(f"/api/v1/library/files/{file_id}")
    assert res.status_code == 200


def test_global_search_unified():
    res = client.get("/api/v1/search/global?q=test")
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert "chats" in data["results"]
    assert "projects" in data["results"]
    assert "files" in data["results"]
    assert "images" in data["results"]
    assert "tasks" in data["results"]


def test_image_gallery_endpoint():
    res = client.get("/api/v1/image/gallery")
    assert res.status_code == 200
    assert "images" in res.json()
