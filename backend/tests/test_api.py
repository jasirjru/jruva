"""Unit and Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify keep-warm health endpoint responds with 200 and status healthy."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_health_db_endpoint():
    """Verify database health endpoint responds with 200 and positive proof."""
    response = client.get("/api/v1/health/db")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert "backend" in data
    assert "timestamp" in data


def test_root_endpoint():
    """Verify root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_chat_validation_empty_message():
    """Verify 422 Unprocessable Entity when message prompt is missing or empty."""
    response = client.post(
        "/api/v1/chat",
        json={"message": ""}
    )
    assert response.status_code == 422


def test_chat_stream_missing_api_key_helpful_warning(monkeypatch):
    """Verify stream gracefully yields notice when API keys are unconfigured."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    from app.services.cache.response_cache import response_cache
    response_cache.clear()
    response = client.post(
        "/api/v1/chat",
        json={"message": "Uncached test stream query for missing key validation", "stream": True}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "API_KEY" in content


def test_chat_non_stream_missing_api_key(monkeypatch):
    """Verify non-streaming request returns 502 problem details when keys are missing."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    from app.services.cache.response_cache import response_cache
    response_cache.clear()
    response = client.post(
        "/api/v1/chat",
        json={"message": "Uncached test query for missing key validation", "stream": False}
    )
    assert response.status_code == 502
    data = response.json()
    assert data["code"] == "LLM_PROVIDER_ERROR"


def test_chat_live_with_configured_key():
    """Verify chat works with the currently configured GEMINI_API_KEY."""
    from app.core.config import settings
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not configured")
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello", "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert len(data["content"]) > 0
