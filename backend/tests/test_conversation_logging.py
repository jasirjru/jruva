"""Unit and Integration tests for Conversation Logging and Feedback System."""
import time
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.conversation_log_service import conversation_log_service

from tests.private_accounts import private_account_headers

@pytest.fixture(autouse=True)
def authenticated_logging_client():
    client.headers.update(private_account_headers("logging-fixture"))
    yield
    conversation_log_service._queue.join()
    client.headers.clear()

client = TestClient(app)


def test_conversation_log_service_async_insert_and_feedback():
    """Verify log_async inserts a record and update_feedback sets signal."""
    test_id = f"test-log-{uuid.uuid4()}"
    logged_id = conversation_log_service.log_async(
        conversation_id=test_id,
        owner_id="logging-fixture",
        user_query="What are the passing criteria for BCA?",
        retrieved_chunks=[{"id": "c1", "text": "Minimum 40% passing marks in ESE and CIA."}],
        provider_used="groq",
        model_response="Students must score at least 40% aggregate in both CIA and ESE.",
        response_time_ms=124.5,
    )
    assert logged_id == test_id

    # Wait briefly for worker thread to persist
    time.sleep(0.3)

    # Verify update feedback
    success = conversation_log_service.update_feedback(test_id, "up", owner_id="logging-fixture")
    assert success is True

    # Update to down
    success_down = conversation_log_service.update_feedback(test_id, "down", owner_id="logging-fixture")
    assert success_down is True

    # Update to regenerated
    success_regen = conversation_log_service.update_feedback(test_id, "regenerated", owner_id="logging-fixture")
    assert success_regen is True


def test_feedback_endpoint_root():
    """Verify POST /feedback updates signal on conversation record."""
    test_id = f"test-ep-{uuid.uuid4()}"
    conversation_log_service.log_async(
        conversation_id=test_id,
        owner_id="logging-fixture",
        user_query="Where is JRU campus?",
        retrieved_chunks=[],
        provider_used="gemini",
        model_response="JRU is located at Kamre, Ratu Road, Ranchi, Jharkhand 835222.",
        response_time_ms=88.2,
    )
    time.sleep(0.3)

    response = client.post(
        "/feedback",
        json={"conversation_id": test_id, "signal": "up"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["conversation_id"] == test_id
    assert data["feedback"] == "up"
    assert data["updated"] is True


def test_feedback_endpoint_v1():
    """Verify POST /api/v1/feedback updates signal on conversation record."""
    test_id = f"test-ep-v1-{uuid.uuid4()}"
    conversation_log_service.log_async(
        conversation_id=test_id,
        owner_id="logging-fixture",
        user_query="Tell me about lateral entry",
        retrieved_chunks=[],
        provider_used="groq",
        model_response="Lateral entry is available for 3-year diploma holders.",
        response_time_ms=65.0,
    )
    time.sleep(0.3)

    response = client.post(
        "/api/v1/feedback",
        json={"conversation_id": test_id, "signal": "down"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["conversation_id"] == test_id
    assert data["feedback"] == "down"
    assert data["updated"] is True


def test_feedback_invalid_signal():
    """Verify invalid signal returns 422 Unprocessable Entity."""
    response = client.post(
        "/feedback",
        json={"conversation_id": "any-id", "signal": "invalid_vote"}
    )
    assert response.status_code == 422


def test_get_recent_logs():
    """Verify get_recent_logs returns list of dicts with expected columns."""
    logs = conversation_log_service.get_recent_logs(limit=5)
    assert isinstance(logs, list)
    assert len(logs) > 0
    first = logs[0]
    for required_key in ["id", "timestamp", "user_query", "retrieved_chunks", "provider_used", "model_response", "feedback"]:
        assert required_key in first

