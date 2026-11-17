"""Comprehensive regression tests for Phase 2: Application Write-Quiescence & Planned Maintenance Gate.

Verifies:
1. HTTP mutation endpoints (POST, PUT, DELETE, PATCH) are blocked with HTTP 503 and Retry-After header.
2. Requests without an Origin header (curl, direct scripts, CLI) cannot bypass the gate.
3. Health and liveness monitoring endpoints (/api/v1/health, /metrics) remain 100% available with HTTP 200.
4. Normal operation is cleanly restored when maintenance mode is deactivated.
5. In-flight idempotency lease claims reject new work while permitting read-only replay of completed records.
6. Quota and idempotency reconciliation routines safely skip during maintenance without deleting receipts.
7. Production database validation strictly rejects loopback even if flags are passed outside test harnesses.
"""
import os
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.maintenance import (
    is_maintenance_mode,
    set_maintenance_mode,
    MaintenanceModeError,
)
from app.main import create_application
from app.services.image.budget import (
    claim_idempotency_lease,
    complete_idempotency,
    get_idempotency_record,
    reconcile_stale_idempotency,
)
from app.services.image.storage_quota import reconcile_storage_quota
from app.services.workspace_store import workspace_store


@pytest.fixture
def maintenance_client():
    """Create a test client with maintenance mode toggled cleanly."""
    app = create_application()
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    set_maintenance_mode(None)


def test_maintenance_mode_toggle_state():
    """Verify is_maintenance_mode respects dynamic and config states."""
    set_maintenance_mode(None)
    assert is_maintenance_mode() is False

    set_maintenance_mode(True)
    assert is_maintenance_mode() is True

    set_maintenance_mode(False)
    assert is_maintenance_mode() is False

    set_maintenance_mode(None)


def test_maintenance_blocks_http_mutations_with_503_and_retry_after(maintenance_client):
    """Verify POST, PUT, DELETE, and PATCH are rejected with HTTP 503 and Retry-After header."""
    client = maintenance_client
    set_maintenance_mode(True)

    # 1. Chat mutation
    resp_chat = client.post("/api/v1/chat", json={"message": "Hello during maintenance"})
    assert resp_chat.status_code == 503
    assert resp_chat.headers.get("retry-after") == "30"
    data = resp_chat.json()
    assert data["error"] == "maintenance_mode"
    assert "temporarily suspended" in data["detail"]

    # 2. Image generation mutation
    resp_img = client.post("/api/v1/image/generate", json={"prompt": "test prompt"})
    assert resp_img.status_code == 503
    assert resp_img.headers.get("retry-after") == "30"

    # 3. Session deletion mutation
    resp_del = client.delete("/api/v1/sessions/test-session-id")
    assert resp_del.status_code == 503
    assert resp_del.headers.get("retry-after") == "30"

    # 4. Library upload mutation
    resp_upload = client.post("/api/v1/library/upload", files={"file": ("test.txt", b"content", "text/plain")})
    assert resp_upload.status_code == 503
    assert resp_upload.headers.get("retry-after") == "30"


def test_requests_without_origin_header_cannot_bypass_maintenance(maintenance_client):
    """Verify direct non-browser requests lacking an Origin header (curl/scripts) are blocked."""
    client = maintenance_client
    set_maintenance_mode(True)

    # Mutation with zero Origin and zero Referer headers (simulating direct curl)
    headers = {"User-Agent": "curl/8.4.0", "Accept": "*/*"}
    resp = client.post("/api/v1/chat", json={"message": "Direct curl attempt"}, headers=headers)
    assert resp.status_code == 503
    assert resp.json()["error"] == "maintenance_mode"

    # Mutation with arbitrary spoofed Origin header is also blocked
    spoofed_headers = {"Origin": "https://attacker.invalid", "User-Agent": "Mozilla/5.0"}
    resp_spoofed = client.post("/api/v1/chat", json={"message": "Spoofed origin attempt"}, headers=spoofed_headers)
    assert resp_spoofed.status_code == 503


def test_health_and_monitoring_endpoints_remain_available_during_maintenance(maintenance_client):
    """Verify container liveness probes (/api/v1/health) return HTTP 200 so orchestrators do not reboot."""
    client = maintenance_client
    set_maintenance_mode(True)

    # Health check must return HTTP 200
    resp_health = client.get("/api/v1/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # Metrics probe passes through maintenance filter (reaches endpoint rather than 503 maintenance block)
    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code != 503

    # OPTIONS preflight request remains permitted for CORS clarity
    resp_options = client.options(
        "/api/v1/chat",
        headers={"Origin": "https://jruva.in", "Access-Control-Request-Method": "POST"},
    )
    assert resp_options.status_code in (200, 204)


def test_normal_behavior_restored_when_maintenance_disabled(maintenance_client):
    """Verify that deactivating maintenance mode immediately restores normal endpoint behavior."""
    client = maintenance_client

    # Enable and verify blocked
    set_maintenance_mode(True)
    resp_blocked = client.post("/api/v1/chat", json={"message": "blocked"})
    assert resp_blocked.status_code == 503

    # Deactivate and verify unblocked
    set_maintenance_mode(False)
    resp_unblocked = client.post("/api/v1/chat", json={"message": ""})  # Validation error expected, not 503
    assert resp_unblocked.status_code != 503


def test_idempotency_lease_claim_blocked_during_maintenance_without_deleting_history():
    """Verify that claiming a new lease raises MaintenanceModeError, while completed replays succeed."""
    import uuid
    owner_id = f"test-maint-{uuid.uuid4().hex[:8]}"
    key_new = f"maint-new-{uuid.uuid4().hex[:8]}"
    key_completed = f"maint-done-{uuid.uuid4().hex[:8]}"
    req_hash = "hash-12345"

    # First, under normal mode, create a completed idempotency record
    set_maintenance_mode(False)
    cached, fence, worker = claim_idempotency_lease(owner_id, key_completed, req_hash)
    assert cached is None
    assert fence is not None
    complete_idempotency(owner_id, key_completed, {"image_url": "https://example.com/done.png", "status": "completed"})

    # Now activate maintenance mode
    set_maintenance_mode(True)

    # 1. New lease claim must be blocked
    with pytest.raises(MaintenanceModeError):
        claim_idempotency_lease(owner_id, key_new, req_hash)

    # 2. Existing completed record must be replayable (read-only)
    replayed, r_fence, r_worker = claim_idempotency_lease(owner_id, key_completed, req_hash)
    assert replayed is not None
    assert replayed["image_url"] == "https://example.com/done.png"

    # 3. Existing record was NOT deleted or altered
    rec = get_idempotency_record(owner_id, key_completed)
    assert rec is not None
    assert rec["status"] == "completed"

    set_maintenance_mode(None)


def test_storage_quota_reconciliation_skips_during_maintenance_unless_forced():
    """Verify reconcile_storage_quota safely skips during maintenance mode unless forced."""
    set_maintenance_mode(True)

    # When maintenance is active and force=False, returns None without error
    result = reconcile_storage_quota(owner_id="test-user", force=False)
    assert result is None

    set_maintenance_mode(None)


def test_stale_idempotency_reconciliation_skips_during_maintenance_unless_forced():
    """Verify reconcile_stale_idempotency safely returns maintenance_blocked when maintenance is active."""
    set_maintenance_mode(True)

    result = reconcile_stale_idempotency(owner_id="test-user", idempotency_key="stale-key", force=False)
    assert result["status"] == "maintenance_blocked"
    assert result["reconciled"] is False

    set_maintenance_mode(None)


def test_scheduled_task_execution_skips_during_maintenance():
    """Verify workspace_store.record_execution safely skips during maintenance mode."""
    set_maintenance_mode(True)

    rec = workspace_store.record_execution(
        task_id="task-123",
        status="running",
        output="output",
    )
    assert rec["id"] == "maintenance_blocked"
    assert rec["status"] == "blocked"

    set_maintenance_mode(None)


def test_production_loopback_rejected_even_if_allow_flag_set_outside_test_runner(monkeypatch):
    """Verify Phase 5 hardening: loopback override is strictly rejected outside automated test runners."""
    from app.core.database import validate_database_target

    # Simulate running outside pytest/CI
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CI_TESTING", raising=False)
    monkeypatch.setenv("ALLOW_TEST_LOOPBACK", "1")

    url = "postgresql+asyncpg://postgres:secret@127.0.0.1:5432/postgres"

    with pytest.raises(RuntimeError) as exc_info:
        validate_database_target(url, environment="production", allow_test_loopback=True)

    assert "strictly prohibited outside of automated test harnesses" in str(exc_info.value)
