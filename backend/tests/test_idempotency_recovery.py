"""Tests for idempotency lifecycle, stale reconciliation, and recovery (R09).

Verifies active vs stale states, reconciliation without duplicate inference,
replays after deletion, changed payloads, and account erasure.
"""

from datetime import datetime, timedelta, timezone

import pytest
from app.services.image import budget
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import workspace_store


@pytest.fixture(autouse=True)
def clean_idempotency():
    """Ensure clean idempotency table for tests."""
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(budget.IDEMPOTENCY_DDL)
        conn.execute("DELETE FROM image_idempotency WHERE owner_id LIKE 'idem_%'")
        conn.execute("DELETE FROM image_records WHERE user_id LIKE 'idem_%'")
        conn.commit()
    yield
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(budget.IDEMPOTENCY_DDL)
        conn.execute("DELETE FROM image_idempotency WHERE owner_id LIKE 'idem_%'")
        conn.execute("DELETE FROM image_records WHERE user_id LIKE 'idem_%'")
        conn.commit()


def test_active_pending_returns_in_progress():
    """An active pending attempt returns HTTP 409 idempotency_in_progress."""
    owner = "idem_user_active"
    key = "key-active"
    req_hash = "hash-1"

    # Claim
    claim = budget.check_or_claim_idempotency(owner, key, req_hash)
    assert claim is None

    # Immediate second attempt should return 409
    with pytest.raises(ImageProviderError) as exc_info:
        budget.check_or_claim_idempotency(owner, key, req_hash)
    assert exc_info.value.status == 409
    assert exc_info.value.kind == "idempotency_in_progress"


def test_stale_pending_request_reconciles_and_allows_retry():
    """A stale pending attempt (>180s) reconciles lock and allows retry instead of permanent deadlock."""
    owner = "idem_user_stale"
    key = "key-stale"
    req_hash = "hash-1"

    budget.check_or_claim_idempotency(owner, key, req_hash)

    # Expire the renewable lease; creation age no longer revokes a live lease.
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).strftime("%Y-%m-%d %H:%M:%S")
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(
            "UPDATE image_idempotency SET lease_expires_at = ? WHERE owner_id = ? AND idempotency_key = ?",
            (old_time, owner, key),
        )
        conn.commit()

    # First retry encounters stale lock, cleans it, and informs user
    with pytest.raises(ImageProviderError) as exc_info:
        budget.check_or_claim_idempotency(owner, key, req_hash)
    assert exc_info.value.status == 409
    assert exc_info.value.kind == "idempotency_needs_reconciliation"

    # Second retry now succeeds in acquiring fresh claim!
    fresh_claim = budget.check_or_claim_idempotency(owner, key, req_hash)
    assert fresh_claim is None


def test_completed_request_returns_cached_response():
    """A completed attempt returns cached response without duplicate provider execution."""
    owner = "idem_user_completed"
    key = "key-completed"
    req_hash = "hash-1"

    budget.check_or_claim_idempotency(owner, key, req_hash)
    
    # Record actual image in workspace_store
    rec = workspace_store.record_image(
        prompt="cached", file_url="local://cached.png", model="m", user_id=owner
    )
    resp_data = {"image_id": rec["id"], "file_url": "local://cached.png"}
    budget.complete_idempotency(owner, key, resp_data)

    cached = budget.check_or_claim_idempotency(owner, key, req_hash)
    assert cached["image_id"] == rec["id"]


def test_replay_after_image_deletion_returns_410():
    """If referenced image was deleted, subsequent idempotency checks return HTTP 410."""
    owner = "idem_user_del"
    key = "key-del"
    req_hash = "hash-1"

    budget.check_or_claim_idempotency(owner, key, req_hash)
    rec = workspace_store.record_image(
        prompt="to-delete", file_url="local://del.png", model="m", user_id=owner
    )
    budget.complete_idempotency(owner, key, {"image_id": rec["id"]})

    # Delete the image
    workspace_store.delete_image(rec["id"])

    # Idempotency replay should now return 410 image_deleted
    with pytest.raises(ImageProviderError) as exc_info:
        budget.check_or_claim_idempotency(owner, key, req_hash)
    assert exc_info.value.status == 410
    assert exc_info.value.kind == "image_deleted"


def test_changed_payload_with_same_key_returns_409_conflict():
    """Reusing the same key with different request parameters returns HTTP 409 idempotency_conflict."""
    owner = "idem_user_conflict"
    key = "key-conflict"

    budget.check_or_claim_idempotency(owner, key, "hash-original")

    with pytest.raises(ImageProviderError) as exc_info:
        budget.check_or_claim_idempotency(owner, key, "hash-changed")
    assert exc_info.value.status == 409
    assert exc_info.value.kind == "idempotency_conflict"


def test_reconcile_stale_idempotency_helper():
    """reconcile_stale_idempotency explicitly transitions stale state."""
    owner = "idem_user_helper"
    key = "key-helper"

    budget.check_or_claim_idempotency(owner, key, "hash-1")
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("UPDATE image_idempotency SET lease_expires_at='2000-01-01 00:00:00' WHERE owner_id=?", (owner,))
    res = budget.reconcile_stale_idempotency(owner, key)
    assert res["status"] == "stale_expired"
    assert res["reconciled"] is True
