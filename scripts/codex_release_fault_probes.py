"""Remaining release acceptance failures. Run ONLY with the offline test runner.

These are deliberately separate from backend compatibility tests: failures are
release blockers, not reasons to contact staging or production.
"""

import asyncio
import threading

import pytest
from app.services.image import budget, genuine_service, storage_quota
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store
from tests.test_independent_image_audit import audit_context as inherited_context
from tests.test_independent_image_audit import generate


@pytest.fixture(name="audit_context")
def audit_context_fixture(monkeypatch, tmp_path):
    return inherited_context.__wrapped__(monkeypatch, tmp_path)


def test_reconciliation_preserves_in_flight_storage_reservation(audit_context):
    storage_quota.reserve_storage_quota("audit-alice", 100)
    storage_quota.reconcile_storage_quota()
    assert storage_quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 100


@pytest.mark.asyncio
async def test_cancelled_storage_thread_leaves_no_blob(audit_context, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    save = storage_service.save_bytes
    written = []

    def blocked_save(*args):
        entered.set()
        assert release.wait(5), "Probe did not release storage thread"
        try:
            path = save(*args)
            written.append(path)
            return path
        finally:
            finished.set()

    monkeypatch.setattr(storage_service, "save_bytes", blocked_save)
    task = asyncio.create_task(genuine_service.save_owned_image(
        audit_context.png, "audit-alice", "prompt", "prompt", "model", "B", "style"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    assert storage_quota.get_storage_quota_usage("audit-alice")["total_bytes"] == 0
    assert written
    with pytest.raises(FileNotFoundError):
        storage_service.read_bytes(written[0])


def test_rejected_fencing_has_no_published_image(audit_context, monkeypatch):
    monkeypatch.setattr(budget, "complete_idempotency", lambda *args: False)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "fenced"})
    assert response.status_code == 409
    assert workspace_store.list_images("audit-alice") == []


def test_provider_ambiguous_timeout_cannot_auto_repeat(audit_context, monkeypatch):
    from app.services.image.providers import CloudflareImageProvider, ImageProviderError

    calls = []

    async def ambiguous(*args, **kwargs):
        calls.append(True)  # Provider may have accepted/billed before response timeout.
        raise ImageProviderError("provider_timeout", "Timed out after possible acceptance", retryable=True)

    monkeypatch.setattr(CloudflareImageProvider, "generate", ambiguous)
    headers = audit_context.alice | {"Idempotency-Key": "ambiguous"}
    assert generate(audit_context, headers=headers).status_code == 503
    assert generate(audit_context, headers=headers).status_code == 503
    with workspace_store._get_sqlite_connection() as conn:
        usage = [row[0] for row in conn.execute("SELECT used FROM image_generation_budgets ORDER BY used")]
    assert usage == [1, 120], "One uncertain external attempt must remain charged once; a blocked retry must not reserve again"
    assert len(calls) == 1, "A failed local state automatically retried an externally indeterminate attempt"
