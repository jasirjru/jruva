"""Durable externally ambiguous outcomes; all provider I/O is deterministic fake I/O."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from app.services.image import budget
from app.services.image.providers import CloudflareImageProvider, ImageProviderError
from app.services.workspace_store import workspace_store

from tests.test_independent_image_audit import audit_context as inherited_context
from tests.test_independent_image_audit import generate


@pytest.fixture(name="audit_context")
def audit_context_fixture(monkeypatch, tmp_path):
    return inherited_context.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.parametrize("failure", ["timeout", "connection", "unexpected", "unrecognized"])
def test_unknown_outcome_persists_and_never_reissues(audit_context, monkeypatch, failure):
    accepted = []

    async def provider(*args, **kwargs):
        accepted.append(True)
        if failure == "unexpected":
            raise RuntimeError("synthetic lost response")
        if failure == "unrecognized":
            raise ImageProviderError("future_provider_error", "unclassified response", retryable=True)
        raise ImageProviderError(
            "provider_timeout" if failure == "timeout" else "provider_unavailable", "unknown", retryable=True
        )

    monkeypatch.setattr(CloudflareImageProvider, "generate", provider)
    headers = audit_context.alice | {"Idempotency-Key": "unknown"}
    assert generate(audit_context, headers=headers).status_code == 503
    assert generate(audit_context, headers=headers).status_code == 503
    row = budget.get_idempotency_record("audit-alice", "unknown")
    assert row["status"] == "indeterminate"
    assert row["operation_id"]
    assert row["external_state"] in {"dispatched", "indeterminate"}
    assert len(accepted) == 1
    with workspace_store._get_sqlite_connection() as conn:
        assert sorted(r[0] for r in conn.execute("SELECT used FROM image_generation_budgets")) == [1, 120]


def test_expired_dispatched_worker_cannot_be_reclaimed(audit_context):
    _, fence, worker = budget.claim_idempotency_lease("audit-alice", "crash", "hash", "crashed-worker")
    assert hasattr(budget, "begin_external_dispatch"), "Missing durable pre-dispatch marker"
    operation_id = budget.begin_external_dispatch("audit-alice", "crash", worker, fence, "synthetic")
    old = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("UPDATE image_idempotency SET lease_expires_at=?, created_at=?", (old, old))
    for _ in range(3):
        with pytest.raises(ImageProviderError) as error:
            budget.claim_idempotency_lease("audit-alice", "crash", "hash", "replacement")
        assert error.value.kind == "operation_indeterminate"
    assert not budget.fail_idempotency("audit-alice", "crash", worker, fence)
    assert budget.get_idempotency_record("audit-alice", "crash")["operation_id"] == operation_id


def test_provider_success_followed_by_save_failure_never_reissues(audit_context, monkeypatch):
    from app.services.storage import storage_service

    monkeypatch.setattr(
        storage_service, "save_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full"))
    )
    headers = audit_context.alice | {"Idempotency-Key": "success-storage-failure"}
    assert generate(audit_context, headers=headers).status_code == 503
    assert generate(audit_context, headers=headers).status_code == 503
    assert audit_context.calls == 1
    assert budget.get_idempotency_record("audit-alice", "success-storage-failure")["external_state"] == "succeeded"


@pytest.mark.asyncio
async def test_cancellation_does_not_clear_dispatch_marker(audit_context):
    _, fence, worker = budget.claim_idempotency_lease("audit-alice", "cancel", "hash")
    assert hasattr(budget, "begin_external_dispatch")
    budget.begin_external_dispatch("audit-alice", "cancel", worker, fence, "synthetic")
    assert not await asyncio.to_thread(budget.fail_idempotency, "audit-alice", "cancel", worker, fence)
    assert budget.get_idempotency_record("audit-alice", "cancel")["status"] == "indeterminate"


def test_ambiguous_primary_never_dispatches_fallback(audit_context, monkeypatch):
    from app.core.config import settings
    from app.services.image.providers import KLEIN

    monkeypatch.setattr(settings, "IMAGE_CF_KLEIN_ENABLED", True)
    monkeypatch.setattr(settings, "IMAGE_CF_KLEIN_FREE_VERIFIED", True)
    accepted = []

    async def provider(self, prompt, seed, model):
        accepted.append(model)
        raise ImageProviderError("provider_timeout", "Possibly accepted", retryable=True)

    monkeypatch.setattr(CloudflareImageProvider, "generate", provider)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "no-fallback"})
    assert response.status_code == 503
    assert accepted == [KLEIN]
    with workspace_store._get_sqlite_connection() as conn:
        assert sorted(row[0] for row in conn.execute("SELECT used FROM image_generation_budgets")) == [1, 106]


def test_missing_key_unknown_retry_keeps_same_request_identity(audit_context, monkeypatch):
    accepted = []

    async def provider(*args, **kwargs):
        accepted.append(True)
        raise ImageProviderError("provider_timeout", "Unknown accepted request", retryable=True)

    monkeypatch.setattr(CloudflareImageProvider, "generate", provider)
    assert generate(audit_context).status_code == 503
    assert generate(audit_context).status_code == 503
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_shared_chat_generation_unknown_retry_keeps_identity(audit_context, monkeypatch):
    from app.services.image import genuine_service

    accepted = []

    async def provider(*args, **kwargs):
        accepted.append(True)
        raise ImageProviderError("provider_timeout", "Unknown accepted request", retryable=True)

    monkeypatch.setattr(CloudflareImageProvider, "generate", provider)
    assert not (await genuine_service.generate("boat", owner="audit-alice"))["success"]
    assert not (await genuine_service.generate("boat", owner="audit-alice"))["success"]
    assert len(accepted) == 1
