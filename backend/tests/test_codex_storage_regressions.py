"""Real thread cancellation and concurrent account erasure against local stores."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from app.services.image import budget, genuine_service, storage_quota
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store

from tests.test_independent_image_audit import audit_context as inherited_context
from tests.test_independent_image_audit import generate


@pytest.fixture(name="audit_context")
def audit_context_fixture(monkeypatch, tmp_path):
    return inherited_context.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["blob", "record"])
async def test_cancelled_mutation_finishes_and_compensates_once(audit_context, monkeypatch, operation):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    storage_quota.reserve_storage_quota("audit-alice", 100)
    workspace_store.record_image("existing", "existing-path", "model", user_id="audit-alice", bytes=100)
    target = storage_service if operation == "blob" else workspace_store
    method = "save_bytes" if operation == "blob" else "record_image"
    original = getattr(target, method)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(target, method, blocked)
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
    usage = storage_quota.get_storage_quota_usage("audit-alice")
    assert (usage["image_count"], usage["total_bytes"], usage["project_bytes"]) == (1, 100, 100)
    assert len(workspace_store.list_images("audit-alice")) == 1
    assert list(audit_context.media.iterdir()) == []


def test_fenced_completion_compensates_only_owned_attempt(audit_context, monkeypatch):
    monkeypatch.setattr(budget, "complete_idempotency", lambda *args: False)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "fenced"})
    assert response.status_code == 409
    assert workspace_store.list_images("audit-alice") == []
    assert list(audit_context.media.iterdir()) == []
    assert storage_quota.get_storage_quota_usage("audit-alice")["total_bytes"] == 0


def test_account_erasure_preserves_other_owners_pending_reservation(audit_context, monkeypatch):
    from app.services.account_erasure import erase_account
    from app.services.auth_service import auth_service
    from app.services.conversation_log_service import conversation_log_service
    from app.services.memory_service import memory_service
    from app.services.rag import qdrant_vector_store
    from app.services.session_store import session_store

    for service in (conversation_log_service, memory_service, session_store):
        monkeypatch.setattr(service, "db_path", audit_context.db)
    monkeypatch.setattr(qdrant_vector_store, "QdrantVectorStore", lambda: SimpleNamespace(
        delete_document_chunks=lambda *args, **kwargs: None,
        delete_owner_chunks=lambda *args, **kwargs: None,
    ))
    storage_quota.reserve_storage_quota("audit-bob", 100)
    assert erase_account("audit-alice", auth_service)
    assert storage_quota.get_storage_quota_usage("audit-bob")["project_bytes"] == 100
