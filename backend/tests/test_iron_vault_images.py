"""Synthetic image storage commit-uncertainty and recovery regressions."""

import pytest
from app.services.image.genuine_service import save_owned_image
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store

from tests.test_independent_image_audit import audit_context as audit_context


@pytest.mark.asyncio
async def test_committed_image_survives_lost_metadata_acknowledgement(audit_context, monkeypatch):
    original = workspace_store.record_image

    def lost_ack(**kwargs):
        original(**kwargs)
        raise OSError("Synthetic commit succeeded; acknowledgement lost")

    monkeypatch.setattr(workspace_store, "record_image", lost_ack)
    result = await save_owned_image(audit_context.png, "audit-alice", "synthetic boat", "synthetic boat", "synthetic", "B", "plain")
    images = workspace_store.list_images("audit-alice")
    assert len(images) == 1
    assert result["image_id"] == images[0]["id"]
    assert storage_service.exists(images[0]["file_url"]), "Commit-ack uncertainty deleted a committed image object"
    with workspace_store._get_sqlite_connection() as conn:
        assert conn.execute("SELECT state FROM image_storage_reservations").fetchone()[0] == "committed"


@pytest.mark.asyncio
async def test_confirmed_metadata_rollback_still_compensates(audit_context, monkeypatch):
    from app.services.image.storage_quota import get_storage_quota_usage

    def failed_write(**kwargs):
        raise OSError("Synthetic rolled-back metadata write")

    monkeypatch.setattr(workspace_store, "record_image", failed_write)
    with pytest.raises(OSError):
        await save_owned_image(audit_context.png, "audit-alice", "synthetic", "synthetic", "synthetic", "B", "plain")
    assert workspace_store.list_images("audit-alice") == []
    assert list(audit_context.media.iterdir()) == []
    assert get_storage_quota_usage("audit-alice")["total_bytes"] == 0


@pytest.mark.asyncio
async def test_unverifiable_metadata_retains_object_and_capacity_for_explicit_recovery(audit_context, monkeypatch):
    from app.services.image import storage_recovery
    from app.services.image.providers import ImageProviderError
    from app.services.image.storage_quota import get_storage_quota_usage, recover_storage_reservation

    def fail(*args, **kwargs):
        raise OSError("Synthetic database unavailable")

    monkeypatch.setattr(workspace_store, "record_image", fail)
    monkeypatch.setattr(storage_recovery, "resolve_image_metadata_write", fail)
    with pytest.raises(ImageProviderError) as error:
        await save_owned_image(audit_context.png, "audit-alice", "synthetic", "synthetic", "synthetic", "B", "plain")
    assert error.value.kind == "storage_metadata_indeterminate"
    with workspace_store._get_sqlite_connection() as conn:
        identifier, path, state = conn.execute("SELECT reservation_id,object_path,state FROM image_storage_reservations").fetchone()
    assert state == "writing" and storage_service.exists(path)
    assert get_storage_quota_usage("audit-alice")["total_bytes"] > 0
    recover_storage_reservation("audit-alice", identifier, confirmed_writer_stopped=True)
    assert not storage_service.exists(path)
    assert get_storage_quota_usage("audit-alice")["total_bytes"] == 0
