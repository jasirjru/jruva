"""Reservation ledger correctness; PostgreSQL companion exercises actual locks."""

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest
from app.services.image import storage_quota as quota
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import workspace_store

from tests.test_independent_image_audit import audit_context as inherited_context


@pytest.fixture(name="audit_context")
def audit_context_fixture(monkeypatch, tmp_path):
    return inherited_context.__wrapped__(monkeypatch, tmp_path)


def reservation(owner="audit-alice", size=100, **kwargs):
    value = quota.reserve_storage_quota(owner, size, **kwargs)
    assert isinstance(value, str), "Reservation needs a durable identity, not anonymous counter increments"
    return value


def test_online_reconcile_preserves_user_and_project_reservations(audit_context):
    reservation()
    reservation("audit-bob", 200)
    quota.reconcile_storage_quota("audit-alice")
    a, b = quota.get_storage_quota_usage("audit-alice"), quota.get_storage_quota_usage("audit-bob")
    assert (a["image_count"], a["total_bytes"], a["project_bytes"]) == (1, 100, 300)
    assert (b["image_count"], b["total_bytes"], b["project_bytes"]) == (1, 200, 300)


def test_conversion_does_not_double_count_overlap(audit_context):
    first, second = reservation(), reservation()
    record = workspace_store.record_image("p", "path", "model", user_id="audit-alice", bytes=100, reservation_id=first)
    quota.reconcile_storage_quota()
    usage = quota.get_storage_quota_usage("audit-alice")
    assert (usage["image_count"], usage["total_bytes"], usage["project_bytes"]) == (2, 200, 200)
    quota.release_storage_quota("audit-alice", 100, reservation_id=second)
    quota.release_storage_quota("audit-alice", 100, reservation_id=second)
    assert quota.get_storage_quota_usage("audit-alice")["total_bytes"] == 100
    assert workspace_store.delete_image(record["id"])
    assert quota.get_storage_quota_usage("audit-alice")["total_bytes"] == 0


def test_expired_prewrite_reservation_is_abandoned_and_fenced(audit_context):
    identifier = reservation()
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(
            "UPDATE image_storage_reservations SET expires_at=? WHERE reservation_id=?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), identifier),
        )
    quota.reconcile_storage_quota()
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 0
    with pytest.raises(ImageProviderError):
        quota.begin_storage_write("audit-alice", identifier, "path")


def test_writing_reservation_never_expires_by_timer(audit_context):
    identifier = reservation()
    quota.begin_storage_write("audit-alice", identifier, "path")
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(
            "UPDATE image_storage_reservations SET expires_at='2000-01-01' WHERE reservation_id=?", (identifier,)
        )
    quota.reconcile_storage_quota()
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 100


def test_reconcile_cannot_open_capacity_for_another_request(audit_context):
    reservation(max_project_bytes=100)
    quota.reconcile_storage_quota()
    with pytest.raises(ImageProviderError) as error:
        quota.reserve_storage_quota("audit-bob", 1, max_project_bytes=100)
    assert error.value.kind == "storage_quota_exhausted"


def test_cross_owner_release_cannot_change_reservation(audit_context):
    identifier = reservation()
    with pytest.raises(ImageProviderError):
        quota.release_storage_quota("audit-bob", 100, reservation_id=identifier)
    assert quota.get_storage_quota_usage("audit-alice")["total_bytes"] == 100


def test_failed_crash_cleanup_is_durable_and_does_not_refund(audit_context, monkeypatch):
    from app.services.storage import storage_service

    identifier = reservation()
    quota.begin_storage_write("audit-alice", identifier, "path")
    monkeypatch.setattr(
        storage_service, "delete_verified", lambda *args: (_ for _ in ()).throw(OSError("cleanup unavailable"))
    )
    with pytest.raises(OSError):
        quota.recover_storage_reservation("audit-alice", identifier, confirmed_writer_stopped=True)
    with workspace_store._get_sqlite_connection() as conn:
        assert (
            conn.execute(
                "SELECT state FROM image_storage_reservations WHERE reservation_id=?", (identifier,)
            ).fetchone()[0]
            == "cleanup"
        )
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 100
    with pytest.raises(ImageProviderError):
        workspace_store.record_image(
            "late", "path", "model", user_id="audit-alice", bytes=100, reservation_id=identifier
        )


def test_crashed_write_cleanup_reclaims_capacity_once(audit_context):
    from app.services.storage import storage_service
    from app.services.storage.base import image_write_scope

    identifier = reservation()
    path = storage_service.reserved_image_path(identifier)
    quota.begin_storage_write("audit-alice", identifier, path)
    with image_write_scope(identifier):
        assert storage_service.save_bytes("generated.png", b"x" * 100) == path
    with pytest.raises(ValueError):
        quota.recover_storage_reservation("audit-alice", identifier)
    quota.recover_storage_reservation("audit-alice", identifier, confirmed_writer_stopped=True)
    quota.recover_storage_reservation("audit-alice", identifier, confirmed_writer_stopped=True)
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 0
    with pytest.raises(FileNotFoundError):
        storage_service.read_bytes(path)


@pytest.mark.asyncio
async def test_repeated_cancellation_observes_writer_before_refund(audit_context, monkeypatch):
    from app.services.image import genuine_service
    from app.services.storage import storage_service

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = storage_service.save_bytes

    def writer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(storage_service, "save_bytes", writer)
    task = asyncio.create_task(
        genuine_service.save_owned_image(audit_context.png, "audit-alice", "p", "p", "m", "B", "style")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert quota.get_storage_quota_usage("audit-alice")["total_bytes"] == len(audit_context.png)
    finally:
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert await asyncio.to_thread(finished.wait, 5)
    assert list(audit_context.media.iterdir()) == []
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 0


@pytest.mark.asyncio
async def test_cancellation_after_writing_transition_reclaims_empty_receipt(audit_context, monkeypatch):
    from app.services.image import genuine_service

    entered, release = threading.Event(), threading.Event()
    original = quota.begin_storage_write

    def transition(*args, **kwargs):
        original(*args, **kwargs)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(quota, "begin_storage_write", transition)
    task = asyncio.create_task(
        genuine_service.save_owned_image(audit_context.png, "audit-alice", "p", "p", "m", "B", "style")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == 0


@pytest.mark.asyncio
async def test_erased_account_cannot_reserve_after_decode(audit_context, monkeypatch):
    from app.services.auth_service import auth_service
    from app.services.image import genuine_service

    entered, release = threading.Event(), threading.Event()
    original = genuine_service.validated_png

    def decode(data):
        entered.set()
        assert release.wait(5)
        return original(data)

    monkeypatch.setattr(genuine_service, "validated_png", decode)
    task = asyncio.create_task(
        genuine_service.save_owned_image(audit_context.png, "audit-alice", "p", "p", "m", "B", "style")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with auth_service._get_connection() as conn:
            conn.execute("DELETE FROM users WHERE id='audit-alice'")
        release.set()
        with pytest.raises(ImageProviderError) as error:
            await task
        assert error.value.kind == "authentication_required"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    with workspace_store._get_sqlite_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM image_storage_reservations").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_unknown_remote_storage_write_keeps_capacity_until_explicit_recovery(audit_context, monkeypatch):
    from app.services.image import genuine_service
    from app.services.storage.supabase import SupabaseStorageBackend

    class LostStorageResponse(SupabaseStorageBackend):
        def save_bytes(self, *args, **kwargs):
            raise OSError("Remote server may still finish an accepted upload")

        def delete_verified(self, path):
            return None  # Absence now does not prove the accepted upload cannot arrive later.

    cloud = LostStorageResponse(supabase_url="https://storage.invalid", service_key="synthetic", bucket_name="private")
    monkeypatch.setattr(genuine_service, "storage_service", cloud)
    with pytest.raises(OSError):
        await genuine_service.save_owned_image(audit_context.png, "audit-alice", "p", "p", "m", "B", "style")
    assert quota.get_storage_quota_usage("audit-alice")["project_bytes"] == len(audit_context.png)
    with workspace_store._get_sqlite_connection() as conn:
        assert conn.execute("SELECT state FROM image_storage_reservations").fetchone()[0] == "cleanup"
