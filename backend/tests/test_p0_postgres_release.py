"""Real PostgreSQL READ COMMITTED transactions; requires the guarded private runner."""

import asyncio
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.services.image import budget
from app.services.image import storage_quota as quota
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import workspace_store

if not os.environ.get("CODEX_PRIVATE_PG_DB"):
    pytest.skip("Private PostgreSQL runner required; no environment database URL accepted", allow_module_level=True)

import psycopg


def connect():
    return psycopg.connect(host="127.0.0.1", port=55439, user="codex_audit", dbname=os.environ["CODEX_PRIVATE_PG_DB"])


def test_main_event_loop_and_sync_worker_do_not_share_pooled_connections():
    from app.core.database import get_async_session, run_sync_db
    from sqlalchemy import text

    async def ping():
        async with get_async_session() as session:
            return (await session.execute(text("SELECT 1"))).scalar()

    assert run_sync_db(ping) == 1
    assert asyncio.run(ping()) == 1
    assert run_sync_db(ping) == 1


@pytest.fixture(autouse=True)
def isolated_tables():
    with connect() as conn:
        assert conn.execute("SHOW transaction_isolation").fetchone()[0] == "read committed"
        conn.execute("TRUNCATE image_records,image_storage_reservations,image_storage_quotas,image_idempotency")
        conn.execute("CREATE TABLE IF NOT EXISTS codex_fake_provider_receipts(operation_id TEXT PRIMARY KEY)")
        conn.execute("TRUNCATE codex_fake_provider_receipts")
    quota.clear_reservation_context()
    yield


def test_concurrent_reserve_and_reconcile_cannot_drop_or_exceed_capacity():
    barrier = threading.Barrier(6)

    def worker(index):
        barrier.wait(timeout=5)
        if index < 4:
            try:
                return quota.reserve_storage_quota(f"pg-owner-{index}", 100, max_project_bytes=200)
            except ImageProviderError as exc:
                assert exc.kind == "storage_quota_exhausted"
                return None
        quota.reconcile_storage_quota()
        return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(worker, range(6)))
    assert sum(isinstance(value, str) for value in results) == 2
    quota.reconcile_storage_quota()
    with connect() as conn:
        assert (
            conn.execute("SELECT SUM(bytes) FROM image_storage_reservations WHERE state='active'").fetchone()[0] == 200
        )
        assert (
            conn.execute("SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__'").fetchone()[0]
            == 200
        )


def test_reconciler_waits_for_project_transaction_and_sees_committed_reservation():
    identifier = quota.reserve_storage_quota("pg-owner", 100)
    holder = connect()
    holder.execute("SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__' FOR UPDATE")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(quota.reconcile_storage_quota)
        try:
            deadline = time.monotonic() + 5
            with connect() as observer:
                while time.monotonic() < deadline:
                    observer.commit()
                    blocked = observer.execute(
                        "SELECT COUNT(*) FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock'"
                    ).fetchone()[0]
                    if blocked:
                        break
                    time.sleep(0.01)
                assert blocked, "Reconciliation did not acquire the project transaction barrier"
            assert not future.done()
        finally:
            holder.commit()
            holder.close()
        future.result(timeout=5)
    assert quota.get_storage_quota_usage("pg-owner")["project_bytes"] == 100
    quota.release_storage_quota("pg-owner", 100, identifier)


def test_record_conversion_and_deletion_are_atomic():
    identifier = quota.reserve_storage_quota("pg-owner", 100)
    record = workspace_store.record_image(
        "synthetic", "path", "model", user_id="pg-owner", bytes=100, reservation_id=identifier
    )
    quota.reconcile_storage_quota()
    assert quota.get_storage_quota_usage("pg-owner")["total_bytes"] == 100
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(workspace_store.delete_image, [record["id"], record["id"]]))
    assert sorted(results) == [False, True]
    quota.reconcile_storage_quota()
    assert quota.get_storage_quota_usage("pg-owner")["project_bytes"] == 0


def test_failed_record_transaction_preserves_reservation(monkeypatch):
    identifier = quota.reserve_storage_quota("pg-owner", 100)
    original = quota.commit_storage_reservation

    async def failing(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("Injected after conversion before transaction commit")

    monkeypatch.setattr(quota, "commit_storage_reservation", failing)
    with pytest.raises(RuntimeError, match="Injected"):
        workspace_store.record_image(
            "synthetic", "path", "model", user_id="pg-owner", bytes=100, reservation_id=identifier
        )
    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM image_records").fetchone()[0] == 0
        assert conn.execute("SELECT state FROM image_storage_reservations").fetchone()[0] == "active"
    assert quota.get_storage_quota_usage("pg-owner")["project_bytes"] == 100


def test_expiry_fences_prewrite_but_keeps_writing_receipts():
    active = quota.reserve_storage_quota("pg-owner", 100)
    writing = quota.reserve_storage_quota("pg-owner", 200)
    quota.begin_storage_write("pg-owner", writing, "known-path")
    with connect() as conn:
        conn.execute("UPDATE image_storage_reservations SET expires_at=clock_timestamp()-INTERVAL '1 second'")
    quota.reconcile_storage_quota()
    assert quota.get_storage_quota_usage("pg-owner")["project_bytes"] == 200
    with pytest.raises(ImageProviderError):
        quota.begin_storage_write("pg-owner", active, "late-path")


def test_fenced_dispatch_and_unknown_retry_against_postgres():
    _, fence, worker = budget.claim_idempotency_lease("pg-owner", "unknown", "hash", "original")
    identifier = budget.begin_external_dispatch("pg-owner", "unknown", worker, fence, "fake")
    with connect() as conn:
        conn.execute("UPDATE image_idempotency SET lease_expires_at=clock_timestamp()-INTERVAL '10 seconds'")
    with pytest.raises(ImageProviderError) as error:
        budget.claim_idempotency_lease("pg-owner", "unknown", "hash", "retry")
    assert error.value.kind == "operation_indeterminate"
    assert not budget.fail_idempotency("pg-owner", "unknown", worker, fence)
    assert budget.get_idempotency_record("pg-owner", "unknown")["operation_id"] == identifier


WORKER = """
import sys,time,json,psycopg
from app.services.image import budget
from app.services.image.providers import ImageProviderError
try:
 _,f,w=budget.claim_idempotency_lease('pg-owner','crashed','hash','killed')
 operation=budget.begin_external_dispatch('pg-owner','crashed',w,f,'fake')
 with psycopg.connect(host='127.0.0.1',port=55439,user='codex_audit',dbname=__import__('os').environ['CODEX_PRIVATE_PG_DB']) as c:
  c.execute('INSERT INTO codex_fake_provider_receipts VALUES (%s)',(operation,))
 print(json.dumps({'accepted':operation}),flush=True)
 if len(sys.argv)>1 and sys.argv[1]=='hold': time.sleep(60)
except ImageProviderError as e:
 print(json.dumps({'error':e.kind}),flush=True)
"""


def test_independent_process_death_preserves_dispatch_receipt():
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", WORKER, "hold"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            assert line, process.stderr.read()
            if '"accepted"' in line:
                break
        else:
            pytest.fail("Worker never committed dispatch")
        process.kill()
        process.communicate(timeout=5)
        with connect() as conn:
            conn.execute("UPDATE image_idempotency SET lease_expires_at=clock_timestamp()-INTERVAL '10 seconds'")
        with pytest.raises(ImageProviderError) as error:
            budget.claim_idempotency_lease("pg-owner", "crashed", "hash", "replacement")
        assert error.value.kind == "operation_indeterminate"
        assert budget.get_idempotency_record("pg-owner", "crashed")["provider_attempts"] == 1
        with connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM codex_fake_provider_receipts").fetchone()[0] == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_concurrent_independent_processes_have_one_external_acceptance():
    processes = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", WORKER], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for _ in range(3)
    ]
    try:
        results = [process.communicate(timeout=20) for process in processes]
        assert all(process.returncode == 0 for process in processes), results
        with connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM codex_fake_provider_receipts").fetchone()[0] == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def test_strict_reservation_rejects_missing_account():
    with pytest.raises(ImageProviderError) as error:
        quota.reserve_storage_quota("missing-account", 100, require_active_owner=True)
    assert error.value.kind == "authentication_required"
    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM image_storage_reservations").fetchone()[0] == 0


def test_account_erasure_preserves_other_owner_reservations():
    from app.core.database import run_sync_db
    from app.services.account_erasure import _erase_postgres, _mark_pending_postgres

    from tests.private_accounts import private_account_headers

    private_account_headers("pg-erased-owner")
    quota.reserve_storage_quota("pg-erased-owner", 100, require_active_owner=True)
    quota.reserve_storage_quota("pg-other-owner", 200)
    assert run_sync_db(_mark_pending_postgres, "pg-erased-owner")
    quota.cancel_owner_reservations("pg-erased-owner")
    run_sync_db(_erase_postgres, "pg-erased-owner")
    quota.reconcile_storage_quota()
    assert quota.get_storage_quota_usage("pg-other-owner")["project_bytes"] == 200


def test_concurrent_process_quota_reservations_cannot_bypass_limit():
    worker = """
import sys,json
from app.services.image import storage_quota as q
from app.services.image.providers import ImageProviderError
try:
 r=q.reserve_storage_quota('process-'+sys.argv[1],100,max_project_bytes=200)
 q.reconcile_storage_quota()
 print(json.dumps({'reserved':r}),flush=True)
except ImageProviderError as e:
 print(json.dumps({'error':e.kind}),flush=True)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", worker, str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for i in range(3)
    ]
    try:
        outputs = [process.communicate(timeout=20) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        with connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*),SUM(bytes) FROM image_storage_reservations WHERE state='active'"
            ).fetchone() == (2, 200)
            assert (
                conn.execute("SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__'").fetchone()[0]
                == 200
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def test_migration_head_and_schema():
    with connect() as conn:
        assert (
            conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "007_operation_reservation_ledger"
        )
        assert (
            conn.execute(
                "SELECT is_nullable FROM information_schema.columns WHERE table_name='image_storage_reservations' AND column_name='bytes'"
            ).fetchone()[0]
            == "NO"
        )
