"""Independent controlled-time lease regressions; no external service access."""

import asyncio
import json
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from app.services.image import budget
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import workspace_store

from tests.test_independent_image_audit import audit_context as inherited_context
from tests.test_independent_image_audit import generate


@pytest.fixture(name="audit_context")
def audit_context_fixture(monkeypatch, tmp_path):
    return inherited_context.__wrapped__(monkeypatch, tmp_path)


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        current = datetime.now(timezone.utc).replace(microsecond=0)

        @classmethod
        def now(cls, tz=None):
            return cls.current if tz else cls.current.replace(tzinfo=None)

    monkeypatch.setattr(budget, "datetime", Clock)
    return Clock


def expect_kind(kind, worker="second"):
    with pytest.raises(ImageProviderError) as exc:
        budget.claim_idempotency_lease("audit-alice", "lease", "hash", worker)
    assert exc.value.kind == kind


def test_heartbeat_beats_original_age_across_workers(audit_context, clock):
    _, fence, worker = budget.claim_idempotency_lease("audit-alice", "lease", "hash", "first")
    clock.current += timedelta(seconds=170)
    assert budget.heartbeat_idempotency_lease("audit-alice", "lease", worker, fence)
    clock.current += timedelta(seconds=20)
    expect_kind("idempotency_in_progress")
    assert budget.complete_idempotency("audit-alice", "lease", {"success": True}, worker, fence)


@pytest.mark.asyncio
async def test_postgres_live_lease_beats_original_age(monkeypatch, clock):
    """SQL path model, explicitly not a PostgreSQL integration test."""
    clock.current += timedelta(seconds=190)
    row = ("hash", "pending", None, clock.current - timedelta(seconds=190),
           "first", clock.current + timedelta(seconds=160), 1)

    class Session:
        def begin(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement, parameters=None):
            return SimpleNamespace(fetchone=lambda: row, rowcount=1)

    @asynccontextmanager
    async def session():
        yield Session()

    monkeypatch.setattr(budget, "get_async_session", session)
    with pytest.raises(ImageProviderError) as exc:
        await budget._claim_idempotency_lease_postgres("owner", "lease", "hash", "second")
    assert exc.value.kind == "idempotency_in_progress"


def test_reclaimed_lease_does_not_inherit_old_age(audit_context, clock):
    budget.claim_idempotency_lease("audit-alice", "lease", "hash", "first")
    clock.current += timedelta(seconds=181)
    expect_kind("idempotency_needs_reconciliation")
    _, fence, worker = budget.claim_idempotency_lease("audit-alice", "lease", "hash", "second")
    expect_kind("idempotency_in_progress", "third")
    assert not budget.complete_idempotency("audit-alice", "lease", {}, "first", 1)
    assert not budget.fail_idempotency("audit-alice", "lease", "first", 1)
    assert not budget.heartbeat_idempotency_lease("audit-alice", "lease", "first", 1)
    assert budget.complete_idempotency("audit-alice", "lease", {"ok": True}, worker, fence)
    assert budget.claim_idempotency_lease("audit-alice", "lease", "hash", "third")[0] == {"ok": True}


def test_reconciliation_cannot_revoke_live_lease(audit_context, clock):
    budget.claim_idempotency_lease("audit-alice", "lease", "hash", "first")
    result = budget.reconcile_stale_idempotency("audit-alice", "lease")
    assert result["reconciled"] is False
    assert result["status"] == "pending"


def test_legacy_null_lease_still_recovers(audit_context, clock):
    budget.claim_idempotency_lease("audit-alice", "lease", "hash", "first")
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("UPDATE image_idempotency SET lease_expires_at=NULL, created_at=?",
                     ((clock.current - timedelta(seconds=181)).isoformat(),))
    expect_kind("idempotency_needs_reconciliation")


WORKER_CODE = """
import json, sys, os
if os.path.exists("backend"): sys.path.insert(0, os.path.abspath("backend"))
from app.services.workspace_store import workspace_store
from app.services.image import budget
from app.services.image.providers import ImageProviderError
workspace_store.db_path = sys.argv[1]
try:
    value = budget.claim_idempotency_lease('audit-alice', 'lease', 'hash', sys.argv[2])
    print(json.dumps({'claimed': value}))
except ImageProviderError as exc:
    print(json.dumps({'error': exc.kind}))
"""


def test_independent_process_rejects_old_but_renewed_lease(audit_context):
    _, fence, worker = budget.claim_idempotency_lease("audit-alice", "lease", "hash", "first")
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("UPDATE image_idempotency SET created_at=?",
                     ((datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat(),))
    assert budget.heartbeat_idempotency_lease("audit-alice", "lease", worker, fence)
    result = subprocess.run([sys.executable, "-c", WORKER_CODE, audit_context.db, "second"],
                            capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {"error": "idempotency_in_progress"}


def test_concurrent_processes_have_one_claim(audit_context):
    processes = [subprocess.Popen([sys.executable, "-c", WORKER_CODE, audit_context.db, str(i)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(3)]
    results = []
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, err
        results.append(json.loads(out.splitlines()[-1]))
    assert sum("claimed" in result for result in results) == 1
    assert sum(result.get("error") == "idempotency_in_progress" for result in results) == 2


def test_killed_worker_recovers_and_cannot_complete(audit_context, clock):
    code = WORKER_CODE + "\nimport time\nsys.stdout.flush()\ntime.sleep(60)\n"
    process = subprocess.Popen([sys.executable, "-u", "-c", code, audit_context.db, "killed"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Read the flushed claim before terminating the worker holding it.
        while True:
            line = process.stdout.readline()
            assert line, "Worker exited before claiming"
            if line.startswith('{"claimed"'):
                break
        process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    clock.current += timedelta(seconds=181)
    expect_kind("idempotency_needs_reconciliation")
    _, fence, worker = budget.claim_idempotency_lease("audit-alice", "lease", "hash", "recovery")
    assert not budget.complete_idempotency("audit-alice", "lease", {}, "killed", 1)
    assert budget.complete_idempotency("audit-alice", "lease", {}, worker, fence)


def test_endpoint_stops_when_heartbeat_loses_ownership(audit_context, monkeypatch):
    from app.api.v1.endpoints import image
    from app.services.image.providers import CloudflareImageProvider

    cancelled = []

    async def provider(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(image, "IDEMPOTENCY_HEARTBEAT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(budget, "heartbeat_idempotency_lease", lambda *args: False)
    monkeypatch.setattr(CloudflareImageProvider, "generate", provider)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "lost-heartbeat"})
    assert response.status_code == 409
    assert cancelled


def test_endpoint_rejects_fenced_completion(audit_context, monkeypatch):
    monkeypatch.setattr(budget, "complete_idempotency", lambda *args: False)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "lost"})
    assert response.status_code == 409


def test_endpoint_renews_lease_during_generation(audit_context, monkeypatch):
    from app.api.v1.endpoints import image
    from app.services.image.providers import SCHNELL, CloudflareImageProvider, GeneratedImage

    calls = []
    original = budget.heartbeat_idempotency_lease

    def heartbeat(*args):
        calls.append(args)
        return original(*args)

    async def slow_provider(*args, **kwargs):
        await asyncio.sleep(0.08)
        return GeneratedImage(audit_context.png, SCHNELL, "synthetic")

    monkeypatch.setattr(image, "IDEMPOTENCY_HEARTBEAT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(budget, "heartbeat_idempotency_lease", heartbeat)
    monkeypatch.setattr(CloudflareImageProvider, "generate", slow_provider)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "slow"})
    assert response.status_code == 200
    assert calls, "Endpoint never renewed the distributed lease"


def test_heartbeat_does_not_revoke_just_committed_completion(audit_context, monkeypatch):
    from app.api.v1.endpoints import image

    complete = budget.complete_idempotency

    def delayed_return(*args):
        result = complete(*args)
        time.sleep(0.08)  # Commit succeeded; worker thread has not returned yet.
        return result

    monkeypatch.setattr(image, "IDEMPOTENCY_HEARTBEAT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(budget, "complete_idempotency", delayed_return)
    response = generate(audit_context, headers=audit_context.alice | {"Idempotency-Key": "completion-race"})
    assert response.status_code == 200
