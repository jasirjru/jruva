"""Independent remaining release criteria and diagnostic probes. No live services.

Failures are audit evidence, not xfail. PostgreSQL race test uses a modeled SQL
interleaving; it is not a hosted PostgreSQL test. Shared original fixtures unchanged.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from app.api.v1.endpoints.image import ImageGenerateRequest
from app.core.config import Settings, settings
from app.main import app, create_application
from app.services.image import budget
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import WorkspaceStore, workspace_store
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from tests.test_independent_image_audit import audit_context as original_audit_fixture
from tests.test_independent_image_audit import generate


@pytest.fixture(name="audit_context")
def inherited_audit_context(monkeypatch, tmp_path):
    return original_audit_fixture.__wrapped__(monkeypatch, tmp_path)


def request_hash(prompt="a boat"):
    payload = ImageGenerateRequest(prompt=prompt).model_dump()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("case", ["pending", "conflict"])
def test_idempotency_errors_return_409_without_inference(audit_context, case):
    c = audit_context
    budget.check_or_claim_idempotency(
        "audit-alice",
        "retry",
        request_hash("different boat" if case == "conflict" else "a boat"),
    )
    c.client = TestClient(app, raise_server_exceptions=False)
    r = generate(c, headers=c.alice | {"Idempotency-Key": "retry"})
    assert r.status_code == 409, "Idempotency domain error became internal server error"
    assert c.calls == 0


@pytest.mark.asyncio
async def test_postgres_claim_race_returns_domain_conflict(monkeypatch):
    """Force two SELECT-miss snapshots before INSERT under READ COMMITTED semantics."""
    barrier = asyncio.Event()
    selected = 0
    inserted = False

    class Session:
        def begin(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement, parameters=None):
            nonlocal selected, inserted
            sql = str(statement)
            if sql.startswith("SELECT"):
                selected += 1
                if selected == 2:
                    barrier.set()
                await barrier.wait()
                return SimpleNamespace(fetchone=lambda: None)
            if sql.startswith("INSERT"):
                if inserted:
                    raise IntegrityError(
                        "synthetic duplicate", {}, Exception("unique constraint")
                    )
                inserted = True
            return SimpleNamespace()

    @asynccontextmanager
    async def sessions():
        yield Session()

    monkeypatch.setattr(budget, "get_async_session", sessions)
    results = await asyncio.gather(
        *(
            budget._check_or_claim_idempotency_postgres(
                "synthetic", "same-key", "a" * 64
            )
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    assert results.count(None) == 1
    assert any(
        isinstance(r, ImageProviderError) and r.status == 409 for r in results
    ), "Concurrent PG claim leaks unique-constraint failure instead of domain conflict"


def test_completed_idempotency_survives_reopen_and_is_owner_scoped(
    audit_context, monkeypatch
):
    c = audit_context
    key = {"Idempotency-Key": "restart-key"}
    first = generate(c, headers=c.alice | key)
    monkeypatch.setattr(budget, "workspace_store", WorkspaceStore(c.db))
    again = generate(c, headers=c.alice | key)
    assert again.status_code == 200
    assert first.json()["image_id"] == again.json()["image_id"] and c.calls == 1
    other = generate(c, headers=c.bob | key)
    assert (
        other.status_code == 200
        and other.json()["image_id"] != first.json()["image_id"]
    )


def test_stale_pending_claim_has_explicit_reconciliation_state(
    audit_context, monkeypatch
):
    c = audit_context
    budget.check_or_claim_idempotency("audit-alice", "orphaned-key", request_hash())
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("UPDATE image_idempotency SET lease_expires_at='2000-01-01 00:00:00'")
    monkeypatch.setattr(budget, "workspace_store", WorkspaceStore(c.db))
    with pytest.raises(ImageProviderError) as exc:
        budget.check_or_claim_idempotency("audit-alice", "orphaned-key", request_hash())
    assert exc.value.kind == "idempotency_needs_reconciliation", (
        "Restart leaves stale pending work indistinguishable from active work indefinitely"
    )


def test_account_erasure_removes_identifiable_idempotency_records(audit_context):
    from app.services.account_erasure import _erase_sqlite_database

    c = audit_context
    assert (
        generate(c, headers=c.alice | {"Idempotency-Key": "private-prompt"}).status_code
        == 200
    )
    _erase_sqlite_database(c.db, "audit-alice")  # Only the per-test synthetic database.
    with workspace_store._get_sqlite_connection() as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM image_idempotency WHERE owner_id='audit-alice'"
        ).fetchone()[0]
    assert remaining == 0, (
        "Erasure retains owner ID, prompt, enhanced prompt and cached private URL"
    )


def test_replaying_key_after_image_delete_does_not_claim_success(audit_context):
    c = audit_context
    headers = c.alice | {"Idempotency-Key": "deleted-image"}
    first = generate(c, headers=headers).json()
    assert (
        c.client.delete(
            "/api/v1/image/" + first["image_id"], headers=c.alice
        ).status_code
        == 200
    )
    response = generate(c, headers=headers)
    assert response.status_code in (409, 410), (
        "Cached success points to an image that was deleted"
    )


@pytest.mark.parametrize("backend", ["sqlite", "unsupported"])
def test_production_rejects_ephemeral_or_unknown_database_backend(
    audit_context, monkeypatch, backend
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DATABASE_BACKEND", backend)
    with pytest.raises(RuntimeError):
        create_application()


def test_production_rejects_unconfigured_durable_storage(audit_context, monkeypatch):
    monkeypatch.setenv("ALLOW_TEST_LOOPBACK", "1")
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DATABASE_BACKEND", "postgres")
    monkeypatch.setattr(
        settings,
        "DATABASE_URL",
        "postgresql+asyncpg://fixture:fixture@127.0.0.1/not-used",
    )
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "supabase")
    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setenv("SUPABASE_KEY", "")
    with pytest.raises(RuntimeError, match="[Ss]torage"):
        create_application()


def test_configured_bucket_is_used_without_process_env(monkeypatch):
    import app.services.storage as module

    monkeypatch.delenv("STORAGE_BUCKET_NAME", raising=False)
    monkeypatch.setattr(settings, "STORAGE_BUCKET_NAME", "synthetic-private-bucket")
    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    monkeypatch.setattr(module, "_active_backend", None)
    assert module.get_storage_backend().bucket_name == "synthetic-private-bucket"


def test_explicit_origin_allowlist_is_not_broadened():
    cfg = Settings(_env_file=None, ALLOWED_ORIGINS="https://approved.example.invalid")
    assert cfg.cors_origins == ["https://approved.example.invalid"]


@pytest.mark.asyncio
async def test_chunked_oversized_body_is_bounded_before_json_parsing(audit_context):
    c = audit_context
    body = json.dumps(
        {"prompt": "a boat", "unused_padding": "x" * (256 * 1024)}
    ).encode()
    sent = 0

    async def chunks():
        nonlocal sent
        for i in range(0, len(body), 4096):
            sent += 4096
            yield body[i : i + 4096]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/image/generate",
            headers=c.alice
            | {"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
            content=chunks(),
        )
    assert response.status_code == 413, (
        f"Body was fully parsed; status={response.status_code}, streamed bytes~{sent}"
    )
    assert c.calls == 0


@pytest.mark.asyncio
async def test_resize_decode_work_is_inside_admission_bound(audit_context, monkeypatch):
    from app.services.image import upscaler

    c = audit_context
    original = generate(c).json()
    monkeypatch.setattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 100)
    validated = upscaler.validated_png
    lock = threading.Lock()
    active = peak = 0

    def held_decode(data):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        try:
            return validated(data)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(upscaler, "validated_png", held_decode)
    await asyncio.gather(
        *(
            upscaler.upscaler_service.upscale(
                original["image_url"], owner_id="audit-alice"
            )
            for _ in range(8)
        )
    )
    assert peak <= 2, (
        f"Pre-admission decode peak={peak}; resize semaphore does not cover decode"
    )


def test_daily_limits_do_not_bound_retained_storage_diagnostic(
    audit_context, monkeypatch
):
    """Passing diagnostic demonstrates accumulation, not a successful storage cap."""
    c = audit_context
    monkeypatch.setattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 1)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(10):

        class Clock:
            @staticmethod
            def now(zone=None, current_day=day):
                return start + timedelta(days=current_day)

        monkeypatch.setattr(budget, "datetime", Clock)
        assert generate(c, headers=c.alice | {"Idempotency-Key": f"new-image-day-{day}"}).status_code == 200
    assert len(workspace_store.list_images("audit-alice")) == 10
    assert len(list(c.media.iterdir())) == 10


def test_sync_bridge_reentrancy_does_not_deadlock(audit_context):
    backend = str(Path(__file__).resolve().parents[1])
    script = """
import asyncio,sys
sys.path.insert(0,sys.argv[1])
from app.core.database import run_sync_db
async def nested():
 print("ENTERED_NESTED",flush=True)
 return run_sync_db(asyncio.sleep(0,result=1))
assert run_sync_db(nested())==1
"""
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, backend],
            capture_output=True,
            timeout=2,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        assert b"ENTERED_NESTED" in (exc.stdout or b""), (
            "Probe did not reach the reentrant call"
        )
        pytest.fail(
            "Dedicated DB thread deadlocked on reentrant run_sync_db; child terminated safely"
        )
    assert result.returncode == 0


def test_idempotency_table_has_versioned_migration():
    folder = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    scripts = [p.read_text() for p in folder.glob("*.py")]
    assert any(
        "image_idempotency" in source and "create_table" in source for source in scripts
    ), "Idempotency table is runtime DDL only; migration history incomplete"


def test_sqlite_concurrent_claims_allow_exactly_one(audit_context):
    def claim(_):
        try:
            budget.check_or_claim_idempotency(
                "audit-alice", "concurrent", request_hash()
            )
            return "claimed"
        except ImageProviderError as exc:
            assert exc.status == 409
            return "pending"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert results.count("claimed") == 1 and results.count("pending") == 7


def test_completed_cache_survives_actual_process_restart(audit_context):
    c = audit_context
    expected = generate(
        c, headers=c.alice | {"Idempotency-Key": "process-restart"}
    ).json()["image_id"]
    backend = str(Path(__file__).resolve().parents[1])
    script = """
import socket,sys
def blocked(*a,**k): raise OSError("network disabled")
socket.socket.connect=blocked
sys.path.insert(0,sys.argv[1])
from app.services.image import budget
from app.services.workspace_store import WorkspaceStore
budget.workspace_store=WorkspaceStore(sys.argv[2])
result=budget.check_or_claim_idempotency("audit-alice","process-restart",sys.argv[3])
print("IMAGE="+result["image_id"])
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, backend, c.db, request_hash()],
        capture_output=True,
        text=True,
        timeout=10,
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 0, (
        "Synthetic restart child failed; no raw diagnostics printed"
    )
    assert result.stdout.split("IMAGE=")[-1].strip() == expected
    assert c.calls == 1
