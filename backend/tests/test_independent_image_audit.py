"""Independent release criteria. Failing assertions are retained as audit evidence.

Only synthetic accounts, temporary local data and mocked providers are used.
Run through scripts/verify_operations_security.py to block real network access.
"""

import asyncio
import base64
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from app.core.config import Settings, settings
from app.main import app, create_application
from app.services.auth_service import auth_service
from app.services.image import budget, genuine_service
from app.services.image.providers import (
    SCHNELL,
    CloudflareImageProvider,
    GeneratedImage,
    ImageProviderError,
)
from app.services.storage import storage_service
from app.services.workspace_store import WorkspaceStore, workspace_store
from fastapi.testclient import TestClient
from PIL import Image

from tests.private_accounts import private_account_headers


class AuditContext(SimpleNamespace):
    def __repr__(self):
        return f"AuditContext(synthetic, provider_calls={self.calls}, credentials=redacted)"


@pytest.fixture
def audit_context(monkeypatch, tmp_path):
    db = str(tmp_path / "audit.db")
    monkeypatch.setattr(workspace_store, "db_path", db)
    monkeypatch.setattr(auth_service, "db_path", db)
    workspace_store._init_sqlite_db()
    auth_service._init_db()
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(storage_service, "base_dir", str(media))
    for name, value in {
        "ENVIRONMENT": "test",
        "DATABASE_BACKEND": "sqlite",
        "IMAGE_GENERATION_ENABLED": True,
        "IMAGE_CF_FREE_PLAN_VERIFIED": True,
        "CLOUDFLARE_ACCOUNT_ID": "c" * 32,
        "CLOUDFLARE_API_TOKEN": "synthetic-audit-no-network",
        "IMAGE_DAILY_NEURON_LIMIT": 1000,
        "IMAGE_USER_DAILY_ATTEMPTS": 5,
    }.items():
        monkeypatch.setattr(settings, name, value)
    out = io.BytesIO()
    Image.new("RGB", (64, 64), "navy").save(out, "PNG")
    context = AuditContext(
        client=TestClient(app),
        alice=private_account_headers("audit-alice"),
        bob=private_account_headers("audit-bob"),
        png=out.getvalue(),
        calls=0,
        media=media,
        db=db,
    )

    async def mocked_generate(*args, **kwargs):
        context.calls += 1
        return GeneratedImage(context.png, SCHNELL, "synthetic-audit")

    monkeypatch.setattr(CloudflareImageProvider, "generate", mocked_generate)
    return context


def generate(ctx, headers=None, **fields):
    return ctx.client.post(
        "/api/v1/image/generate",
        headers=headers or ctx.alice,
        json={"prompt": "a boat", **fields},
    )


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("POST", "/api/v1/image/generate", {"prompt": "a boat"}),
        ("POST", "/api/v1/image/upscale", {"image_url": "0" * 36}),
        ("GET", "/api/v1/image/gallery", None),
        ("GET", "/api/v1/image/unknown/content", None),
        ("DELETE", "/api/v1/image/unknown", None),
    ],
)
def test_unauthenticated_and_spoofed_identity_denied(
    audit_context, method, path, payload
):
    c = audit_context
    r = c.client.request(
        method, path, json=payload, headers={"X-Client-Id": "audit-alice"}
    )
    assert r.status_code == 401
    assert c.calls == 0


def test_cross_user_cannot_read_delete_or_resample(audit_context):
    c = audit_context
    r = generate(c).json()
    assert c.client.get(r["image_url"], headers=c.bob).status_code == 404
    assert (
        c.client.delete("/api/v1/image/" + r["image_id"], headers=c.bob).status_code
        == 404
    )
    assert (
        c.client.post(
            "/api/v1/image/upscale", headers=c.bob, json={"image_url": r["image_url"]}
        ).status_code
        == 400
    )
    assert c.client.get("/api/v1/image/gallery", headers=c.bob).json()["images"] == []


def test_prompt_limit_prevents_provider_call(audit_context):
    assert generate(audit_context, prompt="x" * 2001).status_code == 422
    assert audit_context.calls == 0


def test_oversized_unused_json_payload_is_rejected_before_provider(audit_context):
    r = generate(audit_context, unused_padding="x" * (256 * 1024))
    assert r.status_code in (413, 422), "Unbounded unused JSON field reached generation"
    assert audit_context.calls == 0


def test_production_with_nonlocal_storage_config_must_reach_provider(
    audit_context, monkeypatch
):
    # We check the pre-inference gate only. This does not claim real storage durability.
    c = audit_context
    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    r = generate(c, headers=c.alice | {"Origin": "https://www.jruva.in"})
    assert c.calls == 1, (
        f"Production gate stopped before provider: HTTP {r.status_code}"
    )


def test_missing_postgres_url_must_reject_application_startup(
    audit_context, monkeypatch
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DATABASE_BACKEND", "postgres")
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "SUPABASE_DATABASE_URL", "")
    with pytest.raises(RuntimeError, match="DATABASE|database"):
        create_application()


def test_unknown_storage_backend_must_not_silently_fall_back(
    audit_context, monkeypatch
):
    import app.services.storage as module

    monkeypatch.setattr(module, "_active_backend", None)
    monkeypatch.setenv("STORAGE_BACKEND", "supabsae")
    with pytest.raises((ValueError, RuntimeError)):
        module.get_storage_backend()


def test_gcs_backend_can_be_constructed_or_reports_configuration_failure():
    from app.services.storage.s3_gcs import CloudObjectStorageBackend

    # Abstract-method TypeError is a broken factory contract, not a configuration denial.
    try:
        backend = CloudObjectStorageBackend("synthetic-bucket")
    except RuntimeError:
        return
    assert backend is not None


def test_duplicate_request_id_must_not_generate_twice(audit_context):
    c = audit_context
    headers = c.alice | {"Idempotency-Key": "synthetic-retry-001"}
    first = generate(c, headers=headers)
    second = generate(c, headers=headers)
    assert first.status_code == second.status_code == 200
    assert c.calls == 1, "Identical retried request consumed two provider attempts"
    assert first.json()["image_id"] == second.json()["image_id"]


def test_resampling_cannot_create_unbounded_records(audit_context, monkeypatch):
    c = audit_context
    monkeypatch.setattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 1)
    original = generate(c).json()
    responses = [
        c.client.post(
            "/api/v1/image/upscale",
            headers=c.alice,
            json={"image_url": original["image_url"], "scale": 2},
        )
        for _ in range(7)
    ]
    # A separate media-write budget is acceptable; seven successes with a limit of one is not.
    assert any(r.status_code == 429 for r in responses), (
        "Resampling bypassed daily write budget"
    )


@pytest.mark.asyncio
async def test_generation_has_small_inflight_bound(audit_context, monkeypatch):
    c = audit_context
    monkeypatch.setattr(settings, "IMAGE_DAILY_NEURON_LIMIT", 10000)
    monkeypatch.setattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 100)
    active = peak = 0

    async def held(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.12)
        active -= 1
        return GeneratedImage(c.png, SCHNELL, "synthetic-audit")

    monkeypatch.setattr(CloudflareImageProvider, "generate", held)
    await asyncio.gather(
        *(genuine_service.generate("a boat", owner="audit-alice") for _ in range(8))
    )
    assert peak <= 2, (
        f"Provider inflight peak {peak}; no small 512-MB-safe image admission bound"
    )


def test_concurrent_attempts_stop_before_provider_at_owner_limit(audit_context):
    c = audit_context
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda i: generate(c, headers=c.alice | {"Idempotency-Key": f"distinct-attempt-{i}"}).status_code, range(10)))
    assert statuses.count(200) == c.calls == 5
    assert statuses.count(429) == 5


def test_failure_recovery_does_not_refund_unknown_cost(audit_context, monkeypatch):
    c = audit_context

    async def timeout(*args, **kwargs):
        raise ImageProviderError("provider_timeout", "Please retry.", retryable=True)

    monkeypatch.setattr(CloudflareImageProvider, "generate", timeout)
    assert generate(c).status_code == 503

    async def good(*args, **kwargs):
        return GeneratedImage(c.png, SCHNELL)

    monkeypatch.setattr(CloudflareImageProvider, "generate", good)
    assert generate(c).status_code == 503
    with workspace_store._get_sqlite_connection() as conn:
        assert sorted(
            r[0] for r in conn.execute("SELECT used FROM image_generation_budgets")
        ) == [1, 120]


def test_metadata_failure_removes_new_blob(audit_context, monkeypatch):
    c = audit_context

    def failed_record(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DATABASE_DETAIL")

    monkeypatch.setattr(workspace_store, "record_image", failed_record)
    r = generate(c)
    assert r.status_code == 503
    assert "SYNTHETIC_PRIVATE_DATABASE_DETAIL" not in r.text
    assert not list(c.media.iterdir())
    assert workspace_store.list_images("audit-alice") == []


def test_failed_blob_delete_keeps_metadata_for_retry(audit_context, monkeypatch):
    c = audit_context
    original = generate(c).json()

    def refused(*args, **kwargs):
        raise OSError("SYNTHETIC_PRIVATE_STORAGE_DETAIL")

    monkeypatch.setattr(storage_service, "delete_verified", refused)
    r = c.client.delete("/api/v1/image/" + original["image_id"], headers=c.alice)
    assert r.status_code == 503
    assert "SYNTHETIC_PRIVATE_STORAGE_DETAIL" not in r.text
    assert workspace_store.get_image(original["image_id"], "audit-alice")
    assert c.client.get(original["image_url"], headers=c.alice).status_code == 200


def test_owner_denial_rolls_back_account_counter_and_restart_preserves_it(
    audit_context, monkeypatch
):
    c = audit_context
    monkeypatch.setattr(settings, "IMAGE_USER_DAILY_ATTEMPTS", 1)
    assert generate(c).status_code == 200
    assert generate(c, headers=c.alice | {"Idempotency-Key": "new-quota-denied-attempt"}).status_code == 429
    reopened = WorkspaceStore(c.db)
    monkeypatch.setattr(budget, "workspace_store", reopened)
    with pytest.raises(ImageProviderError):
        budget.reserve_image_budget("c" * 32, "audit-alice", 120, 1000, 1)
    with reopened._get_sqlite_connection() as conn:
        assert sorted(
            r[0] for r in conn.execute("SELECT used FROM image_generation_budgets")
        ) == [1, 120]


def test_multiple_processes_share_one_atomic_budget(audit_context):
    c = audit_context
    backend_path = str(Path(__file__).resolve().parents[1])
    code = """
import socket,sys
def blocked(*a,**k): raise OSError("network disabled")
socket.socket.connect=blocked
sys.path.insert(0,sys.argv[1])
from app.services.image import budget
from app.services.workspace_store import WorkspaceStore
from app.services.image.providers import ImageProviderError
budget.workspace_store=WorkspaceStore(sys.argv[2])
n=0
for i in range(4):
 try:
  budget.reserve_image_budget("process-account",sys.argv[3],120,240,100)
  n+=1
 except ImageProviderError: pass
print("AUDIT_ACCEPTED="+str(n))
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-B", "-c", code, backend_path, c.db, f"process-{i}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        for i in range(4)
    ]
    accepted = 0
    for p in processes:
        out, _err = p.communicate(timeout=30)
        assert p.returncode == 0, "Synthetic child failed (no credential output)"
        accepted += int(out.split("AUDIT_ACCEPTED=")[-1].strip())
    assert accepted == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body",
    [
        (200, {"success": True, "result": {"image": "not-base64"}}),
        (
            200,
            {
                "success": True,
                "result": {"image": base64.b64encode(b"fake png").decode()},
            },
        ),
        (302, {}),
        (500, {"errors": [{"code": 1, "message": "SYNTHETIC_SECRET"}]}),
    ],
)
async def test_bad_provider_response_is_sanitized(status, body):
    cfg = Settings(
        _env_file=None,
        IMAGE_GENERATION_ENABLED=True,
        IMAGE_CF_FREE_PLAN_VERIFIED=True,
        CLOUDFLARE_ACCOUNT_ID="d" * 32,
        CLOUDFLARE_API_TOKEN="synthetic",
    )

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"success": True, "result": []})
        return httpx.Response(
            status, json=body, headers={"location": "http://169.254.169.254/"}
        )

    with pytest.raises(ImageProviderError) as error:
        await CloudflareImageProvider(
            cfg, transport=httpx.MockTransport(respond)
        ).generate("a boat", 1, SCHNELL)
    assert "SYNTHETIC_SECRET" not in str(error.value)


def test_sync_database_bridge_reuses_single_loop_for_pooled_engine():
    from app.core.database import run_sync_db

    async def loop_identity():
        return asyncio.get_running_loop()

    first = run_sync_db(loop_identity())
    second = run_sync_db(loop_identity())
    assert first is second, (
        "Pooled AsyncEngine would be reused across different closed loops"
    )
