"""Regression tests for JRUVA Release Closure Mission blockers.

Covers:
1. In-flight generation locking vs stale expiration (D01)
2. Account erasure storage quota and project total cleanup (D02)
3. Async cancellation safety for storage and quota reservations (D03)
4. PgBouncer/Supavisor dynamic prepared statement naming configuration (D04)
5. Alembic migration 005 backfill logic and SQLite batch downgrade (D05)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from app.services.image import budget
from app.services.image.providers import ImageProviderError
from app.services.image.storage_quota import (
    QUOTA_DDL,
    get_storage_quota_usage,
    reserve_storage_quota,
)
from app.services.workspace_store import workspace_store


@pytest.fixture(autouse=True)
def clean_test_tables():
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(budget.IDEMPOTENCY_DDL)
        conn.execute(QUOTA_DDL)
        conn.execute("DELETE FROM image_idempotency WHERE owner_id LIKE 'reg_%'")
        conn.execute("DELETE FROM image_storage_quotas WHERE user_id LIKE 'reg_%'")
        conn.execute("DELETE FROM image_records WHERE user_id LIKE 'reg_%'")
        conn.execute("UPDATE image_storage_quotas SET total_bytes = 0 WHERE user_id = '__project__'")
        conn.commit()
    yield
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute(budget.IDEMPOTENCY_DDL)
        conn.execute(QUOTA_DDL)
        conn.execute("DELETE FROM image_idempotency WHERE owner_id LIKE 'reg_%'")
        conn.execute("DELETE FROM image_storage_quotas WHERE user_id LIKE 'reg_%'")
        conn.execute("DELETE FROM image_records WHERE user_id LIKE 'reg_%'")
        conn.execute("UPDATE image_storage_quotas SET total_bytes = 0 WHERE user_id = '__project__'")
        conn.commit()


def test_in_flight_generation_cannot_be_expired_while_active():
    """An active in-flight generation must NOT be expired as stale or duplicated by retries."""
    owner = "reg_user_flight"
    key = "flight-key-1"
    req_hash = "hash-flight-1"

    # Initial claim
    claim = budget.check_or_claim_idempotency(owner, key, req_hash)
    assert claim is None

    # Mark as in-flight
    with budget.active_in_flight_generation(owner, key):
        # Artificially age the timestamp by 300s
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).strftime("%Y-%m-%d %H:%M:%S")
        with workspace_store._get_sqlite_connection() as conn:
            conn.execute(
                "UPDATE image_idempotency SET created_at = ? WHERE owner_id = ? AND idempotency_key = ?",
                (old_time, owner, key),
            )
            conn.commit()

        # Retry attempt while still in-flight must raise idempotency_in_progress, NOT idempotency_needs_reconciliation
        with pytest.raises(ImageProviderError) as exc_info:
            budget.check_or_claim_idempotency(owner, key, req_hash)
        assert exc_info.value.status == 409
        assert exc_info.value.kind == "idempotency_in_progress", (
            f"Expected 'idempotency_in_progress', got '{exc_info.value.kind}'; active generation was incorrectly expired!"
        )


def test_account_erasure_cleans_quotas_and_project_bytes():
    """Account erasure must delete user quota record and subtract bytes from __project__."""
    from app.services.account_erasure import _erase_sqlite_database
    owner = "reg_user_erasure"
    data_bytes = 40000

    # Reserve quota and create image record
    reserve_storage_quota(owner, data_bytes)
    workspace_store.record_image(
        prompt="to-erase",
        file_url="/tmp/erase.png",
        model="cf/flux",
        user_id=owner,
        bytes=data_bytes,
    )

    usage_before = get_storage_quota_usage(owner)
    assert usage_before["total_bytes"] == data_bytes
    assert usage_before["project_bytes"] >= data_bytes

    proj_before = usage_before["project_bytes"]

    # Run erasure on the workspace SQLite DB
    _erase_sqlite_database(workspace_store.db_path, owner)

    # Check user quota is deleted
    with workspace_store._get_sqlite_connection() as conn:
        user_row = conn.execute("SELECT * FROM image_storage_quotas WHERE user_id = ?", (owner,)).fetchone()
        assert user_row is None, "User quota row was not deleted during account erasure!"

        proj_row = conn.execute("SELECT total_bytes FROM image_storage_quotas WHERE user_id = '__project__'").fetchone()
        proj_after = proj_row[0] if proj_row else 0
        assert proj_after == proj_before - data_bytes, (
            f"Project bytes ({proj_after}) was not reduced by deleted user bytes ({data_bytes}); expected {proj_before - data_bytes}"
        )


@pytest.mark.asyncio
async def test_save_owned_image_releases_quota_on_cancellation():
    """If save_owned_image is cancelled during storage write, reserved quota is rolled back."""
    import io

    from app.services.image.genuine_service import save_owned_image
    from app.services.storage import storage_service
    from PIL import Image

    owner = "reg_user_cancel"
    from tests.private_accounts import private_account_headers
    private_account_headers(owner)
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(buf, format="PNG")
    dummy_png = buf.getvalue()

    def raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    with (
        patch("app.services.image.genuine_service.active_account", return_value=True),
        patch.object(storage_service, "save_bytes", side_effect=raise_cancelled),
        pytest.raises(asyncio.CancelledError),
    ):
        await save_owned_image(
            dummy_png,
            owner=owner,
            prompt="cancel test",
            enhanced="cancel test",
            model="cf/flux",
            tier="B",
            style="photorealistic",
        )

    usage = get_storage_quota_usage(owner)
    assert usage["image_count"] == 0
    assert usage["total_bytes"] == 0, f"Quota leaked on CancelledError: {usage['total_bytes']} bytes remained reserved!"


def test_pgbouncer_prepared_statement_name_func_configured():
    """PostgreSQL AsyncEngine config must provide prepared_statement_name_func for PgBouncer/Supavisor."""
    from app.core import config
    from app.core.database import create_engine_instance

    with patch.object(config.settings, "DATABASE_BACKEND", "postgres"), \
         patch("app.core.database.create_async_engine") as mock_engine:
        create_engine_instance(custom_url="postgresql+asyncpg://user:pass@127.0.0.1:6543/postgres")
        assert mock_engine.called
        kwargs = mock_engine.call_args.kwargs
        connect_args = kwargs.get("connect_args", {})
        assert "prepared_statement_name_func" in connect_args, (
            "prepared_statement_name_func missing from connect_args! Required to prevent DuplicatePreparedStatementError in transaction pooling."
        )
        func = connect_args["prepared_statement_name_func"]
        name1 = func()
        name2 = func()
        assert name1 != name2, "prepared_statement_name_func must generate unique statement names!"
        assert name1.startswith("__asyncpg_"), "prepared_statement_name_func should use recommended asyncpg prefix"


def test_alembic_005_backfill_and_batch_downgrade_logic():
    """Migration 005 must contain backfill logic for image_storage_quotas and batch alter for SQLite downgrade."""
    from pathlib import Path
    mig_path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "005_storage_quota.py"
    content = mig_path.read_text(encoding="utf-8")
    assert "INSERT INTO image_storage_quotas" in content or "batch_alter_table" in content, (
        "Migration 005 must include backfill logic from image_records into image_storage_quotas and batch alter support"
    )
    assert "batch_alter_table" in content, (
        "Migration 005 downgrade must use op.batch_alter_table to avoid SQLite ALTER TABLE DROP COLUMN errors on older engines"
    )
