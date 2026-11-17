"""Real PostgreSQL 006-to-007 rehearsals in disposable synthetic schemas."""

import os
from uuid import uuid4

import pytest

if not os.environ.get("CODEX_PRIVATE_PG_DB"):
    pytest.skip("Guarded private PostgreSQL runner required", allow_module_level=True)

from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.test_p0_ledger_migrations import load


@pytest.fixture
def legacy_connection():
    engine = create_engine(
        f"postgresql+psycopg://codex_audit@127.0.0.1:55439/{os.environ['CODEX_PRIVATE_PG_DB']}"
    )
    with engine.connect() as conn:
        transaction = conn.begin()
        schema = "assembly_" + uuid4().hex
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        paths = sorted((Path(__file__).resolve().parents[1] / "alembic/versions").glob("00[1-7]_*.py"))
        with Operations.context(MigrationContext.configure(conn)):
            for path in paths[:-1]:
                load(path).upgrade()
            yield conn, load(paths[-1])
        transaction.rollback()  # DDL, records and private schema vanish together.
    engine.dispose()


def test_006_to_007_quarantines_legacy_preserves_completed_and_supports_resume(legacy_connection):
    conn, migration = legacy_connection
    for state in ("pending", "failed", "stale_expired", "completed"):
        conn.execute(text(
            "INSERT INTO image_idempotency(owner_id,idempotency_key,request_hash,status,created_at) "
            "VALUES('synthetic',:state,'hash',:state,now())"
        ), {"state": state})
    checkpoint = conn.begin_nested()
    migration.upgrade()
    assert "image_storage_reservations" in inspect(conn).get_table_names()
    checkpoint.rollback()  # Simulated transactional interruption after DDL.
    assert "image_storage_reservations" not in inspect(conn).get_table_names()
    assert conn.execute(text("SELECT COUNT(*) FROM image_idempotency WHERE status='pending'")).scalar() == 1
    migration.upgrade()
    rows = conn.execute(text("SELECT idempotency_key,status,operation_id FROM image_idempotency")).all()
    assert all(row.status == "indeterminate" and row.operation_id for row in rows if row.idempotency_key != "completed")
    assert next(row for row in rows if row.idempotency_key == "completed").status == "completed"
    assert conn.execute(text("SELECT COUNT(*) FROM image_storage_reservations")).scalar() == 0
    with pytest.raises(RuntimeError, match="indeterminate"):
        migration.downgrade()
    # Repeated verification is read-only and stable; rerunning upgrade DDL is not supported.
    assert conn.execute(text("SELECT COUNT(*) FROM image_idempotency WHERE status='indeterminate'")).scalar() == 3


def test_007_refuses_legacy_quota_excess_before_ddl_and_can_retry_after_recovery(legacy_connection):
    conn, migration = legacy_connection
    conn.execute(text("UPDATE image_storage_quotas SET total_bytes=100 WHERE user_id='__project__'"))
    with pytest.raises(RuntimeError, match="legacy storage reservations"):
        migration.upgrade()
    assert "image_storage_reservations" not in inspect(conn).get_table_names()
    assert "operation_id" not in {c["name"] for c in inspect(conn).get_columns("image_idempotency")}
    conn.execute(text("UPDATE image_storage_quotas SET total_bytes=0 WHERE user_id='__project__'"))
    migration.upgrade()
    conn.execute(text(
        "INSERT INTO image_storage_reservations(reservation_id,user_id,bytes,state,expires_at) "
        "VALUES('synthetic','owner',100,'writing',now())"
    ))
    with pytest.raises(RuntimeError, match="outstanding storage"):
        migration.downgrade()
    conn.execute(text("UPDATE image_storage_reservations SET state='released'"))
    migration.downgrade()
    assert "image_storage_reservations" not in inspect(conn).get_table_names()
    migration.upgrade()
    assert "image_storage_reservations" in inspect(conn).get_table_names()
