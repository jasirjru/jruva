"""Actual SQLite migration sequence and durable schema/rollback contracts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.models import Base
from sqlalchemy import create_engine, inspect, text


def load(path):
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_sqlite_upgrade_all_release_migrations():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            for path in sorted((Path(__file__).resolve().parents[1] / "alembic/versions").glob("00[1-7]_*.py")):
                load(path).upgrade()
        assert "image_storage_reservations" in inspect(conn).get_table_names()
        assert "operation_id" in {column["name"] for column in inspect(conn).get_columns("image_idempotency")}
    engine.dispose()


def test_ledger_is_part_of_canonical_metadata():
    assert "image_storage_reservations" in Base.metadata.tables
    assert "image_idempotency" in Base.metadata.tables
    assert "operation_id" in Base.metadata.tables["image_idempotency"].columns


@pytest.mark.parametrize("legacy_status", ["pending", "failed", "stale_expired"])
def test_007_quarantines_uncompleted_legacy_operations(legacy_status):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        paths = sorted((Path(__file__).resolve().parents[1] / "alembic/versions").glob("00[1-7]_*.py"))
        with Operations.context(MigrationContext.configure(conn)):
            for path in paths[:-1]:
                load(path).upgrade()
            conn.execute(
                text(
                    "INSERT INTO image_idempotency(owner_id,idempotency_key,request_hash,status,created_at) "
                    "VALUES('owner','legacy','hash',:status,CURRENT_TIMESTAMP)"
                ),
                {"status": legacy_status},
            )
            load(paths[-1]).upgrade()
        row = conn.execute(text("SELECT status,external_state,operation_id FROM image_idempotency")).one()
        assert tuple(row[:2]) == ("indeterminate", "indeterminate")
        assert row[2]
    engine.dispose()


def test_007_refuses_untracked_legacy_storage_reservations():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        paths = sorted((Path(__file__).resolve().parents[1] / "alembic/versions").glob("00[1-7]_*.py"))
        with Operations.context(MigrationContext.configure(conn)):
            for path in paths[:-1]:
                load(path).upgrade()
            conn.execute(
                text("INSERT INTO image_storage_quotas(user_id,image_count,total_bytes) VALUES('owner',1,100)")
            )
            with pytest.raises(RuntimeError, match="legacy storage"):
                load(paths[-1]).upgrade()
        assert "image_storage_reservations" not in inspect(conn).get_table_names()
    engine.dispose()


def test_007_downgrade_refuses_unresolved_receipts():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        paths = sorted((Path(__file__).resolve().parents[1] / "alembic/versions").glob("00[1-7]_*.py"))
        with Operations.context(MigrationContext.configure(conn)):
            for path in paths:
                load(path).upgrade()
            conn.execute(
                text(
                    "INSERT INTO image_storage_reservations(reservation_id,user_id,bytes,state,expires_at) "
                    "VALUES('receipt','owner',100,'writing',CURRENT_TIMESTAMP)"
                )
            )
            with pytest.raises(RuntimeError, match="outstanding storage"):
                load(paths[-1]).downgrade()
        assert "image_storage_reservations" in inspect(conn).get_table_names()
    engine.dispose()
