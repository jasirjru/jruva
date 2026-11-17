"""Transaction boundary regressions using explicit failure injection, no remote DB."""

import sqlite3
from contextlib import asynccontextmanager
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.services import workspace_store as workspace_module
from app.services.image import budget, storage_quota
from sqlalchemy import create_engine, inspect, text


@pytest.mark.asyncio
async def test_postgres_replay_stays_in_claim_session(monkeypatch):
    class Session:
        def begin(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement, parameters=None):
            if "FROM image_idempotency" in str(statement):
                return SimpleNamespace(fetchone=lambda: ("hash", "completed", '{"image_id":"image"}', None, "first", None, 1))
            assert "image_records" in str(statement)
            assert parameters == {"id": "image", "owner": "owner"}
            return SimpleNamespace(scalar_one_or_none=lambda: "image")

    @asynccontextmanager
    async def session():
        yield Session()

    def reject_sync(*args):
        raise AssertionError("Async claim reentered synchronous DB bridge")

    monkeypatch.setattr(budget, "get_async_session", session)
    monkeypatch.setattr(budget.workspace_store, "get_image", reject_sync)
    cached, _, _ = await budget._claim_idempotency_lease_postgres("owner", "key", "hash")
    assert cached == {"image_id": "image"}


@pytest.mark.asyncio
async def test_postgres_image_delete_cannot_commit_quota_separately(monkeypatch):
    separate_commits = []
    updates = []
    locked_selects = []

    class OuterSession:
        def begin(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise RuntimeError("Injected image deletion commit failure")

        async def execute(self, statement, parameters=None):
            sql = str(statement)
            if sql.startswith("SELECT"):
                locked_selects.append("FOR UPDATE" in sql)
                return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(user_id="owner", bytes=10))
            if sql.startswith("UPDATE image_storage_quotas"):
                updates.append(sql)

        async def delete(self, record):
            pass

    class SeparateSession(OuterSession):
        async def __aexit__(self, *args):
            separate_commits.append(True)

        async def execute(self, *args):
            pass

    @asynccontextmanager
    async def outer():
        yield OuterSession()

    @asynccontextmanager
    async def separate():
        yield SeparateSession()

    monkeypatch.setattr(workspace_module, "get_async_session", outer)
    monkeypatch.setattr(storage_quota, "get_async_session", separate)
    with pytest.raises(RuntimeError, match="commit failure"):
        await workspace_module.workspace_store.async_delete_image("image")
    assert not separate_commits, "Quota release committed although image deletion rolled back"
    assert len(updates) == 2
    assert all(locked_selects), "Concurrent deletions must serialize on the image row"


def migration_006():
    path = Path(__file__).resolve().parents[1] / "alembic/versions/006_distributed_idempotency.py"
    spec = spec_from_file_location("audit_migration_006", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_006_preserves_legacy_rows_and_downgrades_locally():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE image_idempotency (owner_id TEXT, idempotency_key TEXT, "
                          "status TEXT, created_at TIMESTAMP, PRIMARY KEY(owner_id, idempotency_key))"))
        conn.execute(text("INSERT INTO image_idempotency VALUES ('owner', 'key', 'pending', '2026-01-01')"))
        with Operations.context(MigrationContext.configure(conn)):
            migration_006().upgrade()
            row = conn.execute(text("SELECT worker_id, lease_expires_at, fencing_token FROM image_idempotency")).one()
            assert tuple(row) == (None, None, 1)
            migration_006().downgrade()
        assert "worker_id" not in {column["name"] for column in inspect(conn).get_columns("image_idempotency")}
        assert conn.execute(text("SELECT status FROM image_idempotency")).scalar() == "pending"
    engine.dispose()


def test_migration_006_compiles_postgres_ddl_offline():
    output = StringIO()
    context = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output})
    with Operations.context(context):
        migration_006().upgrade()
    sql = output.getvalue()
    assert "TIMESTAMP WITH TIME ZONE" in sql
    assert "BIGINT DEFAULT '1' NOT NULL" in sql


def test_sqlite_runtime_parity_upgrades_populated_legacy_table():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE image_idempotency (owner_id TEXT, idempotency_key TEXT, request_hash TEXT, "
                     "status TEXT, response_json TEXT, created_at TIMESTAMP)")
        conn.execute("INSERT INTO image_idempotency VALUES ('owner','key','hash','pending',NULL,'2026-01-01')")
        budget._ensure_sqlite_idempotency_schema(conn)
        assert conn.execute("SELECT fencing_token, updated_at FROM image_idempotency").fetchone()[0] == 1
