"""Failure Mode and Split-Brain Prevention Tests for Database Layer.

CRITICAL REQUIREMENTS (Phase 10):
1. In production (DATABASE_BACKEND=postgres): NEVER silently fall back to local SQLite.
2. If DATABASE_BACKEND=postgres and PostgreSQL is unavailable, fail clearly with controlled errors.
3. Transaction rollback on failure must leave data intact.
4. Startup with missing URL when DATABASE_BACKEND=postgres in production must fail immediately.
"""

from unittest.mock import patch
import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.core.database import (
    check_database_health,
    create_engine_instance,
    get_async_engine,
    is_postgres_configured,
    reset_engine_for_testing,
)


@pytest.mark.asyncio
async def test_production_mode_strictly_rejects_missing_postgres_url():
    """Verify that setting DATABASE_BACKEND=postgres with missing URL in production raises RuntimeError."""
    fake_settings = Settings(
        DATABASE_BACKEND="postgres",
        DATABASE_URL="",
        SUPABASE_DATABASE_URL="",
        ENVIRONMENT="production",
    )
    with patch("app.core.database.settings", fake_settings):
        reset_engine_for_testing()
        with pytest.raises(RuntimeError) as exc_info:
            create_engine_instance()
        assert "Refusing silent fallback to ephemeral SQLite to prevent split-brain data loss" in str(exc_info.value)
        reset_engine_for_testing()


@pytest.mark.asyncio
async def test_postgres_unreachable_reports_unhealthy_without_fallback():
    """When PostgreSQL host is unreachable, health check reports unhealthy rather than falling back."""
    unreachable_settings = Settings(
        DATABASE_BACKEND="postgres",
        DATABASE_URL="postgresql+asyncpg://dummy_user:dummy_pass@127.0.0.1:54329/nonexistent_db",
        DB_POOL_TIMEOUT=1.0,
    )
    with patch("app.core.database.settings", unreachable_settings):
        reset_engine_for_testing()
        report = await check_database_health()
        assert report["status"] == "unhealthy"
        assert report["connected"] is False
        assert report["backend"] == "postgres"
        # Confirm that database remains configured as postgres (no silent switch to sqlite)
        assert is_postgres_configured() is True
        reset_engine_for_testing()


@pytest.mark.asyncio
async def test_database_url_normalization_handles_various_driver_prefixes():
    """Verify standard postgresql:// URLs are safely adapted to asyncpg for async engine."""
    s1 = Settings(DATABASE_URL="postgresql://user:pass@host:5432/db")
    assert s1.effective_database_url.startswith("postgresql+asyncpg://")

    s2 = Settings(DATABASE_URL="postgres://user:pass@host:5432/db")
    assert s2.effective_database_url.startswith("postgresql+asyncpg://")

    s3 = Settings(DATABASE_URL="sqlite:///chatbot.db")
    assert s3.effective_database_url.startswith("sqlite+aiosqlite://")


@pytest.mark.asyncio
async def test_transaction_rollback_guarantee_on_simulated_error():
    """Verify atomic transaction rollback when an operation within the transaction fails."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.core.database import Base
    from app.models import Project

    # In-memory test engine for transaction semantics
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSession = async_sessionmaker(test_engine, expire_on_commit=False)

    # 1. Insert a baseline project
    async with TestSession() as session:
        async with session.begin():
            session.add(Project(id="p1", name="Permanent Project", user_id="u1"))

    # 2. Attempt a transaction that inserts one item and then crashes
    with pytest.raises(ValueError):
        async with TestSession() as session:
            async with session.begin():
                session.add(Project(id="p2", name="Should Be Rolled Back", user_id="u1"))
                raise ValueError("Simulated unexpected failure mid-transaction")

    # 3. Verify p2 was rolled back and p1 still exists
    async with TestSession() as session:
        stmt = text("SELECT id, name FROM projects WHERE id = 'p2'")
        res = await session.execute(stmt)
        p2_row = res.fetchone()
        assert p2_row is None, "Failed transaction was not rolled back!"

        stmt_p1 = text("SELECT id, name FROM projects WHERE id = 'p1'")
        res_p1 = await session.execute(stmt_p1)
        p1_row = res_p1.fetchone()
        assert p1_row is not None, "Baseline record was lost!"

    await test_engine.dispose()


def test_validate_database_target_rejects_staging_in_production():
    """Verify production environment strictly rejects targeting staging database."""
    from app.core.database import validate_database_target, STAGING_PROJECT_REF

    # Staging ref in URL
    staging_url = f"postgresql+asyncpg://postgres.{STAGING_PROJECT_REF}:secret@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"
    with pytest.raises(RuntimeError) as exc_info:
        validate_database_target(staging_url, environment="production")
    assert "Production environment cannot target staging project ref" in str(exc_info.value)

    # Localhost in production
    local_url = "postgresql+asyncpg://postgres:secret@127.0.0.1:5432/postgres"
    with pytest.raises(RuntimeError) as exc_info2:
        validate_database_target(local_url, environment="production")
    assert "Production environment cannot target a local loopback database" in str(exc_info2.value)

    # SQLite in production
    sqlite_url = "sqlite+aiosqlite:///chatbot.db"
    with pytest.raises(RuntimeError) as exc_info3:
        validate_database_target(sqlite_url, environment="production")
    assert "Production environment cannot use SQLite" in str(exc_info3.value)


def test_validate_database_target_rejects_production_in_non_production():
    """Verify test, dev, and staging environments strictly reject targeting production database."""
    from app.core.database import validate_database_target, PRODUCTION_PROJECT_REF

    prod_url = f"postgresql+asyncpg://postgres.{PRODUCTION_PROJECT_REF}:secret@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"

    for env in ("test", "development", "staging"):
        with pytest.raises(RuntimeError) as exc_info:
            validate_database_target(prod_url, environment=env)
        assert "cannot target production project ref" in str(exc_info.value)


def test_mask_database_url_protects_credentials():
    """Verify mask_database_url scrubs password and masks username in connection strings."""
    from app.core.database import mask_database_url

    url = "postgresql+asyncpg://postgres.stagingprojectref000:ExamplePass123!@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?ssl=require"
    masked = mask_database_url(url)
    assert "ExamplePass123!" not in masked
    assert "******" in masked
    assert "aws-0-ap-south-1.pooler.supabase.com" in masked
    assert mask_database_url("") == "<EMPTY>"

