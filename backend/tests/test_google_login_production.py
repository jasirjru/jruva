"""Offline Google login regressions against real persistence and signed test tokens.

The ORM cases also run against a disposable loopback PostgreSQL database when
JRUVA_AUTH_TEST_POSTGRES_URL is supplied. Remote targets are always rejected.
"""

import os
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1.endpoints import auth
from app.core import database
from app.core.config import Settings
from app.models import Message, Session, User
from app.services.auth_service import auth_service

pytest_plugins = ["tests.google_identity_fixtures"]


@pytest.fixture(params=["sqlite", "orm"])
def login_store(request, google_token_factory, monkeypatch, tmp_path):
    backend = request.param
    engine = None
    if backend == "orm":
        url = os.environ.get("JRUVA_AUTH_TEST_POSTGRES_URL")
        if url:
            parsed = make_url(url)
            assert parsed.host == "127.0.0.1" and parsed.database.startswith("jruva_auth_test_")
            assert parsed.port and parsed.port != 5432
        else:
            url = f"sqlite+aiosqlite:///{tmp_path / 'orm.db'}"
        engine = create_async_engine(url)
        async def setup():
            async with engine.begin() as conn:
                await conn.run_sync(lambda c: User.metadata.create_all(c, tables=[User.__table__, Session.__table__, Message.__table__]))
        database.run_sync_db(setup)
        monkeypatch.setattr(database, "_ASYNC_ENGINE", engine)
        monkeypatch.setattr(database, "_ASYNC_SESSION_MAKER", async_sessionmaker(engine, expire_on_commit=False))
        monkeypatch.setattr(database, "_DB_SESSION_SLOTS", None)
        # Exercise the PostgreSQL public contract, worker loop and transaction wrapper.
        monkeypatch.setattr(database, "settings", Settings(_env_file=None, DATABASE_BACKEND="postgres", DATABASE_URL=url))
    else:
        with auth_service._get_connection() as conn:
            conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, user_id TEXT, title TEXT)")
            conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)")

    def execute(sql, params=None):
        if backend == "orm":
            async def run():
                async with database.get_async_session() as session:
                    result = await session.execute(text(sql), params or {})
                    return [dict(row) for row in result.mappings()] if result.returns_rows else []
            return database.run_sync_db(run)
        with auth_service._get_connection() as conn:
            cursor = conn.execute(sql, params or {})
            return [dict(row) for row in cursor.fetchall()]

    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")
    with TestClient(app) as client:
        yield client, execute, engine
    if engine:
        async def teardown():
            async with engine.begin() as conn:
                await conn.run_sync(lambda c: User.metadata.drop_all(c, tables=[Message.__table__, Session.__table__, User.__table__]))
            await engine.dispose()
        database.run_sync_db(teardown)


def seed_account(execute, subject="synthetic-google-sub", pending=False):
    execute("""INSERT INTO users
        (id, username, email, full_name, role, password_hash, salt, google_sub, deletion_pending, created_at, last_login)
        VALUES ('legacy-id', 'legacy', 'synthetic-oidc@example.invalid', 'Saved Profile', 'student',
                'synthetic-hash', 'synthetic-salt', :subject, :pending, '2000-01-01', '2000-01-01')""",
            {"subject": subject, "pending": pending})
    execute("INSERT INTO sessions (session_id, user_id, title) VALUES ('history', 'legacy-id', 'Saved History')")
    execute("INSERT INTO messages (session_id, role, content) VALUES ('history', 'user', 'Synthetic historical message')")


def snapshot(execute):
    return [execute(f"SELECT * FROM {table}") for table in ("users", "sessions", "messages")]


def test_existing_google_user_keeps_identity_and_history(login_store, google_token_factory):
    client, execute, _ = login_store
    seed_account(execute)
    history = snapshot(execute)[1:]
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
    assert response.status_code == 200, response.text
    assert response.json()["user"]["id"] == "legacy-id"
    assert response.json()["user"]["full_name"] == "Saved Profile"
    assert auth_service.verify_jwt_token(response.json()["access_token"])["sub"] == "legacy-id"
    assert client.get("/api/v1/auth/me").json()["id"] == "legacy-id"
    assert snapshot(execute)[1:] == history
    last_login = execute("SELECT last_login FROM users")[0]["last_login"]
    assert last_login is not None and str(last_login).startswith(str(datetime.now(timezone.utc).year))


@pytest.mark.parametrize("subject", [None, "another-bound-subject"])
def test_email_match_requires_account_recovery_without_mutating_history(login_store, google_token_factory, subject):
    client, execute, _ = login_store
    seed_account(execute, subject=subject)
    before = snapshot(execute)
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
    assert response.status_code == 403
    assert "account recovery" in response.json()["detail"].lower()
    assert "google credentials" not in response.json()["detail"].lower()
    assert "set-cookie" not in response.headers and "access_token" not in response.json()
    assert snapshot(execute) == before


def test_pending_erasure_is_denied_without_a_session(login_store, google_token_factory):
    client, execute, _ = login_store
    seed_account(execute, pending=True)
    before = snapshot(execute)
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
    assert response.status_code == 403
    assert "deletion" in response.json()["detail"].lower()
    assert "set-cookie" not in response.headers
    assert snapshot(execute) == before


def test_new_account_returns_persisted_profile_and_secure_cookie(login_store, google_token_factory, monkeypatch):
    client, execute, _ = login_store
    monkeypatch.setattr(auth.settings, "ENVIRONMENT", "production")
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
    assert response.status_code == 200, response.text
    user = response.json()["user"]
    assert user["created_at"]
    assert user["role"] == "student"
    assert len(execute("SELECT id FROM users")) == 1
    assert auth_service.verify_jwt_token(response.json()["access_token"])["sub"] == user["id"]
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "secure" in cookie and "samesite=none" in cookie
    assert "synthetic-hash" not in response.text and "salt" not in user


@pytest.mark.parametrize("error", [OperationalError("SELECT private", {}, RuntimeError("synthetic-sensitive-database-detail")),
                                 sqlite3.OperationalError("synthetic-sensitive-database-detail"),
                                 TimeoutError("synthetic-sensitive-database-detail"),
                                 ConnectionRefusedError("synthetic-sensitive-database-detail")])
def test_database_failures_are_retryable_and_redacted(login_store, google_token_factory, monkeypatch, caplog, error):
    client, execute, _ = login_store
    seed_account(execute)
    before = snapshot(execute)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(auth_service, "authenticate_or_register_google_user", fail)
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
    assert response.status_code == 503
    assert response.headers["retry-after"]
    assert "google credentials" not in response.json()["detail"].lower()
    assert "set-cookie" not in response.headers
    assert "synthetic-sensitive" not in response.text + caplog.text
    assert snapshot(execute) == before


def test_orm_commit_failure_rolls_back_new_account(login_store, google_token_factory):
    client, execute, engine = login_store
    if engine is None:
        pytest.skip("ORM transaction failure is covered only by the ORM implementation")
    seed_account(execute)
    before = snapshot(execute)
    # Flush empties session.new, so intercept the actual database COMMIT instead.
    def fail_connection_commit(conn):
        raise OperationalError("COMMIT", {}, RuntimeError("synthetic-sensitive-commit-detail"))
    event.listen(engine.sync_engine, "commit", fail_connection_commit)
    try:
        response = client.post("/api/v1/auth/google", json={"credential": google_token_factory(sub="new-sub", email="new@example.invalid")})
    finally:
        event.remove(engine.sync_engine, "commit", fail_connection_commit)
    assert response.status_code == 503
    assert "set-cookie" not in response.headers
    assert snapshot(execute) == before


@pytest.mark.parametrize("stage", ["jwt", "response", "verifier"])
def test_internal_failures_are_server_errors_without_credentials_in_logs(login_store, google_token_factory, monkeypatch, caplog, stage):
    client, _, _ = login_store
    def fail(*args, **kwargs):
        raise ValueError("synthetic-sensitive-internal-detail")
    if stage == "jwt":
        monkeypatch.setattr(auth_service, "create_jwt_token", fail)
    elif stage == "response":
        monkeypatch.setattr(auth, "AuthResponse", fail)
    else:
        def verifier_bug(*args, **kwargs):
            raise RuntimeError("synthetic-sensitive-internal-detail")
        monkeypatch.setattr(auth, "verify_google_identity", verifier_bug)
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
    assert response.status_code == 500
    assert "google credentials" not in response.json()["detail"].lower()
    assert "set-cookie" not in response.headers
    assert "synthetic-sensitive" not in response.text + caplog.text
