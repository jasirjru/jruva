"""Independent cryptographic identity and private-data authorization regressions."""

import base64
import json

import pytest
from app.api.v1.endpoints import auth
from app.main import create_application
from app.services.auth_service import auth_service
from fastapi.testclient import TestClient
from google.auth.transport import requests as google_requests

from tests.test_integrated_release_adversarial import token, upload

pytest_plugins = ["tests.test_integrated_release_adversarial", "tests.google_identity_fixtures"]







def test_real_google_signature_and_claim_verification(google_token_factory):
    with TestClient(create_application()) as client:
        response = client.post("/api/v1/auth/google", json={"credential": google_token_factory()})
        assert response.status_code == 200, response.text
        assert response.json()["user"]["role"] == "student"


@pytest.mark.parametrize("changes", [
    {"aud": "other-app.apps.googleusercontent.com"}, {"iss": "https://attacker.invalid"},
    {"exp": 1}, {"iat": 4102444800}, {"email_verified": False}, {"sub": ""},
])
def test_google_invalid_claims_denied(google_token_factory, changes):
    with TestClient(create_application()) as client:
        assert client.post("/api/v1/auth/google", json={"credential": google_token_factory(**changes)}).status_code in (400, 401, 403)


@pytest.mark.parametrize("kind", ["tampered", "unsigned", "malformed"])
def test_google_invalid_signature_denied(google_token_factory, kind):
    credential = google_token_factory()
    if kind == "tampered":
        header, payload, signature = credential.split(".")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
        claims["email"] = "victim@example.invalid"
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        credential = f"{header}.{payload}.{signature}"
    elif kind == "unsigned":
        credential = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ2aWN0aW0ifQ."
    else:
        credential = "malformed"
    with TestClient(create_application()) as client:
        assert client.post("/api/v1/auth/google", json={"credential": credential}).status_code in (400, 401, 403)


def test_missing_audience_fails_closed(google_token_factory, monkeypatch):
    monkeypatch.setattr(auth.settings, "GOOGLE_CLIENT_ID", "")
    with TestClient(create_application()) as client:
        assert client.post("/api/v1/auth/google", json={"credential": google_token_factory()}).status_code == 503


def test_google_subject_is_stable_identity(google_token_factory):
    with TestClient(create_application()) as client:
        first = client.post("/api/v1/auth/google", json={"credential": google_token_factory(sub="stable-sub", email="first-sub@example.invalid")})
        second = client.post("/api/v1/auth/google", json={"credential": google_token_factory(sub="stable-sub", email="updated-sub@example.invalid")})
        assert first.status_code == second.status_code == 200
        assert first.json()["user"]["id"] == second.json()["user"]["id"]


def test_different_google_subject_cannot_take_over_email(google_token_factory):
    with TestClient(create_application()) as client:
        first = client.post("/api/v1/auth/google", json={"credential": google_token_factory(sub="original-sub", email="no-takeover@example.invalid")})
        assert first.status_code == 200
        assert client.post("/api/v1/auth/google", json={"credential": google_token_factory(sub="another-sub", email="no-takeover@example.invalid")}).status_code in (400, 401, 403)


def test_transient_attachment_retrieval_does_not_persist_private_vectors(sandbox):
    _, _, qclient = sandbox
    from app.services.rag.document_rag import document_rag_service
    chunks, _ = document_rag_service.retrieve(
        [{"name": "private_attachment.txt", "mime_type": "text/plain", "data": b"Synthetic temporary glacier station TR891 opens in June 2036."}],
        "When does glacier station TR891 open?",
    )
    assert chunks
    assert not qclient.get_collections().collections


@pytest.mark.parametrize("audience", ["", "placeholder", "https://example.invalid"])
def test_production_rejects_missing_or_invalid_auth_configuration(monkeypatch, audience):
    import app.main as main_module
    monkeypatch.setattr(main_module.settings, "ENVIRONMENT", "production")
    # Existing model has no GOOGLE_CLIENT_ID yet: update its underlying dictionary
    # so the pre-implementation test fails for absent validation, not fixture setup.
    monkeypatch.setitem(main_module.settings.__dict__, "GOOGLE_CLIENT_ID", audience)
    with pytest.raises(RuntimeError):
        create_application()


@pytest.mark.parametrize("path", ["/api/v1/library/files", "/api/v1/library/search?q=private",
                                  "/api/v1/sessions", "/api/v1/memory", "/api/v1/search/global?q=private&user_id=alice"])
@pytest.mark.parametrize("headers", [{}, {"X-Client-Id": "alice", "X-Role": "admin"},
                                     {"Authorization": "Bearer invalid", "X-Client-Id": "alice"}])
def test_private_routes_require_verified_accounts(sandbox, path, headers):
    client, _, _ = sandbox
    assert client.get(path, headers=headers).status_code == 401


def test_global_search_ignores_forged_owner(sandbox):
    client, _, _ = sandbox
    assert upload(client, token("alice")).status_code == 200
    response = client.get("/api/v1/search/global?q=release_beacon&user_id=alice", headers=token("bob"))
    assert response.status_code == 200
    assert "release_beacon.txt" not in response.text


def test_deleted_account_token_cannot_access_private_data(sandbox, monkeypatch):
    client, _, _ = sandbox
    stale_token = token("alice")
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda uid: None)
    assert client.get("/api/v1/library/files", headers={**stale_token, "X-Client-Id": "alice"}).status_code == 401


def test_traversal_is_denied(sandbox, tmp_path):
    _, storage, _ = sandbox
    outside = tmp_path / "outside.txt"
    outside.write_text("synthetic outside storage")
    with pytest.raises(PermissionError):
        storage.get("../outside.txt")


def test_cross_user_memory_write_denied(sandbox, monkeypatch):
    client, _, _ = sandbox
    from app.services.memory_service import memory_service
    monkeypatch.setattr(memory_service, "db_path", auth_service.db_path)
    memory_service._init_sqlite_db()
    response = client.post("/api/v1/memory", headers=token("alice"), json={"fact": "Synthetic private preference", "category": "general"})
    assert response.status_code == 201
    mid = response.json()["id"]
    assert client.put(f"/api/v1/memory/{mid}", headers=token("bob"), json={"fact": "Forged replacement", "category": "general"}).status_code == 404
    assert client.delete(f"/api/v1/memory/{mid}", headers=token("bob")).status_code == 404


@pytest.mark.parametrize("path", ["/feedback", "/api/v1/feedback"])
def test_feedback_requires_matching_log_owner(sandbox, monkeypatch, tmp_path, path):
    client, _, _ = sandbox
    from app.services.conversation_log_service import conversation_log_service
    conversation_log_service._queue.join()
    monkeypatch.setattr(conversation_log_service, "db_path", str(tmp_path / "logs.db"))
    conversation_log_service._init_sqlite_db()
    # Keep account records in the same database as logs.
    monkeypatch.setattr(auth_service, "db_path", conversation_log_service.db_path)
    auth_service._init_db()
    with auth_service._get_connection() as conn:
        for uid in ("alice", "bob"):
            conn.execute("INSERT INTO users (id, username, email, full_name, password_hash, salt) VALUES (?, ?, ?, ?, ?, ?)", (uid, uid, uid + "@example.invalid", uid, "unused", "unused"))
    cid = "owned-log"
    conversation_log_service.log_async(conversation_id=cid, user_query="synthetic-private-query",
        retrieved_chunks=[{"text": "synthetic-private-passage"}], provider_used="synthetic",
        model_response="synthetic-private-response", response_time_ms=1, owner_id="alice")
    conversation_log_service._queue.join()
    assert client.post(path, headers=token("bob"), json={"conversation_id": cid, "signal": "down"}).status_code == 404
    assert client.post(path, headers=token("alice"), json={"conversation_id": cid, "signal": "up"}).status_code == 200
    with conversation_log_service._get_sqlite_connection() as conn:
        row = dict(conn.execute("SELECT * FROM conversation_logs WHERE id = ?", (cid,)).fetchone())
    assert row["user_id"] == "alice" and row["feedback"] == "up"
    assert "synthetic-private" not in json.dumps(row)


def test_pending_erasure_revokes_access_but_permits_retry(sandbox, monkeypatch):
    client, _, qclient = sandbox
    assert upload(client, token("alice")).status_code == 200
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic vector service outage")
    original = qclient.delete
    monkeypatch.setattr(qclient, "delete", unavailable)
    assert client.delete("/api/v1/auth/me", headers=token("alice")).status_code == 503
    assert client.get("/api/v1/library/files", headers=token("alice")).status_code == 401
    assert client.get("/api/v1/auth/me", headers=token("alice")).status_code == 401
    monkeypatch.setattr(qclient, "delete", original)
    assert client.delete("/api/v1/auth/me", headers=token("alice")).status_code == 200
    assert auth_service.get_user_by_id("alice") is None


@pytest.mark.parametrize("stream", [False, True])
def test_public_anonymous_chat_is_stateless(sandbox, monkeypatch, stream):
    client, _, _ = sandbox
    from app.services.conversation_log_service import conversation_log_service
    from app.services.memory_service import memory_service
    from app.services.session_store import session_store
    conversation_log_service._queue.join()
    for service in (session_store, memory_service, conversation_log_service):
        monkeypatch.setattr(service, "db_path", auth_service.db_path)
    session_store._init_db()
    memory_service._init_sqlite_db()
    conversation_log_service._init_sqlite_db()
    response = client.post("/api/v1/chat", headers={"X-Client-Id": "alice"}, json={"message": "Hello", "stream": stream})
    assert response.status_code == 200, response.text
    conversation_log_service._queue.join()
    with auth_service._get_connection() as conn:
        for table in ("sessions", "messages", "user_memories", "conversation_logs"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_anonymous_private_attachment_requires_account(sandbox):
    client, _, _ = sandbox
    response = client.post("/api/v1/chat", json={"message": "Read this", "attachments": [{"name": "private.txt", "mime_type": "text/plain", "data": "cHJpdmF0ZQ=="}]})
    assert response.status_code == 401


def test_queue_cannot_restore_deleted_account_log(sandbox, monkeypatch):
    _, _, _ = sandbox
    from app.services.conversation_log_service import conversation_log_service
    conversation_log_service._queue.join()
    monkeypatch.setattr(conversation_log_service, "db_path", auth_service.db_path)
    conversation_log_service._init_sqlite_db()
    with auth_service._get_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = 'alice'")
    conversation_log_service._insert_log_sqlite({"id": "stale-log", "user_id": "alice"})
    with auth_service._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_logs").fetchone()[0] == 0


def test_project_and_gallery_metadata_are_private(sandbox):
    client, _, _ = sandbox
    assert client.get("/api/v1/projects").status_code == 401
    assert client.get("/api/v1/image/gallery").status_code == 401
    project = client.post("/api/v1/projects", headers=token("alice"), json={"name": "Private workspace", "instructions": "Synthetic private instructions"})
    assert project.status_code == 200
    pid = project.json()["project"]["id"]
    assert client.get(f"/api/v1/projects/{pid}", headers=token("bob")).status_code == 403
    assert client.delete(f"/api/v1/projects/{pid}", headers=token("bob")).status_code == 403


def test_anonymous_integration_toggle_denied(sandbox):
    client, _, _ = sandbox
    assert client.post("/api/v1/plugins/web_search/toggle", json={"is_enabled": True}).status_code == 401


def test_privacy_migration_upgrades_and_downgrades_sqlite():
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect, text
    directory = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    def load(name):
        spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    first, privacy = load("001_initial_schema"), load("002_tenant_privacy")
    engine = create_engine("sqlite://")
    with engine.begin() as connection, Operations.context(MigrationContext.configure(connection)):
        first.upgrade()
        connection.execute(text("INSERT INTO conversation_logs (id,timestamp,user_query,retrieved_chunks,provider_used,model_response,response_time_ms) VALUES ('legacy','2026-10-08 00:00:00','private','[]','synthetic','private',1)"))
        privacy.upgrade()
        assert "deletion_pending" in {c["name"] for c in inspect(connection).get_columns("users")}
        assert "user_id" in {c["name"] for c in inspect(connection).get_columns("conversation_logs")}
        assert connection.execute(text("SELECT user_query,model_response FROM conversation_logs WHERE id='legacy'")).one() == ("", "")
        privacy.downgrade()
        assert "user_id" not in {c["name"] for c in inspect(connection).get_columns("conversation_logs")}
        assert connection.execute(text("SELECT user_query FROM conversation_logs WHERE id='legacy'")).scalar() == ""
    engine.dispose()


def test_pending_erasure_prevents_delayed_session_write(sandbox, monkeypatch):
    _, _, _ = sandbox
    from uuid import uuid4

    from app.services.session_store import session_store
    monkeypatch.setattr(session_store, "db_path", auth_service.db_path)
    session_store._init_db()
    sid = uuid4()
    with auth_service._get_connection() as conn:
        conn.execute("UPDATE users SET deletion_pending=1 WHERE id='alice'")
    session_store.ensure_private_session(sid, "Late callback", "alice")
    session_store.record_private_turn(sid, "assistant", "Sensitive late answer", "alice")
    with auth_service._get_connection() as conn:
        assert not conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (str(sid),)).fetchone()
        assert not conn.execute("SELECT 1 FROM messages WHERE session_id=?", (str(sid),)).fetchone()


def test_erasure_clears_all_owned_stores_and_preserves_other_account(sandbox, monkeypatch):
    client, storage, qclient = sandbox
    from uuid import uuid4

    from app.services.cache.response_cache import response_cache
    from app.services.conversation_log_service import conversation_log_service
    from app.services.memory_service import memory_service
    from app.services.rag.qdrant_vector_store import query_embedding_cache
    from app.services.session_store import session_store
    from app.services.workspace_store import workspace_store
    conversation_log_service._queue.join()
    for service in (session_store, memory_service, conversation_log_service):
        monkeypatch.setattr(service, "db_path", auth_service.db_path)
    session_store._init_db()
    memory_service._init_sqlite_db()
    conversation_log_service._init_sqlite_db()
    for uid in ("alice", "bob"):
        assert upload(client, token(uid)).status_code == 200
        sid = uuid4()
        session_store.ensure_private_session(sid, "Synthetic owned session", uid)
        session_store.record_private_turn(sid, "user", "Synthetic private message", uid)
        memory_service.add_memory(user_id=uid, fact="Synthetic owned memory", category="general")
        workspace_store.create_project("Synthetic owned project", user_id=uid)
        task = workspace_store.create_scheduled_task(name="Synthetic owned task", prompt="Synthetic private task", user_id=uid)
        workspace_store.record_execution(task["id"], status="success", output="Synthetic owned output")
        workspace_store.toggle_integration("web_search", True, user_id=uid)
        asset = storage.put(uid + ".png", b"synthetic-image-bytes")
        workspace_store.record_image(prompt="Synthetic private image prompt", file_url=asset, model="synthetic", user_id=uid)
        conversation_log_service.log_async(conversation_id=uid + "-log", user_query="private", retrieved_chunks=[],
            provider_used="synthetic", model_response="private", response_time_ms=1, owner_id=uid)
    conversation_log_service._queue.join()
    response_cache.set("B", "auto", "synthetic cached prompt", "synthetic response", "synthetic", "synthetic", 1)
    query_embedding_cache.set("synthetic private query", [1.0])
    alice_asset = next(i["file_url"] for i in workspace_store.list_images(user_id="alice"))
    assert client.delete("/api/v1/auth/me", headers=token("alice")).status_code == 200
    assert not storage.exists(alice_asset)
    with auth_service._get_connection() as conn:
        for table in ("sessions", "user_memories", "projects", "scheduled_tasks", "user_integrations", "image_records", "library_files", "conversation_logs"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id='alice'").fetchone()[0] == 0
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id='bob'").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_executions").fetchone()[0] == 1
    assert auth_service.get_user_by_id("bob")
    assert not response_cache.get_stats()["total_entries"] and not query_embedding_cache.size()
    assert qclient.count("user_knowledge_base").count == 1


def test_google_key_fetch_failure_fails_closed(google_token_factory, monkeypatch):
    from google.auth.exceptions import TransportError
    def unavailable(*args, **kwargs):
        raise TransportError("synthetic certificate outage")
    monkeypatch.setattr(google_requests, "Request", lambda **kwargs: unavailable)
    with TestClient(create_application()) as client:
        assert client.post("/api/v1/auth/google", json={"credential": google_token_factory()}).status_code == 503


def test_failed_index_cleanup_still_blocks_retrieval(sandbox, monkeypatch):
    client, _, qclient = sandbox
    actual_upsert = qclient.upsert
    def partial(*args, **kwargs):
        actual_upsert(*args, **kwargs)
        raise RuntimeError("synthetic partial write")
    def no_cleanup(*args, **kwargs):
        raise RuntimeError("synthetic cleanup outage")
    monkeypatch.setattr(qclient, "upsert", partial)
    monkeypatch.setattr(qclient, "delete", no_cleanup)
    assert upload(client, token("alice")).status_code == 422
    assert qclient.count("user_knowledge_base").count > 0
    for uid in ("alice", "bob"):
        response = client.get("/api/v1/library/search?q=VX731", headers=token(uid))
        assert response.status_code == 200 and response.json()["count"] == 0


def test_unknown_blob_absence_does_not_acknowledge_deletion(sandbox, monkeypatch):
    client, storage, _ = sandbox
    response = upload(client, token("alice"))
    assert response.status_code == 200
    fid = response.json()["file"]["id"]
    monkeypatch.setattr(storage, "delete_file", lambda *args: False)
    monkeypatch.setattr(storage, "exists", lambda *args: False)
    def uncertain(*args):
        raise OSError("synthetic metadata service outage")
    monkeypatch.setattr(storage, "metadata", uncertain)
    assert client.delete(f"/api/v1/library/files/{fid}", headers=token("alice")).status_code == 503
    assert client.post(f"/api/v1/library/files/{fid}/reindex", headers=token("alice")).status_code == 409


def test_legacy_guest_record_is_quarantined(sandbox):
    client, storage, _ = sandbox
    from app.services.workspace_store import workspace_store
    path = storage.put("legacy.txt", b"Synthetic legacy guest private content")
    record = workspace_store.record_library_file(filename="legacy.txt", file_type="TXT", file_size=38,
                                               storage_path=path, user_id="guest", status="indexed")
    fid = record["id"]
    for uid in ("alice", "bob"):
        assert client.get(f"/api/v1/library/files/{fid}/content", headers=token(uid)).status_code == 403
        assert client.post(f"/api/v1/library/files/{fid}/reindex", headers=token(uid)).status_code == 403
        assert client.delete(f"/api/v1/library/files/{fid}", headers=token(uid)).status_code == 403


def test_metadata_commit_failure_never_reports_indexed(sandbox, monkeypatch):
    client, _, _ = sandbox
    from app.services.workspace_store import workspace_store
    update = workspace_store.update_library_file_status
    monkeypatch.setattr(workspace_store, "update_library_file_status",
                        lambda *args, **kwargs: False if kwargs.get("status") == "indexed" else update(*args, **kwargs))
    response = upload(client, token("alice"))
    assert response.status_code in (422, 503)
    assert client.get("/api/v1/library/search?q=VX731", headers=token("alice")).json()["count"] == 0
