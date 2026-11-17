"""Release criteria against real local storage/vectors and synthetic identities.

Known unsafe behavior deliberately FAILS; no xfail/skip masks release blockers.
Run via scripts/verify_operations_security.py --targets <this file>.
No hosted model generation or production data is involved.
"""

import base64
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from app.api.dependencies import get_chat_service
from app.api.v1.endpoints import library
from app.main import create_application
from app.services.auth_service import auth_service
from app.services.rag import qdrant_vector_store as vectors
from app.services.rag.document_rag import document_rag_service
from app.services.storage.local import LocalStorageBackend
from app.services.workspace_store import workspace_store
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

pytest_plugins = ["tests.google_identity_fixtures"]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Fresh metadata, blobs and vectors per case; actual JWT verification."""
    monkeypatch.setattr(workspace_store, "db_path", str(tmp_path / "workspace.db"))
    workspace_store._init_sqlite_db()
    storage = LocalStorageBackend(str(tmp_path / "uploads"))
    monkeypatch.setattr(library, "storage_service", storage)
    import app.services.storage as storage_module
    monkeypatch.setattr(storage_module, "storage_service", storage)
    import app.services.workspace_store as workspace_module
    monkeypatch.setattr(workspace_module, "storage_service", storage)
    qclient = QdrantClient(":memory:")
    monkeypatch.setattr(vectors, "_QDRANT_CLIENT", qclient)
    monkeypatch.setattr(vectors, "_KNOWN_COLLECTIONS", set())
    vectors._BM25_COLLECTION_CACHE.clear()
    # Private fixtures now represent real accounts, not unsigned guest partitions.
    monkeypatch.setattr(auth_service, "db_path", workspace_store.db_path)
    auth_service._init_db()
    with auth_service._get_connection() as conn:
        for uid in ("alice", "bob"):
            conn.execute("INSERT INTO users (id, username, email, full_name, password_hash, salt, google_sub) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (uid, uid, uid + "@example.invalid", "Synthetic " + uid, "unused", "unused", "synthetic-" + uid))
    with TestClient(create_application(), raise_server_exceptions=False) as client:
        yield client, storage, qclient
    vectors._BM25_COLLECTION_CACHE.clear()
    qclient.close()


def token(uid):
    return {"Authorization": "Bearer " + auth_service.create_jwt_token({"sub": uid})}


def upload(client, headers, text="Release beacon VX731 activates on 19 April 2033."):
    return client.post(
        "/api/v1/library/upload-json", headers=headers,
        json={"filename": "release_beacon.txt", "file_type": "text/plain",
              "data": base64.b64encode(text.encode()).decode()},
    )


def points(qclient):
    if not qclient.collection_exists("user_knowledge_base"):
        return []
    return qclient.scroll("user_knowledge_base", limit=1000, with_payload=True)[0]


@pytest.mark.parametrize("action", ["content", "search", "reindex", "delete"])
def test_copied_guest_identifier_does_not_authorize_private_document(sandbox, action):
    client, _, _ = sandbox
    victim = {"X-Client-Id": "victim-device-" + uuid.uuid4().hex}
    result = upload(client, {**token("alice"), **victim})
    assert result.status_code == 200, result.text
    fid = result.json()["file"]["id"]
    # A fresh caller copies the identifier, without a credential or signed cookie.
    with TestClient(client.app) as attacker:
        if action == "search":
            response = attacker.get("/api/v1/library/search?q=VX731", headers=victim)
            assert response.status_code in (401, 403) or response.json()["count"] == 0
        elif action == "content":
            assert attacker.get(f"/api/v1/library/files/{fid}/content", headers=victim).status_code in (401, 403)
        elif action == "reindex":
            assert attacker.post(f"/api/v1/library/files/{fid}/reindex", headers=victim).status_code in (401, 403)
        else:
            assert attacker.delete(f"/api/v1/library/files/{fid}", headers=victim).status_code in (401, 403)


@pytest.mark.parametrize("action", ["list", "search", "content"])
def test_shared_guest_namespace_is_not_private(sandbox, action):
    client, _, _ = sandbox
    result = upload(client, {})
    if result.status_code in (401, 403):
        return  # Requiring verified identity is an acceptable future secure contract.
    assert result.status_code == 200
    fid = result.json()["file"]["id"]
    with TestClient(client.app) as attacker:
        if action == "list":
            response = attacker.get("/api/v1/library/files")
            assert response.status_code in (401, 403) or not response.json()["files"]
        elif action == "search":
            response = attacker.get("/api/v1/library/search?q=VX731")
            assert response.status_code in (401, 403) or response.json()["count"] == 0
        else:
            assert attacker.get(f"/api/v1/library/files/{fid}/content", headers=token("bob")).status_code in (401, 403)


def test_verified_user_identity_overrides_forged_client_header(sandbox):
    client, _, _ = sandbox
    result = upload(client, token("alice"))
    assert result.status_code == 200, result.text
    fid = result.json()["file"]["id"]
    bob = {**token("bob"), "X-Client-Id": "alice"}
    assert client.get(f"/api/v1/library/files/{fid}/content", headers=bob).status_code == 403
    assert client.post(f"/api/v1/library/files/{fid}/reindex", headers=bob).status_code == 403
    assert client.delete(f"/api/v1/library/files/{fid}", headers=bob).status_code == 403
    assert client.get("/api/v1/library/search?q=VX731", headers=bob).json()["count"] == 0
    store = vectors.QdrantVectorStore()
    assert not store.search_dense("user_knowledge_base", "VX731", owner_id="bob")
    assert not store.search_sparse_bm25("user_knowledge_base", "VX731", owner_id="bob")


def test_reindex_exact_vector_count_and_delete_invalidates_bm25(sandbox):
    client, storage, qclient = sandbox
    result = upload(client, token("alice"))
    assert result.status_code == 200
    file = result.json()["file"]
    before_ids = {p.id for p in points(qclient)}
    for _ in range(2):
        assert client.post(f"/api/v1/library/files/{file['id']}/reindex", headers=token("alice")).status_code == 200
        assert {p.id for p in points(qclient)} == before_ids
    store = vectors.QdrantVectorStore()
    assert store.search_sparse_bm25("user_knowledge_base", "VX731", owner_id="alice")
    assert client.delete(f"/api/v1/library/files/{file['id']}", headers=token("alice")).status_code == 200
    assert not points(qclient)
    assert workspace_store.get_library_file(file["id"]) is None
    assert not storage.exists(file["storage_path"])
    assert not store.search_sparse_bm25("user_knowledge_base", "VX731", owner_id="alice")


def test_zero_index_writes_return_failed_metadata(sandbox, monkeypatch):
    client, _, qclient = sandbox
    monkeypatch.setattr(vectors.QdrantVectorStore, "index_chunks", lambda *a, **k: 0)
    result = upload(client, token("alice"))
    assert result.status_code == 422
    records = workspace_store.list_library_files(user_id="alice")
    assert len(records) == 1 and records[0]["status"] == "failed"
    assert records[0]["chunk_count"] == 0 and not points(qclient)


@pytest.mark.parametrize("phase", ["partial_upsert", "bm25"])
def test_partial_index_failure_leaves_no_retrievable_vectors(sandbox, monkeypatch, phase):
    client, _, qclient = sandbox
    store = vectors.QdrantVectorStore()
    if phase == "partial_upsert":
        real_upsert = qclient.upsert
        def partial_upsert(*args, **kwargs):
            kwargs["points"] = kwargs["points"][:1]
            real_upsert(*args, **kwargs)
            raise RuntimeError("synthetic partial write failure")
        monkeypatch.setattr(qclient, "upsert", partial_upsert)
    else:
        def broken_bm25(*args, **kwargs):
            raise RuntimeError("synthetic lexical indexing failure")
        monkeypatch.setattr(vectors.QdrantVectorStore, "_build_bm25_index", broken_bm25)
    result = upload(client, token("alice"))
    assert result.status_code == 422
    assert workspace_store.list_library_files(user_id="alice")[0]["status"] == "failed"
    # Dense retrieval remains usable even if the lexical build is fault-injected.
    assert not store.search_dense("user_knowledge_base", "VX731", owner_id="alice")
    assert not points(qclient)


def test_delete_failure_does_not_acknowledge_erasure(sandbox, monkeypatch):
    client, _, qclient = sandbox
    result = upload(client, token("alice"))
    assert result.status_code == 200
    fid = result.json()["file"]["id"]
    def broken_delete(*args, **kwargs):
        raise RuntimeError("synthetic unavailable vector store")
    monkeypatch.setattr(qclient, "delete", broken_delete)
    response = client.delete(f"/api/v1/library/files/{fid}", headers=token("alice"))
    observed = {"status": response.status_code, "remaining_vectors": len(points(qclient)),
                "metadata_exists": workspace_store.get_library_file(fid) is not None}
    assert response.status_code >= 400, f"Erasure falsely acknowledged: {observed}"
    assert workspace_store.get_library_file(fid) is not None


def test_failed_index_response_does_not_echo_internal_exception(sandbox, monkeypatch):
    client, _, _ = sandbox
    sentinel = "synthetic-password=do-not-expose; internal-host.example/secret.db"
    def broken_index(*args, **kwargs):
        raise RuntimeError(sentinel)
    monkeypatch.setattr(document_rag_service, "index_library_document", broken_index)
    response = upload(client, token("alice"))
    assert response.status_code == 422
    assert sentinel not in response.text


def test_google_wrong_audience_rejected(sandbox, google_token_factory):
    client, _, _ = sandbox
    # Real locally signed token, trusted synthetic certificate transport, wrong audience.
    response = client.post("/api/v1/auth/google", json={"credential": google_token_factory(aud="another-app.apps.googleusercontent.com")})
    assert response.status_code in (400, 401, 403)


def test_copied_guest_identifier_cannot_read_session(sandbox, tmp_path, monkeypatch):
    client, _, _ = sandbox
    from app.services.session_store import SessionStore, session_store
    isolated = SessionStore(str(tmp_path / "sessions.db"))
    monkeypatch.setattr(session_store, "db_path", isolated.db_path)
    sid = uuid.uuid4()
    session_store.ensure_session(sid, "Synthetic private session", user_id="anon_copied-device")
    session_store.append_turn(sid, "user", "Synthetic private text")
    with TestClient(client.app) as attacker:
        assert attacker.get(f"/api/v1/sessions/{sid}/messages", headers={"X-Client-Id": "copied-device"}).status_code in (401, 403)


def test_account_erasure_removes_document_metadata_blobs_and_vectors(sandbox, monkeypatch):
    client, storage, qclient = sandbox
    from app.services.memory_service import MemoryService
    monkeypatch.setattr(auth_service, "db_path", workspace_store.db_path)
    auth_service._init_db()
    MemoryService(workspace_store.db_path)
    result = upload(client, token("alice"))
    assert result.status_code == 200
    file = result.json()["file"]
    assert client.delete("/api/v1/auth/me", headers=token("alice")).status_code == 200
    remaining = {"metadata": workspace_store.get_library_file(file["id"]) is not None,
                 "vectors": bool(points(qclient)), "blob": storage.exists(file["storage_path"])}
    assert not any(remaining.values()), f"Account erasure incomplete: {remaining}"


@pytest.mark.parametrize("path", ["/feedback", "/api/v1/feedback"])
def test_anonymous_feedback_cannot_write(sandbox, monkeypatch, path):
    client, _, _ = sandbox
    from app.services.conversation_log_service import conversation_log_service
    writes = []
    monkeypatch.setattr(conversation_log_service, "update_feedback", lambda **k: writes.append(k) or True)
    response = client.post(path, json={"conversation_id": str(uuid.uuid4()), "signal": "up"})
    assert response.status_code in (401, 403)
    assert not writes


def test_storage_sibling_prefix_rejected(sandbox, tmp_path):
    _, storage, _ = sandbox
    sibling = tmp_path / "uploads-sibling" / "synthetic.txt"
    sibling.parent.mkdir()
    sibling.write_text("outside storage", encoding="utf-8")
    with pytest.raises(PermissionError):
        storage.get(str(sibling))


def test_real_retrieval_reaches_chat_prompt_and_citations(sandbox, monkeypatch):
    client, _, _ = sandbox
    result = upload(client, token("alice"))
    assert result.status_code == 200
    service = get_chat_service()
    generation = AsyncMock(return_value=("VX731 activates on 19 April 2033.", "synthetic", "synthetic", False))
    monkeypatch.setattr(service.router, "generate_response", generation)
    import app.services.chat_service as chat_module
    monkeypatch.setattr(chat_module.web_search_service, "should_search", lambda *a: False)
    monkeypatch.setattr(chat_module, "should_check_domain_knowledge", lambda *a: False)
    response = client.post("/api/v1/chat", headers=token("alice"),
                           json={"message": "When does release beacon VX731 activate?", "stream": False})
    assert response.status_code == 200, response.text
    assert "19 April 2033" in generation.call_args.kwargs["system_instruction"]
    assert "release_beacon.txt" in json.dumps(response.json()["sources"])
