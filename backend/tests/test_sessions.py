from tests.private_accounts import private_account_headers
"""Unit and integration tests for sessions management endpoints and privacy isolation."""
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.session_store import session_store

client = TestClient(app)


def test_unauthenticated_without_client_id_is_denied():
    """Verify that requests without client ID or auth token return empty list and do not leak sessions."""
    res = client.get("/api/v1/sessions")
    assert res.status_code == 401


def test_sessions_lifecycle_with_client_id():
    """Test full CRUD lifecycle for session management using anonymous client ID."""
    test_client_id = f"test-client-{uuid.uuid4()}"
    headers = private_account_headers(test_client_id)
    owner_id = test_client_id
    test_id = uuid.uuid4()

    # 1. Create a session in store tied to owner
    session_store.ensure_session(test_id, "Test Exploration Session", user_id=owner_id)
    session_store.append_turn(test_id, "user", "Hello there!")
    session_store.append_turn(test_id, "assistant", "Hello! How can I help you today?")

    # 2. List sessions for this client
    res = client.get("/api/v1/sessions", headers=headers)
    assert res.status_code == 200
    sessions_data = res.json()
    assert "sessions" in sessions_data
    found = [s for s in sessions_data["sessions"] if s["id"] == str(test_id)]
    assert len(found) == 1
    assert found[0]["title"] == "Test Exploration Session"

    # 3. Get session messages
    res_msg = client.get(f"/api/v1/sessions/{test_id}/messages", headers=headers)
    assert res_msg.status_code == 200
    msg_data = res_msg.json()
    assert len(msg_data["messages"]) == 2
    assert msg_data["messages"][0]["role"] == "user"
    assert msg_data["messages"][0]["content"] == "Hello there!"
    assert msg_data["messages"][1]["role"] == "assistant"

    # 4. Rename session
    res_rename = client.patch(
        f"/api/v1/sessions/{test_id}",
        json={"title": "Renamed Session Title"},
        headers=headers,
    )
    assert res_rename.status_code == 200
    assert res_rename.json()["title"] == "Renamed Session Title"

    # 5. Delete session
    res_delete = client.delete(f"/api/v1/sessions/{test_id}", headers=headers)
    assert res_delete.status_code == 200
    assert res_delete.json()["status"] == "deleted"

    # 6. Verify session is gone
    res_msg_after = client.get(f"/api/v1/sessions/{test_id}/messages", headers=headers)
    assert res_msg_after.status_code == 404


def test_user_chat_history_isolation():
    """Critical privacy test: User B cannot see or access User A's session or messages."""
    alice_client = f"alice-{uuid.uuid4()}"
    bob_client = f"bob-{uuid.uuid4()}"

    alice_headers = private_account_headers(alice_client)
    bob_headers = private_account_headers(bob_client)

    alice_owner = alice_client
    alice_session_id = uuid.uuid4()

    # Alice creates a private conversation
    session_store.ensure_session(alice_session_id, "Alice's Private Secrets", user_id=alice_owner)
    session_store.append_turn(alice_session_id, "user", "My secret password is xyz")
    session_store.append_turn(alice_session_id, "assistant", "I will keep your secret safe.")

    # 1. Bob lists sessions: Alice's session must NOT appear in Bob's session list
    bob_list_res = client.get("/api/v1/sessions", headers=bob_headers)
    assert bob_list_res.status_code == 200
    bob_sessions = bob_list_res.json()["sessions"]
    alice_found_by_bob = [s for s in bob_sessions if s["id"] == str(alice_session_id)]
    assert len(alice_found_by_bob) == 0, "Privacy violation: Bob can see Alice's session in his sidebar!"

    # 2. Bob attempts to directly fetch Alice's message history by guessing/sharing UUID
    bob_msg_res = client.get(f"/api/v1/sessions/{alice_session_id}/messages", headers=bob_headers)
    assert bob_msg_res.status_code == 403

    # 3. Bob attempts to rename Alice's session
    bob_rename_res = client.patch(
        f"/api/v1/sessions/{alice_session_id}",
        json={"title": "Hacked by Bob"},
        headers=bob_headers,
    )
    assert bob_rename_res.status_code == 403

    # 4. Bob attempts to delete Alice's session
    bob_del_res = client.delete(f"/api/v1/sessions/{alice_session_id}", headers=bob_headers)
    assert bob_del_res.status_code == 403

    # 5. Alice can still access her own messages
    alice_msg_res = client.get(f"/api/v1/sessions/{alice_session_id}/messages", headers=alice_headers)
    assert alice_msg_res.status_code == 200
    assert len(alice_msg_res.json()["messages"]) == 2
    assert alice_msg_res.json()["messages"][0]["content"] == "My secret password is xyz"

    # Cleanup
    session_store.delete_session(alice_session_id, user_id=alice_owner)


def test_anonymous_migration_to_authenticated_user():
    """Verify that guest sessions can be migrated to a signed-in account."""
    anon_client = f"guest-{uuid.uuid4()}"
    anon_owner = anon_client
    auth_user_id = str(uuid.uuid4())

    sess_1 = uuid.uuid4()
    sess_2 = uuid.uuid4()

    session_store.ensure_session(sess_1, "Guest Chat 1", user_id=anon_owner)
    session_store.ensure_session(sess_2, "Guest Chat 2", user_id=anon_owner)

    # Migrate to authenticated user
    count = session_store.migrate_anonymous_sessions(anon_owner, auth_user_id)
    assert count == 2

    # Verify they now belong to auth_user_id
    auth_sessions = session_store.list_sessions(user_id=auth_user_id)
    assert len(auth_sessions) == 2
    titles = [s["title"] for s in auth_sessions]
    assert "Guest Chat 1" in titles
    assert "Guest Chat 2" in titles

    # Cleanup
    session_store.delete_session(sess_1, user_id=auth_user_id)
    session_store.delete_session(sess_2, user_id=auth_user_id)


def test_end_to_end_chat_endpoint_privacy(monkeypatch):
    from unittest.mock import AsyncMock
    from app.api.dependencies import get_chat_service
    monkeypatch.setattr(get_chat_service().router, "generate_response", AsyncMock(return_value=("Synthetic private grades reply.", "synthetic", "synthetic", False)))
    """Simulate Alice chatting, sharing link with Bob, and verifying 100% isolation."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"
    alice_session = str(uuid.uuid4())

    # 1. Alice sends a chat message
    chat_res = client.post(
        "/api/v1/chat",
        headers=private_account_headers(alice_id),
        json={
            "session_id": alice_session,
            "message": "Hi Jruva, this is Alice's private inquiry about my grades",
            "model": "auto",
            "stream": False,
        },
    )
    assert chat_res.status_code == 200

    # 2. Alice checks her session list
    alice_sessions = client.get("/api/v1/sessions", headers=private_account_headers(alice_id)).json()["sessions"]
    assert any(s["id"] == alice_session for s in alice_sessions)

    # 3. Bob opens the chatbot link (fresh client ID) and checks sessions
    bob_sessions = client.get("/api/v1/sessions", headers=private_account_headers(bob_id)).json()["sessions"]
    assert not any(s["id"] == alice_session for s in bob_sessions), "Bob must not see Alice's chat session!"

    # 4. Bob tries to access Alice's messages (must be rejected with HTTP 403 Forbidden)
    bob_resp = client.get(f"/api/v1/sessions/{alice_session}/messages", headers=private_account_headers(bob_id))
    assert bob_resp.status_code == 403, "Bob must receive 403 Forbidden when accessing Alice's session!"

    # 5. A completely headerless / random visitor checks sessions
    assert client.get("/api/v1/sessions").status_code == 401

    # Cleanup
    client.delete(f"/api/v1/sessions/{alice_session}", headers=private_account_headers(alice_id))
