from tests.private_accounts import private_account_headers
"""Comprehensive Security & Negative Authorization Hardening Tests.

Verifies strict tenant isolation across:
1. Conversation reading (403 on foreign session)
2. Conversation appending (403 on foreign session append)
3. Conversation deletion (403 on foreign session delete)
4. Uploaded document retrieval / listing (User A cannot see User B files)
5. Uploaded document deletion (403 on foreign document delete)
6. Central RAG vector & lexical search leakage (User A query never retrieves User B private chunks)
7. Client-supplied identity spoofing rejection
"""
import uuid
import inspect
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.main import app
from app.services.session_store import session_store
from app.services.workspace_store import workspace_store
from app.services.rag.qdrant_vector_store import QdrantVectorStore


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("client_id", [None, "   ", "x" * 129])
async def test_guest_payload_cannot_impersonate_session_owner(monkeypatch, stream, client_id):
    from app.api.v1.endpoints.chat import chat_endpoint
    from app.schemas.chat import ChatRequest

    victim_id = "verified-victim"
    monkeypatch.setattr(session_store, "get_session", lambda sid: {"userId": victim_id})
    payload = ChatRequest(
        session_id=uuid.uuid4(), user_id=victim_id,
        message="Read my private conversation", stream=stream,
    )
    request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1)})

    class StubChat:
        async def generate_chat_response(self, payload):
            return None

        async def stream_chat_response(self, payload):
            yield ""

    with pytest.raises(HTTPException) as caught:
        await inspect.unwrap(chat_endpoint)(
            request=request, payload=payload, chat_service=StubChat(),
            current_user=None, x_client_id=client_id,
        )
    assert caught.value.status_code == 403
    assert payload.user_id != victim_id


@pytest.fixture
def client():
    return TestClient(app)


def test_user_cannot_read_another_users_conversation(client):
    """User B must be rejected with 403 Forbidden when attempting to read User A's session."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"
    alice_session = uuid.uuid4()

    # Alice creates a private session with turns
    session_store.ensure_session(alice_session, "Alice Private Grades", user_id=alice_id)
    session_store.append_turn(alice_session, "user", "What is my grade in CS501?")
    session_store.append_turn(alice_session, "assistant", "Your grade is A+.")

    # Bob attempts to read Alice's conversation history
    res = client.get(
        f"/api/v1/sessions/{alice_session}/messages",
        headers=private_account_headers(bob_id),
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json().get("detail", "")


def test_user_cannot_append_to_another_users_conversation(client):
    """User B must be rejected with 403 Forbidden when attempting to append to User A's session."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"
    alice_session = str(uuid.uuid4())

    # Alice initializes session
    session_store.ensure_session(uuid.UUID(alice_session), "Alice Thesis", user_id=alice_id)

    # Bob attempts to send a chat message into Alice's session
    res = client.post(
        "/api/v1/chat",
        headers=private_account_headers(bob_id),
        json={
            "session_id": alice_session,
            "message": "Bob malicious injection into Alice's transcript",
            "model": "auto",
            "stream": False,
        },
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json().get("detail", "")


def test_user_cannot_delete_another_users_conversation(client):
    """User B must be rejected with 403 Forbidden when attempting to delete User A's session."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"
    alice_session = uuid.uuid4()

    # Alice creates session
    session_store.ensure_session(alice_session, "Alice Important Session", user_id=alice_id)

    # Bob attempts to delete Alice's session
    res = client.delete(
        f"/api/v1/sessions/{alice_session}",
        headers=private_account_headers(bob_id),
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json().get("detail", "")

    # Confirm Alice's session was not deleted
    meta = session_store.get_session(alice_session)
    assert meta is not None
    assert meta["id"] == str(alice_session)


def test_user_cannot_list_another_users_uploaded_documents(client):
    """User B listing library documents must never see User A's private documents."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"

    # Alice uploads a document
    doc_alice = workspace_store.record_library_file(
        filename="alice_confidential_payroll.pdf",
        file_type="PDF",
        file_size=1024,
        storage_path="/tmp/fake_alice.pdf",
        user_id=alice_id,
    )

    # Bob lists documents
    res_bob = client.get("/api/v1/library/files", headers=private_account_headers(bob_id))
    assert res_bob.status_code == 200
    bob_files = res_bob.json().get("files", [])
    assert not any(f["id"] == doc_alice["id"] for f in bob_files), "Bob must not see Alice's document!"

    # Alice lists documents
    res_alice = client.get("/api/v1/library/files", headers=private_account_headers(alice_id))
    assert res_alice.status_code == 200
    alice_files = res_alice.json().get("files", [])
    assert any(f["id"] == doc_alice["id"] for f in alice_files), "Alice must see her own document."


def test_user_cannot_delete_another_users_document(client):
    """User B must receive 403 Forbidden when attempting to delete User A's document."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"

    # Alice records document
    doc_alice = workspace_store.record_library_file(
        filename="alice_private_medical.pdf",
        file_type="PDF",
        file_size=2048,
        storage_path="/tmp/fake_alice_med.pdf",
        user_id=alice_id,
    )

    # Bob attempts to delete Alice's document
    res = client.delete(
        f"/api/v1/library/files/{doc_alice['id']}",
        headers=private_account_headers(bob_id),
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json().get("detail", "")

    # Confirm Alice's document still exists
    record = workspace_store.get_library_file(doc_alice["id"])
    assert record is not None


def test_rag_cross_tenant_search_leakage_prevented():
    """Verify that Qdrant & BM25 hybrid search strictly isolates private tenant chunks."""
    store = QdrantVectorStore()
    col = f"test_tenant_col_{uuid.uuid4().hex[:8]}"

    alice_user = f"user_alice_{uuid.uuid4().hex[:6]}"
    bob_user = f"user_bob_{uuid.uuid4().hex[:6]}"

    chunks = [
        {
            "id": "alice-doc-1",
            "doc_name": "alice_secret.txt",
            "owner_id": alice_user,
            "text": "Alice Private Token: 998822XYZ confidential research data for Department A.",
        },
        {
            "id": "bob-doc-1",
            "doc_name": "bob_secret.txt",
            "owner_id": bob_user,
            "text": "Bob Proprietary Formula: 443311ABC chemical reaction synthesis for Department B.",
        },
        {
            "id": "public-doc-1",
            "doc_name": "campus_rules.txt",
            "owner_id": "public",
            "text": "Campus rules: Library opens at 8am and closes at 8pm for all students.",
        },
    ]

    store.index_chunks(col, chunks)

    # 1. Alice searches for confidential research data
    alice_results = store.search_hybrid(
        collection_name=col,
        query="confidential research data chemical reaction synthesis",
        limit=5,
        owner_id=alice_user,
    )
    alice_ids = [r["id"] for r in alice_results]
    assert "alice-doc-1" in alice_ids, "Alice should find her own chunk."
    assert "bob-doc-1" not in alice_ids, "Cross-tenant leakage! Alice retrieved Bob's chunk!"

    # 2. Bob searches for confidential research data
    bob_results = store.search_hybrid(
        collection_name=col,
        query="confidential research data chemical reaction synthesis",
        limit=5,
        owner_id=bob_user,
    )
    bob_ids = [r["id"] for r in bob_results]
    assert "bob-doc-1" in bob_ids, "Bob should find his own chunk."
    assert "alice-doc-1" not in bob_ids, "Cross-tenant leakage! Bob retrieved Alice's chunk!"

    # 3. Both can find public document
    pub_alice = store.search_hybrid(col, "Library opens at 8am", limit=5, owner_id=alice_user)
    pub_bob = store.search_hybrid(col, "Library opens at 8am", limit=5, owner_id=bob_user)
    assert any(r["id"] == "public-doc-1" for r in pub_alice)
    assert any(r["id"] == "public-doc-1" for r in pub_bob)


def test_user_cannot_rename_another_users_conversation(client):
    """User B must be rejected with 403 Forbidden when attempting to rename User A's session."""
    alice_id = f"client-alice-{uuid.uuid4()}"
    bob_id = f"client-bob-{uuid.uuid4()}"
    alice_session = uuid.uuid4()

    # Alice initializes session
    session_store.ensure_session(alice_session, "Alice Original Title", user_id=alice_id)

    # Bob attempts to rename Alice's session
    res = client.put(
        f"/api/v1/sessions/{alice_session}",
        headers=private_account_headers(bob_id),
        json={"title": "Hacked Title by Bob"},
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json().get("detail", "")

    # Confirm title unchanged
    meta = session_store.get_session(alice_session)
    assert meta["title"] == "Alice Original Title"


def test_user_cannot_read_or_delete_another_users_memories():
    """Verify that User B cannot read or delete memories owned by User A."""
    from app.services.memory_service import memory_service
    alice_uid = f"alice_mem_{uuid.uuid4().hex[:6]}"
    bob_uid = f"bob_mem_{uuid.uuid4().hex[:6]}"

    # Add memory for Alice
    mem_alice = memory_service.add_memory(
        user_id=alice_uid,
        fact="Enrolled in B.Tech CSE 3rd Year",
        category="academic",
    )
    assert mem_alice is not None

    # Bob attempts to fetch Alice's memories -> should return only Bob's (empty)
    bob_mems = memory_service.get_user_memories(user_id=bob_uid)
    assert not any(m["id"] == mem_alice["id"] for m in bob_mems)

    # Bob attempts to delete Alice's memory -> returns False
    del_res = memory_service.delete_memory(memory_id=mem_alice["id"], user_id=bob_uid)
    assert del_res is False

    # Confirm Alice's memory remains intact
    alice_mems = memory_service.get_user_memories(user_id=alice_uid)
    assert any(m["id"] == mem_alice["id"] for m in alice_mems)

    # Clean up
    memory_service.delete_memory(mem_alice["id"], user_id=alice_uid)


def test_global_search_tenant_isolation_prevents_leakage():
    """Verify global_search strictly filters chats and projects by user_id."""
    alice_uid = f"alice_search_{uuid.uuid4().hex[:6]}"
    bob_uid = f"bob_search_{uuid.uuid4().hex[:6]}"

    # Alice creates a private project and private session
    p_alice = workspace_store.create_project(
        name="Alice Confidential Algorithm Research",
        description="Proprietary neural architecture",
        user_id=alice_uid,
    )
    sid_alice = uuid.uuid4()
    session_store.ensure_session(sid_alice, "Alice Confidential Session Topic", user_id=alice_uid)

    # Bob performs global search for "Confidential"
    bob_results = workspace_store.global_search(query="Confidential", user_id=bob_uid)
    assert not any(p["id"] == p_alice["id"] for p in bob_results["results"]["projects"]), "Bob found Alice's project!"
    assert not any(c["id"] == str(sid_alice) for c in bob_results["results"]["chats"]), "Bob found Alice's session!"

    # Alice performs search and finds her project and session
    alice_results = workspace_store.global_search(query="Confidential", user_id=alice_uid)
    assert any(p["id"] == p_alice["id"] for p in alice_results["results"]["projects"])
    assert any(c["id"] == str(sid_alice) for c in alice_results["results"]["chats"])

    # Clean up
    workspace_store.delete_project(p_alice["id"])
    session_store.delete_session(sid_alice, user_id=alice_uid)

