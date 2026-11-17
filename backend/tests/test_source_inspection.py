"""Tests for Ground-Truth Studio authenticated source-inspection endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.auth_service import auth_service
from app.services.rag.qdrant_vector_store import QdrantVectorStore
from app.services.workspace_store import workspace_store
from app.services.storage import storage_service


@pytest.fixture
def test_users_and_tokens():
    user_a = auth_service.authenticate_or_register_google_user(
        email="alice.evidence@example.com",
        full_name="Alice Evidence",
        google_sub="sub-alice-evidence",
    )
    user_b = auth_service.authenticate_or_register_google_user(
        email="bob.adversary@example.com",
        full_name="Bob Adversary",
        google_sub="sub-bob-adversary",
    )
    return {
        "alice": {"user": user_a["user"], "token": user_a["token"]},
        "bob": {"user": user_b["user"], "token": user_b["token"]},
    }


def test_user_can_inspect_own_passage(test_users_and_tokens):
    """Verify that an authenticated user can inspect a grounded passage from their uploaded document."""
    alice = test_users_and_tokens["alice"]
    client = TestClient(app)

    # 1. Setup a file in workspace_store for Alice
    storage_path = storage_service.save_bytes("syllabus.pdf", b"%PDF-1.4 sample syllabus content")
    file_record = workspace_store.record_library_file(
        filename="syllabus.pdf",
        file_type="PDF",
        file_size=1024,
        storage_path=storage_path,
        chunk_count=1,
        status="indexed",
        user_id=alice["user"]["id"],
    )
    file_id = file_record["id"]

    # 2. Index a chunk into Qdrant user_knowledge_base tagged with Alice's user_id
    store = QdrantVectorStore()
    chunk = {
        "id": "chunk-exam-01",
        "file_id": file_id,
        "doc_name": "syllabus.pdf",
        "page": 4,
        "section": "Grading Policy",
        "breadcrumb": "Course Regulations > Grading Policy",
        "text": "The final examination comprises 40% of the total semester grade.",
        "owner_id": alice["user"]["id"],
    }
    store.index_chunks("user_knowledge_base", [chunk], owner_id=alice["user"]["id"])

    # 3. Alice requests passage inspection
    headers = {"Authorization": f"Bearer {alice['token']}"}
    response = client.get(f"/api/v1/library/files/{file_id}/passages/chunk-exam-01", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["passage_id"] == "chunk-exam-01"
    assert data["file_id"] == file_id
    assert data["page"] == 4
    assert data["doc_name"] == "syllabus.pdf"
    assert "40% of the total semester grade" in data["text"]
    assert data["download_url"] == f"/api/v1/library/files/{file_id}/content"
    assert data["verified_owner"] is True


def test_cross_tenant_passage_inspection_blocked(test_users_and_tokens):
    """Verify that Bob cannot inspect Alice's passage (must return 404 Not Found to prevent enumeration)."""
    alice = test_users_and_tokens["alice"]
    bob = test_users_and_tokens["bob"]
    client = TestClient(app)

    # Alice's file
    storage_path = storage_service.save_bytes("confidential.pdf", b"Confidential financial figures.")
    file_record = workspace_store.record_library_file(
        filename="confidential.pdf",
        file_type="PDF",
        file_size=2048,
        storage_path=storage_path,
        chunk_count=1,
        status="indexed",
        user_id=alice["user"]["id"],
    )
    file_id = file_record["id"]

    store = QdrantVectorStore()
    chunk = {
        "id": "confidential-chunk-99",
        "file_id": file_id,
        "doc_name": "confidential.pdf",
        "page": 1,
        "text": "Secret operational runway: $500,000 remaining.",
        "owner_id": alice["user"]["id"],
    }
    store.index_chunks("user_knowledge_base", [chunk], owner_id=alice["user"]["id"])

    # Bob attempts to inspect Alice's passage
    headers_bob = {"Authorization": f"Bearer {bob['token']}"}
    response = client.get(f"/api/v1/library/files/{file_id}/passages/confidential-chunk-99", headers=headers_bob)
    assert response.status_code == 404
    assert "Passage not found" in response.json()["detail"]


def test_nonexistent_passage_returns_404(test_users_and_tokens):
    """Verify that querying a nonexistent passage ID returns 404."""
    alice = test_users_and_tokens["alice"]
    client = TestClient(app)

    storage_path = storage_service.save_bytes("doc.txt", b"Regular text.")
    file_record = workspace_store.record_library_file(
        filename="doc.txt",
        file_type="TXT",
        file_size=50,
        storage_path=storage_path,
        chunk_count=1,
        status="indexed",
        user_id=alice["user"]["id"],
    )
    file_id = file_record["id"]

    headers = {"Authorization": f"Bearer {alice['token']}"}
    response = client.get(f"/api/v1/library/files/{file_id}/passages/does-not-exist-999", headers=headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Passage not found"
