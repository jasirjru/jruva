from tests.private_accounts import private_account_headers
"""Unit and Integration Tests for Durable Storage Service and Tenant Access Boundaries."""

import io
import os
import shutil
import tempfile
import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage.local import LocalStorageBackend, sanitize_filename
from app.services.storage.supabase import SupabaseStorageBackend, sanitize_storage_key
from app.services.storage import reset_storage_backend_for_testing
from app.services.workspace_store import workspace_store


@pytest.fixture
def temp_storage_dir():
    temp_dir = tempfile.mkdtemp(prefix="jruva_test_storage_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def local_storage(temp_storage_dir):
    backend = LocalStorageBackend(base_dir=temp_storage_dir)
    reset_storage_backend_for_testing(backend)
    yield backend
    reset_storage_backend_for_testing(None)


def test_sanitize_filename():
    assert sanitize_filename("safe_report.pdf") == "safe_report.pdf"
    # Path traversal patterns sanitized
    assert ".." not in sanitize_filename("../../../etc/passwd")
    assert "passwd" in sanitize_filename("../../../etc/passwd")
    # Null bytes stripped
    assert "\x00" not in sanitize_filename("malicious\x00file.exe")


def test_local_storage_crud(local_storage):
    test_content = b"Jruva AI Research Paper Content 2026"
    filename = "research.pdf"

    # Put
    path = local_storage.put(filename, test_content, content_type="application/pdf")
    assert local_storage.exists(path)

    # Get
    retrieved = local_storage.get(path)
    assert retrieved == test_content

    # Metadata
    meta = local_storage.metadata(path)
    assert meta["size_bytes"] == len(test_content)
    assert "pdf" in meta["content_type"]

    # Delete
    deleted = local_storage.delete(path)
    assert deleted is True
    assert not local_storage.exists(path)

    with pytest.raises(FileNotFoundError):
        local_storage.get(path)


def test_local_storage_path_traversal_blocked(local_storage, temp_storage_dir):
    """Verify that attempting to access files outside base_dir raises PermissionError."""
    outside_file = os.path.abspath(os.path.join(temp_storage_dir, "..", "secret.txt"))
    with pytest.raises(PermissionError):
        local_storage.get(outside_file)

    with pytest.raises(PermissionError):
        local_storage.delete(outside_file)


def test_supabase_storage_mock_transport():
    """Verify Supabase Storage REST client protocol using mock HTTP transport."""
    mock_responses = {
        "/object/jruva-documents/": httpx.Response(200, json={"Key": "uploaded"}),
        "/object/authenticated/jruva-documents/test_file.pdf": httpx.Response(200, content=b"Supabase Content"),
        "/object/info/authenticated/jruva-documents/test_file.pdf": httpx.Response(
            200, json={"size": 16, "mimetype": "application/pdf", "created_at": "2026-10-08T00:00:00Z"}
        ),
        "/object/jruva-documents": httpx.Response(200, json=[{"name": "test_file.pdf"}]),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.replace("/storage/v1", "")
        for pattern, resp in mock_responses.items():
            if pattern in path:
                return resp
        return httpx.Response(404, json={"error": "Not Found"})

    client = httpx.Client(base_url="https://xyz.supabase.co/storage/v1", transport=httpx.MockTransport(handler))
    backend = SupabaseStorageBackend(
        supabase_url="https://xyz.supabase.co",
        service_key="mock_secret_key",
        bucket_name="jruva-documents",
        http_client=client,
    )

    # Put
    uri = backend.put("test_file.pdf", b"Supabase Content", content_type="application/pdf")
    assert uri.startswith("supabase://jruva-documents/")

    # Get
    data = backend.get("supabase://jruva-documents/test_file.pdf")
    assert data == b"Supabase Content"

    # Metadata
    meta = backend.metadata("supabase://jruva-documents/test_file.pdf")
    assert meta["size_bytes"] == 16
    assert meta["content_type"] == "application/pdf"

    # Delete
    deleted = backend.delete("supabase://jruva-documents/test_file.pdf")
    assert deleted is True


def test_tenant_isolation_on_library_file_content(local_storage):
    """User B must be strictly forbidden from downloading or deleting User A's uploaded documents."""
    client = TestClient(app)

    # 1. User A uploads a document
    file_bytes = b"CONFIDENTIAL FINANCIAL STRATEGY FOR TENANT A"
    upload_resp = client.post(
        "/api/v1/library/upload",
        files={"file": ("strategy.pdf", io.BytesIO(file_bytes), "application/pdf")},
        headers=private_account_headers("tenant_user_a"),
    )
    assert upload_resp.status_code == 200
    file_id = upload_resp.json()["file"]["id"]

    # 2. User A can download content
    user_a_get = client.get(
        f"/api/v1/library/files/{file_id}/content",
        headers=private_account_headers("tenant_user_a"),
    )
    assert user_a_get.status_code == 200
    assert user_a_get.content == file_bytes

    # 3. User B attempts to download User A's file content -> 403 Forbidden
    user_b_get = client.get(
        f"/api/v1/library/files/{file_id}/content",
        headers=private_account_headers("tenant_user_b"),
    )
    assert user_b_get.status_code == 403
    assert "Forbidden" in user_b_get.json()["detail"]

    # 4. User B attempts to delete User A's file -> 403 Forbidden
    user_b_del = client.delete(
        f"/api/v1/library/files/{file_id}",
        headers=private_account_headers("tenant_user_b"),
    )
    assert user_b_del.status_code == 403
    assert "Forbidden" in user_b_del.json()["detail"]

    # 5. User A deletes their own file -> 200 OK
    user_a_del = client.delete(
        f"/api/v1/library/files/{file_id}",
        headers=private_account_headers("tenant_user_a"),
    )
    assert user_a_del.status_code == 200

    # 6. Subsequent download is 404
    after_del = client.get(
        f"/api/v1/library/files/{file_id}/content",
        headers=private_account_headers("tenant_user_a"),
    )
    assert after_del.status_code == 404


def test_document_uploads_disabled_when_flag_false(monkeypatch, local_storage):
    """When ENABLE_DOCUMENT_UPLOADS is false, upload endpoints must return 503."""
    monkeypatch.setenv("ENABLE_DOCUMENT_UPLOADS", "false")
    client = TestClient(app)

    file_bytes = b"TEST CONTENT"
    resp = client.post(
        "/api/v1/library/upload",
        files={"file": ("test.pdf", io.BytesIO(file_bytes), "application/pdf")},
        headers=private_account_headers("any_user"),
    )
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()

    # Also test JSON endpoint
    resp_json = client.post(
        "/api/v1/library/upload-json",
        json={"filename": "test.txt", "data": "VEVTVCBDT05URU5U"},
        headers=private_account_headers("any_user"),
    )
    assert resp_json.status_code == 503
    assert "disabled" in resp_json.json()["detail"].lower()


def test_document_uploads_disabled_in_production_with_local_storage(monkeypatch, local_storage):
    """In production with local storage, document uploads must fail-closed to 503."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.delenv("ENABLE_DOCUMENT_UPLOADS", raising=False)
    client = TestClient(app)

    file_bytes = b"TEST CONTENT"
    resp = client.post(
        "/api/v1/library/upload",
        files={"file": ("test.pdf", io.BytesIO(file_bytes), "application/pdf")},
        headers=private_account_headers("any_user"),
    )
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()
