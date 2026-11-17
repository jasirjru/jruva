"""Offline Supabase adapter contracts; every HTTP request uses MockTransport."""

import hashlib
from types import SimpleNamespace

import httpx
import pytest

from app.services.storage import base
from app.services.storage.supabase import SupabaseStorageBackend


BUCKET = "synthetic-private"
FILE_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def adapter_factory():
    clients = []

    def build(handler):
        client = httpx.Client(
            base_url="https://synthetic.invalid/storage/v1",
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        return SupabaseStorageBackend(
            supabase_url="https://synthetic.invalid", service_key="synthetic-secret",
            bucket_name=BUCKET, http_client=client,
        )

    yield build
    for client in clients:
        client.close()


def test_settings_credentials_work_without_process_environment(monkeypatch):
    from app.core import config

    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY", "STORAGE_BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "settings", SimpleNamespace(
        SUPABASE_URL="https://synthetic.invalid/", SUPABASE_SERVICE_ROLE_KEY="synthetic-key",
        STORAGE_BUCKET_NAME=BUCKET,
    ))
    adapter = SupabaseStorageBackend()
    assert adapter.is_configured
    assert adapter.supabase_url == "https://synthetic.invalid"
    assert adapter.service_key == "synthetic-key"


@pytest.mark.parametrize("path", [
    f"supabase://{BUCKET}/../other-bucket/image.png", "../other-bucket/image.png",
    f"supabase://other-bucket/image.png", "https://evil.invalid/image.png", "/image.png",
    "folder/./image.png", "folder//image.png", "folder/../image.png", "folder\\image.png",
    "folder/%2e%2e/image.png", "folder/%252e%252e/image.png", "image.png?token=secret",
    "image.png#fragment", "image.png\x00", "image.png\n", "", ".", "..",
])
@pytest.mark.parametrize("operation", ["get", "metadata", "exists", "delete"])
def test_invalid_references_never_issue_privileged_requests(adapter_factory, path, operation):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"size": 8, "content_type": "image/png"})

    adapter = adapter_factory(handler)
    with pytest.raises((ValueError, PermissionError)):
        getattr(adapter, operation)(path)
    assert requests == []


@pytest.mark.parametrize("path", [f"supabase://{BUCKET}/folder/image.png", "folder/image.png"])
def test_safe_legacy_keys_remain_readable(adapter_factory, path):
    def handler(request):
        assert request.url.path == f"/storage/v1/object/authenticated/{BUCKET}/folder/image.png"
        return httpx.Response(200, content=b"synthetic-image")

    assert adapter_factory(handler).get(path) == b"synthetic-image"


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_exists_does_not_disguise_storage_failure_as_absence(adapter_factory, status):
    adapter = adapter_factory(lambda request: httpx.Response(status, text="synthetic-secret"))
    with pytest.raises(IOError) as error:
        adapter.exists("image.png")
    assert "synthetic-secret" not in str(error.value)


def test_exists_reports_confirmed_absence(adapter_factory):
    assert not adapter_factory(lambda request: httpx.Response(404)).exists("image.png")


@pytest.mark.parametrize("operation", ["get", "metadata", "delete", "put"])
def test_response_bodies_are_not_disclosed_by_errors(adapter_factory, operation):
    adapter = adapter_factory(lambda request: httpx.Response(503, text="synthetic-secret and private document"))
    with pytest.raises(IOError) as error:
        if operation == "put":
            adapter.put("image.png", b"synthetic-image")
        else:
            getattr(adapter, operation)("image.png")
    assert "synthetic-secret" not in str(error.value)
    assert "private document" not in str(error.value)
    assert "503" in str(error.value)


@pytest.mark.parametrize("response", [
    {"size": 8, "content_type": "image/png", "last_modified": "2026-10-09", "created_at": "2026-10-08"},
    {"size": 8, "mimetype": "image/png", "updated_at": "2026-10-09", "created_at": "2026-10-08"},
])
def test_official_and_legacy_metadata_fields(adapter_factory, response):
    metadata = adapter_factory(lambda request: httpx.Response(200, json=response)).metadata("image.png")
    assert metadata["size_bytes"] == 8
    assert metadata["content_type"] == "image/png"
    assert metadata["modified_at"] == "2026-10-09"


def test_upload_does_not_overwrite_existing_object(adapter_factory):
    objects = {}

    def handler(request):
        if request.url.path in objects and request.headers.get("x-upsert") != "true":
            return httpx.Response(409)
        objects[request.url.path] = request.content
        return httpx.Response(200)

    adapter = adapter_factory(handler)
    with base.image_write_scope(FILE_ID):
        adapter.put("generated.png", b"first-image")
        with pytest.raises(IOError):
            adapter.put("generated.png", b"replacement-image")
    assert list(objects.values()) == [b"first-image"]


def test_document_scope_retains_deterministic_tenant_reference(adapter_factory):
    owner = "synthetic-owner"
    expected = f"supabase://{BUCKET}/documents/{hashlib.sha256(owner.encode()).hexdigest()}/{FILE_ID}_report.pdf"
    received = []

    def handler(request):
        received.append(request)
        return httpx.Response(201)

    adapter = adapter_factory(handler)
    assert adapter.reserved_document_path(owner, FILE_ID, "report.pdf") == expected
    with base.document_write_scope(owner, FILE_ID, "report.pdf"):
        assert adapter.put("report.pdf", b"synthetic-pdf") == expected
    assert received[0].headers["content-type"] == "application/pdf"
    assert received[0].headers["x-upsert"] == "false"


@pytest.mark.parametrize("filename,mime,kind", [
    ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "mime"),
    ("script.html", "text/html", "mime"),
    ("report.pdf", "application/pdf", "size"),
])
def test_bucket_policy_rejects_upload_before_request(adapter_factory, monkeypatch, filename, mime, kind):
    monkeypatch.setenv("STORAGE_MAX_OBJECT_BYTES", "8")
    monkeypatch.delenv("STORAGE_ALLOWED_MIME_TYPES", raising=False)
    requests = []
    adapter = adapter_factory(lambda request: (requests.append(request) or httpx.Response(201)))
    content = b"ninebytes" if kind == "size" else b"synthetic"
    if kind == "mime":
        monkeypatch.setenv("STORAGE_MAX_OBJECT_BYTES", "100")
    with pytest.raises(base.StorageAdmissionError) as error:
        adapter.put(filename, content, content_type=mime)
    assert error.value.kind == kind
    assert requests == []


def test_policy_validation_is_available_before_creating_write_receipt(adapter_factory, monkeypatch):
    monkeypatch.delenv("STORAGE_ALLOWED_MIME_TYPES", raising=False)
    adapter = adapter_factory(lambda request: pytest.fail("Validation must not access storage"))
    with pytest.raises(base.StorageAdmissionError) as error:
        adapter.validate_upload("report.docx", b"synthetic", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert error.value.kind == "mime"


def test_operator_can_explicitly_match_bucket_docx_policy(adapter_factory, monkeypatch):
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    monkeypatch.setenv("STORAGE_ALLOWED_MIME_TYPES", f"application/pdf, {mime}")
    monkeypatch.setenv("STORAGE_MAX_OBJECT_BYTES", "8")
    adapter = adapter_factory(lambda request: httpx.Response(201))
    assert adapter.put("report.docx", b"12345678", content_type=mime).startswith(f"supabase://{BUCKET}/")


def test_transport_diagnostics_are_sanitized(adapter_factory):
    def handler(request):
        raise httpx.ConnectError("synthetic-secret and private document", request=request)

    with pytest.raises(IOError) as error:
        adapter_factory(handler).get("image.png")
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize("body", [b"not-json synthetic-secret", b"[1,2,3]"])
def test_malformed_metadata_is_a_sanitized_storage_failure(adapter_factory, body):
    with pytest.raises(IOError) as error:
        adapter_factory(lambda request: httpx.Response(200, content=body)).metadata("image.png")
    assert "synthetic-secret" not in str(error.value)
