"""Library uploads that hit a quota must say so, not ask the user to "recover" an upload.

Before the fix, DocumentQuotaError fell into the generic handler and returned
503 "Document upload could not be confirmed ... may require recovery".
"""

import uuid
from unittest.mock import patch

import pytest
from app.main import app
from app.services.storage.document_quota import DocumentQuotaError
from fastapi.testclient import TestClient

from tests.private_accounts import private_account_headers

client = TestClient(app)


@pytest.mark.parametrize(
    "code,expected_status",
    [
        ("user_file_count_exceeded", 409),
        ("user_storage_quota_exceeded", 413),
        ("object_size_exceeded", 413),
        ("project_storage_quota_exceeded", 503),
    ],
)
def test_quota_errors_map_to_specific_statuses(code, expected_status):
    message = f"quota message for {code}"
    user_id = f"user_quota_{uuid.uuid4().hex[:8]}"
    with patch(
        "app.services.storage.document_quota.reserve_document_upload",
        side_effect=DocumentQuotaError(code, message),
    ):
        res = client.post(
            "/api/v1/library/upload",
            headers=private_account_headers(user_id),
            files={"file": ("notes.txt", b"Plain readable notes.", "text/plain")},
        )
    assert res.status_code == expected_status
    assert res.json()["detail"] == message
    assert "recovery" not in res.json()["detail"].lower()


@pytest.mark.parametrize("enabled", [True, False])
def test_file_list_reports_upload_policy(enabled):
    """The Library disables its upload control when the server will refuse uploads."""
    from app.core.config import settings

    user_id = f"user_policy_{uuid.uuid4().hex[:8]}"
    with patch.object(type(settings), "effective_enable_document_uploads", property(lambda self: enabled)):
        res = client.get("/api/v1/library/files", headers=private_account_headers(user_id))
    assert res.status_code == 200
    body = res.json()
    assert body["uploads_enabled"] is enabled
    assert body["max_file_bytes"] == 15 * 1024 * 1024
    assert body["files"] == []
