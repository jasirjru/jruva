"""Regression tests for Staging Smoke Runner Database Connectivity Verification.

Ensures:
1. HTTP 404 is strictly rejected as a FAILURE (false-positive bug eliminated).
2. HTTP 500, 503, connection errors are rejected as FAILURES.
3. HTTP 200 requires positive proof in payload.
4. Production targets are strictly rejected by the direct SQL safety gate.
5. Direct SQL verification passes on positive PostgreSQL query confirmation.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path so scripts can be imported whether pytest runs from backend/ or repo root
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import httpx  # noqa: E402

from scripts.staging_smoke_runner import (  # noqa: E402
    check_database_liveness,
    check_direct_database_connectivity,
    check_input_validation_contract,
    check_protected_resource_auth,
)


def test_http_404_rejected_as_failure():
    """Verify that HTTP 404 from /api/v1/health/db is NOT counted as success."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 404
    mock_resp.text = '{"detail": "Not Found"}'
    client.get.return_value = mock_resp

    res = check_database_liveness(client, db_url=None)
    assert res["passed"] is False, "HTTP 404 must NOT pass database connectivity check!"
    assert res["status"] == 404
    assert "404 is NOT positive proof" in res["detail"]


def test_http_500_rejected_as_failure():
    """Verify that HTTP 500 is counted as failure."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    client.get.return_value = mock_resp

    res = check_database_liveness(client, db_url=None)
    assert res["passed"] is False
    assert res["status"] == 500


def test_http_503_rejected_as_failure():
    """Verify that HTTP 503 is counted as failure."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    client.get.return_value = mock_resp

    res = check_database_liveness(client, db_url=None)
    assert res["passed"] is False
    assert res["status"] == 503


def test_network_error_rejected_as_failure():
    """Verify that network exceptions are counted as failure."""
    client = MagicMock(spec=httpx.Client)
    client.get.side_effect = httpx.ConnectError("Connection refused")

    res = check_database_liveness(client, db_url=None)
    assert res["passed"] is False
    assert res["status"] == "ERROR"


def test_http_200_positive_proof_accepted():
    """Verify that HTTP 200 with positive confirmation passes."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "database": "connected"}
    client.get.return_value = mock_resp

    res = check_database_liveness(client, db_url=None)
    assert res["passed"] is True
    assert res["status"] == 200
    assert "positive HTTP 200 proof" in res["detail"]


def test_http_200_without_proof_rejected():
    """Verify that HTTP 200 with empty or unconfirmed payload is rejected."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"unrelated": "payload"}
    client.get.return_value = mock_resp

    res = check_database_liveness(client, db_url=None)
    assert res["passed"] is False
    assert "missing positive database confirmation" in res["detail"]


def test_direct_sql_rejects_production_target():
    """Verify that staging direct SQL check refuses production Supabase URL."""
    prod_url = "postgresql://postgres:pass@db.prodprojectref000000.supabase.co:5432/postgres"
    res = check_direct_database_connectivity(prod_url)
    assert res["passed"] is False
    assert res["status"] == "SAFETY_VIOLATION"
    assert "matches production reference" in res["detail"]


def test_direct_sql_positive_proof():
    """Verify that successful SQL query confirms connectivity."""
    staging_url = "postgresql://postgres:pass@db.stagingprojectref000.supabase.co:5432/postgres"

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ["008_doc_storage_daily_quotas"]
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
        res = check_direct_database_connectivity(staging_url)
        assert res["passed"] is True
        assert res["status"] == 200
        assert "008_doc_storage_daily_quotas" in res["detail"]


def test_gallery_auth_200_fails_as_security_violation():
    """Verify that HTTP 200 without credentials strictly FAILS the auth gate."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"images": [{"id": "leak"}]}
    client.get.return_value = mock_resp

    res = check_protected_resource_auth(client)
    assert res["passed"] is False, "HTTP 200 on protected resource MUST NOT pass auth gate!"
    assert res["status"] == 200
    assert "SECURITY VIOLATION" in res["detail"]


def test_gallery_auth_401_passes():
    """Verify that HTTP 401 correctly passes auth gate."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    client.get.return_value = mock_resp

    res = check_protected_resource_auth(client)
    assert res["passed"] is True
    assert res["status"] == 401


def test_gallery_auth_403_passes():
    """Verify that HTTP 403 correctly passes auth gate."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 403
    client.get.return_value = mock_resp

    res = check_protected_resource_auth(client)
    assert res["passed"] is True
    assert res["status"] == 403


def test_input_validation_422_with_detail_passes():
    """Verify that HTTP 422 with structured detail contract passes validation check."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": [{"loc": ["body", "message"], "msg": "Field required"}]}
    client.post.return_value = mock_resp

    res = check_input_validation_contract(client)
    assert res["passed"] is True
    assert res["status"] == 422
    assert "structured details" in res["detail"]


def test_input_validation_503_fails_as_not_validation_proof():
    """Verify that HTTP 503 is rejected and NOT counted as validation success."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 503
    mock_resp.text = '{"error": "maintenance_mode"}'
    client.post.return_value = mock_resp

    res = check_input_validation_contract(client)
    assert res["passed"] is False, "HTTP 503 must NOT be counted as input validation proof!"
    assert res["status"] == 503
    assert "NOT input-validation proof" in res["detail"]


def test_input_validation_200_fails_as_validation_bypassed():
    """Verify that HTTP 200 on invalid input is rejected as validation bypass."""
    client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    client.post.return_value = mock_resp

    res = check_input_validation_contract(client)
    assert res["passed"] is False, "HTTP 200 on invalid payload must fail validation gate!"
    assert "SECURITY VIOLATION" in res["detail"]


def test_input_validation_sends_production_origin_for_csrf():
    """Production validation must reach Pydantic rather than fail CSRF first."""
    client = MagicMock(spec=httpx.Client)
    client.base_url = httpx.URL("https://www.jruva.in")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": [{"loc": ["body", "message"], "msg": "String should have at least 1 character"}]}
    client.post.return_value = mock_resp

    result = check_input_validation_contract(client)

    assert result["passed"] is True
    client.post.assert_called_once_with(
        "/api/v1/chat",
        json={"message": ""},
        headers={"Origin": "https://www.jruva.in"},
    )


def test_input_validation_preserves_localhost_port_in_origin():
    """Local/staging smoke checks retain the actual target origin and port."""
    client = MagicMock(spec=httpx.Client)
    client.base_url = httpx.URL("http://127.0.0.1:8001")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": [{"loc": ["body", "message"], "msg": "Invalid prompt"}]}
    client.post.return_value = mock_resp

    result = check_input_validation_contract(client)

    assert result["passed"] is True
    assert client.post.call_args.kwargs["headers"] == {"Origin": "http://127.0.0.1:8001"}
