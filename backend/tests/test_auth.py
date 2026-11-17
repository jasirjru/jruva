"""Unit and Integration Tests for User Authentication and Profile Endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_removed_endpoints_return_404():
    """Verify that local /register and /login endpoints no longer exist."""
    reg_resp = client.post("/api/v1/auth/register", json={"email": "a@b.com", "password": "123"})
    assert reg_resp.status_code in (404, 405)

    login_resp = client.post("/api/v1/auth/login", json={"identifier": "a@b.com", "password": "123"})
    assert login_resp.status_code in (404, 405)


def test_google_auth_endpoint_flow(monkeypatch):
    """Verify Google OAuth token verification and account registration/login."""

    google_email = f"google_user_{id([])}@gmail.com"
    fake_payload = {
        "email": google_email,
        "email_verified": "true",
        "name": "Google Tester",
        "sub": "1234567890987654321",
        "picture": "https://example.com/avatar.jpg"
    }

    # Account/cookie tests consume already cryptographically verified claims.
    # Real RSA verification and adversarial claims are tested independently.
    fake_payload["sub"] = google_email
    monkeypatch.setattr("app.api.v1.endpoints.auth.verify_google_identity", lambda credential, audience: fake_payload)

    # 1. First time Google login -> auto registration
    resp = client.post("/api/v1/auth/google", json={"credential": "mock_valid_id_token"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == google_email
    assert data["user"]["full_name"] == "Google Tester"

    # 2. Second time Google login -> login existing
    resp2 = client.post("/api/v1/auth/google", json={"credential": "mock_valid_id_token"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["user"]["id"] == data["user"]["id"]


def test_cookie_auth_flow(monkeypatch):
    """Verify that Google auth sets httpOnly cookie, /me accepts cookie, and /logout clears it."""

    google_email = f"cookie_user_{id([])}@gmail.com"
    fake_payload = {
        "email": google_email,
        "email_verified": "true",
        "name": "Cookie Tester",
        "sub": "9876543210123456789",
        "picture": "https://example.com/avatar2.jpg"
    }

    # Account/cookie tests consume already cryptographically verified claims.
    # Real RSA verification and adversarial claims are tested independently.
    fake_payload["sub"] = google_email
    monkeypatch.setattr("app.api.v1.endpoints.auth.verify_google_identity", lambda credential, audience: fake_payload)

    test_client = TestClient(app)
    login_resp = test_client.post("/api/v1/auth/google", json={"credential": "mock_token"})
    assert login_resp.status_code == 200
    assert "access_token" in login_resp.cookies

    # Access /me using cookie automatically handled by TestClient session
    me_resp = test_client.get("/api/v1/auth/me")
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == google_email

    # Logout clears cookie
    logout_resp = test_client.post("/api/v1/auth/logout")
    assert logout_resp.status_code == 200
    assert logout_resp.json()["status"] == "success"


def test_delete_account_endpoint(monkeypatch):
    """Verify DELETE /api/v1/auth/me permanently deletes the authenticated user."""

    google_email = f"del_user_{id([])}@gmail.com"
    fake_payload = {
        "email": google_email,
        "email_verified": "true",
        "name": "Delete Target",
        "sub": "5555555555555555555",
        "picture": "https://example.com/del.jpg"
    }

    # Account/cookie tests consume already cryptographically verified claims.
    # Real RSA verification and adversarial claims are tested independently.
    fake_payload["sub"] = google_email
    monkeypatch.setattr("app.api.v1.endpoints.auth.verify_google_identity", lambda credential, audience: fake_payload)

    test_client = TestClient(app)
    login_resp = test_client.post("/api/v1/auth/google", json={"credential": "mock_token"})
    assert login_resp.status_code == 200

    del_resp = test_client.delete("/api/v1/auth/me")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # Profile should now return 401 Unauthorized
    after_resp = test_client.get("/api/v1/auth/me")
    assert after_resp.status_code == 401


