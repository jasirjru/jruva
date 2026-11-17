"""CSRF Origin Middleware Verification Tests.

Tests that the CsrfOriginMiddleware correctly blocks, allows, and handles
edge cases for cross-origin mutating requests.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def prod_app():
    """Create a production-mode app with CSRF middleware active."""
    import os
    orig_env = os.environ.get("ENVIRONMENT")
    orig_origins = os.environ.get("ALLOWED_ORIGINS")
    orig_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    orig_jwt = os.environ.get("JWT_SECRET_KEY")
    orig_db = os.environ.get("DATABASE_BACKEND")
    orig_db_url = os.environ.get("DATABASE_URL")

    import importlib
    import app.core.config
    import app.main

    try:
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DATABASE_BACKEND"] = "postgres"
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
        os.environ["ALLOW_TEST_LOOPBACK"] = "1"
        os.environ["ALLOWED_ORIGINS"] = "https://my-app.vercel.app,https://staging.vercel.app"
        os.environ["GOOGLE_CLIENT_ID"] = "test-prod-audience.apps.googleusercontent.com"
        os.environ["JWT_SECRET_KEY"] = "super-secret-jwt-key-with-at-least-32-characters-test"

        # Force reimport to pick up production env
        importlib.reload(app.core.config)
        importlib.reload(app.main)

        client = TestClient(app.main.create_application())
        yield client
    finally:
        os.environ.pop("ALLOW_TEST_LOOPBACK", None)
        # Restore dev mode
        if orig_env is not None:
            os.environ["ENVIRONMENT"] = orig_env
        else:
            os.environ["ENVIRONMENT"] = "development"
        if orig_db is not None:
            os.environ["DATABASE_BACKEND"] = orig_db
        else:
            os.environ.pop("DATABASE_BACKEND", None)
        if orig_db_url is not None:
            os.environ["DATABASE_URL"] = orig_db_url
        else:
            os.environ.pop("DATABASE_URL", None)
        if orig_origins is not None:
            os.environ["ALLOWED_ORIGINS"] = orig_origins
        else:
            os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3121,http://127.0.0.1:3121"
        if orig_client_id is not None:
            os.environ["GOOGLE_CLIENT_ID"] = orig_client_id
        else:
            os.environ.pop("GOOGLE_CLIENT_ID", None)
        if orig_jwt is not None:
            os.environ["JWT_SECRET_KEY"] = orig_jwt
        else:
            os.environ.pop("JWT_SECRET_KEY", None)

        importlib.reload(app.core.config)
        importlib.reload(app.main)


@pytest.fixture
def dev_app():
    """Create a development-mode app (CSRF middleware should be inactive)."""
    import os
    os.environ["ENVIRONMENT"] = "development"

    import importlib
    import app.core.config
    importlib.reload(app.core.config)
    import app.main
    importlib.reload(app.main)

    client = TestClient(app.main.create_application())
    yield client


class TestCsrfMiddleware:
    """Verify CSRF Origin middleware behavior."""

    def test_spoofed_origin_gets_403(self, prod_app):
        """A POST from an attacker origin must be rejected with 403."""
        resp = prod_app.post(
            "/api/v1/chat",
            json={"message": "test"},
            headers={"Origin": "https://evil-attacker.com"},
        )
        assert resp.status_code == 403
        assert "CSRF" in resp.json()["detail"]
        assert "evil-attacker.com" in resp.json()["detail"]

    def test_missing_origin_and_referer_gets_403(self, prod_app):
        """A POST with NO Origin and NO Referer must be rejected (deliberate safe default)."""
        resp = prod_app.post(
            "/api/v1/chat",
            json={"message": "test"},
            # No Origin or Referer headers
        )
        assert resp.status_code == 403
        assert "Origin header is required" in resp.json()["detail"]

    def test_valid_origin_passes(self, prod_app):
        """A POST from an allowed origin should pass CSRF check (may fail on auth/other, but not 403 CSRF)."""
        resp = prod_app.post(
            "/api/v1/chat",
            json={"message": "test"},
            headers={"Origin": "https://my-app.vercel.app"},
        )
        # Should NOT be 403 CSRF. Might be 401 (no auth) or 422 (schema) or 200 — but not CSRF blocked.
        assert resp.status_code != 403 or "CSRF" not in resp.json().get("detail", "")

    def test_referer_fallback_passes(self, prod_app):
        """A POST with no Origin but a valid Referer should pass CSRF check."""
        resp = prod_app.post(
            "/api/v1/chat",
            json={"message": "test"},
            headers={"Referer": "https://my-app.vercel.app/chat?q=hello"},
        )
        assert resp.status_code != 403 or "CSRF" not in resp.json().get("detail", "")

    def test_referer_fallback_bad_origin_gets_403(self, prod_app):
        """A POST with no Origin but a bad Referer should be rejected."""
        resp = prod_app.post(
            "/api/v1/chat",
            json={"message": "test"},
            headers={"Referer": "https://phishing-site.com/steal"},
        )
        assert resp.status_code == 403
        assert "CSRF" in resp.json()["detail"]

    def test_get_request_always_passes(self, prod_app):
        """GET requests should always pass CSRF check (safe method)."""
        resp = prod_app.get(
            "/api/v1/health",
            headers={"Origin": "https://evil-attacker.com"},
        )
        # Should NOT be 403 regardless of Origin
        assert resp.status_code != 403

    def test_google_auth_requires_origin(self, prod_app):
        """The JSON Google login flow requires the same origin protection as mutations."""
        resp = prod_app.post(
            "/api/v1/auth/google",
            json={"credential": "fake-token"},
            # Deliberately no Origin header — this should NOT trigger CSRF
        )
        # Should fail on auth validation (400), not CSRF (403)
        assert resp.status_code == 403

    def test_dev_mode_skips_csrf(self, dev_app):
        """In development mode, CSRF middleware should not block anything."""
        resp = dev_app.post(
            "/api/v1/chat",
            json={"message": "test"},
            headers={"Origin": "https://evil-attacker.com"},
        )
        # Should NOT be 403 in dev mode
        assert resp.status_code != 403
