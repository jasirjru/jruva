"""Limiter keys must require verified authentication or explicit benchmark opt-in."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import Settings
import app.core.security as security
from app.services.auth_service import auth_service


def request_with(headers):
    return Request({"type": "http", "client": ("203.0.113.10", 1000), "headers": headers})


def test_benchmark_defaults_off():
    mode = Settings.model_fields.get("BENCHMARK_MODE")
    assert mode is not None and mode.default is False
    assert Settings.model_fields["BENCHMARK_BYPASS_SECRET"].default == ""


def test_production_compose_disables_benchmark_mode():
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.prod.yml").read_text()
    assert "BENCHMARK_MODE=false" in compose
    assert "BENCHMARK_BYPASS_SECRET=${BENCHMARK_BYPASS_SECRET:-}" in compose


@pytest.mark.parametrize("mode,secret,supplied", [
    (False, "review-secret", "review-secret"),
    (True, "", ""),
    (True, "review-secret", "wrong-secret"),
])
def test_bypass_requires_mode_and_nonempty_matching_secret(monkeypatch, mode, secret, supplied):
    monkeypatch.setattr(security, "settings", SimpleNamespace(BENCHMARK_MODE=mode, BENCHMARK_BYPASS_SECRET=secret))
    request = request_with([(b"x-benchmark-bypass", supplied.encode())])
    assert security.get_user_or_ip_identifier(request) == "ip:203.0.113.10"


def test_explicit_benchmark_mode_allows_configured_bypass(monkeypatch):
    monkeypatch.setattr(security, "settings", SimpleNamespace(BENCHMARK_MODE=True, BENCHMARK_BYPASS_SECRET="review-secret"))
    request = request_with([(b"x-benchmark-bypass", b"review-secret")])
    key = security.get_user_or_ip_identifier(request)
    assert key.startswith("bypass:")
    assert key != security.get_user_or_ip_identifier(request)


@pytest.mark.parametrize("cookie", [False, True])
def test_verified_jwt_identity_used_for_limits(monkeypatch, cookie):
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda uid: {"id": uid})
    token = auth_service.create_jwt_token({"sub": "verified-user"})
    headers = [(b"cookie", f"access_token={token}".encode())] if cookie else [(b"authorization", f"Bearer {token}".encode())]
    headers.append((b"x-user-id", b"spoofed-user"))
    assert security.get_user_or_ip_identifier(request_with(headers)) == "user:verified-user"


def test_rotating_unverified_headers_cannot_evade_http_limit(monkeypatch):
    monkeypatch.setattr(security, "settings", SimpleNamespace(BENCHMARK_MODE=False, BENCHMARK_BYPASS_SECRET="review-secret"))
    application = FastAPI()
    limiter = Limiter(key_func=security.get_user_or_ip_identifier)
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @application.get("/probe")
    @limiter.limit("2/minute")
    def probe(request: Request):
        return {"ok": True}

    with TestClient(application) as client:
        statuses = [client.get("/probe", headers={
            "X-User-ID": f"user-{i}", "X-Session-ID": f"session-{i}",
            "X-Benchmark-Bypass": "review-secret",
        }).status_code for i in range(3)]
    assert statuses == [200, 200, 429]
