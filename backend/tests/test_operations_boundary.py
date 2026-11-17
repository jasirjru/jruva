"""Synthetic HTTP verification of trusted identity and operational privacy boundaries."""

from unittest.mock import Mock

import pytest
from app.core import config
from app.main import create_application
from app.services.auth_service import auth_service
from app.services.cache.response_cache import response_cache
from app.services.telemetry.telemetry_service import RequestTrace, telemetry_service
from fastapi.testclient import TestClient

READ_ROUTES = [
    "/api/v1/telemetry/metrics",
    "/api/v1/telemetry/traces",
    "/metrics",
    "/api/v1/metrics",
]


@pytest.fixture
def identities(monkeypatch):
    # Only the account repository is synthetic: JWT signing and verification are real.
    users = {
        "ops-admin": {"id": "ops-admin", "role": "admin"},
        "ordinary": {"id": "ordinary", "role": "student"},
    }
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda uid: users.get(uid))
    monkeypatch.setattr(config.settings, "ENVIRONMENT", "test")
    monkeypatch.setattr(config.settings, "ALLOWED_ORIGINS", "https://www.jruva.in")
    yield users


@pytest.fixture
def client(identities):
    return TestClient(create_application(), raise_server_exceptions=False)


def bearer(uid="ops-admin", **claims):
    return {
        "Authorization": "Bearer "
        + auth_service.create_jwt_token({"sub": uid, **claims})
    }


@pytest.mark.parametrize("path", READ_ROUTES)
def test_anonymous_operations_access_denied(client, path):
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", READ_ROUTES)
def test_regular_user_cannot_forge_admin_role(client, path):
    headers = {
        **bearer("ordinary", role="admin"),
        "X-Role": "admin",
        "X-Admin": "true",
        "X-Client-Id": "ops-admin",
    }
    assert client.get(path, headers=headers).status_code == 403


@pytest.mark.parametrize("path", READ_ROUTES)
def test_verified_server_admin_can_read(client, path):
    response = client.get(path, headers=bearer(role="student"))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "token",
    [
        "not-a-token",
        "Bearer admin",
        "",
        "eyJhbGciOiJub25lIn0.eyJzdWIiOiJvcHMtYWRtaW4ifQ.",
    ],
)
def test_invalid_or_unsigned_token_denied(client, token):
    assert (
        client.get(
            READ_ROUTES[0], headers={"Authorization": "Bearer " + token}
        ).status_code
        == 401
    )


def test_expired_and_tampered_tokens_denied(client):
    token = auth_service.create_jwt_token({"sub": "ops-admin"}, expires_in_seconds=-1)
    assert (
        client.get(
            READ_ROUTES[0], headers={"Authorization": "Bearer " + token}
        ).status_code
        == 401
    )
    token = auth_service.create_jwt_token({"sub": "ops-admin"})
    parts = token.split(".")
    parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
    assert (
        client.get(
            READ_ROUTES[0], headers={"Authorization": "Bearer " + ".".join(parts)}
        ).status_code
        == 401
    )


def test_role_revocation_and_missing_account_fail_closed(client, identities):
    headers = bearer()
    identities["ops-admin"]["role"] = "student"
    assert client.get(READ_ROUTES[0], headers=headers).status_code == 403
    identities.pop("ops-admin")
    assert client.get(READ_ROUTES[0], headers=headers).status_code == 401


def test_admin_cookie_and_invalid_cookie_precedence(client):
    client.cookies.set(
        "access_token", auth_service.create_jwt_token({"sub": "ops-admin"})
    )
    assert client.get(READ_ROUTES[0]).status_code == 200
    client.cookies.set("access_token", "invalid")
    assert client.get(READ_ROUTES[0], headers=bearer()).status_code == 401


@pytest.mark.parametrize(
    "environment", ["production", " Production ", "staging", "unknown", ""]
)
def test_reset_never_enabled_outside_explicit_local_environments(
    client, monkeypatch, environment
):
    monkeypatch.setattr(config.settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(
        config.settings, "OPERATIONS_RESET_ENABLED", True, raising=False
    )
    reset = Mock()
    clear = Mock()
    monkeypatch.setattr(telemetry_service, "reset_metrics", reset)
    monkeypatch.setattr(response_cache, "clear", clear)
    response = client.post(
        "/api/v1/telemetry/reset",
        headers={**bearer(), "Origin": "https://www.jruva.in"},
    )
    assert response.status_code == 403
    reset.assert_not_called()
    clear.assert_not_called()


@pytest.mark.parametrize("identity,status", [(None, 401), ("ordinary", 403)])
def test_unauthorized_reset_has_no_side_effects(client, monkeypatch, identity, status):
    reset = Mock()
    clear = Mock()
    monkeypatch.setattr(telemetry_service, "reset_metrics", reset)
    monkeypatch.setattr(response_cache, "clear", clear)
    assert (
        client.post(
            "/api/v1/telemetry/reset", headers=bearer(identity) if identity else {}
        ).status_code
        == status
    )
    reset.assert_not_called()
    clear.assert_not_called()


def test_reset_opt_in_and_local_admin(client, monkeypatch):
    monkeypatch.setattr(
        config.settings, "OPERATIONS_RESET_ENABLED", False, raising=False
    )
    assert client.post("/api/v1/telemetry/reset", headers=bearer()).status_code == 403
    monkeypatch.setattr(config.settings, "OPERATIONS_RESET_ENABLED", True)
    assert client.post("/api/v1/telemetry/reset", headers=bearer()).status_code == 200


def test_production_csrf_and_auth_are_independent(client, monkeypatch):
    monkeypatch.setattr(config.settings, "ENVIRONMENT", "production")
    headers = bearer()
    for origin in (None, "https://attacker.invalid"):
        response = client.post(
            "/api/v1/telemetry/reset",
            headers={**headers, **({"Origin": origin} if origin else {})},
        )
        assert response.status_code == 403
        assert "CSRF" in response.json()["detail"]
    assert (
        client.post(
            "/api/v1/telemetry/reset", headers={"Origin": "https://www.jruva.in"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/telemetry/reset",
            headers={**headers, "Origin": "https://www.jruva.in"},
        ).status_code
        == 403
    )


def test_diagnostics_drop_sensitive_fields_and_bound_trace_limit(client):
    telemetry_service.reset_metrics()
    for _ in range(205):
        telemetry_service.record_trace(
            RequestTrace(
                session_id="private-user@example.invalid",
                requested_model="secret-token",
                effective_model="secret-token",
                endpoint="/private/customer",
                routed_provider="secret-token",
                error_message="password=secret-token internal traceback",
                failover_reason="secret-token",
                tier="secret-token",
                status="error",
                total_latency_ms=12,
                total_tokens=5,
            )
        )
    traces = client.get(READ_ROUTES[1] + "?limit=200", headers=bearer())
    assert traces.status_code == 200
    assert len(traces.json()) == 200
    summary = client.get(READ_ROUTES[0], headers=bearer())
    assert summary.status_code == 200
    for response in (traces, summary):
        assert "secret-token" not in response.text
        assert "private-user" not in response.text
        assert "internal traceback" not in response.text
    forbidden = {
        "session_id",
        "trace_id",
        "endpoint",
        "error_message",
        "requested_model",
        "effective_model",
        "failover_reason",
    }
    assert not forbidden.intersection(traces.json()[0])
    assert (
        client.get(READ_ROUTES[1] + "?limit=201", headers=bearer()).status_code == 422
    )
    assert (
        client.get(
            READ_ROUTES[1] + "?provider=private-user", headers=bearer()
        ).status_code
        == 422
    )


def test_operational_exception_is_generic(client, monkeypatch):
    monkeypatch.setattr(
        telemetry_service,
        "get_traces",
        Mock(side_effect=RuntimeError("password=secret-token internal traceback")),
    )
    response = client.get(READ_ROUTES[1], headers=bearer())
    assert response.status_code == 503
    assert "secret-token" not in response.text
    assert "traceback" not in response.text


def test_prometheus_omits_unapproved_collectors_and_labels(client):
    from app.api.v1.endpoints.metrics import REQUESTS_TOTAL
    from prometheus_client import REGISTRY, Gauge

    secret = "private-user@example.invalid"
    extra = Gauge("synthetic_private_diagnostic", secret)
    REQUESTS_TOTAL.labels(endpoint=secret, method="GET", status="200").inc()
    try:
        response = client.get("/metrics", headers=bearer())
        assert response.status_code == 200
        assert secret not in response.text
        assert "synthetic_private_diagnostic" not in response.text
        assert "jru_provider_health_status" in response.text
        from prometheus_client.parser import text_string_to_metric_families

        assert list(text_string_to_metric_families(response.text))
    finally:
        REGISTRY.unregister(extra)
        REQUESTS_TOTAL.remove(secret, "GET", "200")


def test_health_remains_public(client):
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("role", [None, "ADMIN", "superadmin", "", ["admin"]])
def test_unrecognized_server_role_denied(client, identities, role):
    identities["ops-admin"]["role"] = role
    response = client.get(READ_ROUTES[0], headers=bearer())
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"


def test_account_lookup_failure_fails_closed(client, monkeypatch):
    monkeypatch.setattr(
        auth_service,
        "get_user_by_id",
        Mock(side_effect=RuntimeError("synthetic-private-account-error")),
    )
    response = client.get(READ_ROUTES[0], headers=bearer())
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert "synthetic-private-account-error" not in response.text


def test_trace_projection_rejects_nonfinite_and_free_text_values(client, monkeypatch):
    malicious = {
        "status": "private-error",
        "tier": "private-tier",
        "routed_provider": "private-provider",
        "total_latency_ms": float("nan"),
        "prompt_tokens": float("inf"),
        "ttft_ms": -1,
        "total_tokens": 10**400,
        "estimated_cost_usd": "private-cost",
        "cached": "private-cache",
    }
    monkeypatch.setattr(
        telemetry_service, "get_traces", lambda **kwargs: [malicious] * 300
    )
    response = client.get(READ_ROUTES[1] + "?limit=1", headers=bearer())
    assert response.status_code == 200
    assert len(response.json()) == 1
    trace = response.json()[0]
    assert (
        trace["total_latency_ms"]
        == trace["prompt_tokens"]
        == trace["total_tokens"]
        == trace["ttft_ms"]
        == 0
    )
    assert trace["cached"] is False
    assert "private-" not in response.text


@pytest.mark.parametrize(
    "path", ["/api/v1/telemetry/metrics", "/metrics", "/api/v1/metrics"]
)
def test_other_operational_errors_do_not_leak(client, monkeypatch, path):
    if path.endswith("telemetry/metrics"):
        monkeypatch.setattr(
            telemetry_service,
            "get_metrics_summary",
            Mock(side_effect=RuntimeError("synthetic-private-error")),
        )
    else:
        from app.api.v1.endpoints.metrics import image_service

        monkeypatch.setattr(
            image_service.health,
            "is_healthy",
            Mock(side_effect=RuntimeError("synthetic-private-error")),
        )
    response = client.get(path, headers=bearer())
    assert response.status_code == 503
    assert "synthetic-private-error" not in response.text


def test_reset_failure_response_is_generic(client, monkeypatch):
    monkeypatch.setattr(config.settings, "OPERATIONS_RESET_ENABLED", True)
    monkeypatch.setattr(
        telemetry_service,
        "reset_metrics",
        Mock(side_effect=RuntimeError("synthetic-private-reset-error")),
    )
    response = client.post("/api/v1/telemetry/reset", headers=bearer())
    assert response.status_code == 503
    assert "synthetic-private-reset-error" not in response.text


def test_invalid_diagnostic_query_does_not_echo_credentials(client):
    response = client.get(
        READ_ROUTES[1],
        params={"provider": "synthetic-private-key" * 100},
        headers=bearer(),
    )
    assert response.status_code == 422
    assert "synthetic-private-key" not in response.text
    assert len(response.content) < 512
    assert response.headers["cache-control"] == "no-store"


def test_public_chat_does_not_require_operations_admin(client):
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from app.api.dependencies import get_chat_service
    from app.schemas.chat import ChatResponse

    service = Mock()
    service.generate_chat_response = AsyncMock(
        return_value=ChatResponse(session_id=uuid4(), content="Synthetic public reply")
    )
    client.app.dependency_overrides[get_chat_service] = lambda: service
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "synthetic public question", "stream": False},
        )
        assert response.status_code == 200
        assert response.json()["content"] == "Synthetic public reply"
        service.generate_chat_response.assert_awaited_once()
    finally:
        client.app.dependency_overrides.clear()
