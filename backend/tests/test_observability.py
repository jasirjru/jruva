"""Observability tests: Prometheus scrape endpoint, structured logging, and alert definitions."""
import json
import logging
from fastapi.testclient import TestClient
from app.main import app
from app.core.logging_config import JSONLogFormatter
from app.core.alerts import PRODUCTION_ALERTS


def test_prometheus_metrics_endpoint(monkeypatch):
    """Verify that /metrics endpoint returns valid Prometheus scrape exposition."""
    client = TestClient(app)
    from app.services.auth_service import auth_service
    monkeypatch.setattr(auth_service, "get_user_by_id", lambda uid: {"id": uid, "role": "admin"})
    client.headers["Authorization"] = "Bearer " + auth_service.create_jwt_token({"sub": "synthetic-metrics-admin"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    body = response.text
    # Assert standard Prometheus metric families are exposed
    assert "jru_requests_total" in body or "jru_provider_health_status" in body
    assert "jru_provider_health_status" in body


def test_structured_json_logging():
    """Verify that JSONLogFormatter outputs parseable, single-line JSON with required fields."""
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="Induced test error occurred",
        args=(),
        exc_info=None,
    )
    record.error_type = "timeout"
    record.status_code = 504
    record.provider = "flux"

    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "ERROR"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Induced test error occurred"
    assert parsed["error_type"] == "timeout"
    assert parsed["status_code"] == 504
    assert parsed["provider"] == "flux"
    assert "timestamp" in parsed


def test_production_alert_definitions():
    """Verify production alert rules are defined with valid severity and expressions."""
    assert len(PRODUCTION_ALERTS) >= 3
    alert_names = [a.name for a in PRODUCTION_ALERTS]
    assert "HighErrorRate" in alert_names
    assert "HighLatencyP95" in alert_names
    assert "PrimaryProviderDegraded" in alert_names

    for alert in PRODUCTION_ALERTS:
        assert alert.severity in ("critical", "warning", "info")
        assert len(alert.expr) > 5
        assert alert.duration.endswith(("m", "s", "h"))
