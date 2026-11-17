"""Unit tests for Prometheus Alert Rules and Simulation Verification."""
import os
import yaml
import pytest
from app.core.alerts import (
    PRODUCTION_ALERTS,
    evaluate_alert_condition,
    render_prometheus_alert_yaml,
)


def test_alerts_yml_valid_structure():
    """Verify backend/alerts.yml exists, parses as valid YAML, and contains all production alerts."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "alerts.yml")
    assert os.path.exists(yaml_path), "backend/alerts.yml file missing"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "groups" in data
    assert len(data["groups"]) > 0
    group = data["groups"][0]
    assert group["name"] == "jru_chatbot_alerts"
    rule_names = [r["alert"] for r in group["rules"]]

    for alert in PRODUCTION_ALERTS:
        assert alert.name in rule_names
        rule = next(r for r in group["rules"] if r["alert"] == alert.name)
        assert rule["labels"]["severity"] == alert.severity
        assert rule["for"] == alert.duration


def test_high_error_rate_alert_simulation():
    """Simulate error rate conditions: verify alert fires when error rate > 2% (e.g. 5%) and stays quiet when healthy."""
    # Healthy: 1 error out of 100 requests = 1.0% (< 2.0% threshold)
    metrics_healthy = {"requests_total": 100.0, "requests_5xx_total": 1.0}
    assert evaluate_alert_condition("HighErrorRate", metrics_healthy) is False

    # Unhealthy: 5 errors out of 100 requests = 5.0% (> 2.0% threshold)
    metrics_unhealthy = {"requests_total": 100.0, "requests_5xx_total": 5.0}
    assert evaluate_alert_condition("HighErrorRate", metrics_unhealthy) is True

    # Zero requests edge-case: no false positives
    assert evaluate_alert_condition("HighErrorRate", {"requests_total": 0.0, "requests_5xx_total": 0.0}) is False


def test_high_latency_p95_alert_simulation():
    """Simulate p95 latency: verify alert fires when p95 > 1.5s."""
    assert evaluate_alert_condition("HighLatencyP95", {"latency_p95_seconds": 0.85}) is False
    assert evaluate_alert_condition("HighLatencyP95", {"latency_p95_seconds": 1.50}) is False
    assert evaluate_alert_condition("HighLatencyP95", {"latency_p95_seconds": 2.10}) is True


def test_primary_provider_degraded_alert_simulation():
    """Simulate provider health: verify alert fires when provider status drops to 0 (degraded)."""
    assert evaluate_alert_condition("PrimaryProviderDegraded", {"provider_health_status": 1.0}) is False
    assert evaluate_alert_condition("PrimaryProviderDegraded", {"provider_health_status": 0.0}) is True


def test_cache_hit_rate_degraded_alert_simulation():
    """Simulate cache ratio: verify alert fires when hit ratio drops below 10% on sustained traffic."""
    # Good cache ratio: 50 hits, 50 misses = 50%
    assert evaluate_alert_condition("CacheHitRateDegraded", {"cache_hits_total": 50.0, "cache_misses_total": 50.0}) is False

    # Degraded cache ratio: 5 hits, 95 misses = 5% (< 10%)
    assert evaluate_alert_condition("CacheHitRateDegraded", {"cache_hits_total": 5.0, "cache_misses_total": 95.0}) is True

    # Zero cache traffic
    assert evaluate_alert_condition("CacheHitRateDegraded", {"cache_hits_total": 0.0, "cache_misses_total": 0.0}) is False
