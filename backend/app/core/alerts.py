"""Alerting Threshold Definitions & Prometheus Alert Rules for Production."""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AlertRule:
    name: str
    expr: str
    duration: str
    severity: str
    description: str
    runbook_url: str


# Formal Prometheus Alert Rules
PRODUCTION_ALERTS: List[AlertRule] = [
    AlertRule(
        name="HighErrorRate",
        expr='rate(jru_requests_total{status=~"5.."}[5m]) / rate(jru_requests_total[5m]) > 0.02',
        duration="3m",
        severity="critical",
        description="HTTP 5xx error rate exceeds 2% over 5-minute rolling window.",
        runbook_url="https://runbooks.internal/jru/high-error-rate",
    ),
    AlertRule(
        name="HighLatencyP95",
        expr='histogram_quantile(0.95, sum(rate(jru_request_duration_seconds_bucket[5m])) by (le, endpoint)) > 1.5',
        duration="5m",
        severity="warning",
        description="P95 request latency exceeds 1.5s SLO target over 5-minute window.",
        runbook_url="https://runbooks.internal/jru/high-latency",
    ),
    AlertRule(
        name="PrimaryProviderDegraded",
        expr='jru_provider_health_status == 0',
        duration="1m",
        severity="warning",
        description="Primary image or reasoning provider marked degraded after 3 consecutive failures.",
        runbook_url="https://runbooks.internal/jru/provider-failover",
    ),
    AlertRule(
        name="CacheHitRateDegraded",
        expr='rate(jru_cache_hits_total[10m]) / (rate(jru_cache_hits_total[10m]) + rate(jru_cache_misses_total[10m])) < 0.10',
        duration="15m",
        severity="info",
        description="Cache hit ratio dropped below 10% on sustained traffic.",
        runbook_url="https://runbooks.internal/jru/cache-tuning",
    ),
]


def evaluate_alert_condition(rule_name: str, metrics: dict) -> bool:
    """Evaluate whether an alert condition triggers given simulated metric telemetry."""
    if rule_name == "HighErrorRate":
        total = metrics.get("requests_total", 0.0)
        errors = metrics.get("requests_5xx_total", 0.0)
        if total <= 0:
            return False
        return (errors / total) > 0.02

    elif rule_name == "HighLatencyP95":
        p95 = metrics.get("latency_p95_seconds", 0.0)
        return p95 > 1.5

    elif rule_name == "PrimaryProviderDegraded":
        health = metrics.get("provider_health_status", 1.0)
        return health == 0.0

    elif rule_name == "CacheHitRateDegraded":
        hits = metrics.get("cache_hits_total", 0.0)
        misses = metrics.get("cache_misses_total", 0.0)
        total_cache = hits + misses
        if total_cache <= 0:
            return False
        return (hits / total_cache) < 0.10

    raise KeyError(f"Alert rule '{rule_name}' not registered in evaluation engine")


def render_prometheus_alert_yaml() -> str:
    """Render formal Prometheus YAML alerting rules configuration."""
    lines = [
        "# Prometheus Alerting Rules for JRU AI Chatbot",
        "# Auto-generated from app.core.alerts.PRODUCTION_ALERTS",
        "groups:",
        "  - name: jru_chatbot_alerts",
        "    rules:",
    ]
    for alert in PRODUCTION_ALERTS:
        lines.extend([
            f"      - alert: {alert.name}",
            f"        expr: {alert.expr}",
            f"        for: {alert.duration}",
            "        labels:",
            f"          severity: {alert.severity}",
            "        annotations:",
            f"          summary: \"{alert.description}\"",
            f"          runbook_url: \"{alert.runbook_url}\"",
        ])
    return "\n".join(lines) + "\n"

