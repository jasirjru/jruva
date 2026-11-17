"""Unit tests for Service Level Objectives (SLO) definitions and drift prevention."""
import pytest
from app.core.slo import (
    PRODUCTION_SLOS,
    EndpointSLO,
    get_slo,
    is_latency_compliant,
    is_availability_compliant,
)


def test_production_slo_catalog_integrity():
    """Ensure all required endpoints have defined SLOs with valid bounds."""
    required_keys = [
        "chat_cached",
        "chat_stream_ttft",
        "image_cached",
        "image_synthesis",
        "health_and_meta",
    ]
    for key in required_keys:
        assert key in PRODUCTION_SLOS, f"Missing required SLO key: {key}"
        slo = PRODUCTION_SLOS[key]
        assert isinstance(slo, EndpointSLO)
        assert 0.0 < slo.target_availability_pct <= 100.0
        assert 0.0 <= slo.target_error_rate_pct < 100.0
        assert slo.latency_p50_ms < slo.latency_p95_ms <= slo.latency_p99_ms
        assert len(slo.description) > 0


def test_slo_exact_targets_prevent_silent_drift():
    """Assert exact target numbers to prevent silent drift between code and documentation."""
    # Chat cached
    chat_cached = get_slo("chat_cached")
    assert chat_cached.latency_p50_ms == 5.0
    assert chat_cached.latency_p95_ms == 15.0
    assert chat_cached.latency_p99_ms == 30.0
    assert chat_cached.target_availability_pct == 99.99

    # Chat streaming TTFT
    chat_stream = get_slo("chat_stream_ttft")
    assert chat_stream.latency_p50_ms == 450.0
    assert chat_stream.latency_p95_ms == 1200.0
    assert chat_stream.latency_p99_ms == 2500.0
    assert chat_stream.target_availability_pct == 99.5

    # Image cached
    image_cached = get_slo("image_cached")
    assert image_cached.latency_p50_ms == 5.0
    assert image_cached.latency_p95_ms == 15.0
    assert image_cached.latency_p99_ms == 30.0

    # Image live synthesis
    image_synth = get_slo("image_synthesis")
    assert image_synth.latency_p50_ms == 4500.0
    assert image_synth.latency_p95_ms == 12000.0
    assert image_synth.latency_p99_ms == 18000.0

    # Health and metadata
    health = get_slo("health_and_meta")
    assert health.latency_p50_ms == 3.0
    assert health.latency_p95_ms == 10.0
    assert health.latency_p99_ms == 25.0


def test_slo_compliance_checkers():
    """Test latency and availability compliance helper functions."""
    # Chat cached: p95 target is 15.0ms
    assert is_latency_compliant("chat_cached", 10.0) is True
    assert is_latency_compliant("chat_cached", 15.0) is True
    assert is_latency_compliant("chat_cached", 15.1) is False

    # Chat cached: error rate target is 0.01%
    assert is_availability_compliant("chat_cached", 0.005) is True
    assert is_availability_compliant("chat_cached", 0.02) is False

    # Invalid key raises KeyError
    with pytest.raises(KeyError):
        get_slo("non_existent_key")
