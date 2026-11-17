"""Regression and unit tests for Gemini model-aware thinking configuration and 429 quota handling."""

from app.services.llm.circuit_breaker import HealthState, model_health_tracker
from app.services.llm.gemini_provider import GeminiProvider


class TestGeminiThinkingConfiguration:
    def test_gemini_3_flash_thinking_level(self):
        """Gemini 3.6 Flash must use thinkingLevel and NEVER thinkingBudget."""
        provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash")
        cfg = provider._build_thinking_config("gemini-3.6-flash", is_deep_thinking=False)
        assert cfg == {"thinkingLevel": "LOW"}
        assert "thinkingBudget" not in cfg

    def test_gemini_3_flash_lite_thinking_level(self):
        """Gemini 3.5 Flash Lite must use thinkingLevel and NEVER thinkingBudget."""
        provider = GeminiProvider(api_key="test-key", model="gemini-3.5-flash-lite")
        cfg = provider._build_thinking_config("gemini-3.5-flash-lite", is_deep_thinking=False)
        assert cfg == {"thinkingLevel": "LOW"}
        assert "thinkingBudget" not in cfg

    def test_gemini_3_deep_thinking_level(self):
        """Gemini 3.x deep reasoning requests should use MEDIUM thinkingLevel."""
        provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash")
        cfg = provider._build_thinking_config("gemini-3.6-flash", is_deep_thinking=True)
        assert cfg == {"thinkingLevel": "MEDIUM"}

    def test_gemini_2_5_thinking_budget(self):
        """Gemini 2.5 Flash must use thinkingBudget and NEVER thinkingLevel."""
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        cfg = provider._build_thinking_config("gemini-2.5-flash", is_deep_thinking=False)
        assert cfg == {"thinkingBudget": 0}
        assert "thinkingLevel" not in cfg

    def test_gemini_2_5_deep_thinking_budget(self):
        """Gemini 2.5 deep reasoning should provide positive token budget."""
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        cfg = provider._build_thinking_config("gemini-2.5-flash", is_deep_thinking=True)
        assert cfg == {"thinkingBudget": 2048}

    def test_gemini_1_5_no_thinking_config(self):
        """Legacy Gemini 1.5 models do not support thinkingConfig; must return None."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-flash")
        cfg = provider._build_thinking_config("gemini-1.5-flash", is_deep_thinking=False)
        assert cfg is None

    def test_build_payload_model_specific(self):
        """_build_payload must configure generationConfig specifically for the target model."""
        provider = GeminiProvider(api_key="test-key", model="gemini-3.6-flash")

        # When building for 3.5-flash-lite
        payload_lite = provider._build_payload(
            prompt="Hello", model="gemini-3.5-flash-lite"
        )
        assert payload_lite["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}

        # When building for 1.5-flash
        payload_15 = provider._build_payload(
            prompt="Hello", model="gemini-1.5-flash"
        )
        assert "thinkingConfig" not in payload_15["generationConfig"]


class TestGeminiQuotaAndHealthTracking:
    def setup_method(self):
        model_health_tracker.reset_all()

    def test_retry_after_cooldown_recorded(self):
        """ModelHealthTracker must respect dynamic cooldown from upstream Retry-After."""
        model_health_tracker.record_failure(
            "gemini", "gemini-3.6-flash", "quota_429", status_code=429, cooldown_seconds=45.0
        )
        rec = model_health_tracker.get_record("gemini", "gemini-3.6-flash")
        assert rec.state == HealthState.RATE_LIMITED
        assert rec.cooldown_seconds == 45.0
        assert not model_health_tracker.is_eligible("gemini", "gemini-3.6-flash")

    def test_candidate_models_failover_eligibility(self):
        """When primary is rate-limited, secondary eligible model is attempted."""
        model_health_tracker.record_failure(
            "gemini", "gemini-3.6-flash", "quota_429", status_code=429, cooldown_seconds=60.0
        )
        assert not model_health_tracker.is_eligible("gemini", "gemini-3.6-flash")
        assert model_health_tracker.is_eligible("gemini", "gemini-3.5-flash-lite")
