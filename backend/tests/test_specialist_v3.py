"""Tests for Retirement of JRU Specialist v3/v4 in Favor of Cloud Frontier Compute."""
import pytest
from unittest.mock import MagicMock, PropertyMock
from app.services.llm.router import LLMRouter, AVAILABLE_MODELS, TIER_CONFIGS
from app.services.llm.groq_provider import GroqProvider

def test_specialist_v3_and_v4_not_in_public_models():
    """Verify that jru-specialist:v3, v4, v5, v6 are NOT advertised in AVAILABLE_MODELS."""
    model_ids = [m["id"] for m in AVAILABLE_MODELS]
    assert "ollama/jru-specialist:v3" not in model_ids
    assert "ollama/jru-specialist:v4" not in model_ids
    assert "ollama/jru-specialist:v5" not in model_ids
    assert "ollama/jru-specialist:v6" not in model_ids
    assert TIER_CONFIGS["S"]["primary_model"] == "qwen/qwen3.8-27b"

def test_specialist_legacy_routing_resolution():
    """Verify that legacy requests for jru-specialist:v4 gracefully resolve to verified cloud models."""
    mock_groq = MagicMock(spec=GroqProvider)
    type(mock_groq).is_available = PropertyMock(return_value=True)
    type(mock_groq).provider_name = PropertyMock(return_value="groq")

    router = LLMRouter(groq_provider=mock_groq)
    
    # Direct legacy model request for v4 maps to cloud
    primary, backup, model_target = router._resolve_providers(requested_model="ollama/jru-specialist:v4")
    assert primary == mock_groq
    assert model_target == "openai/gpt-oss-120b"

    # Direct legacy model request for v3 maps to cloud
    primary, backup, model_target = router._resolve_providers(requested_model="ollama/jru-specialist:v3")
    assert primary == mock_groq
    assert model_target == "openai/gpt-oss-120b"

    # Tier S request resolves to Groq Qwen cloud
    primary, backup, model_target = router._resolve_providers(requested_model="auto", tier="S")
    assert primary == mock_groq
    assert model_target == "qwen/qwen3.8-27b"
