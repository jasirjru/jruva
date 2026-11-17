"""Tests for Retirement of Local Qwen3-8B in Favor of Groq LPU Cloud Qwen."""
import pytest
from unittest.mock import MagicMock, PropertyMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import LLMRouter, AVAILABLE_MODELS

client = TestClient(app)


def test_qwen3_local_not_in_available_models():
    """Verify qwen3-8b-local and RTX 4060 laptop badges are NOT in AVAILABLE_MODELS."""
    qwen3_entry = next((m for m in AVAILABLE_MODELS if m["id"] == "qwen3-8b-local"), None)
    assert qwen3_entry is None, "qwen3-8b-local must not be surfaced in AVAILABLE_MODELS"
    for m in AVAILABLE_MODELS:
        assert "RTX 4060" not in m.get("badge", ""), f"Laptop GPU advertised in badge for {m['id']}"
        assert m.get("provider") != "local_qwen", f"local_qwen provider surfaced for {m['id']}"


def test_router_resolves_qwen3_local_to_cloud_qwen():
    """Verify router gracefully maps legacy requests for qwen3-8b-local to Groq cloud Qwen."""
    mock_groq = MagicMock(spec=GroqProvider)
    type(mock_groq).is_available = PropertyMock(return_value=True)
    type(mock_groq).provider_name = PropertyMock(return_value="groq")

    router = LLMRouter(groq_provider=mock_groq)
    primary, backup, effective_id = router._resolve_providers(requested_model="qwen3-8b-local")
    assert primary == mock_groq
    assert effective_id == "qwen/qwen3.8-27b"


def test_router_catalog_excludes_local_qwen():
    """Verify get_catalog excludes local_qwen provider and exposes only cloud runtimes."""
    router = LLMRouter()
    catalog = router.get_catalog(include_hidden=True)
    local_items = [m for m in catalog if m.get("provider") == "local_qwen" or m.get("runtime") == "local"]
    assert len(local_items) == 0, f"Found local items in catalog: {local_items}"
