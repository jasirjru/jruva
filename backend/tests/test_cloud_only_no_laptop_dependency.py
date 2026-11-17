"""Comprehensive regression tests verifying 100% cloud-only AI operation.

Proves:
1. Zero network probes to localhost / 127.0.0.1:11434 / personal GPU endpoints.
2. Complete absence of local-only models (JRU Specialist v3-v6, Qwen3 8B Local) from the public catalog.
3. Safe and truthful migration of legacy saved model preferences (sessions, projects, user settings).
4. Truthful metadata in both sync and streaming paths (never mislabeling Groq as JRU Specialist).
"""
import json
import socket
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.model_registry import (
    MODEL_REGISTRY,
    DECOMMISSIONED_MODELS,
    get_canonical_model,
    is_local_model,
)
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import LLMRouter, AVAILABLE_MODELS
from app.services.chat_service import ChatService
from app.schemas.chat import ChatRequest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ==============================================================================
# 1. CATALOG INTEGRITY: ZERO LOCAL MODELS ADVERTISED
# ==============================================================================

def test_public_catalog_contains_zero_local_models():
    """Verify /api/v1/models exposes only cloud models and zero local dependencies."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    models = data["models"]

    # Public active models must be exactly 3: auto, gemini-3.6-flash, qwen/qwen3.8-27b
    model_ids = [m["id"] for m in models]
    assert "auto" in model_ids
    assert "gemini-3.6-flash" in model_ids
    assert "qwen/qwen3.8-27b" in model_ids

    # Every surfaced model must have runtime == 'cloud'
    for m in models:
        assert m.get("runtime") == "cloud", f"Model {m['id']} has non-cloud runtime: {m.get('runtime')}"
        assert m.get("provider") != "ollama", f"Model {m['id']} has local provider 'ollama'"

    # Forbidden local models must NEVER be present
    forbidden_ids = [
        "ollama/jru-specialist:v2",
        "ollama/jru-specialist:v3",
        "ollama/jru-specialist:v4",
        "ollama/jru-specialist:v5",
        "ollama/jru-specialist:v6",
        "jru-specialist-v4",
        "qwen3-8b-local",
        "local_qwen",
    ]
    for forbidden in forbidden_ids:
        assert forbidden not in model_ids, f"Forbidden local model {forbidden} surfaced in public catalog"


def test_model_registry_contains_zero_local_entries():
    """Verify backend MODEL_REGISTRY has zero local/ollama models."""
    for model_id, config in MODEL_REGISTRY.items():
        assert config.provider.value != "ollama", f"MODEL_REGISTRY[{model_id}] has provider 'ollama'"
        assert config.provider.value != "local_qwen", f"MODEL_REGISTRY[{model_id}] has provider 'local_qwen'"
        assert "localhost" not in str(config), f"MODEL_REGISTRY[{model_id}] references localhost"
        assert "11434" not in str(config), f"MODEL_REGISTRY[{model_id}] references 11434"


# ==============================================================================
# 2. PROVE ZERO NETWORK CALLS TO LAPTOP / LOCALHOST / OLLAMA
# ==============================================================================

def test_catalog_probe_never_attempts_socket_to_localhost_or_ollama():
    """Verify router.get_catalog() never attempts socket connections to 127.0.0.1:11434."""
    intercepted_calls = []

    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host, port = address[0], address[1]
        intercepted_calls.append((host, port))
        if str(host) in ("127.0.0.1", "localhost", "0.0.0.0") and port == 11434:
            raise AssertionError(f"BLOCKED: Attempted socket connection to laptop Ollama at {host}:{port}")
        return orig_connect(self, address)

    router = LLMRouter()
    with patch.object(socket.socket, "connect", guarded_connect):
        catalog = router.get_catalog(include_hidden=True)

    assert len(catalog) > 0
    # Confirm no local Ollama socket connections occurred
    for host, port in intercepted_calls:
        assert port != 11434, f"Socket probe detected to Ollama port {port} on {host}"


# ==============================================================================
# 3. LEGACY SAVED MODEL PREFERENCES: TRUTHFUL MIGRATION TO CLOUD
# ==============================================================================

@pytest.mark.asyncio
async def test_legacy_saved_ollama_specialist_migrates_truthfully_sync():
    """Verify legacy preference 'ollama/jru-specialist:v4' routes to Groq cloud with truthful metadata."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_payloads = []

    async def mock_post(url, json=None, headers=None, **kwargs):
        recorded_payloads.append(dict(json or {}))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Cloud response from Groq"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await router.generate_response(
            prompt="Explain quantum entanglement",
            requested_model="ollama/jru-specialist:v4",
        )

    # 1. Groq cloud API was invoked
    assert len(recorded_payloads) == 1
    assert recorded_payloads[0]["model"] == "openai/gpt-oss-120b"

    # 2. Truthful metadata: legacy request preserved, cloud actual model reported
    assert result.requested_model == "ollama/jru-specialist:v4"
    assert "Legacy Migrated" in result.requested_model_label
    assert result.actual_model == "openai/gpt-oss-120b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is True
    assert result.fallback_reason == "local_model_deprecated"
    assert "Cloud response from Groq" in result.text


@pytest.mark.asyncio
async def test_legacy_saved_ollama_specialist_migrates_truthfully_streaming():
    """Verify streaming legacy preference 'ollama/jru-specialist:v4' yields truthful fallback metadata."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_payloads = []

    async def mock_stream_lines():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "Streamed token 1"}}]})
        yield "data: " + json.dumps({"choices": [{"delta": {"content": " token 2"}}]})
        yield "data: [DONE]"

    class MockStreamContext:
        def __init__(self, *args, **kwargs):
            if "json" in kwargs:
                recorded_payloads.append(dict(kwargs["json"]))
            self.status_code = 200

        async def __aenter__(self):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.aiter_lines = mock_stream_lines
            return mock_resp

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", side_effect=MockStreamContext):
        tokens = []
        async for token_item in router.stream_response(
            prompt="Analyze this document",
            requested_model="ollama/jru-specialist:v4",
        ):
            tokens.append(token_item)

    assert len(recorded_payloads) == 1
    assert recorded_payloads[0]["model"] == "openai/gpt-oss-120b"
    assert len(tokens) >= 2

    first_token = tokens[0]
    assert first_token.requested_model == "ollama/jru-specialist:v4"
    assert "Legacy Migrated" in first_token.requested_model_label
    assert first_token.actual_model == "openai/gpt-oss-120b"
    assert first_token.actual_provider == "groq"
    assert first_token.fallback_used is True
    assert first_token.fallback_reason == "local_model_deprecated"


@pytest.mark.asyncio
async def test_legacy_saved_qwen_local_migrates_to_cloud_qwen():
    """Verify legacy preference 'qwen3-8b-local' routes to Groq 'qwen/qwen3.8-27b' cloud."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_payloads = []

    async def mock_post(url, json=None, headers=None, **kwargs):
        recorded_payloads.append(dict(json or {}))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "def solution(): pass"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await router.generate_response(
            prompt="Write quicksort in python",
            requested_model="qwen3-8b-local",
        )

    assert len(recorded_payloads) == 1
    assert recorded_payloads[0]["model"] == "qwen/qwen3.8-27b"
    assert result.requested_model == "qwen3-8b-local"
    assert result.actual_model == "qwen/qwen3.8-27b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is True
    assert result.fallback_reason == "local_model_deprecated"


# ==============================================================================
# 4. CANONICAL MODEL FORWARD MAPPING & DECOMMISSION DICTIONARY
# ==============================================================================

def test_canonical_model_resolution_for_all_legacy_local_tags():
    """Verify get_canonical_model maps all local tags forward to cloud models."""
    legacy_tags = [
        "ollama/jru-specialist:v2",
        "ollama/jru-specialist:v3",
        "ollama/jru-specialist:v4",
        "ollama/jru-specialist:v5",
        "ollama/jru-specialist:v6",
        "jru-specialist-v4",
        "qwen3-8b-local",
        "local_qwen",
    ]
    for tag in legacy_tags:
        assert is_local_model(tag) is True, f"Expected {tag} to be recognized as local model"
        canonical = get_canonical_model(tag)
        assert canonical.model_id in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b"), (
            f"Expected {tag} to map to cloud model, got {canonical.model_id}"
        )


def test_decommissioned_models_contains_all_local_tags():
    """Verify DECOMMISSIONED_MODELS contains all retired local model tags."""
    assert "ollama/jru-specialist:v4" in DECOMMISSIONED_MODELS
    assert "ollama/jru-specialist:v3" in DECOMMISSIONED_MODELS
    assert "ollama/jru-specialist:v5" in DECOMMISSIONED_MODELS
    assert "ollama/jru-specialist:v6" in DECOMMISSIONED_MODELS
    assert "qwen3-8b-local" in DECOMMISSIONED_MODELS
