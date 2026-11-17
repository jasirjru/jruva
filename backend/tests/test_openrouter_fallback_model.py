"""OpenRouter fallback uses Gemma 4 31B (free), not the old Liquid LFM 2.6B.

Gemma 4 31B is a much larger model, accepts images, and is listed by OpenRouter's
public /models endpoint (checked 10 Nov 2026). The old vision fallback
inclusionai/ling-3.0-flash-vl:free is no longer listed there.
Saved settings that still name the Liquid model keep working through aliases.
"""
from unittest.mock import MagicMock

import pytest
from app.core.model_registry import get_canonical_model
from app.services.llm import router as router_module
from app.services.llm.openrouter_provider import OpenRouterProvider
from app.services.llm.router import LLMRouter

GEMMA = "google/gemma-4-31b-it:free"
OLD_LIQUID = "liquid/lfm-2.5-2.6b:free"


def provider(name: str, available: bool = True) -> MagicMock:
    p = MagicMock()
    p.name = name
    p.is_available = available
    return p


def openrouter_only_router() -> LLMRouter:
    return LLMRouter(gemini_provider=provider("gemini", available=False),
                     groq_provider=provider("groq", available=False),
                     openrouter_provider=provider("openrouter"))


def test_provider_default_model_is_gemma():
    assert OpenRouterProvider(api_key="sk-or-test").model == GEMMA


@pytest.mark.parametrize("saved", [OLD_LIQUID, "qwen-2.5-coder-32b", "openrouter/" + OLD_LIQUID])
def test_legacy_model_names_map_to_gemma(saved):
    assert OpenRouterProvider(api_key="sk-or-test", model=saved).model == GEMMA


@pytest.mark.parametrize("saved", ["liquid", OLD_LIQUID, "openrouter", GEMMA])
def test_registry_aliases_resolve_to_gemma(saved):
    assert get_canonical_model(saved).model_id == GEMMA


def test_explicit_openrouter_request_uses_gemma():
    r = openrouter_only_router()

    primary, _, model = r._resolve_providers(requested_model=OLD_LIQUID, prompt="hello")

    assert (primary, model) == (r.openrouter_provider, GEMMA)


def test_everyday_chat_falls_back_to_gemma_when_only_openrouter_is_up():
    r = openrouter_only_router()

    primary, _, model = r._resolve_providers(requested_model="auto", prompt="What is the capital of Australia?")

    assert (primary, model) == (r.openrouter_provider, GEMMA)


def test_image_task_falls_back_to_gemma_when_gemini_is_down(monkeypatch):
    monkeypatch.setattr(router_module, "detect_task_type", lambda _prompt: "image")
    r = openrouter_only_router()

    primary, _, model = r._resolve_providers(requested_model="auto", prompt="draw a cat")

    assert (primary, model) == (r.openrouter_provider, GEMMA)
