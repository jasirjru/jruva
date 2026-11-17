"""Everyday chat goes to the fastest free model first.

Phase 1 (PROJECT_CONTEXT.md): Groq qwen3.8-27b shows visible text in 0.1-0.3 s; Gemini
3.5 Flash takes 1.4-2.7 s typical with 14-15 s spikes. Independent quality scores are
comparable (Qwen low 26 vs Gemini 3.6 Flash minimal 24 / high 33). Images and documents
still need Gemini's vision, and Gemini stays the backup.
"""
from unittest.mock import MagicMock

import pytest
from app.services.llm.router import LLMRouter


def provider(name: str, available: bool = True) -> MagicMock:
    p = MagicMock()
    p.name = name
    p.is_available = available
    return p


@pytest.fixture
def router():
    return LLMRouter(gemini_provider=provider("gemini"), groq_provider=provider("groq"),
                     openrouter_provider=provider("openrouter"))


@pytest.mark.parametrize("prompt", ["What is the capital of Australia?", "Tell me a fun fact about octopuses.",
                                     "Plan a 3-day trip to Goa."])
def test_everyday_chat_tries_groq_qwen_first_with_gemini_backup(router, prompt):
    primary, backup, model = router._resolve_providers(requested_model="auto", tier="S+++", prompt=prompt)

    assert (primary, model) == (router.groq_provider, "qwen/qwen3.8-27b")
    assert backup is router.gemini_provider


def test_everyday_chat_uses_gemini_when_groq_unavailable():
    r = LLMRouter(gemini_provider=provider("gemini"), groq_provider=provider("groq", available=False),
                  openrouter_provider=provider("openrouter"))

    primary, _, model = r._resolve_providers(requested_model="auto", prompt="What is the capital of Australia?")

    assert (primary, model) == (r.gemini_provider, "gemini-3.6-flash")


def test_images_still_route_to_gemini_vision(router):
    image = {"name": "photo.jpg", "mime_type": "image/jpeg", "data": "AAAA"}

    primary, _, _ = router._resolve_providers(requested_model="auto", prompt="What is in this photo?",
                                              attachments=[image])

    assert primary is router.gemini_provider
