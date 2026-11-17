"""Direct local model selection must allow the provider's bounded cold start."""
import asyncio

from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.router import LLMRouter


async def test_explicit_local_selection_survives_a_15_second_cold_start(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "ALLOW_LOCAL_DEV_OLLAMA", True)
    provider = OllamaProvider(model="jru-specialist:v4")
    monkeypatch.setattr(type(provider), "is_available", property(lambda self: True))

    async def synthetic_cold_start(**kwargs):
        await asyncio.sleep(15.2)
        provider.last_active_model = "ollama/jru-specialist:v4"
        return "Completed bounded local inference"

    monkeypatch.setattr(provider, "generate_response", synthetic_cold_start)
    result = await LLMRouter(ollama_provider=provider).generate_response(
        "An uncached long-form local inference probe", requested_model="ollama/jru-specialist:v4"
    )
    assert result.text == "Completed bounded local inference"
    assert result.actual_provider == "ollama"
    assert not result.fallback_used
