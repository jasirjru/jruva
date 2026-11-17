"""Model discovery must not stall the event loop or advertise absent local models."""
import asyncio
import time
from unittest.mock import patch

import httpx
from app.api.v1.endpoints.models import list_models
from app.core.config import Settings
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.router import LLMRouter


def test_saved_project_specialist_alias_maps_safely_to_cloud_model(monkeypatch):
    local = OllamaProvider()
    monkeypatch.setattr(type(local), "is_available", property(lambda self: True))
    router = LLMRouter(ollama_provider=local, gemini_provider=GeminiProvider(api_key="synthetic-audit"))
    provider, _, model = router._resolve_providers(requested_model="jru-specialist-v4", prompt="Explain a short concept")
    # In cloud-only JRUVA, deprecated local models migrate forward to verified cloud providers
    assert model in ("openai/gpt-oss-120b", "gemini-3.6-flash", "qwen/qwen3.8-27b")
    assert provider is not local


async def test_model_discovery_does_not_block_event_loop():
    class SlowCatalog:
        def get_catalog(self, include_hidden=False):
            time.sleep(0.25)  # Represents synchronous daemon probes.
            return []

    started = time.perf_counter()
    request = asyncio.create_task(list_models(SlowCatalog(), Settings(_env_file=None)))
    await asyncio.sleep(0.01)
    elapsed = time.perf_counter() - started
    await request
    assert elapsed < 0.15, f"Model catalog blocked loop for {elapsed:.3f}s"


def test_catalog_does_not_advertise_missing_ollama_models():
    """In cloud-only architecture, local Ollama models are completely excluded from public catalog."""
    client_class = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]}))
    with patch("httpx.Client", side_effect=lambda **kw: client_class(transport=transport, **kw)):
        catalog = LLMRouter(ollama_provider=OllamaProvider()).get_catalog(include_hidden=True)
    local = [m for m in catalog if m["provider"] == "ollama"]
    assert len(local) == 0
