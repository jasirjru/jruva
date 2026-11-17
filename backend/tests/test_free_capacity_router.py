"""Free-capacity routing: spread over per-model quotas, honor cooldowns, optional extra tiers, never empty.

All HTTP is mocked with httpx.MockTransport; no live provider calls.
"""
import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from app.core.config import settings
from app.core.exceptions import QuotaExhaustedException
from app.schemas.chat import ChatRequest
from app.services.chat_service import ALL_MODELS_BUSY_MESSAGE, ChatService
from app.services.llm.circuit_breaker import HealthState, model_health_tracker
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.openai_compat_provider import (
    OpenAICompatibleProvider,
    build_openai_compatible_providers,
)
from app.services.llm.rate_limit import MAX_COOLDOWN_SECONDS, cooldown_from_response
from app.services.llm.router import LLMRouter

pytestmark = pytest.mark.asyncio

COMPAT_ENV_FIELDS = [
    f"{vendor}_{suffix}"
    for vendor in ("MISTRAL", "CLOUDFLARE", "CEREBRAS", "NVIDIA", "SAMBANOVA")
    for suffix in ("API_KEY", "BASE_URL", "MODELS")
]


def _sse(*pieces: str) -> str:
    body = "\n\n".join("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) for p in pieces)
    return body + "\n\ndata: [DONE]\n\n"


def _patched_client(handler):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    return patch("httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw))


@pytest.fixture
def no_compat_keys():
    with patch.multiple(settings, **{name: "" for name in COMPAT_ENV_FIELDS}):
        yield


def _unavailable(name: str) -> MagicMock:
    mock = MagicMock(is_available=False)
    mock.provider_name = name
    return mock


# --- (a) spread across Groq models, honoring Retry-After -------------------------------------------------


async def test_groq_spreads_to_next_model_on_429_and_records_retry_after_cooldown():
    seen_models = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen_models.append(model)
        if model in ("qwen/qwen3.8-27b", "openai/gpt-oss-120b"):
            return httpx.Response(429, headers={"retry-after": "971"}, json={"error": "rate limited"})
        return httpx.Response(200, text=_sse("Hello", " there"))

    provider = GroqProvider(api_key="synthetic-test")
    with _patched_client(handler):
        text = "".join([t async for t in provider.stream_response("hi", model_override="qwen/qwen3.8-27b")])

    assert text == "Hello there"
    assert seen_models == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert provider.last_active_model == "openai/gpt-oss-20b"
    record = model_health_tracker.get_record("groq", "qwen/qwen3.8-27b")
    assert record.state == HealthState.RATE_LIMITED
    assert record.cooldown_seconds == 971.0  # was a fixed 60 s before


async def test_cooled_down_groq_model_is_skipped_without_http_call():
    model_health_tracker.record_failure("groq", "qwen/qwen3.8-27b", "quota_429", status_code=429, cooldown_seconds=900)
    seen_models = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_models.append(json.loads(request.content)["model"])
        return httpx.Response(200, text=_sse("ok"))

    provider = GroqProvider(api_key="synthetic-test")
    started = time.monotonic()
    with _patched_client(handler):
        text = "".join([t async for t in provider.stream_response("hi", model_override="qwen/qwen3.8-27b")])

    assert text == "ok"
    assert "qwen/qwen3.8-27b" not in seen_models  # skipped, not waited on
    assert time.monotonic() - started < 2.0


async def test_groq_fallback_models_configurable_by_env():
    with patch.object(settings, "GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b, openai/gpt-oss-20b ,"):
        provider = GroqProvider(api_key="synthetic-test")
    assert provider.FALLBACK_MODELS == ["openai/gpt-oss-20b"]
    assert "openai/gpt-oss-20b" in GroqProvider(api_key="synthetic-test").FALLBACK_MODELS


def test_cooldown_parsing_header_body_and_clamp():
    assert cooldown_from_response({"retry-after": "84"}) == 84.0
    body = json.dumps({"error": {"details": [{"@type": "RetryInfo", "retryDelay": "34s"}]}})
    assert cooldown_from_response({}, body) == 34.0
    assert cooldown_from_response({"retry-after": "999999"}) == MAX_COOLDOWN_SECONDS
    assert cooldown_from_response({}, "not json") == 60.0


# --- (b) generic OpenAI-compatible adapter ---------------------------------------------------------------


def test_compat_providers_disabled_without_keys(no_compat_keys):
    assert build_openai_compatible_providers() == []


def test_compat_provider_enabled_by_key_with_defaults_and_overrides(no_compat_keys):
    with patch.multiple(
        settings,
        MISTRAL_API_KEY="synthetic-mistral",
        SAMBANOVA_API_KEY="synthetic-samba",
        SAMBANOVA_MODELS="gpt-oss-120b",
        SAMBANOVA_BASE_URL="https://example.test/v1/",
    ):
        providers = {p.provider_name: p for p in build_openai_compatible_providers()}

    assert set(providers) == {"mistral", "sambanova"}
    assert providers["mistral"].chat_url == "https://api.mistral.ai/v1/chat/completions"
    assert providers["mistral"].models == ["mistral-small-latest"]
    assert providers["sambanova"].chat_url == "https://example.test/v1/chat/completions"
    assert providers["sambanova"].models == ["gpt-oss-120b"]


def test_cloudflare_needs_account_id(no_compat_keys):
    with patch.multiple(settings, CLOUDFLARE_API_KEY="synthetic-cf", CLOUDFLARE_ACCOUNT_ID=""):
        assert build_openai_compatible_providers() == []
    with patch.multiple(settings, CLOUDFLARE_API_KEY="synthetic-cf", CLOUDFLARE_ACCOUNT_ID="acct123"):
        (provider,) = build_openai_compatible_providers()
    assert provider.chat_url == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1/chat/completions"


async def test_compat_provider_streams_and_cascades_models_on_429():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer synthetic-key"
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "model-a":
            return httpx.Response(429, headers={"retry-after": "120"})
        return httpx.Response(200, text=_sse("Hi", " <think>secret</think>", "!"))

    provider = OpenAICompatibleProvider("mistral", "https://example.test/v1", "synthetic-key", ["model-a", "model-b"])
    with _patched_client(handler):
        text = "".join([t async for t in provider.stream_response("hi")])

    assert text == "Hi !"
    assert seen == ["model-a", "model-b"]
    assert model_health_tracker.get_record("mistral", "model-a").cooldown_seconds == 120.0


async def test_compat_provider_raises_quota_when_all_models_limited():
    provider = OpenAICompatibleProvider("cerebras", "https://example.test/v1", "synthetic-key", ["only"])
    with _patched_client(lambda request: httpx.Response(429)):
        with pytest.raises(QuotaExhaustedException):
            await provider.generate_response("hi")


async def test_router_falls_back_to_env_enabled_compat_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        if "groq.com" in request.url.host:
            return httpx.Response(429, headers={"retry-after": "600"})
        return httpx.Response(200, text=_sse("From", " Mistral"))

    compat = OpenAICompatibleProvider("mistral", "https://example.test/v1", "synthetic-key", ["mistral-small-latest"])
    router = LLMRouter(
        gemini_provider=_unavailable("gemini"),
        groq_provider=GroqProvider(api_key="synthetic-test"),
        openrouter_provider=_unavailable("openrouter"),
        extra_providers=[compat],
    )
    with _patched_client(handler):
        tokens = [t async for t in router.stream_response("hello there friend", requested_model="auto")]

    assert "".join(t.token for t in tokens) == "From Mistral"
    assert tokens[0].actual_provider == "mistral"
    assert tokens[0].fallback_used is True
    for model in GroqProvider(api_key="x").FALLBACK_MODELS:
        assert model_health_tracker.get_health("groq", model) == HealthState.RATE_LIMITED


# --- (c) never an empty answer ---------------------------------------------------------------------------


async def test_whitespace_only_provider_is_treated_as_empty_and_cascades():
    def handler(request: httpx.Request) -> httpx.Response:
        if "groq.com" in request.url.host:
            return httpx.Response(200, text=_sse("  ", "\n"))
        return httpx.Response(200, text=_sse("Real answer"))

    compat = OpenAICompatibleProvider("mistral", "https://example.test/v1", "synthetic-key", ["m"])
    router = LLMRouter(
        gemini_provider=_unavailable("gemini"),
        groq_provider=GroqProvider(api_key="synthetic-test"),
        openrouter_provider=_unavailable("openrouter"),
        extra_providers=[compat],
    )
    with _patched_client(handler):
        tokens = [t async for t in router.stream_response("hello there friend", requested_model="auto")]

    assert "".join(t.token for t in tokens) == "Real answer"
    assert all(t.token.strip() for t in tokens)


async def test_stream_sends_clear_busy_message_when_every_provider_is_rate_limited():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "300"})

    service = ChatService()
    service.router = LLMRouter(
        gemini_provider=_unavailable("gemini"),
        groq_provider=GroqProvider(api_key="synthetic-test"),
        openrouter_provider=_unavailable("openrouter"),
        extra_providers=[OpenAICompatibleProvider("mistral", "https://example.test/v1", "k", ["m"])],
    )
    with _patched_client(handler):
        chunks = [c async for c in service.stream_chat_response(ChatRequest(message="hello there friend", stream=True))]

    error_frames = [c for c in chunks if c.startswith("event: error")]
    assert error_frames, chunks
    payload = json.loads(error_frames[-1].split("data: ", 1)[1])
    assert payload["content"] == ALL_MODELS_BUSY_MESSAGE
    assert payload["finish_reason"] == "capacity"
    assert not any('"type":"token"' in c and '"content":""' in c for c in chunks)
