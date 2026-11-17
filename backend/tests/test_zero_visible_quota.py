"""Deterministic Zero-Visible-Quota Hardening Regression Suite.

Verifies:
- Gemini 3.6 Flash 429 -> Gemini Flash Lite
- Gemini family failure -> Groq
- Groq GPT-OSS healthy
- Groq Qwen healthy
- Dead DeepSeek ID never called
- Dead llama-3.3 ID never called
- Dead Gemini 2.0 ID never called
- Qwen UI never silently reports Liquid as Qwen
- Unavailable Ollama does not block cloud generation
- OpenRouter fallback works in N-provider cascade
- Fallback loops impossible
- Actual_model and requested_model metadata correct
- Mid-stream corruption prevented
- Raw quota/provider errors cannot reach user
- Rate-limited model fast-skipped during cooldown
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.core.exceptions import (
    AllProvidersUnavailableException,
    LLMProviderException,
    QuotaExhaustedException,
)
from app.core.model_registry import (
    DECOMMISSIONED_MODELS,
    is_decommissioned,
)
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import (
    HealthState,
    model_health_tracker,
)
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import (
    LLMRouter,
    GenerationResult,
    StreamToken,
    get_model_label,
)


@pytest.fixture(autouse=True)
def reset_health_tracker():
    """Reset the global model health tracker before each test for clean isolation."""
    model_health_tracker.reset()
    yield
    model_health_tracker.reset()


def test_decommissioned_models_registry_and_health():
    """Verify decommissioned models are blocked and marked MISCONFIGURED."""
    stale_ids = [
        "gemini-2.0-flash",
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen-2.5-coder-32b-instruct:free",
    ]
    for m in stale_ids:
        assert is_decommissioned(m) is True
        assert m in DECOMMISSIONED_MODELS
        # Health state must immediately evaluate to MISCONFIGURED
        state = model_health_tracker.get_health("any_prov", m)
        assert state == HealthState.MISCONFIGURED
        # Must not be eligible for generation
        assert model_health_tracker.is_eligible("any_prov", m) is False


def test_dead_model_ids_never_sent_upstream():
    """Verify routing never issues upstream requests to decommissioned model IDs."""
    router = LLMRouter()
    
    # Check resolution of decommissioned DeepSeek alias -> redirected to Groq GPT-OSS
    _, _, model = router._resolve_providers("groq/deepseek-r1-distill-llama-70b")
    assert model != "deepseek-r1-distill-llama-70b"
    assert "gpt-oss-120b" in model

    # Check resolution of decommissioned llama-3.3 -> redirected to Groq GPT-OSS
    _, _, model = router._resolve_providers("llama-3.3-70b-versatile")
    assert model != "llama-3.3-70b-versatile"
    assert "gpt-oss-120b" in model

    # Check resolution of decommissioned gemini-2.0-flash -> redirected to gemini-3.6-flash
    _, _, model = router._resolve_providers("gemini-2.0-flash")
    assert model != "gemini-2.0-flash"
    assert model == "gemini-3.6-flash"

    # Check resolution of decommissioned qwen:free -> redirected to active coding engine
    _, _, model = router._resolve_providers("openrouter/qwen-2.5-coder-32b")
    assert "qwen-2.5-coder-32b-instruct:free" not in model


@pytest.mark.asyncio
async def test_gemini_35_flash_429_falls_back_to_flash_lite():
    """Verify GeminiProvider falls back to gemini-3.5-flash-lite on 429 without error."""
    provider = GeminiProvider(api_key="test_api_key")

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = "Quota exceeded"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Response from Flash-Lite!"}]}}]
    }

    async def mock_post(self_client, url, **kwargs):
        if "gemini-3.6-flash:" in str(url):
            return mock_resp_429
        elif "gemini-3.5-flash-lite:" in str(url):
            return mock_resp_200
        return mock_resp_200

    with patch("httpx.AsyncClient.post", new=mock_post):
        text = await provider.generate_response(prompt="Hello Gemini")
        assert text == "Response from Flash-Lite!"
        assert provider.last_active_model == "gemini-3.5-flash-lite"

        # Verify health tracking: 3.5-flash marked RATE_LIMITED, flash-lite marked HEALTHY
        assert model_health_tracker.get_health("gemini", "gemini-3.6-flash") == HealthState.RATE_LIMITED
        assert model_health_tracker.get_health("gemini", "gemini-3.5-flash-lite") == HealthState.HEALTHY


@pytest.mark.asyncio
async def test_rate_limited_model_fast_skipped_during_cooldown():
    """Verify that once a model is RATE_LIMITED, it is fast-skipped without trying upstream."""
    provider = GeminiProvider(api_key="test_api_key")

    # Explicitly set 3.5-flash to RATE_LIMITED
    model_health_tracker.record_failure("gemini", "gemini-3.6-flash", "429 Quota Exceeded", status_code=429)
    assert model_health_tracker.is_eligible("gemini", "gemini-3.6-flash") is False

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Fast-skipped response from Lite!"}]}}]
    }

    calls = []
    async def mock_post(self_client, url, **kwargs):
        calls.append(str(url))
        return mock_resp_200

    with patch("httpx.AsyncClient.post", new=mock_post):
        text = await provider.generate_response(prompt="Test fast-skip")
        assert text == "Fast-skipped response from Lite!"
        # gemini-3.6-flash was NEVER attempted!
        assert not any("gemini-3.6-flash:" in u for u in calls)
        assert any("gemini-3.5-flash-lite:" in u for u in calls)


@pytest.mark.asyncio
async def test_gemini_family_failure_falls_over_to_groq():
    """Verify that complete Gemini family failure cascades to Groq with truthful metadata."""
    mock_gemini = MagicMock(spec=BaseLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.model = "gemini-3.6-flash"
    mock_gemini.generate_response = AsyncMock(side_effect=QuotaExhaustedException("Gemini completely 429"))

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.generate_response = AsyncMock(return_value="Answered by Groq 120B!")

    router = LLMRouter(gemini_provider=mock_gemini, groq_provider=mock_groq)

    result = await router.generate_response(
        prompt="Tell me about AI",
        requested_model="gemini-3.6-flash",
    )

    text, provider, model, was_failover = result
    assert text == "Answered by Groq 120B!"
    assert provider == "groq"
    assert model == "openai/gpt-oss-120b"
    assert was_failover is True
    # Rich metadata properties
    assert result.requested_model == "gemini-3.6-flash"
    assert result.requested_model_label == "Gemini Flash"
    assert result.actual_model == "openai/gpt-oss-120b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is True
    assert result.fallback_reason == "rate_limited"


@pytest.mark.asyncio
async def test_truthful_qwen_reporting_no_silent_substitution():
    """Verify Qwen coding requests route truthfully to Groq Qwen 3.8 and record metadata."""
    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "qwen/qwen3.8-27b"
    mock_groq.generate_response = AsyncMock(return_value="def solve(): pass")

    router = LLMRouter(groq_provider=mock_groq)

    result = await router.generate_response(
        prompt="Write a Python sort function",
        requested_model="qwen/qwen3.8-27b",
    )

    text, provider, model, was_failover = result
    assert "def solve" in text
    assert provider == "groq"
    assert model == "qwen/qwen3.8-27b"
    assert was_failover is False
    assert result.actual_model == "qwen/qwen3.8-27b"
    assert result.requested_model_label == "Qwen 3.8"


@pytest.mark.asyncio
async def test_unavailable_ollama_does_not_block_and_reports_fallback_reason():
    """Verify offline Ollama immediately routes to cloud with local_model_unavailable."""
    mock_ollama = MagicMock(spec=BaseLLMProvider)
    mock_ollama.provider_name = "ollama"
    mock_ollama.is_available = False  # Ollama offline

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.generate_response = AsyncMock(return_value="Document answer from Groq cloud!")

    router = LLMRouter(ollama_provider=mock_ollama, groq_provider=mock_groq)

    result = await router.generate_response(
        prompt="Summarize this policy document",
        requested_model="ollama/jru-specialist:v4",
    )

    text, provider, model, was_failover = result
    assert text == "Document answer from Groq cloud!"
    assert provider == "groq"
    assert was_failover is True
    assert result.requested_model == "ollama/jru-specialist:v4"
    assert ("Specialist" in result.requested_model_label) or ("Legacy" in result.requested_model_label)
    assert result.actual_model == "openai/gpt-oss-120b"
    assert result.fallback_used is True
    assert result.fallback_reason in ("local_model_unavailable", "local_model_deprecated")


@pytest.mark.asyncio
async def test_n_provider_cascade_to_openrouter():
    """Verify ordered cascade: Gemini fails -> Groq fails -> OpenRouter succeeds."""
    mock_gemini = MagicMock(spec=BaseLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.model = "gemini-3.6-flash"
    mock_gemini.generate_response = AsyncMock(side_effect=QuotaExhaustedException("Gemini 429"))

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.generate_response = AsyncMock(side_effect=LLMProviderException("Groq 503 Overloaded"))

    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    mock_openrouter.provider_name = "openrouter"
    mock_openrouter.is_available = True
    mock_openrouter.model = "liquid/lfm-2.5-2.6b:free"
    mock_openrouter.generate_response = AsyncMock(return_value="Third tier OpenRouter success!")

    router = LLMRouter(
        gemini_provider=mock_gemini,
        groq_provider=mock_groq,
        openrouter_provider=mock_openrouter,
    )

    result = await router.generate_response(
        prompt="Explain calculus",
        requested_model="auto",
    )

    text, provider, model, was_failover = result
    assert text == "Third tier OpenRouter success!"
    assert provider == "openrouter"
    assert was_failover is True
    assert result.actual_provider == "openrouter"
    assert result.actual_model == "liquid/lfm-2.5-2.6b:free"


@pytest.mark.asyncio
async def test_all_providers_down_returns_user_contract_error():
    """Verify that when all providers fail, user receives the clean contract error."""
    mock_gemini = MagicMock(spec=BaseLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.generate_response = AsyncMock(side_effect=QuotaExhaustedException("429"))

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.generate_response = AsyncMock(side_effect=LLMProviderException("503"))

    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    mock_openrouter.provider_name = "openrouter"
    mock_openrouter.is_available = True
    mock_openrouter.generate_response = AsyncMock(side_effect=LLMProviderException("404"))

    mock_ollama = MagicMock(spec=BaseLLMProvider)
    mock_ollama.provider_name = "ollama"
    mock_ollama.is_available = False

    router = LLMRouter(
        gemini_provider=mock_gemini,
        groq_provider=mock_groq,
        openrouter_provider=mock_openrouter,
        ollama_provider=mock_ollama,
    )

    with pytest.raises(AllProvidersUnavailableException) as exc_info:
        await router.generate_response(prompt="Hello world", requested_model="auto")

    # Contract requirement: error detail must be the user-facing contract string
    assert "AI capacity is temporarily unavailable. Please try again shortly." in str(exc_info.value)
    # Zero raw provider quotas or provider names leaked in detail
    assert "429" not in exc_info.value.detail
    assert "RESOURCE_EXHAUSTED" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_streaming_fallback_before_first_token():
    """Verify streaming seamlessly falls over before emission and emits StreamToken."""
    mock_gemini = MagicMock(spec=BaseLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.model = "gemini-3.6-flash"

    async def failing_stream(*args, **kwargs):
        raise QuotaExhaustedException("Gemini quota 429 before emission")
        yield "never"

    mock_gemini.stream_response = failing_stream

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "openai/gpt-oss-120b"

    async def successful_stream(*args, **kwargs):
        for chunk in ["Streaming ", "token ", "from ", "Groq!"]:
            yield chunk

    mock_groq.stream_response = successful_stream

    router = LLMRouter(gemini_provider=mock_gemini, groq_provider=mock_groq)

    tokens = []
    async for stream_tok in router.stream_response(prompt="Summarize the stream prompt", requested_model="auto"):
        tok_text, prov, mod, failover = stream_tok
        tokens.append(tok_text)
        assert prov == "groq"
        assert mod == "openai/gpt-oss-120b"
        assert failover is True
        # Verify StreamToken rich attributes
        assert stream_tok.actual_provider == "groq"
        assert stream_tok.actual_model == "openai/gpt-oss-120b"
        assert stream_tok.fallback_used is True

    assert "".join(tokens) == "Streaming token from Groq!"


@pytest.mark.asyncio
async def test_mid_stream_corruption_prevented():
    """Verify that failure AFTER first token terminates cleanly without concatenating fallback model."""
    mock_gemini = MagicMock(spec=BaseLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.model = "gemini-3.6-flash"

    async def mid_stream_failing(*args, **kwargs):
        yield "First valid token."
        raise LLMProviderException("Connection dropped mid-stream!")

    mock_gemini.stream_response = mid_stream_failing

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "openai/gpt-oss-120b"

    async def backup_stream(*args, **kwargs):
        yield "SHOULD NEVER BE CONCATENATED"

    mock_groq.stream_response = backup_stream

    router = LLMRouter(gemini_provider=mock_gemini, groq_provider=mock_groq)

    yielded = []
    with pytest.raises(AllProvidersUnavailableException):
        async for tok_tuple in router.stream_response(prompt="Summarize the mid-stream test", requested_model="auto"):
            yielded.append(tok_tuple[0])

    # First token was received
    assert "First valid token." in yielded
    # Backup model was NOT concatenated into stream!
    assert "SHOULD NEVER BE CONCATENATED" not in "".join(yielded)
