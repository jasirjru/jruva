"""Tests for Multi-Provider Router, Groq Provider, and Models Catalog Endpoint."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.exceptions import AllProvidersUnavailableException, QuotaExhaustedException
from app.services.llm.base import BaseLLMProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import LLMRouter, RequestBudget

client = TestClient(app)


def test_models_catalog_endpoint():
    """Verify /api/v1/models returns active tiers + Auto, hides decommissioned models, and exposes metadata."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default_provider" in data
    assert "active_count" in data

    model_ids = [m["id"] for m in data["models"]]
    # Active Tiers + Auto
    assert "auto" in model_ids
    assert "gemini-3.6-flash" in model_ids
    assert "qwen/qwen3.8-27b" in model_ids
    assert "ollama/jru-specialist:v4" not in model_ids

    # Surfaced count is 3 (auto, gemini-3.6-flash, qwen/qwen3.8-27b)
    assert len(model_ids) == 3

    # Stale/decommissioned and internal fallback models must NOT be surfaced in the public picker
    assert "gemini-2.0-flash" not in model_ids
    assert "openrouter/qwen-2.5-coder-32b" not in model_ids
    assert "groq/deepseek-r1-distill-llama-70b" not in model_ids
    assert "qwen3-8b-local" not in model_ids
    assert "ollama/jru-specialist:v5" not in model_ids
    assert "ollama/jru-specialist:v3" not in model_ids

    # Verify each model exposes health, runtime, fallback_capable, and actual_backend_model
    for m in data["models"]:
        assert "health" in m
        assert "runtime" in m
        assert "fallback_capable" in m
        assert "actual_backend_model" in m


def test_groq_provider_build_messages():
    """Verify GroqProvider creates standard OpenAI compatible messages."""
    provider = GroqProvider(api_key="mock_key", model="llama-3.3-70b-versatile")
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    messages = provider._build_messages(
        prompt="Tell me more",
        system_instruction="Be concise",
        history=history,
    )
    assert len(messages) == 4
    assert messages[0] == {"role": "system", "content": "Be concise"}
    assert messages[1] == {"role": "user", "content": "What is Python?"}
    assert messages[2] == {"role": "assistant", "content": "Python is a programming language."}
    assert messages[3] == {"role": "user", "content": "Tell me more"}


@pytest.mark.asyncio
async def test_router_automatic_failover():
    """Verify router automatically fails over to backup provider when primary hits 429 quota exhaustion."""
    # Mock primary provider that throws QuotaExhaustedException
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "groq"
    mock_primary.is_available = True
    mock_primary.model = "qwen/qwen3.8-27b"
    mock_primary.generate_response = AsyncMock(side_effect=QuotaExhaustedException("Rate limit 429"))

    # Mock backup provider that succeeds
    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "gemini"
    mock_backup.is_available = True
    mock_backup.model = "gemini-3.6-flash"
    mock_backup.generate_response = AsyncMock(return_value="Failover response from Gemini!")

    router = LLMRouter(gemini_provider=mock_backup, groq_provider=mock_primary)
    
    # Everyday chat routes Groq Qwen first -> Gemini backup
    text, provider, model, was_failover = await router.generate_response(
        prompt="Hello",
        requested_model="auto",
        tier="A",
    )

    assert text == "Failover response from Gemini!"
    assert provider == "gemini"
    assert was_failover is True
    assert mock_primary.generate_response.called
    assert mock_backup.generate_response.called


@pytest.mark.asyncio
async def test_router_tier_b_automatic_failover():
    """Verify everyday chat routes to Groq first and automatically fails over to Gemini."""
    mock_gemini = MagicMock(spec=BaseLLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.is_available = True
    mock_gemini.model = "gemini-3.6-flash"
    mock_gemini.generate_response = AsyncMock(return_value="Failover response from Gemini!")

    mock_groq = MagicMock(spec=BaseLLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.is_available = True
    mock_groq.model = "qwen/qwen3.8-27b"
    mock_groq.generate_response = AsyncMock(side_effect=QuotaExhaustedException("Groq rate limit"))

    router = LLMRouter(gemini_provider=mock_gemini, groq_provider=mock_groq)

    text, provider, model, was_failover = await router.generate_response(
        prompt="Hello Daily",
        requested_model="auto",
    )

    assert text == "Failover response from Gemini!"
    assert provider == "gemini"
    assert was_failover is True
    assert mock_groq.generate_response.called
    assert mock_gemini.generate_response.called



@pytest.mark.asyncio
async def test_router_streaming_failover():
    """Verify router automatically fails over to backup stream when primary fails before yielding tokens."""
    # Mock primary that fails immediately on stream_response
    mock_primary = MagicMock(spec=BaseLLMProvider)
    mock_primary.provider_name = "groq"
    mock_primary.is_available = True
    mock_primary.model = "qwen/qwen3.8-27b"

    async def mock_failing_stream(*args, **kwargs):
        raise QuotaExhaustedException("429 Too Many Requests")
        yield "never"

    mock_primary.stream_response = mock_failing_stream

    # Mock backup that streams normally
    mock_backup = MagicMock(spec=BaseLLMProvider)
    mock_backup.provider_name = "gemini"
    mock_backup.is_available = True
    mock_backup.model = "gemini-3.6-flash"

    async def mock_successful_stream(*args, **kwargs):
        yield "Hello "
        yield "from "
        yield "backup!"

    mock_backup.stream_response = mock_successful_stream

    router = LLMRouter(gemini_provider=mock_backup, groq_provider=mock_primary)

    tokens = []
    providers = []
    failovers = []

    async for token, provider, model, is_failover in router.stream_response(
        prompt="Test streaming",
        requested_model="auto",
        tier="A",
    ):
        tokens.append(token)
        providers.append(provider)
        failovers.append(is_failover)

    assert "".join(tokens) == "Hello from backup!"
    assert all(p == "gemini" for p in providers)
    assert all(f is True for f in failovers)


def test_task_detection_heuristics():
    """Verify detect_task_type correctly categorizes user prompts into the 4 tiers."""
    from app.services.llm.router import detect_task_type

    # Coding tier
    assert detect_task_type("Write a Python function to parse JSON") == "coding"
    assert detect_task_type("How to debug this React hook? const [val, setVal] = useState()") == "coding"
    assert detect_task_type("```def hello(): pass```") == "coding"

    # Deep Reasoning tier
    assert detect_task_type("Solve the differential equation dy/dx = 2x step by step") == "reasoning"
    assert detect_task_type("Prove that the square root of 2 is irrational") == "reasoning"
    assert detect_task_type("Calculate the probability of getting 3 heads in a row") == "reasoning"

    # Documentation tier
    assert detect_task_type("Summarize this marks card and calculate the CGPA") == "documentation"
    assert detect_task_type("Please extract the tabular syllabus from this document") == "documentation"
    assert detect_task_type("Give me a summary of the university CBCS ordinance") == "documentation"

    # Daily tier (default)
    assert detect_task_type("Hello, how are you?") == "daily"
    assert detect_task_type("What is the capital of France?") == "daily"
    assert detect_task_type("Tell me a quick joke for today") == "daily"


def test_router_auto_routes_by_task():
    """Verify router._resolve_providers routes prompt to the corresponding tier in auto mode."""
    from unittest.mock import PropertyMock

    mock_gemini = MagicMock(spec=BaseLLMProvider)
    type(mock_gemini).is_available = PropertyMock(return_value=True)
    type(mock_gemini).provider_name = PropertyMock(return_value="gemini")

    mock_groq = MagicMock(spec=BaseLLMProvider)
    type(mock_groq).is_available = PropertyMock(return_value=True)
    type(mock_groq).provider_name = PropertyMock(return_value="groq")

    mock_ollama = MagicMock(spec=BaseLLMProvider)
    type(mock_ollama).is_available = PropertyMock(return_value=True)
    type(mock_ollama).provider_name = PropertyMock(return_value="ollama")

    mock_openrouter = MagicMock(spec=BaseLLMProvider)
    type(mock_openrouter).is_available = PropertyMock(return_value=True)
    type(mock_openrouter).provider_name = PropertyMock(return_value="openrouter")

    router = LLMRouter(
        gemini_provider=mock_gemini,
        groq_provider=mock_groq,
        ollama_provider=mock_ollama,
        openrouter_provider=mock_openrouter,
    )

    # Coding -> Groq Qwen 3.8 27B
    primary, backup, model = router._resolve_providers(
        requested_model="auto",
        prompt="Write a Python script to sort an array",
    )
    assert primary == mock_groq
    assert "qwen" in model

    # Reasoning -> Groq GPT-OSS 120B (DeepSeek decommissioned upstream)
    primary, backup, model = router._resolve_providers(
        requested_model="auto",
        prompt="Solve the system of linear equations and prove uniqueness",
    )
    assert primary == mock_groq
    assert "gpt-oss-120b" in model

    # Documentation -> Gemini Flash (Cloud-backed)
    primary, backup, model = router._resolve_providers(
        requested_model="auto",
        prompt="Summarize the university syllabus and ordinance",
    )
    assert primary == mock_gemini
    assert "gemini" in model

    # Daily -> Groq Qwen 3.8 27B (fastest first token), Gemini backup
    primary, backup, model = router._resolve_providers(
        requested_model="auto",
        prompt="Hi, what is the weather like today?",
    )
    assert primary == mock_groq
    assert model == "qwen/qwen3.8-27b"
    assert backup == mock_gemini


def test_request_budget_timing_and_candidate_budgeting():
    """Verify RequestBudget accurately calculates remaining time, timeout bounds, and candidate feasibility."""
    budget = RequestBudget(total_seconds=30.0, first_token_target_seconds=8.0)
    assert not budget.is_expired()
    assert 29.0 <= budget.remaining_seconds <= 30.0
    assert budget.elapsed_seconds < 1.0

    to_cand = budget.timeout_for_candidate(default_timeout=15.0)
    assert 14.0 <= to_cand <= 15.0
    assert budget.can_attempt_candidate(min_seconds=3.0) is True


def test_request_budget_expired_state():
    """Verify expired budget returns False for can_attempt_candidate and True for is_expired."""
    import time
    budget = RequestBudget(total_seconds=0.05, first_token_target_seconds=0.02)
    time.sleep(0.06)
    assert budget.is_expired() is True
    assert budget.can_attempt_candidate(min_seconds=1.0) is False
    assert budget.timeout_for_candidate(default_timeout=10.0) == 0.0


@pytest.mark.asyncio
async def test_router_aborts_fallback_when_budget_expired():
    """Verify LLMRouter does not attempt subsequent providers when the request budget has expired."""
    import asyncio
    router = LLMRouter()

    p1 = MagicMock(spec=BaseLLMProvider)
    p1.provider_name = "gemini"
    p1.is_available.return_value = True

    async def slow_stream(*args, **kwargs):
        await asyncio.sleep(0.08)
        raise TimeoutError("Upstream timeout")
        yield "token"

    p1.stream_response = slow_stream

    p2 = MagicMock(spec=BaseLLMProvider)
    p2.provider_name = "groq"
    p2.is_available.return_value = True
    p2_called = False

    async def backup_stream(*args, **kwargs):
        nonlocal p2_called
        p2_called = True
        yield "recovered"

    p2.stream_response = backup_stream

    router.providers = {"gemini": p1, "groq": p2}
    router.provider_order = ["gemini", "groq"]
    router._resolve_providers = MagicMock(return_value=(p1, p2, "gemini-3.6-flash"))

    tight_budget = RequestBudget(total_seconds=0.05, first_token_target_seconds=0.03)

    tokens = []
    with pytest.raises(AllProvidersUnavailableException):
        async for token in router.stream_response(prompt="Hello", tier="tier_a", budget=tight_budget):
            tokens.append(token)

    assert not p2_called
    assert tokens == []

