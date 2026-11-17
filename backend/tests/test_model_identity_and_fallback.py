"""Tests for Direct Model Identity, Same-Provider Truthful Fallback, and RAG Gating."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import LLMRouter, GenerationResult, StreamToken
from app.services.llm.circuit_breaker import model_health_tracker
from app.services.chat_service import ChatService, should_check_domain_knowledge, check_domain_knowledge
from app.schemas.chat import ChatRequest


@pytest.fixture(autouse=True)
def reset_health_tracker():
    """Reset model health before and after each test."""
    model_health_tracker.reset_all()
    yield
    model_health_tracker.reset_all()


# ==============================================================================
# TEST A: Qwen requested + Qwen healthy -> Qwen actually called
# ==============================================================================

@pytest.mark.asyncio
async def test_qwen_requested_and_healthy_calls_qwen_sync():
    """Verify that a direct Qwen request sends 'qwen/qwen3.8-27b' in payload and reports fallback_used=False."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_payloads = []

    async def mock_post(url, json=None, headers=None, **kwargs):
        recorded_payloads.append(dict(json or {}))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Qwen coding solution"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await router.generate_response(
            prompt="Write a Python sort function",
            requested_model="qwen/qwen3.8-27b",
        )

    assert len(recorded_payloads) == 1
    assert recorded_payloads[0]["model"] == "qwen/qwen3.8-27b"
    assert result.requested_model == "qwen/qwen3.8-27b"
    assert result.requested_model_label == "Qwen 3.8"
    assert result.actual_model == "qwen/qwen3.8-27b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is False
    assert result.was_failover is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_qwen_requested_and_healthy_calls_qwen_streaming():
    """Verify streaming direct Qwen request sends 'qwen/qwen3.8-27b' and yields fallback_used=False."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_payloads = []

    async def mock_stream_lines():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "def sort():"}}]})
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
            prompt="Write a sort",
            requested_model="qwen/qwen3.8-27b",
        ):
            tokens.append(token_item)

    assert len(recorded_payloads) == 1
    assert recorded_payloads[0]["model"] == "qwen/qwen3.8-27b"
    assert len(tokens) > 0
    t0 = tokens[0]
    assert t0.requested_model == "qwen/qwen3.8-27b"
    assert t0.requested_model_label == "Qwen 3.8"
    assert t0.actual_model == "qwen/qwen3.8-27b"
    assert t0.actual_provider == "groq"
    assert t0.fallback_used is False
    assert t0.is_failover is False


# ==============================================================================
# TEST B: Qwen requested + Qwen 429 -> GPT-OSS fallback + truthful metadata
# ==============================================================================

@pytest.mark.asyncio
async def test_qwen_requested_429_falls_back_to_gpt_oss_truthful_sync():
    """Verify that when Qwen returns 429, Groq falls back to GPT-OSS with fallback_used=True."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_models = []

    async def mock_post(url, json=None, headers=None, **kwargs):
        model = json.get("model")
        recorded_models.append(model)
        mock_resp = MagicMock()
        if model == "qwen/qwen3.8-27b":
            mock_resp.status_code = 429
            mock_resp.headers = {"retry-after": "5"}
            mock_resp.text = "Rate limit reached"
        else:
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "GPT-OSS fallback answer"}}]
            }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await router.generate_response(
            prompt="Write a Python sort function",
            requested_model="qwen/qwen3.8-27b",
        )

    assert recorded_models == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    assert result.requested_model == "qwen/qwen3.8-27b"
    assert result.actual_model == "openai/gpt-oss-120b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is True
    assert result.was_failover is True
    assert result.fallback_reason in ("rate_limited", "model_fallback")


@pytest.mark.asyncio
async def test_qwen_requested_429_falls_back_to_gpt_oss_truthful_stream():
    """Verify that in streaming, Qwen 429 falls back to GPT-OSS with fallback_used=True."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_models = []

    async def mock_gpt_lines():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "GPT-OSS stream token"}}]})
        yield "data: [DONE]"

    class MockStreamContext:
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("json", {}).get("model")
            recorded_models.append(self.model)

        async def __aenter__(self):
            mock_resp = MagicMock()
            if self.model == "qwen/qwen3.8-27b":
                mock_resp.status_code = 429
                mock_resp.headers = {"retry-after": "5"}
            else:
                mock_resp.status_code = 200
                mock_resp.aiter_lines = mock_gpt_lines
            return mock_resp

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", side_effect=MockStreamContext):
        tokens = []
        async for token_item in router.stream_response(
            prompt="Write a sort",
            requested_model="qwen/qwen3.8-27b",
        ):
            tokens.append(token_item)

    assert recorded_models == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    assert len(tokens) > 0
    t0 = tokens[0]
    assert t0.requested_model == "qwen/qwen3.8-27b"
    assert t0.actual_model == "openai/gpt-oss-120b"
    assert t0.actual_provider == "groq"
    assert t0.fallback_used is True
    assert t0.is_failover is True
    assert t0.fallback_reason in ("rate_limited", "model_fallback")


# ==============================================================================
# TEST C: GPT-OSS requested -> GPT-OSS actually called
# ==============================================================================

@pytest.mark.asyncio
async def test_gpt_oss_requested_calls_gpt_oss_directly():
    """Verify that direct request for openai/gpt-oss-120b calls GPT-OSS first with fallback_used=False."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_models = []

    async def mock_post(url, json=None, headers=None, **kwargs):
        recorded_models.append(json.get("model"))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "GPT-OSS reasoning output"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await router.generate_response(
            prompt="Explain general relativity",
            requested_model="openai/gpt-oss-120b",
        )

    assert recorded_models == ["openai/gpt-oss-120b"]
    assert result.requested_model == "openai/gpt-oss-120b"
    assert result.actual_model == "openai/gpt-oss-120b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is False


# ==============================================================================
# TEST D: Auto coding route -> Qwen actually called when healthy
# ==============================================================================

@pytest.mark.asyncio
async def test_auto_coding_route_calls_qwen_when_healthy():
    """Verify auto mode routing a coding prompt dispatches to Qwen 3.8 on Groq."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    recorded_models = []

    async def mock_post(url, json=None, headers=None, **kwargs):
        recorded_models.append(json.get("model"))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "def binary_search(): pass"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await router.generate_response(
            prompt="Write a Python binary search function",
            requested_model="auto",
        )

    assert recorded_models == ["qwen/qwen3.8-27b"]
    assert result.requested_model == "auto"
    assert result.actual_model == "qwen/qwen3.8-27b"
    assert result.actual_provider == "groq"
    assert result.fallback_used is False


# ==============================================================================
# TEST E: No cross-request contamination between Qwen and GPT-OSS
# ==============================================================================

@pytest.mark.asyncio
async def test_no_cross_request_contamination_sequential():
    """Verify that alternating calls between Qwen and GPT-OSS do not contaminate provider state."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    async def mock_post(url, json=None, headers=None, **kwargs):
        model = json.get("model")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": f"Response from {model}"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        # 1. Qwen request
        r1 = await router.generate_response("sort code", requested_model="qwen/qwen3.8-27b")
        assert r1.actual_model == "qwen/qwen3.8-27b"

        # 2. GPT-OSS request
        r2 = await router.generate_response("deep logic", requested_model="openai/gpt-oss-120b")
        assert r2.actual_model == "openai/gpt-oss-120b"

        # 3. Qwen request again
        r3 = await router.generate_response("sort code again", requested_model="qwen/qwen3.8-27b")
        assert r3.actual_model == "qwen/qwen3.8-27b"


# ==============================================================================
# TEST F: Concurrent Qwen/GPT-OSS calls preserve independent model identity
# ==============================================================================

@pytest.mark.asyncio
async def test_concurrent_qwen_and_gpt_oss_preserve_identity():
    """Verify that simultaneous asynchronous requests maintain strict model isolation."""
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b")
    router = LLMRouter(groq_provider=provider)

    async def mock_post(url, json=None, headers=None, **kwargs):
        model = json.get("model")
        await asyncio.sleep(0.02)  # Simulate network latency overlap
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": f"Answer from {model}"}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        tasks = [
            router.generate_response("Task 1", requested_model="qwen/qwen3.8-27b"),
            router.generate_response("Task 2", requested_model="openai/gpt-oss-120b"),
            router.generate_response("Task 3", requested_model="qwen/qwen3.8-27b"),
            router.generate_response("Task 4", requested_model="openai/gpt-oss-120b"),
        ]
        results = await asyncio.gather(*tasks)

    assert results[0].actual_model == "qwen/qwen3.8-27b"
    assert results[1].actual_model == "openai/gpt-oss-120b"
    assert results[2].actual_model == "qwen/qwen3.8-27b"
    assert results[3].actual_model == "openai/gpt-oss-120b"


# ==============================================================================
# TEST G: General query ("Where is Kuttippuram?") must NOT attach RAG citations
# ==============================================================================

@pytest.mark.asyncio
async def test_general_query_kuttippuram_attaches_zero_rag_citations():
    """Verify that 'Where is Kuttippuram?' without web_search or attachments yields sources=[]."""
    assert should_check_domain_knowledge("Where is Kuttippuram?") is False

    chunks, citations = check_domain_knowledge("Where is Kuttippuram?")
    assert chunks == []
    assert citations == []

    chat_service = ChatService()
    req = ChatRequest(
        message="Where is Kuttippuram?",
        model="auto",
        stream=False,
        web_search=False,
    )

    with patch.object(
        chat_service.router,
        "generate_response",
        return_value=GenerationResult(
            text="Kuttippuram is a town in Malappuram district, Kerala, India.",
            provider="gemini",
            model="gemini-3.6-flash",
            was_failover=False,
            actual_model="gemini-3.6-flash",
            actual_provider="gemini",
        ),
    ):
        resp = await chat_service.generate_chat_response(req)

    assert resp.sources == []
    assert "Kuttippuram" in resp.content
    # Irrelevant STEM terms must never appear in sources or citations
    for source in resp.sources:
        title_lower = source.title.lower()
        for stem_term in ["algebra", "kinetics", "relativity", "quantum", "chemistry", "genetics"]:
            assert stem_term not in title_lower
