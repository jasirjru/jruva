"""Focused regression test suite for end-to-end request deadlines and streaming reliability.

Verifies:
1. Genuine end-to-end deadline budget covering preprocessing, retrieval, streaming chunk waits, and provider cancellation.
2. Partial streaming timeouts produce an explicit error event/status, never a successful completion or cached partial answer.
3. Upstream provider streams are cleanly cancelled via .aclose() when deadlines or chunk timeouts fire.
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.core.config import settings
from app.core.exceptions import AllProvidersUnavailableException
from app.schemas.chat import ChatRequest, SourceCitation
from app.services.cache.response_cache import response_cache
from app.services.chat_service import ChatService
from app.services.llm.base import BaseLLMProvider
from app.services.llm.router import LLMRouter, RequestBudget
from app.services.quota.daily_usage_service import async_get_daily_usage


async def read_usage(uid):
    """Usage writes run in the background when limits are off; flush before reading."""
    from app.services.chat_service import flush_background_usage_writes

    await flush_background_usage_writes()
    return await async_get_daily_usage(uid)


class CancellableHangingProvider(BaseLLMProvider):
    """Mock provider that yields an initial token and then hangs, tracking .aclose()."""

    def __init__(self, name: str = "mock-hanging", model_name: str = "test-model"):
        self._name = name
        self._model_name = model_name
        self.was_closed = False

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model_name

    async def is_available(self) -> bool:
        return True

    async def generate_response(self, prompt: str, **kwargs) -> str:
        return "mock non-streaming response"

    async def stream_response(self, prompt: str, **kwargs):
        try:
            yield "token1 "
            await asyncio.sleep(8.0)  # Hangs longer than candidate budget
            yield "token2"
        finally:
            self.was_closed = True


@pytest.fixture
def chat_service():
    """Create ChatService instance with fresh router."""
    service = ChatService()
    service.router = LLMRouter()
    return service


@pytest.mark.asyncio
async def test_router_stream_inter_chunk_timeout_and_provider_cancellation():
    """Verify router times out waiting for subsequent stream chunks and invokes upstream .aclose()."""
    router = LLMRouter()
    hanging_provider = CancellableHangingProvider()
    router.providers = [hanging_provider]
    router._resolve_providers = MagicMock(return_value=(hanging_provider, None, "test-model"))
    router._collect_fallback_candidates = MagicMock(return_value=[])

    # Create a small budget: 3.5s total, 1.0s first token
    budget = RequestBudget(total_seconds=3.5, first_token_target_seconds=1.0)
    t0 = time.perf_counter()

    tokens = []
    with pytest.raises(AllProvidersUnavailableException) as exc_info:
        async for token_item in router.stream_response(
            prompt="Hello",
            requested_model="auto",
            budget=budget,
        ):
            tokens.append(token_item[0])

    elapsed = time.perf_counter() - t0

    # 1. First token was received
    assert len(tokens) == 1
    assert tokens[0] == "token1 "

    # 2. Raised exception mentions timeout
    assert "timed out waiting for stream chunk" in str(exc_info.value).lower() or "exceeded request deadline" in str(exc_info.value).lower()

    # 3. Provider was cancelled via aclose() in finally block
    assert hanging_provider.was_closed is True

    # 4. Total duration was bounded by the budget, NOT the 8.0s sleep
    assert elapsed < 5.0, f"Expected bounded chunk timeout < 5.0s, got {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_partial_streaming_timeout_emits_explicit_error_and_never_success():
    """Verify partial streaming timeout produces an explicit error event and never caches partial answer."""
    service = ChatService()
    uid = f"stream-partial-timeout-{uuid.uuid4().hex[:8]}"

    # Provider yields two tokens and then raises a timeout
    async def midstream_timeout_generator(*args, **kwargs):
        yield ("First token ", "groq", "test-model", False)
        yield ("second token ", "groq", "test-model", False)
        raise TimeoutError("Deadline exceeded mid-stream")

    service.router = LLMRouter()
    service.router.stream_response = midstream_timeout_generator

    req = ChatRequest(
        message="Give me a very long explanation of cosmology.",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    events = []
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)

    # 1. Initial tokens were yielded to client
    content_chunks = [c for c in events if "First token" in c or "second token" in c]
    assert len(content_chunks) >= 1

    # 2. Terminal event is strictly error, NEVER done or finish_reason: stop
    error_events = [c for c in events if "event: error" in c]
    assert len(error_events) == 1, "Expected exactly 1 event: error event"
    assert "finish_reason" in error_events[0]
    assert '"finish_reason":"error"' in error_events[0]

    # Must NOT have emitted event: done with finish_reason: stop
    done_events = [c for c in events if "event: done" in c]
    assert len(done_events) == 0, f"Expected 0 event: done events on timeout, got {len(done_events)}"
    assert not any('"finish_reason":"stop"' in c for c in events)

    # 3. Response cache MUST NOT have cached the broken partial text
    cached = response_cache.get(tier="FREE", model="auto", prompt=req.message)
    assert cached is None, "Incomplete partial response was erroneously cached!"

    # 4. Quota was refunded because stream suffered a timeout
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 0


@pytest.mark.asyncio
async def test_end_to_end_deadline_covers_preprocessing_and_retrieval():
    """Verify preprocessing / web search exceeding request deadline aborts cleanly without calling LLM."""
    service = ChatService()
    uid = f"e2e-deadline-user-{uuid.uuid4().hex[:8]}"

    req = ChatRequest(
        message="Search and answer the latest updates.",
        model="auto",
        stream=False,
        web_search=True,
    )
    req.user_id = uid

    # Mock web search that sleeps longer than deadline
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(0.5)
        return [SourceCitation(id="src_slow", title="Slow", url="https://example.org", snippet="Slow")]

    # Mock router that should NOT be called if deadline is expired
    mock_router_gen = AsyncMock()
    service.router = LLMRouter()
    service.router.generate_response = mock_router_gen

    with patch.object(settings, "DAILY_PROMPTS_PER_USER", 2):
        with patch("app.services.search.web_search.web_search_service.search", side_effect=slow_search):
            # Pass a mock request budget or configure a small default
            with patch("app.services.chat_service.RequestBudget", lambda *args, **kwargs: RequestBudget(total_seconds=0.3, first_token_target_seconds=0.1)):
                with pytest.raises(AllProvidersUnavailableException) as exc_info:
                    await service.generate_chat_response(req)

                assert "timed out after" in str(exc_info.value).lower() or "preprocessing" in str(exc_info.value).lower()
                # LLM router was not called because preprocessing exhausted the deadline budget
                assert mock_router_gen.call_count == 0

                # Quota was refunded
                usage = await read_usage(uid)
                assert usage["chat_prompts"] == 0


@pytest.mark.asyncio
async def test_gemini_prevents_model_switching_after_partial_tokens():
    """Verify GeminiProvider raises LLMProviderException and never switches models after yielding tokens."""
    import httpx
    from app.core.exceptions import LLMProviderException
    from app.services.llm.gemini_provider import GeminiProvider

    provider = GeminiProvider(api_key="test-key-mock", model="gemini-3.6-flash")
    called_urls = []

    class MockStreamContext:
        def __init__(self, url):
            self.url = url
            self.status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_lines(self):
            if "gemini-3.6-flash" in self.url:
                # Yield 1 token, then simulate mid-stream network drop
                yield 'data: {"candidates": [{"content": {"parts": [{"text": "Hello partial"}]}}]}'
                raise httpx.ReadError("Network connection reset mid-stream")
            else:
                yield 'data: {"candidates": [{"content": {"parts": [{"text": "Different model output"}]}}]}'

    def fake_stream(method, url, *args, **kwargs):
        called_urls.append(str(url))
        return MockStreamContext(str(url))

    with patch("httpx.AsyncClient.stream", side_effect=fake_stream):
        tokens = []
        with pytest.raises(LLMProviderException) as exc_info:
            async for token in provider.stream_response("test prompt"):
                tokens.append(token)

        assert tokens == ["Hello partial"]
        assert "Stream interrupted after yielding tokens" in str(exc_info.value) or "Network connection reset" in str(exc_info.value)
        # MUST NOT have fallen back to second candidate model
        assert len(called_urls) == 1, f"Expected exactly 1 model call, but made {len(called_urls)}: {called_urls}"


@pytest.mark.asyncio
async def test_groq_prevents_model_switching_after_partial_tokens():
    """Verify GroqProvider raises LLMProviderException and never switches models after yielding tokens."""
    import httpx
    from app.core.exceptions import LLMProviderException
    from app.services.llm.groq_provider import GroqProvider

    provider = GroqProvider(api_key="test-key-mock", model="llama-3.3-70b-versatile")
    called_payloads = []

    class MockGroqStreamContext:
        def __init__(self, json_payload):
            self.payload = json_payload
            self.status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_lines(self):
            if len(called_payloads) == 1:
                yield 'data: {"choices": [{"delta": {"content": "First Groq chunk"}}]}'
                raise httpx.ReadError("Groq mid-stream network disconnect")
            else:
                yield 'data: {"choices": [{"delta": {"content": "Fallback Groq chunk"}}]}'

    def fake_groq_stream(method, url, *args, **kwargs):
        payload = kwargs.get("json", {})
        called_payloads.append(payload)
        return MockGroqStreamContext(payload)

    with patch("httpx.AsyncClient.stream", side_effect=fake_groq_stream):
        tokens = []
        with pytest.raises(LLMProviderException) as exc_info:
            async for token in provider.stream_response("test prompt"):
                tokens.append(token)

        assert tokens == ["First Groq chunk"]
        assert "Stream interrupted after yielding tokens" in str(exc_info.value) or "disconnect" in str(exc_info.value)
        # MUST NOT have fallen back to second candidate model
        assert len(called_payloads) == 1, f"Expected 1 candidate call, got {len(called_payloads)}"


def test_request_budget_candidate_timeout_enforces_monotonic_deadline_without_floor():
    """Verify RequestBudget.timeout_for_candidate never floors at 1.0s when deadline is smaller."""
    # 1. When remaining deadline is 0.4s, candidate timeout must not exceed 0.4s
    budget = RequestBudget(total_seconds=0.4, first_token_target_seconds=0.2)
    to_cand = budget.timeout_for_candidate(default_timeout=8.0)
    assert 0.0 < to_cand <= 0.4, f"Expected <= 0.4s, got {to_cand}s (must not floor at 1.0s)"

    # 2. When deadline has expired (0.0s remaining), candidate timeout must be 0.0s
    expired_budget = RequestBudget(total_seconds=0.01)
    time.sleep(0.02)
    assert expired_budget.is_expired()
    assert expired_budget.timeout_for_candidate(default_timeout=8.0) == 0.0


@pytest.mark.asyncio
async def test_partial_streaming_non_timeout_error_refunds_quota_without_double_refund():
    """Verify non-timeout provider failure mid-stream refunds quota and cannot double refund."""
    service = ChatService()
    uid = f"non-timeout-stream-refund-{uuid.uuid4().hex[:8]}"

    # Provider yields token 1, then crashes with non-timeout error
    async def midstream_crash_generator(*args, **kwargs):
        yield ("First token ", "groq", "test-model", False)
        raise RuntimeError("Non-timeout upstream provider socket crash")

    service.router = LLMRouter()
    service.router.stream_response = midstream_crash_generator

    req = ChatRequest(
        message="Hello testing partial non-timeout refund",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    events = []
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)

    # 1. Emitted error event with finish_reason: error
    error_events = [c for c in events if "event: error" in c]
    assert len(error_events) == 1, "Expected exactly 1 event: error"
    assert '"finish_reason":"error"' in error_events[0]

    # 2. Admitted quota was properly refunded to 0 despite having partial output
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 0, f"Expected refunded chat_prompts == 0, got {usage['chat_prompts']}"


@pytest.mark.asyncio
async def test_cancellation_during_stalled_provider_read_after_client_disconnect():
    """Verify stalled provider read cancels promptly when client disconnects, cleaning up upstream."""
    service = ChatService()
    uid = f"disconnect-stalled-user-{uuid.uuid4().hex[:8]}"

    class StalledProvider(BaseLLMProvider):
        def __init__(self):
            self.was_closed = False

        @property
        def provider_name(self) -> str:
            return "mock-stalled"

        @property
        def model(self) -> str:
            return "stalled-model"

        async def is_available(self) -> bool:
            return True

        async def generate_response(self, prompt: str, **kwargs) -> str:
            return "none"

        async def stream_response(self, prompt: str, **kwargs):
            try:
                yield "Initial chunk "
                # Hangs indefinitely simulating stalled upstream provider read
                await asyncio.sleep(15.0)
                yield "Unreachable chunk"
            finally:
                self.was_closed = True

    stalled_p = StalledProvider()
    service.router = LLMRouter()
    service.router.providers = [stalled_p]
    service.router._resolve_providers = MagicMock(return_value=(stalled_p, None, "stalled-model"))
    service.router._collect_fallback_candidates = MagicMock(return_value=[])

    req = ChatRequest(
        message="Test client disconnect during stalled read",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    # Client disconnect checker returns True after 0.1s
    disconnect_state = False

    async def disconnect_checker():
        return disconnect_state

    async def trigger_disconnect_after_delay():
        await asyncio.sleep(0.1)
        nonlocal disconnect_state
        disconnect_state = True

    asyncio.create_task(trigger_disconnect_after_delay())

    t0 = time.perf_counter()
    emitted = []
    async for chunk in service.stream_chat_response(req, client_disconnect_checker=disconnect_checker):
        emitted.append(chunk)
    elapsed = time.perf_counter() - t0

    # 1. Terminated promptly without waiting for 15s provider hang
    assert elapsed < 3.0, f"Expected cancellation in < 3.0s, took {elapsed:.2f}s"

    # 2. Upstream provider stream was cancelled via .aclose()
    assert stalled_p.was_closed is True, "Upstream provider was not cancelled via aclose() on disconnect"

    # 3. Quota was refunded
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 0, f"Expected quota refund on disconnect, got {usage['chat_prompts']}"

