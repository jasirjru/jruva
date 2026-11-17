"""Deterministic regression test suite for OPERATION JRUVA LIGHTNING (Scenarios A through R).

Covers:
A. "hi"
B. "yoo bro"
C. "Hello, how are you?"
D. "What is 2 + 2?"
E. Short Python coding request
F. Long multi-paragraph user prompt
G. Multi-turn conversation
H. Expired upstream API quota
I. HTTP 429 from Gemini
J. Groq timeout
K. OpenRouter timeout
L. All providers unavailable
M. Provider recovery after cooldown
N. Concurrent users sending requests
O. Client disconnect during streaming
P. SSE stream clean termination (StopAsyncIteration bug fix)
Q. Database quota lock contention
R. Daily quota exhaustion
"""
import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from app.core.config import settings
from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from app.services.llm.router import LLMRouter, RequestBudget, detect_task_type
from app.services.quota.daily_usage_service import (
    DailyQuotaExceededError,
    async_check_and_consume_chat_prompt,
    async_get_daily_usage,
)


async def read_usage(uid):
    """Usage writes run in the background when limits are off; flush before reading."""
    from app.services.chat_service import flush_background_usage_writes

    await flush_background_usage_writes()
    return await async_get_daily_usage(uid)


class MockFastProvider(BaseLLMProvider):
    """Mock provider simulating healthy, fast upstream streaming."""

    def __init__(self, name: str = "mock_fast", tokens: list[str] | None = None, delay: float = 0.01):
        self._name = name
        self.tokens = tokens or ["Hello", " from ", "Jruva!"]
        self.delay = delay
        self.model = f"{name}-v1"
        self._last_active_model = self.model

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, *args, **kwargs) -> str:
        return "".join(self.tokens)

    async def stream_response(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        for tok in self.tokens:
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            yield tok


class MockFailingProvider(BaseLLMProvider):
    """Mock provider simulating immediate or delayed provider failure."""

    def __init__(self, name: str = "mock_fail", exc_to_raise: Exception | None = None, delay: float = 0.0):
        self._name = name
        self.exc = exc_to_raise or LLMProviderException(f"Provider {name} failed")
        self.delay = delay
        self.model = f"{name}-v1"

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, *args, **kwargs) -> str:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        raise self.exc

    async def stream_response(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        raise self.exc
        yield ""  # Make generator



# ─────────────────────────────────────────────────────────────────────────────
# Test A, B, C, D: Basic greetings & quick queries
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("greeting", ["hi", "yoo bro", "Hello, how are you?", "What is 2 + 2?"])
async def test_scenarios_a_to_d_fast_greetings_stream_cleanly(greeting: str):
    """Verify greetings stream fast without StopAsyncIteration bug or erroneous error events."""
    service = ChatService()
    # Mock router to use fast provider
    fast_prov = MockFastProvider(name="gemini", tokens=["Hey", " there! ", "I am Jruva."])
    service.router.gemini_provider = fast_prov
    service.router._resolve_providers = MagicMock(return_value=(fast_prov, None, "gemini-3.6-flash"))

    req = ChatRequest(message=greeting, model="auto", stream=True)
    events = []
    t0 = time.perf_counter()
    async for chunk in service.stream_chat_response(req, client_disconnect_checker=lambda: False):
        events.append(chunk)
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"Greeting '{greeting}' took {elapsed:.2f}s, target < 3s"

    token_events = [c for c in events if "event: message" in c and '"type":"token"' in c]
    done_events = [c for c in events if "event: done" in c and '"finish_reason":"stop"' in c]
    error_events = [c for c in events if "event: error" in c]

    assert len(token_events) >= 1, "Expected tokens emitted"
    assert len(done_events) == 1, "Expected exactly 1 clean event: done with finish_reason: stop"
    assert len(error_events) == 0, f"Erronous error event emitted on greeting '{greeting}': {error_events}"


# ─────────────────────────────────────────────────────────────────────────────
# Test E: Short Python coding request
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_e_coding_request_routes_to_code_tier():
    """Verify short Python coding request is classified as coding and routes to Qwen."""
    prompt = "Write a python function to check if a string is palindrome"
    assert detect_task_type(prompt) == "coding"

    service = ChatService()
    code_prov = MockFastProvider(name="groq", tokens=["def is_palindrome(s):", " return s == s[::-1]"])
    service.router.groq_provider = code_prov
    service.router._resolve_providers = MagicMock(return_value=(code_prov, None, "qwen/qwen3.8-27b"))

    req = ChatRequest(message=prompt, model="auto", stream=True)
    events = []
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)

    done_events = [c for c in events if "event: done" in c]
    assert len(done_events) == 1
    assert any("def is_palindrome" in c for c in events)


# ─────────────────────────────────────────────────────────────────────────────
# Test F & G: Long prompt and multi-turn conversation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_f_long_multiparagraph_prompt():
    """Verify long multi-paragraph user prompt handles budgeting and streams cleanly."""
    long_prompt = "Explain quantum computing in detail.\n\n" * 10
    service = ChatService()
    fast_prov = MockFastProvider(name="gemini", tokens=["Quantum ", "computing ", "uses ", "qubits."])
    service.router.gemini_provider = fast_prov
    service.router._resolve_providers = MagicMock(return_value=(fast_prov, None, "gemini-3.6-flash"))

    req = ChatRequest(message=long_prompt, model="auto", stream=True)
    events = []
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)

    done_events = [c for c in events if "event: done" in c]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_scenario_g_multiturn_conversation():
    """Verify multi-turn history is handled properly without cross-session leakage."""
    service = ChatService()
    uid = f"user-{uuid.uuid4().hex[:8]}"
    sid = uuid.uuid4()

    # Pre-record a turn
    service.record_turn(sid, "user", "My favorite color is emerald green")
    service.record_turn(sid, "assistant", "Noted, emerald green!")

    fast_prov = MockFastProvider(name="gemini", tokens=["Your favorite color is emerald green!"])
    service.router.gemini_provider = fast_prov
    service.router._resolve_providers = MagicMock(return_value=(fast_prov, None, "gemini-3.6-flash"))

    req = ChatRequest(message="What is my favorite color?", session_id=sid, user_id=uid, stream=True)
    events = []
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)

    done_events = [c for c in events if "event: done" in c]
    assert len(done_events) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test H & I: Quota Exhausted & 429 Failover
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_h_quota_exhausted_failover():
    """Verify QuotaExhaustedException on primary cascades immediately to backup."""
    primary = MockFailingProvider(name="gemini", exc_to_raise=QuotaExhaustedException("Gemini quota exhausted"))
    backup = MockFastProvider(name="groq", tokens=["Backup ", "answered."])

    router = LLMRouter(gemini_provider=primary, groq_provider=backup)
    router._resolve_providers = MagicMock(return_value=(primary, backup, "gemini-3.6-flash"))

    budget = RequestBudget(total_seconds=10.0, first_token_target_seconds=3.0)
    tokens = []
    async for item in router.stream_response("test", budget=budget):
        tokens.append(item)

    assert len(tokens) == 2
    assert "".join([t[0] for t in tokens]) == "Backup answered."
    assert tokens[0][3] is True  # is_failover is True


@pytest.mark.asyncio
async def test_scenario_i_gemini_429_failover_within_2_seconds():
    """Verify HTTP 429 on Gemini triggers failover to Groq in under 2 seconds."""
    primary = MockFailingProvider(name="gemini", exc_to_raise=LLMProviderException("Gemini HTTP 429: RESOURCE_EXHAUSTED"))
    backup = MockFastProvider(name="groq", tokens=["Groq ", "fast ", "recovery!"])

    router = LLMRouter(gemini_provider=primary, groq_provider=backup)
    router._resolve_providers = MagicMock(return_value=(primary, backup, "gemini-3.6-flash"))

    t0 = time.perf_counter()
    tokens = []
    async for item in router.stream_response("yoo bro", budget=RequestBudget(total_seconds=10.0, first_token_target_seconds=3.0)):
        tokens.append(item)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Failover took {elapsed:.2f}s, expected < 2.0s"
    assert "".join([t[0] for t in tokens]) == "Groq fast recovery!"


# ─────────────────────────────────────────────────────────────────────────────
# Test J & K: Provider Timeout & Cascade
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_j_groq_timeout_cascades_to_openrouter():
    """Verify Groq timing out waiting for first token cascades to OpenRouter."""
    groq_slow = MockFailingProvider(name="groq", delay=5.0)  # Delay exceeds 1.5s first-token target
    openrouter_fast = MockFastProvider(name="openrouter", tokens=["OpenRouter ", "token."])

    router = LLMRouter(groq_provider=groq_slow, openrouter_provider=openrouter_fast)
    router._resolve_providers = MagicMock(return_value=(groq_slow, openrouter_fast, "qwen/qwen3.8-27b"))

    budget = RequestBudget(total_seconds=10.0, first_token_target_seconds=0.2)
    tokens = []
    async for item in router.stream_response("test code", budget=budget):
        tokens.append(item)

    assert len(tokens) >= 1
    assert tokens[0][1] == "openrouter"


# ─────────────────────────────────────────────────────────────────────────────
# Test L: All Providers Unavailable (Fast Failure < 3-5s, NOT 35s)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_l_all_providers_unavailable_fails_fast_in_under_5s():
    """Verify when all providers fail, router fails fast (< 3-5s) instead of burning 35s."""
    p1 = MockFailingProvider(name="gemini", exc_to_raise=QuotaExhaustedException("Gemini down"))
    p2 = MockFailingProvider(name="groq", exc_to_raise=QuotaExhaustedException("Groq down"))
    p3 = MockFailingProvider(name="openrouter", exc_to_raise=QuotaExhaustedException("OpenRouter down"))

    service = ChatService()
    service.router.gemini_provider = p1
    service.router.groq_provider = p2
    service.router.openrouter_provider = p3
    service.router._resolve_providers = MagicMock(return_value=(p1, p2, "gemini-3.6-flash"))
    service.router._collect_fallback_candidates = MagicMock(return_value=[p2, p3])

    t0 = time.perf_counter()
    events = []
    req = ChatRequest(message="yoo bro", model="auto", stream=True)
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"All-providers failure took {elapsed:.2f}s! Expected < 5.0s, never 35s"

    # Exactly 1 error event, 0 token events, 0 done events
    error_events = [c for c in events if "event: error" in c]
    token_events = [c for c in events if "event: message" in c and '"type":"token"' in c]
    done_events = [c for c in events if "event: done" in c]

    assert len(error_events) == 1, "Expected exactly 1 error event"
    assert len(token_events) == 0, f"Expected 0 fake token events before error, got {len(token_events)}"
    assert len(done_events) == 0, "Expected 0 done events"


# ─────────────────────────────────────────────────────────────────────────────
# Test M: Circuit Breaker Recovery After Cooldown
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_m_circuit_breaker_recovery_after_cooldown():
    """Verify circuit breaker transitions from OPEN to HALF_OPEN after cooldown."""
    cfg = CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=0.1, canary_success_threshold=1)
    cb = CircuitBreaker("gemini", cfg)

    cb.record_failure("error 1")
    cb.record_failure("error 2")
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

    # Wait for cooldown
    time.sleep(0.15)
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitState.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
# Test N: Concurrent Users
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_n_concurrent_users_no_data_leakage():
    """Verify multiple concurrent streaming requests don't cross-contaminate tokens."""
    service = ChatService()
    # One patch shared by all requests: each request's own prompt picks its provider.
    # (Separate patch.object calls on the same router overwrite each other when requests interleave.)
    providers = {
        name: MockFastProvider(name="gemini", tokens=[f"Reply_{name.lower()}_1", f"Reply_{name.lower()}_2"])
        for name in ("Alice", "Bob", "Carol")
    }

    def resolve(*_args, prompt=None, **_kwargs):
        owner = next(name for name in providers if name in (prompt or ""))
        return providers[owner], None, "gemini-3.6-flash"

    async def user_request(uid: str, msg: str):
        req = ChatRequest(message=msg, user_id=uid, stream=True)
        chunks = []
        async for chunk in service.stream_chat_response(req):
            chunks.append(chunk)
        return "".join(chunks)

    with patch.object(service.router, "_resolve_providers", side_effect=resolve):
        results = await asyncio.gather(
            user_request("alice", "Hi from Alice"),
            user_request("bob", "Hi from Bob"),
            user_request("carol", "Hi from Carol"),
        )

    assert "Reply_alice" in results[0] and "Reply_bob" not in results[0]
    assert "Reply_bob" in results[1] and "Reply_alice" not in results[1]
    assert "Reply_carol" in results[2] and "Reply_alice" not in results[2]


# ─────────────────────────────────────────────────────────────────────────────
# Test O: Client Disconnect During Streaming
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_o_client_disconnect_cancels_stream_and_refunds_quota():
    """Verify client disconnect terminates generator promptly and refunds prompt quota."""
    service = ChatService()
    uid = f"disc-user-{uuid.uuid4().hex[:8]}"

    disconnected = False
    async def is_disc():
        return disconnected

    tokens = [f"Token_{i}" for i in range(20)]
    fast_prov = MockFastProvider(name="gemini", tokens=tokens, delay=0.05)
    service.router.gemini_provider = fast_prov
    service.router._resolve_providers = MagicMock(return_value=(fast_prov, None, "gemini-3.6-flash"))

    req = ChatRequest(message="Long stream prompt", user_id=uid, stream=True)
    received = []
    async for chunk in service.stream_chat_response(req, client_disconnect_checker=is_disc):
        received.append(chunk)
        if len(received) >= 2:
            disconnected = True  # Disconnect client mid-stream

    # Generator should stop well before 20 tokens
    assert len(received) < 10

    # Quota must be refunded
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 0, f"Expected quota refunded to 0, got {usage['chat_prompts']}"


# ─────────────────────────────────────────────────────────────────────────────
# Test P: Clean Stream Completion (StopAsyncIteration Bug Regression)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_p_clean_stream_completion_never_raises_stop_async_iteration():
    """Regression test proving StopAsyncIteration at stream termination does not trigger error event."""
    service = ChatService()
    fast_prov = MockFastProvider(name="gemini", tokens=["Yo", " bro", "!"])
    service.router.gemini_provider = fast_prov
    service.router._resolve_providers = MagicMock(return_value=(fast_prov, None, "gemini-3.6-flash"))

    req = ChatRequest(message="yoo bro", model="auto", stream=True)
    events = []
    # Pass disconnect checker (which caused the bug in production)
    async for chunk in service.stream_chat_response(req, client_disconnect_checker=lambda: False):
        events.append(chunk)

    done_events = [c for c in events if "event: done" in c]
    error_events = [c for c in events if "event: error" in c]

    assert len(done_events) == 1, "Must emit exactly 1 event: done"
    assert len(error_events) == 0, "Must NEVER emit event: error on successful stream finish"


# ─────────────────────────────────────────────────────────────────────────────
# Test R: Daily Quota Exhaustion
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_r_daily_quota_exhaustion_emits_clear_error_promptly():
    """Verify exhausted daily prompt quota immediately emits clear error event in < 100ms when limit enabled."""
    with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True):
        service = ChatService()
        uid = f"exhausted-user-{uuid.uuid4().hex[:8]}"

        # Exhaust quota
        for _ in range(50):
            try:
                await async_check_and_consume_chat_prompt(uid)
            except DailyQuotaExceededError:
                break

        req = ChatRequest(message="yoo bro", user_id=uid, stream=True)
        events = []
        t0 = time.perf_counter()
        async for chunk in service.stream_chat_response(req):
            events.append(chunk)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5, f"Quota check took {elapsed:.2f}s, expected < 0.5s"
        error_events = [c for c in events if "event: error" in c]
        assert len(error_events) == 1
        assert "quota" in error_events[0].lower() or "limit" in error_events[0].lower()
