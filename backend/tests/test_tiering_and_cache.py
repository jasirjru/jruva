"""Comprehensive test suite for Multi-Tier Compute Architecture (Tiers B, A, S)
and Sub-15ms In-Memory Semantic Query & Response Caching.
"""
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import pytest

from app.schemas.chat import ChatRequest
from app.services.cache.response_cache import ResponseCache, response_cache
from app.services.chat_service import ChatService
from app.services.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.services.llm.router import LLMRouter, TIER_CONFIGS
from app.services.telemetry.telemetry_service import RequestTrace, TelemetryService


def test_response_cache_lru_and_fingerprinting():
    """Verify deterministic hashing, LRU eviction, and hit metrics in ResponseCache."""
    cache = ResponseCache(max_size=2, default_ttl_seconds=3600)

    # 1. Test Deterministic Key Hashing
    k1 = cache._generate_key("B", "openai/gpt-oss-120b", "Hello World", None)
    k2 = cache._generate_key("b", "openai/gpt-oss-120b", " Hello World ", None)
    assert k1 == k2, "SHA-256 fingerprint should normalize tier case and prompt whitespace"

    # 2. Test Set & Get
    cache.set(
        tier="B",
        model="auto",
        prompt="Explain quantum entanglement",
        content="Quantum entanglement is a physical phenomenon...",
        provider="groq",
        model_name="openai/gpt-oss-120b",
        total_tokens=150,
    )

    hit = cache.get(tier="B", model="auto", prompt="Explain quantum entanglement")
    assert hit is not None
    assert hit["cached"] is True
    assert hit["provider"] == "groq"
    assert hit["total_tokens"] == 150
    assert "entanglement" in hit["content"]

    # 3. Test Miss
    miss = cache.get(tier="B", model="auto", prompt="Different prompt")
    assert miss is None

    # 4. Test LRU capacity eviction (max_size=2)
    cache.set("B", "auto", "Prompt 2", "Response 2", "groq", "model2", 50)
    # Access "Explain quantum entanglement" to make "Prompt 2" the oldest
    _ = cache.get(tier="B", model="auto", prompt="Explain quantum entanglement")
    # Add 3rd item -> "Prompt 2" should be evicted
    cache.set("B", "auto", "Prompt 3", "Response 3", "gemini", "model3", 50)

    assert cache.get(tier="B", model="auto", prompt="Prompt 2") is None
    assert cache.get(tier="B", model="auto", prompt="Explain quantum entanglement") is not None
    assert cache.get(tier="B", model="auto", prompt="Prompt 3") is not None

    # 5. Stats verification
    stats = cache.get_stats()
    assert stats["total_entries"] == 2
    assert stats["hits"] >= 2
    assert stats["misses"] >= 1
    assert stats["hit_ratio_pct"] > 0


def test_response_cache_ttl_expiration():
    """Verify expired items return None and increment miss counters."""
    cache = ResponseCache(max_size=10, default_ttl_seconds=1)
    cache.set("A", "gemini", "Expire test", "Ephemeral answer", "gemini", "gemini-3.7-flash", 40, ttl_seconds=1)

    # Force expiration by setting expires_at in the past
    key = cache._generate_key("A", "gemini", "Expire test")
    cache._cache[key].expires_at = time.time() - 10.0

    hit = cache.get("A", "gemini", "Expire test")
    assert hit is None


def test_router_tier_bias_auto_mode():
    """Verify Auto routing resolves providers according to compute tier profiles."""
    cfg = CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=15.0)

    mock_gemini = MagicMock()
    mock_gemini.provider_name = "gemini"
    mock_gemini.model = "gemini-3.7-flash"
    mock_gemini.is_available = True

    mock_groq = MagicMock()
    mock_groq.provider_name = "groq"
    mock_groq.model = "llama-3.3-70b-versatile"
    mock_groq.is_available = True

    mock_ollama = MagicMock()
    mock_ollama.provider_name = "ollama"
    mock_ollama.model = "my-specialist"
    mock_ollama.is_available = True

    router = LLMRouter(
        gemini_provider=mock_gemini,
        groq_provider=mock_groq,
        ollama_provider=mock_ollama,
        circuit_config=cfg,
    )

    # Tier B: Expect Groq primary (high throughput, zero cost)
    primary_b, backup_b, _ = router._resolve_providers(requested_model="auto", tier="B")
    assert primary_b.provider_name == "groq"
    assert backup_b.provider_name == "gemini"

    # Tier A: Expect Gemini primary (deep reasoning, large context)
    primary_a, backup_a, _ = router._resolve_providers(requested_model="auto", tier="A")
    assert primary_a.provider_name == "gemini"
    assert backup_a.provider_name == "groq"

    # Tier S: Expect Groq primary (high-throughput cloud frontier)
    primary_s, backup_s, _ = router._resolve_providers(requested_model="auto", tier="S")
    assert primary_s.provider_name == "groq"
    assert backup_s.provider_name == "gemini"

    # Tier S++: Expect Groq primary
    primary_spp, backup_spp, _ = router._resolve_providers(requested_model="auto", tier="S++")
    assert primary_spp.provider_name == "groq"
    assert backup_spp.provider_name == "gemini"

    # Tier S+++: Expect Groq primary (GPT-OSS 120B / Qwen 27B) with Gemini 3.6 Flash backup
    primary_sppp, backup_sppp, model_sppp = router._resolve_providers(requested_model="auto", tier="S+++")
    assert primary_sppp.provider_name == "groq"
    assert backup_sppp.provider_name == "gemini"
    assert model_sppp == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_chat_service_tier_budget_and_cache_cycle(monkeypatch):
    """Verify end-to-end ChatService tier token sizing, cache hit speed, and $0 cost."""
    # Reset singleton cache
    response_cache.clear()
    # Exercise the stateless cache path, not an unversioned live RAG corpus.
    monkeypatch.setattr("app.services.chat_service.check_domain_knowledge", lambda *a, **k: ([], []))

    mock_router = MagicMock(spec=LLMRouter)
    mock_router.generate_response = AsyncMock(return_value=(
        "Distributed systems require consensus algorithms like Raft.",
        "groq",
        "llama-3.3-70b-versatile",
        False
    ))

    service = ChatService(router=mock_router)

    # 1. First Call: Cache MISS, invokes Router with Tier B token cap
    req1 = ChatRequest(
        message="Explain distributed consensus algorithms.",
        tier="B",
        stream=False,
    )
    t0 = time.perf_counter()
    res1 = await service.generate_chat_response(req1)
    t1 = time.perf_counter()

    assert res1.cached is False
    assert res1.tier == "B"
    assert "consensus" in res1.content
    assert mock_router.generate_response.call_count == 1

    # Verify router called with Tier B max_tokens (2048) and tier="B"
    _, kwargs = mock_router.generate_response.call_args
    assert kwargs["max_tokens"] == 2048
    assert kwargs["tier"] == "B"

    # 2. Second Call with identical prompt: Cache HIT (Router NOT invoked, < 15ms)
    req2 = ChatRequest(
        message="Explain distributed consensus algorithms.",
        tier="B",
        stream=False,
    )
    t2 = time.perf_counter()
    res2 = await service.generate_chat_response(req2)
    cache_latency_ms = (time.perf_counter() - t2) * 1000

    assert res2.cached is True
    assert res2.tier == "B"
    assert res2.content == res1.content
    # Router must NOT have been called a second time
    assert mock_router.generate_response.call_count == 1
    # Cache hit must be sub-50ms inclusive of SQLite turn persistence
    assert cache_latency_ms < 50.0, f"Cache latency was {cache_latency_ms:.2f}ms, expected < 50ms"


@pytest.mark.asyncio
async def test_chat_service_streaming_cache_hit():
    """Verify streaming response caching yields cached tokens and records telemetry."""
    response_cache.clear()

    # Pre-populate cache
    response_cache.set(
        tier="A",
        model="auto",
        prompt="Stream cache benchmark query",
        content="Streaming cache payload verified.",
        provider="gemini",
        model_name="gemini-3.7-flash",
        total_tokens=25,
    )

    service = ChatService()
    req = ChatRequest(
        message="Stream cache benchmark query",
        tier="A",
        stream=True,
    )

    events = []
    async for sse_raw in service.stream_chat_response(req):
        events.append(sse_raw)

    assert len(events) >= 2
    # Check token event
    assert '"cached":true' in events[0]
    assert '"tier":"A"' in events[0]
    # Check done event
    assert 'event: done' in events[1]
    assert '"cached":true' in events[1]


def test_telemetry_tier_accounting():
    """Verify TelemetryService aggregates requests and costs across Tiers B, A, S."""
    telem = TelemetryService(max_traces=50)

    # Record Tier B
    telem.record_trace(RequestTrace(
        tier="B",
        cached=False,
        total_tokens=500,
        estimated_cost_usd=0.0003,
        total_latency_ms=120.0,
    ))

    # Record Tier A
    telem.record_trace(RequestTrace(
        tier="A",
        cached=False,
        total_tokens=1500,
        estimated_cost_usd=0.0015,
        total_latency_ms=340.0,
    ))

    # Record Tier S (Cached)
    telem.record_trace(RequestTrace(
        tier="S",
        cached=True,
        total_tokens=2000,
        estimated_cost_usd=0.0,
        total_latency_ms=2.1,
    ))

    metrics = telem.get_metrics_summary()
    assert "tiers" in metrics
    tier_reqs = metrics["tiers"]["requests_by_tier"]
    assert tier_reqs["B"] == 1
    assert tier_reqs["A"] == 1
    assert tier_reqs["S"] == 1
    assert metrics["tiers"]["cache_hits"] == 1
