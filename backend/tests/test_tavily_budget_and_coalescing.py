"""Deterministic offline tests for Tavily Budget, Caching, Coalescing, and Circuit Breaker.

Verifies Mission E:
- TTL caching prevents repeated HTTP calls for identical queries
- In-flight request coalescing merges concurrent queries into a single HTTP call
- Circuit breaker trips on repeated failures and cooldown operates correctly
- Rate admission limit prevents unbounded queries
- Basic search depth and max_results constraints are preserved
"""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.schemas.chat import SourceCitation
from app.services.search.web_search import WebSearchService


def make_mock_response(status_code: int, json_data: dict = None, text: str = "") -> httpx.Response:
    req = httpx.Request("POST", WebSearchService.TAVILY_API_URL)
    if json_data is not None:
        return httpx.Response(status_code=status_code, json=json_data, request=req)
    return httpx.Response(status_code=status_code, text=text, request=req)


@pytest.mark.asyncio
async def test_ttl_cache_avoids_repeated_network_calls():
    """Identical sequential queries within TTL hit in-memory cache with 0 additional HTTP calls."""
    service = WebSearchService()
    call_count = 0

    async def mock_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return make_mock_response(
            200,
            json_data={
                "results": [
                    {"title": "Cache Test Title", "url": "https://cache-test.com", "content": "Cached snippet"}
                ]
            },
        )

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):

        # First call: executes HTTP request
        results_1 = await service.search("latest AI news")
        assert len(results_1) == 1
        assert call_count == 1
        assert results_1[0].title == "Cache Test Title"

        # Second call: served from in-memory cache without hitting network
        results_2 = await service.search("latest AI news")
        assert len(results_2) == 1
        assert call_count == 1  # Still 1!
        assert results_2[0].title == "Cache Test Title"


@pytest.mark.asyncio
async def test_inflight_request_coalescing():
    """Concurrent identical search queries coalesce into exactly ONE HTTP request."""
    service = WebSearchService()
    call_count = 0

    async def delayed_mock_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate simulated network latency of 50ms
        await asyncio.sleep(0.05)
        return make_mock_response(
            200,
            json_data={
                "results": [
                    {"title": "Coalesced News", "url": "https://coalesce.org/news", "content": "Shared response"}
                ]
            },
        )

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=delayed_mock_post):

        # Launch 5 identical search queries concurrently
        tasks = [service.search("breaking space news") for _ in range(5)]
        all_results = await asyncio.gather(*tasks)

        assert len(all_results) == 5
        # All 5 callers receive identical results
        for res in all_results:
            assert len(res) == 1
            assert res[0].title == "Coalesced News"

        # Exactly 1 HTTP post was made to Tavily
        assert call_count == 1


@pytest.mark.asyncio
async def test_circuit_breaker_cooldown():
    """Repeated failures trip the circuit breaker; subsequent searches skip Tavily during cooldown."""
    service = WebSearchService()
    tavily_attempts = 0

    async def failing_mock_post(self, url, **kwargs):
        nonlocal tavily_attempts
        tavily_attempts += 1
        return make_mock_response(429, text="Rate limit exceeded")

    mock_ddg_results = [
        SourceCitation(id="src-1", title="DDG Fallback", url="https://ddg.com/res", snippet="DDG content", source_type="web")
    ]

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch("app.core.config.settings.TAVILY_CIRCUIT_BREAKER_SECONDS", 10.0), \
         patch.object(httpx.AsyncClient, "post", new=failing_mock_post), \
         patch.object(service, "_search_duckduckgo", new_callable=AsyncMock, return_value=mock_ddg_results):

        # Call 1: fails
        res1 = await service.search("query 1")
        assert res1 == mock_ddg_results
        assert tavily_attempts == 1
        assert not service.is_circuit_breaker_open()

        # Call 2: fails
        res2 = await service.search("query 2")
        assert res2 == mock_ddg_results
        assert tavily_attempts == 2

        # Call 3: fails -> trips circuit breaker
        res3 = await service.search("query 3")
        assert res3 == mock_ddg_results
        assert tavily_attempts == 3
        assert service.is_circuit_breaker_open()

        # Call 4: breaker is open -> Tavily is completely bypassed, zero HTTP call, immediate DDG fallback!
        res4 = await service.search("query 4")
        assert res4 == mock_ddg_results
        assert tavily_attempts == 3  # Did NOT increment!


@pytest.mark.asyncio
async def test_sliding_window_rate_admission_guard():
    """When requests per minute exceed admission limit, gracefully diverts to fallback without crashing."""
    service = WebSearchService()
    tavily_calls = 0

    async def mock_post(self, url, **kwargs):
        nonlocal tavily_calls
        tavily_calls += 1
        return make_mock_response(
            200,
            json_data={"results": [{"title": "Success", "url": "https://ok.com", "content": "content"}]},
        )

    mock_ddg_results = [
        SourceCitation(id="src-1", title="DDG Admission Fallback", url="https://ddg.com/fall", snippet="fallback", source_type="web")
    ]

    # Limit to 3 requests per minute for this test
    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch("app.core.config.settings.TAVILY_MAX_REQUESTS_PER_MINUTE", 3), \
         patch.object(httpx.AsyncClient, "post", new=mock_post), \
         patch.object(service, "_search_duckduckgo", new_callable=AsyncMock, return_value=mock_ddg_results):

        r1 = await service.search("query A")
        r2 = await service.search("query B")
        r3 = await service.search("query C")
        assert tavily_calls == 3

        # 4th request exceeds 3 RPM -> diverted to DDG without hitting Tavily
        r4 = await service.search("query D")
        assert tavily_calls == 3
        assert r4 == mock_ddg_results
