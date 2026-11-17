"""Verification of all 20 search entry points and edge cases.

Verifies Mission B:
1. Search manually enabled.
2. Search manually disabled.
3. Automatic current-information detection.
4. Tavily success.
5. Tavily HTTP 401.
6. Tavily HTTP 403.
7. Tavily HTTP 429.
8. Tavily HTTP 500.
9. Tavily timeout.
10. Tavily malformed response.
11. Tavily zero results.
12. DuckDuckGo fallback.
13. Wikipedia fallback.
14. Every provider unavailable.
15. Missing Tavily API key.
16. Incorrect API key.
17. Concurrent searches.
18. Duplicate requests.
19. User cancellation.
20. Search-query injection attempts.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.schemas.chat import SourceCitation
from app.services.search.web_search import WebSearchService


def make_resp(status_code: int, json_data: dict = None, text: str = "") -> httpx.Response:
    req = httpx.Request("POST", WebSearchService.TAVILY_API_URL)
    if json_data is not None:
        return httpx.Response(status_code=status_code, json=json_data, request=req)
    return httpx.Response(status_code=status_code, text=text, request=req)


@pytest.fixture
def service():
    return WebSearchService()


# 1. Search manually enabled
def test_entry_point_1_manual_toggle_enabled(service):
    assert service.should_search("simple query", manual_toggle=True) is True


# 2. Search manually disabled
def test_entry_point_2_manual_toggle_disabled(service):
    assert service.should_search("latest breaking news today", manual_toggle=False) is False


# 3. Automatic current-information detection
def test_entry_point_3_automatic_detection(service):
    assert service.should_search("What is the latest score in the 2026 match?") is True
    assert service.should_search("ഇന്നത്തെ സ്വർണ്ണ വില എന്താണ്?") is True  # Malayalam temporal intent


# 4. Tavily success
@pytest.mark.asyncio
async def test_entry_point_4_tavily_success(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(200, json_data={"results": [{"title": "Tavily Res", "url": "https://tav.com/1", "content": "Snippet"}]})

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-valid"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service.search("current topic")
        assert len(res) == 1
        assert res[0].id == "tavily-1"
        assert res[0].title == "Tavily Res"


# 5. Tavily HTTP 401
@pytest.mark.asyncio
async def test_entry_point_5_tavily_http_401(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(401, text="Unauthorized key")

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-bad"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 6. Tavily HTTP 403
@pytest.mark.asyncio
async def test_entry_point_6_tavily_http_403(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(403, text="Forbidden access")

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-forbidden"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 7. Tavily HTTP 429
@pytest.mark.asyncio
async def test_entry_point_7_tavily_http_429(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(429, text="Rate limit / quota exceeded")

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-quota-exhausted"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 8. Tavily HTTP 500
@pytest.mark.asyncio
async def test_entry_point_8_tavily_http_500(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(500, text="Internal Server Error")

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 9. Tavily timeout
@pytest.mark.asyncio
async def test_entry_point_9_tavily_timeout(service):
    async def mock_post(self, url, **kwargs):
        raise httpx.TimeoutException("Connection timed out after 5s")

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 10. Tavily malformed response
@pytest.mark.asyncio
async def test_entry_point_10_tavily_malformed(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(200, json_data={"unrelated_key": 123})

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 11. Tavily zero results
@pytest.mark.asyncio
async def test_entry_point_11_tavily_zero_results(service):
    async def mock_post(self, url, **kwargs):
        return make_resp(200, json_data={"results": []})

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []


# 12. DuckDuckGo fallback
@pytest.mark.asyncio
async def test_entry_point_12_ddg_fallback(service):
    ddg_cits = [SourceCitation(id="src-1", title="DDG", url="https://ddg.com", snippet="DDG text", source_type="web")]
    with patch.object(service, "is_tavily_configured", return_value=True), \
         patch.object(service, "_search_tavily", new_callable=AsyncMock, return_value=[]), \
         patch.object(service, "_search_duckduckgo", new_callable=AsyncMock, return_value=ddg_cits):
        res = await service.search("fallback query")
        assert res == ddg_cits


# 13. Wikipedia fallback
@pytest.mark.asyncio
async def test_entry_point_13_wikipedia_fallback(service):
    wiki_cits = [SourceCitation(id="wiki-1", title="Wiki", url="https://wiki.org", snippet="Wiki text", source_type="web")]
    with patch.object(service, "is_tavily_configured", return_value=True), \
         patch.object(service, "_search_tavily", new_callable=AsyncMock, return_value=[]), \
         patch.object(service, "_search_duckduckgo", new_callable=AsyncMock, return_value=[]), \
         patch.object(service, "_search_wikipedia", new_callable=AsyncMock, return_value=wiki_cits):
        res = await service.search("fallback to wiki")
        assert res == wiki_cits


# 14. Every provider unavailable
@pytest.mark.asyncio
async def test_entry_point_14_all_providers_unavailable(service):
    with patch.object(service, "is_tavily_configured", return_value=True), \
         patch.object(service, "_search_tavily", new_callable=AsyncMock, return_value=[]), \
         patch.object(service, "_search_duckduckgo", new_callable=AsyncMock, return_value=[]), \
         patch.object(service, "_search_wikipedia", new_callable=AsyncMock, return_value=[]):
        res = await service.search("everything down")
        assert res == []


# 15. Missing Tavily API key
def test_entry_point_15_missing_api_key(service):
    with patch("app.core.config.settings.TAVILY_API_KEY", ""):
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            assert service.is_tavily_configured() is False


# 16. Incorrect API key handled without leaking secret
@pytest.mark.asyncio
async def test_entry_point_16_incorrect_api_key_safe(service, caplog):
    async def mock_post(self, url, **kwargs):
        return make_resp(401, text="Invalid API token provided")

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-secret-12345"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        res = await service._search_tavily("test query")
        assert res == []
        assert "tvly-secret-12345" not in caplog.text


# 17. Concurrent searches
@pytest.mark.asyncio
async def test_entry_point_17_concurrent_searches(service):
    async def mock_post(self, url, **kwargs):
        payload = kwargs.get("json", {})
        q = payload.get("query", "res")
        return make_resp(200, json_data={"results": [{"title": f"Res {q}", "url": f"https://ex.com/{q}", "content": "c"}]})

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        tasks = [service.search(f"concurrent query {i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        for i, res in enumerate(results):
            assert len(res) == 1


# 18. Duplicate requests (cached)
@pytest.mark.asyncio
async def test_entry_point_18_duplicate_requests(service):
    call_count = 0
    async def mock_post(self, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return make_resp(200, json_data={"results": [{"title": "Cached", "url": "https://cache.com", "content": "text"}]})

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=mock_post):
        r1 = await service.search("duplicate query")
        r2 = await service.search("duplicate query")
        assert call_count == 1
        assert r1 == r2


# 19. User cancellation
@pytest.mark.asyncio
async def test_entry_point_19_user_cancellation(service):
    async def slow_post(self, url, **kwargs):
        await asyncio.sleep(2.0)
        return make_resp(200, json_data={"results": []})

    with patch("app.core.config.settings.TAVILY_API_KEY", "tvly-key"), \
         patch.object(httpx.AsyncClient, "post", new=slow_post):
        task = asyncio.create_task(service.search("slow query"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# 20. Search-query injection attempts
def test_entry_point_20_query_injection_attempt(service):
    malicious_query = "What happened \x00\x08today? \u200B' OR '1'='1' -- DROP TABLE users; " + "a" * 500
    cleaned = service.clean_search_query(malicious_query)

    # Control chars removed, null bytes removed, length capped to 200
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "\u200B" not in cleaned
    assert len(cleaned) <= 200
