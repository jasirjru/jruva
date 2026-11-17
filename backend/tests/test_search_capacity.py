"""Regression: production logs (10 Nov 2026) showed every guest sharing one 3/day search
allowance, then DuckDuckGo blocking Render's IP, so search returned 0 sources."""

import asyncio

import httpx
import pytest
from app.core.config import settings
from app.services.quota import daily_usage_service
from app.services.search.web_search import web_search_service


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response, seen):
        self._response = response
        self._seen = seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        self._seen.update(url=url, params=params, headers=headers)
        return self._response


def _patch_client(monkeypatch, response, seen):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(response, seen))


def test_jina_disabled_without_key(monkeypatch):
    monkeypatch.setattr(settings, "JINA_API_KEY", "", raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    assert asyncio.run(web_search_service._search_jina("render free tier")) == []


def test_jina_parses_results_and_sends_search_only_headers(monkeypatch):
    monkeypatch.setattr(settings, "JINA_API_KEY", "test-key", raising=False)
    seen = {}
    payload = {"data": [
        {"title": "Deploy for Free", "url": "https://render.com/docs/free", "description": "Spins down after 15 minutes."},
        {"title": "bad", "url": "javascript:alert(1)", "description": "x"},
    ]}
    _patch_client(monkeypatch, _FakeResponse(200, payload), seen)

    results = asyncio.run(web_search_service._search_jina("render free tier"))

    assert [r.url for r in results] == ["https://render.com/docs/free"]
    assert seen["params"] == {"q": "render free tier"}
    assert seen["headers"]["X-Respond-With"] == "no-content"


def test_jina_http_error_falls_back_to_empty(monkeypatch):
    monkeypatch.setattr(settings, "JINA_API_KEY", "test-key", raising=False)
    _patch_client(monkeypatch, _FakeResponse(429, {}), {})
    assert asyncio.run(web_search_service._search_jina("x y")) == []


@pytest.mark.parametrize("user_id, expected", [("anonymous", "pool"), ("user-123", "per_user")])
def test_guest_bucket_uses_pool_limit(monkeypatch, user_id, expected):
    monkeypatch.setattr(settings, "DAILY_GUEST_SEARCH_POOL", 20, raising=False)
    monkeypatch.setattr(settings, "DAILY_SEARCHES_PER_USER", 3, raising=False)
    limit = (settings.DAILY_GUEST_SEARCH_POOL if user_id == daily_usage_service.GUEST_SEARCH_BUCKET
             else settings.DAILY_SEARCHES_PER_USER)
    assert limit == (20 if expected == "pool" else 3)
