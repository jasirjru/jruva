"""Regression tests proving Gemini authentication headers, secret isolation, and absence of keys in URLs."""
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.services.llm.gemini_provider import GeminiProvider


SECRET_DUMMY_KEY = "AIzaSyDummySecretKeyForSecurityTesting987654321"


@pytest.mark.asyncio
async def test_gemini_generate_response_uses_headers_not_url():
    """Verify generate_response places the key in x-goog-api-key header and NOT in request URL."""
    provider = GeminiProvider(api_key=SECRET_DUMMY_KEY, model="gemini-3.6-flash")
    captured_requests = []

    async def mock_post(url, *args, **kwargs):
        captured_requests.append({
            "url": str(url),
            "headers": kwargs.get("headers", {}),
            "json": kwargs.get("json", {}),
        })
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello secure world"}]}}]
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        res = await provider.generate_response(prompt="Test prompt")

    assert res == "Hello secure world"
    assert len(captured_requests) >= 1
    for req in captured_requests:
        assert SECRET_DUMMY_KEY not in req["url"], f"SECRET LEAK: API key found in URL query: {req['url']}"
        assert "key=" not in req["url"]
        assert req["headers"].get("x-goog-api-key") == SECRET_DUMMY_KEY
        assert req["headers"].get("Content-Type") == "application/json"


@pytest.mark.asyncio
async def test_gemini_stream_response_uses_headers_not_url():
    """Verify stream_response places the key in x-goog-api-key header and NOT in request URL."""
    provider = GeminiProvider(api_key=SECRET_DUMMY_KEY, model="gemini-3.6-flash")
    captured_stream_calls = []

    class MockAsyncStreamContext:
        def __init__(self, method, url, *args, **kwargs):
            captured_stream_calls.append({
                "method": method,
                "url": str(url),
                "headers": kwargs.get("headers", {}),
            })
            self.status_code = 200

        async def __aenter__(self):
            mock_resp = MagicMock()
            mock_resp.status_code = 200

            async def fake_aiter_lines():
                yield 'data: {"candidates": [{"content": {"parts": [{"text": "Stream chunk"}]}}]}'
                yield 'data: [DONE]'

            mock_resp.aiter_lines = fake_aiter_lines
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream", side_effect=MockAsyncStreamContext):
        tokens = []
        async for token in provider.stream_response(prompt="Test streaming"):
            tokens.append(token)

    assert tokens == ["Stream chunk"]
    assert len(captured_stream_calls) >= 1
    for call in captured_stream_calls:
        assert SECRET_DUMMY_KEY not in call["url"], f"SECRET LEAK: API key found in stream URL: {call['url']}"
        assert "key=" not in call["url"]
        assert call["headers"].get("x-goog-api-key") == SECRET_DUMMY_KEY


@pytest.mark.asyncio
async def test_gemini_error_logging_redacts_api_key(caplog):
    """Verify that error logging and body inspection never exposes the raw API key."""
    provider = GeminiProvider(api_key=SECRET_DUMMY_KEY, model="gemini-3.6-flash")

    async def mock_post_err(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        # Simulate upstream Google API echoing the key in an error message
        mock_resp.text = f"Bad request for key {SECRET_DUMMY_KEY}"
        return mock_resp

    with caplog.at_level(logging.ERROR):
        with patch("httpx.AsyncClient.post", side_effect=mock_post_err):
            with pytest.raises(QuotaExhaustedException):
                await provider.generate_response(prompt="Trigger error")

    # Prove raw key is absent from logs
    assert SECRET_DUMMY_KEY not in caplog.text
    assert "[REDACTED]" in caplog.text


@pytest.mark.asyncio
async def test_gemini_network_exception_does_not_leak_key(caplog):
    """Verify that network exceptions (httpx.RequestError) never interpolate the raw API key."""
    provider = GeminiProvider(api_key=SECRET_DUMMY_KEY, model="gemini-3.6-flash")

    async def mock_post_network_err(url, *args, **kwargs):
        raise httpx.ConnectError(f"Failed to connect with {SECRET_DUMMY_KEY}")

    with caplog.at_level(logging.WARNING):
        with patch("httpx.AsyncClient.post", side_effect=mock_post_network_err):
            with pytest.raises(QuotaExhaustedException):
                await provider.generate_response(prompt="Test network failure")

    assert SECRET_DUMMY_KEY not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_gemini_candidate_models_only_verified():
    """Verify that candidate model list excludes deprecated dead models (2.0-flash, 1.5-flash, 2.5-flash)."""
    provider = GeminiProvider(api_key=SECRET_DUMMY_KEY)
    assert provider.model == "gemini-3.6-flash"
    assert "gemini-2.0-flash" not in provider.FALLBACK_MODELS
    assert "gemini-1.5-flash" not in provider.FALLBACK_MODELS
    assert "gemini-2.5-flash" not in provider.FALLBACK_MODELS
    assert provider.FALLBACK_MODELS == ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
