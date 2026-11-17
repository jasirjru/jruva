"""Comprehensive Unit Tests for Providers and Web Search Hardening.

Pushes coverage on:
1. app/services/search/web_search.py
2. app/services/llm/ollama_provider.py
3. app/services/llm/openrouter_provider.py
4. app/services/llm/gemini_provider.py
"""
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.services.search.web_search import WebSearchService
from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.openrouter_provider import OpenRouterProvider
from app.services.llm.gemini_provider import GeminiProvider


# ─────────────────────────────────────────────────────────────────────────────
# 1. WebSearchService Hardening
# ─────────────────────────────────────────────────────────────────────────────

def test_web_search_clean_query():
    service = WebSearchService()
    # Punctuation stripping & stop word filtering
    assert service.clean_search_query("what is the current price of Bitcoin, please?!") == "current price Bitcoin"
    # Fallback to raw query if short tokens
    assert service.clean_search_query("the") == "the"
    assert service.clean_search_query("    ") == ""


def test_web_search_empty_or_short_query():
    service = WebSearchService()
    assert service.should_search("") is False
    assert service.should_search("abc") is False


@pytest.mark.asyncio
async def test_web_search_empty_clean_query_returns_empty():
    service = WebSearchService()
    res = await service.search("   ")
    assert res == []


@pytest.mark.asyncio
async def test_web_search_ddg_success_and_parsing():
    service = WebSearchService()
    html_content = """
    <div class="result results_links">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews&rut=1">Example News</a>
      <a class="result__snippet">Breaking news snippet for test.</a>
    </div>
    <div class="result results_links">
      <a class="result__a" href="https://duckduckgo.com/about">Internal Link</a>
      <a class="result__snippet">Internal DuckDuckGo page.</a>
    </div>
    <div class="result results_links">
      <a class="result__a" href="https://directlink.org/article">Direct Article</a>
      <a class="result__snippet">Direct link snippet.</a>
    </div>
    """
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = html_content

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        citations = await service.search("latest news", max_results=3)

    assert len(citations) == 2
    assert citations[0].title == "Example News"
    assert citations[0].url == "https://example.com/news"
    assert citations[1].title == "Direct Article"
    assert citations[1].url == "https://directlink.org/article"


@pytest.mark.asyncio
async def test_web_search_ddg_network_error_fallback_to_wikipedia():
    service = WebSearchService()
    wiki_json = {
        "query": {
            "search": [
                {"title": "Quantum computing", "snippet": "<span>Quantum computing</span> is a rapidly-emerging technology."}
            ]
        }
    }
    mock_wiki_resp = MagicMock(spec=httpx.Response)
    mock_wiki_resp.status_code = 200
    mock_wiki_resp.json.return_value = wiki_json

    async def fake_get(url, *args, **kwargs):
        if "duckduckgo.com" in url:
            raise httpx.RequestError("DDG Blocked / Connection Timed Out")
        elif "wikipedia.org" in url:
            return mock_wiki_resp
        raise ValueError(f"Unexpected url: {url}")

    with patch("httpx.AsyncClient.get", side_effect=fake_get):
        citations = await service.search("Quantum computing research 2026", max_results=2)

    assert len(citations) == 1
    assert "Quantum computing" in citations[0].title
    assert "https://en.wikipedia.org/wiki/Quantum_computing" in citations[0].url
    assert "rapidly-emerging technology" in citations[0].snippet


def test_web_search_empty_citations_grounding():
    service = WebSearchService()
    assert service.build_grounding_context([]) == ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. OllamaProvider Hardening
# ─────────────────────────────────────────────────────────────────────────────

def test_ollama_availability_check():
    provider = OllamaProvider()

    # Success 200
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    with patch("httpx.Client.get", return_value=mock_resp_200):
        assert provider.is_available is True

    # Failure 500
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    with patch("httpx.Client.get", return_value=mock_resp_500):
        assert provider.is_available is False

    # Exception / ConnectError
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Daemon offline")):
        assert provider.is_available is False


def test_ollama_build_messages():
    provider = OllamaProvider(model="ollama/llama3.2")
    assert provider.provider_name == "ollama"

    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "model", "content": "more info"},
        {"role": "user", "content": ""},  # empty content should be skipped
    ]
    msgs = provider._build_messages("my prompt", system_instruction="custom sys", history=history)
    assert len(msgs) == 5
    assert msgs[0] == {"role": "system", "content": "custom sys"}
    assert msgs[1] == {"role": "user", "content": "hello"}
    assert msgs[2] == {"role": "assistant", "content": "hi there"}
    assert msgs[3] == {"role": "assistant", "content": "more info"}
    assert msgs[4] == {"role": "user", "content": "my prompt"}


@pytest.mark.asyncio
async def test_ollama_generate_response_success_and_think_stripping():
    provider = OllamaProvider(model="deepseek-r1")
    response_json = {
        "message": {
            "content": "<think>Thinking through step 1, step 2...</think>Here is the finalized code."
        }
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = response_json

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await provider.generate_response("Write code")

    assert result == "Here is the finalized code."


@pytest.mark.asyncio
async def test_ollama_generate_response_errors():
    provider = OllamaProvider()

    # Non-200 error
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Ollama GPU crash"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(LLMProviderException) as exc:
            await provider.generate_response("Hi")
        assert exc.value.status_code == 500

    # ConnectError
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("offline")):
        with pytest.raises(LLMProviderException) as exc:
            await provider.generate_response("Hi")
        assert "not running" in str(exc.value)


@pytest.mark.asyncio
async def test_ollama_stream_response_success_and_thinking_extraction():
    provider = OllamaProvider()

    stream_chunks = [
        b'{"message": {"content": "Intro <think>"}}\n',
        b'{"message": {"content": "deep reasoning"}}\n',
        b'{"message": {"content": "</think> Outro token."}}\n',
        b'{"done": true}\n',
    ]

    class FakeStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            for c in stream_chunks:
                yield c.decode("utf-8")

    with patch("httpx.AsyncClient.stream", return_value=FakeStreamResponse()):
        tokens = []
        async for token in provider.stream_response("Tell story"):
            tokens.append(token)

    assert "Intro " in tokens
    assert " Outro token." in "".join(tokens)
    assert "deep reasoning" not in "".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenRouterProvider Hardening
# ─────────────────────────────────────────────────────────────────────────────

def test_openrouter_init_and_availability():
    p1 = OpenRouterProvider(api_key="sk-or-test", model="openrouter/deepseek-r1:free")
    assert p1.model == "deepseek/deepseek-r1:free"
    assert p1.is_available is True
    assert p1.provider_name == "openrouter"

    p2 = OpenRouterProvider(api_key="", model="qwen-2.5-coder-32b")
    assert p2.model == "google/gemma-4-31b-it:free"

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=True):
        with patch("app.core.config.settings.OPENROUTER_API_KEY", ""):
            p3 = OpenRouterProvider(api_key=None)
            assert p3.is_available is False


@pytest.mark.asyncio
async def test_openrouter_generate_response_no_key():
    p = OpenRouterProvider(api_key="")
    with patch("app.core.config.settings.OPENROUTER_API_KEY", ""):
        with pytest.raises(LLMProviderException) as exc:
            await p.generate_response("Hello")
        assert "OPENROUTER_API_KEY is not configured" in str(exc.value)


@pytest.mark.asyncio
async def test_openrouter_generate_response_success():
    p = OpenRouterProvider(api_key="sk-test")
    resp_data = {
        "choices": [
            {"message": {"content": "OpenRouter response verified."}}
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = resp_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await p.generate_response("Test prompt", history=[{"role": "user", "content": "past"}])
        assert res == "OpenRouter response verified."


@pytest.mark.asyncio
async def test_openrouter_stream_response_success():
    p = OpenRouterProvider(api_key="sk-test")
    sse_lines = [
        ': ping\n',
        'data: {"choices": [{"delta": {"reasoning": "thought 1"}}]}\n',
        'data: {"choices": [{"delta": {"content": "Hello "}}]}\n',
        'data: {"choices": [{"delta": {"content": "world!"}}]}\n',
        'data: [DONE]\n',
    ]

    class FakeSSEStream:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            for l in sse_lines:
                yield l

    with patch("httpx.AsyncClient.stream", return_value=FakeSSEStream()):
        chunks = []
        async for c in p.stream_response("Hello"):
            chunks.append(c)

    assert chunks == ["Hello ", "world!"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. GeminiProvider Hardening
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_init_and_availability():
    p1 = GeminiProvider(api_key="AIzaSyTestKey123", model="gemini-3.6-flash")
    assert p1.model == "gemini-3.6-flash"
    assert p1.provider_name == "gemini"
    assert p1.is_available is True

    # Legacy / deprecated models normalize to active verified gemini-3.6-flash
    p_legacy = GeminiProvider(api_key="AIzaSyTestKey123", model="gemini-2.0-flash")
    assert p_legacy.model == "gemini-3.6-flash"

    p2 = GeminiProvider(api_key="")
    assert p2.is_available is False


def test_gemini_build_payload_with_attachments_and_history():
    p = GeminiProvider(api_key="AIzaTest")
    history = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
    ]
    attachments = [
        {"mime_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg=="},
    ]
    payload = p._build_payload(
        prompt="Describe this image",
        system_instruction="Be helpful",
        history=history,
        temperature=0.5,
        max_tokens=1000,
        attachments=attachments,
    )
    assert "contents" in payload
    assert len(payload["contents"]) == 3
    # Check history
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"
    # Check current prompt + attachment
    user_turn = payload["contents"][2]
    assert user_turn["role"] == "user"
    assert len(user_turn["parts"]) == 2
    assert user_turn["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert user_turn["parts"][1]["text"] == "Describe this image"
    assert payload["generationConfig"]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_gemini_generate_response_success():
    p = GeminiProvider(api_key="AIzaTest")
    resp_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Gemini verified response."}]
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = resp_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await p.generate_response("Hello Gemini")
        assert res == "Gemini verified response."


@pytest.mark.asyncio
async def test_gemini_generate_response_rate_limit_429():
    p = GeminiProvider(api_key="AIzaTest")
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "RESOURCE_EXHAUSTED: quota exceeded"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(QuotaExhaustedException):
            await p.generate_response("Hello Gemini")


@pytest.mark.asyncio
async def test_gemini_stream_response_success():
    p = GeminiProvider(api_key="AIzaTest")
    stream_lines = [
        'data: {"candidates": [{"content": {"parts": [{"text": "Streamed "}]}}]}\n',
        'data: {"candidates": [{"content": {"parts": [{"text": "Gemini "}]}}]}\n',
        'data: {"candidates": [{"content": {"parts": [{"text": "tokens."}]}}]}\n',
    ]

    class FakeGeminiStream:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            for l in stream_lines:
                yield l

    with patch("httpx.AsyncClient.stream", return_value=FakeGeminiStream()):
        tokens = []
        async for token in p.stream_response("Stream prompt"):
            tokens.append(token)

    assert "".join(tokens) == "Streamed Gemini tokens."
