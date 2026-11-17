"""Exercise real provider stream parsers against fragmented protocol fixtures."""
import json
from unittest.mock import patch

import httpx
import pytest
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.openrouter_provider import OpenRouterProvider


@pytest.mark.parametrize("provider_type", [OllamaProvider, GroqProvider, OpenRouterProvider])
@pytest.mark.parametrize("pieces", [
    ["Intro <thi", "nk>PRIVATE", " CHAIN</th", "ink> Answer."],
    ["Intro <think>PRIVATE", " CHAIN</think> Answer."],
])
async def test_fragmented_thinking_delimiters_never_leak(provider_type, pieces):
    if provider_type is OllamaProvider:
        provider = provider_type()
        body = "\n".join(json.dumps({"message": {"content": token}}) for token in pieces)
        body += '\n{"done":true}\n'
    else:
        provider = provider_type(api_key="synthetic-audit")
        body = "\n\n".join('data: ' + json.dumps({"choices": [{"delta": {"content": token}}]}) for token in pieces)
        body += '\n\ndata: [DONE]\n\n'
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with patch("httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
        result = "".join([token async for token in provider.stream_response("Tell me a fact")])
    assert result == "Intro  Answer."


async def test_gemini_generate_response_strips_thinking():
    provider = GeminiProvider(api_key="synthetic-audit")
    payload = {"candidates": [{"content": {"parts": [{"text": "Intro <think>PRIVATE CHAIN</think> Answer."}]}}]}
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with patch("httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
        result = await provider.generate_response("Tell me a fact")
    assert result == "Intro  Answer."


@pytest.mark.parametrize("pieces", [
    ["Intro <thi", "nk>PRIVATE", " CHAIN</th", "ink> Answer."],
    ["Intro <think>PRIVATE", " CHAIN</think> Answer."],
])
async def test_gemini_fragmented_thinking_delimiters_never_leak(pieces):
    provider = GeminiProvider(api_key="synthetic-audit")
    body = "\n\n".join(
        "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": token}]}}]}) for token in pieces
    )
    body += "\n\n"
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with patch("httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
        result = "".join([token async for token in provider.stream_response("Tell me a fact")])
    assert result == "Intro  Answer."
