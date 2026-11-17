"""Qwen on Groq must get the tier's output budget; a 1,000-token clamp cut long answers mid-sentence."""
import json
from unittest.mock import patch

import httpx
import pytest
from app.services.llm.groq_provider import GroqProvider


@pytest.mark.parametrize("stream", [False, True])
async def test_qwen_long_answers_are_not_cut_at_1000_tokens(stream):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        if stream:
            return httpx.Response(200, text='data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    provider = GroqProvider(api_key="synthetic-audit", model="qwen/qwen3.8-27b")
    with patch("httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
        if stream:
            "".join([t async for t in provider.stream_response("Explain TCP in depth", max_tokens=2048)])
        else:
            await provider.generate_response("Explain TCP in depth", max_tokens=2048)
    assert seen["model"] == "qwen/qwen3.8-27b"
    assert seen["max_tokens"] == 2048
