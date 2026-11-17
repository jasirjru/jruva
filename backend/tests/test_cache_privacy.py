"""Negative tests for shared cache reuse across private/contextual requests."""
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from app.schemas.chat import ChatRequest
from app.services.cache.response_cache import ResponseCache
from app.services.chat_service import ChatService
import app.services.chat_service as chat_module


@pytest.fixture
def isolated_chat(monkeypatch):
    cache = ResponseCache()
    monkeypatch.setattr(cache, "_embed_prompt", lambda prompt: np.array([1., 0.], dtype=np.float32))
    monkeypatch.setattr(chat_module, "response_cache", cache)
    monkeypatch.setattr(chat_module, "check_domain_knowledge", lambda *a, **k: ([], []))
    monkeypatch.setattr(chat_module.web_search_service, "search", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat_module.memory_service, "format_memory_prompt_block", lambda uid: "Private memory")
    monkeypatch.setattr(chat_module.memory_service, "extract_and_save_facts", MagicMock())
    monkeypatch.setattr(chat_module.conversation_log_service, "log_async", MagicMock())
    router = MagicMock()
    router.generate_response = AsyncMock(return_value=("Fresh safe reply.", "stub", "stub", False))

    async def stream_response(**kwargs):
        yield "Fresh safe reply.", "stub", "stub", False

    router.stream_response = stream_response
    service = ChatService(router=router)
    service.store = MagicMock()
    service.store.get_session_history.return_value = []
    service.store.generate_title_from_message.return_value = "Test"
    return service, cache


async def answer(service, request):
    if not request.stream:
        response = await service.generate_chat_response(request)
        return response.content, response.cached
    events = []
    async for raw in service.stream_chat_response(request):
        for line in raw.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert not any(e["type"] == "error" for e in events)
    tokens = [e for e in events if e["type"] == "token"]
    return "".join(e.get("content", "") for e in tokens), any(e.get("cached") for e in tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scope", ["cross_user", "cross_conversation", "history", "web_search"])
async def test_contextual_requests_neither_read_nor_write_shared_cache(isolated_chat, stream, scope, monkeypatch):
    service, cache = isolated_chat
    prompt = "Explain cache isolation mechanisms in databases."
    cache.set("B", "auto", prompt, "Other caller's private answer", "stub", "stub", 5)
    get_spy = MagicMock(wraps=cache.get)
    set_spy = MagicMock(wraps=cache.set)
    monkeypatch.setattr(cache, "get", get_spy)
    monkeypatch.setattr(cache, "set", set_spy)

    if scope == "cross_user":
        requests = [ChatRequest(message=prompt, tier="B", stream=stream, user_id=uid) for uid in ("alice", "bob")]
    elif scope == "cross_conversation":
        requests = [ChatRequest(message=prompt, tier="B", stream=stream, session_id=uuid4(), conversation_id=str(uuid4())) for _ in range(2)]
    elif scope == "history":
        service.store.get_session_history.return_value = [{"role": "user", "content": "Private earlier context"}]
        requests = [ChatRequest(message=prompt, tier="B", stream=stream, user_id="alice")]
    else:
        requests = [ChatRequest(message=prompt, tier="B", stream=stream, web_search=True)]

    for request in requests:
        content, cached = await answer(service, request)
        assert content == "Fresh safe reply."
        assert not cached
    get_spy.assert_not_called()
    set_spy.assert_not_called()
