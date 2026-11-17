"""The analytics-only prompt counter must not delay the first token.

With ENABLE_DAILY_PROMPT_LIMIT off (production), the daily-usage write only records
statistics, but it was awaited before generation: one new DB connection to a distant
region (~0.8-1 s). With the limit on, it must still run inline so it can block.
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import app.services.chat_service as chat_module
import app.services.quota.daily_usage_service as usage_module
import pytest
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

SLOW_WRITE_S = 0.8


@pytest.fixture
def chat(monkeypatch):
    monkeypatch.setattr(chat_module, "check_domain_knowledge", lambda *a, **k: ([], []))
    monkeypatch.setattr(chat_module.web_search_service, "search", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat_module.memory_service, "format_memory_prompt_block", lambda uid: "")
    monkeypatch.setattr(chat_module.memory_service, "extract_and_save_facts", MagicMock())
    monkeypatch.setattr(chat_module.conversation_log_service, "log_async", MagicMock())
    monkeypatch.setattr(chat_module.response_cache, "get", MagicMock(return_value=None))
    monkeypatch.setattr(chat_module.response_cache, "set", MagicMock())
    calls = []

    async def slow_consume(user_id):
        await asyncio.sleep(SLOW_WRITE_S)
        calls.append(user_id)

    monkeypatch.setattr(usage_module, "async_check_and_consume_chat_prompt", slow_consume)
    router = MagicMock()

    async def stream_response(**kwargs):
        yield "Canberra.", "stub", "stub", False

    router.stream_response = stream_response
    router.generate_response = AsyncMock(return_value=("Canberra.", "stub", "stub", False))
    router.get_circuit_metrics = MagicMock(return_value={})
    service = ChatService(router=router)
    service.store = MagicMock()
    service.store.get_session_history.return_value = []
    service.store.generate_title_from_message.return_value = "Test"
    return service, calls


async def warm_up(service, calls, monkeypatch):
    """First request pays one-time imports; keep that out of the timed request."""
    monkeypatch.setattr(chat_module.settings, "ENABLE_DAILY_PROMPT_LIMIT", False)
    await seconds_to_first_token(service, "warm up")
    await asyncio.sleep(SLOW_WRITE_S + 0.2)
    calls.clear()


async def seconds_to_first_token(service, message):
    t0 = time.perf_counter()
    async for raw in service.stream_chat_response(ChatRequest(message=message, stream=True)):
        for line in raw.splitlines():
            if line.startswith("data: ") and json.loads(line[6:]).get("type") == "token":
                return time.perf_counter() - t0
    raise AssertionError("no token streamed")


@pytest.mark.asyncio
async def test_disabled_limit_does_not_wait_for_usage_write(chat, monkeypatch):
    service, calls = chat
    await warm_up(service, calls, monkeypatch)
    monkeypatch.setattr(chat_module.settings, "ENABLE_DAILY_PROMPT_LIMIT", False)

    elapsed = await seconds_to_first_token(service, "What is the capital of Australia?")

    assert elapsed < SLOW_WRITE_S / 2
    await asyncio.sleep(SLOW_WRITE_S + 0.2)
    assert calls == ["anonymous"], "usage must still be recorded in the background"


@pytest.mark.asyncio
async def test_disabled_limit_non_stream_does_not_wait(chat, monkeypatch):
    service, calls = chat
    await warm_up(service, calls, monkeypatch)
    monkeypatch.setattr(chat_module.settings, "ENABLE_DAILY_PROMPT_LIMIT", False)

    t0 = time.perf_counter()
    await service.generate_chat_response(ChatRequest(message="What is the capital of Australia?", stream=False))

    assert time.perf_counter() - t0 < SLOW_WRITE_S / 2
    await asyncio.sleep(SLOW_WRITE_S + 0.2)
    assert calls == ["anonymous"]


@pytest.mark.asyncio
async def test_enabled_limit_is_still_checked_before_generation(chat, monkeypatch):
    service, calls = chat
    await warm_up(service, calls, monkeypatch)
    monkeypatch.setattr(chat_module.settings, "ENABLE_DAILY_PROMPT_LIMIT", True)

    elapsed = await seconds_to_first_token(service, "What is the capital of Australia?")

    assert elapsed >= SLOW_WRITE_S
    assert calls == ["anonymous"]
