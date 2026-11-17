"""A slow or unreachable Qdrant must not stall chat.

Production, 10 Nov 2026: Qdrant Cloud reset every connection. A STEM question such as
"What is photosynthesis?" ran the domain-knowledge lookup synchronously inside the async
request, blocked the event loop and hit "Request timed out after 25s during preprocessing".

Knowledge lookups now run in a worker thread with a short time limit. After a timeout the
lookups are skipped for a cooldown, so later requests answer without waiting.
"""
import asyncio
import time

import pytest
from app.services import chat_service

SLOW_SECONDS = 1.0
TEST_TIMEOUT = 0.1


def slow_lookup(*_args, **_kwargs):
    time.sleep(SLOW_SECONDS)
    return ["late chunk"], ["late citation"]


@pytest.fixture(autouse=True)
def short_limits(monkeypatch):
    monkeypatch.setattr(chat_service, "KNOWLEDGE_LOOKUP_TIMEOUT_SECONDS", TEST_TIMEOUT)
    monkeypatch.setattr(chat_service, "_knowledge_lookup_paused_until", 0.0)


def test_slow_lookup_returns_empty_within_time_limit():
    async def scenario() -> tuple:
        start = time.monotonic()
        result = await chat_service.lookup_knowledge_fail_fast(slow_lookup, "q")
        return result, time.monotonic() - start

    # Measured inside the loop: asyncio.run() itself waits for the leftover thread on exit.
    result, elapsed = asyncio.run(scenario())

    assert result == ([], [])
    assert elapsed < SLOW_SECONDS / 2


def test_slow_lookup_does_not_block_event_loop():
    async def scenario() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        await chat_service.lookup_knowledge_fail_fast(slow_lookup, "q")
        task.cancel()
        return ticks

    assert asyncio.run(scenario()) >= 3


def test_timeout_pauses_later_lookups():
    calls = []

    def counting_lookup(*_args, **_kwargs):
        calls.append(1)
        return slow_lookup()

    asyncio.run(chat_service.lookup_knowledge_fail_fast(counting_lookup, "q"))
    start = time.monotonic()
    result = asyncio.run(chat_service.lookup_knowledge_fail_fast(counting_lookup, "q"))

    assert result == ([], [])
    assert len(calls) == 1
    assert time.monotonic() - start < TEST_TIMEOUT


def test_fast_lookup_result_is_returned():
    def fast_lookup(query, top_k=3):
        return [f"chunk for {query}"], [f"cite {top_k}"]

    result = asyncio.run(chat_service.lookup_knowledge_fail_fast(fast_lookup, "photosynthesis", top_k=2))

    assert result == (["chunk for photosynthesis"], ["cite 2"])


def test_lookup_error_returns_empty_without_pausing():
    def broken_lookup(*_args, **_kwargs):
        raise ConnectionResetError("reset by peer")

    assert asyncio.run(chat_service.lookup_knowledge_fail_fast(broken_lookup, "q")) == ([], [])
    assert chat_service._knowledge_lookup_paused_until == 0.0
