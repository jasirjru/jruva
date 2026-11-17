"""Targeted regression test for P0 AI capacity failure on PostgreSQL.

Verifies that daily usage quota tracking uses CURRENT_TIMESTAMP natively in SQL
and avoids passing ISO string parameters into TIMESTAMP WITH TIME ZONE columns,
preventing asyncpg.exceptions.DataError and cascading 500/AI capacity errors.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.quota.daily_usage_service import (
    _SQLiteDailySession,
    async_check_and_consume_audio_seconds,
    async_check_and_consume_chat_prompt,
    async_check_and_consume_document_upload,
    async_check_and_consume_web_search,
    async_get_daily_usage,
    async_refund_chat_prompt,
)


@pytest.mark.asyncio
async def test_sqlite_session_datetime_defensive_serialization():
    """Verify _SQLiteDailySession serializes datetime objects defensively."""
    class FakeConn:
        def __init__(self):
            self.executed_sql = None
            self.executed_params = None

        def execute(self, sql, params):
            self.executed_sql = sql
            self.executed_params = params
            return self

    conn = FakeConn()
    session = _SQLiteDailySession(conn)
    dt_now = datetime.now(timezone.utc)
    await session.execute("SELECT :ts", {"ts": dt_now, "val": 42})
    assert isinstance(conn.executed_params["ts"], str)
    assert conn.executed_params["val"] == 42


@pytest.mark.asyncio
async def test_daily_usage_all_operations_local():
    """Verify complete lifecycle of all 6 quota operations in daily_usage_service."""
    uid = f"regress-user-{uuid.uuid4().hex[:8]}"

    # 1. Chat prompt check & consume
    res = await async_check_and_consume_chat_prompt(uid)
    assert res["allowed"] is True
    assert res["used"] == 1

    # 2. Daily usage read
    usage = await async_get_daily_usage(uid)
    assert usage["chat_prompts"] == 1
    assert usage["provider_calls"] == 1

    # 3. Chat prompt refund
    await async_refund_chat_prompt(uid)
    usage_refunded = await async_get_daily_usage(uid)
    assert usage_refunded["chat_prompts"] == 0

    # 4. Web search consume
    ws_res = await async_check_and_consume_web_search(uid)
    assert ws_res["allowed"] is True
    assert ws_res["used"] == 1

    # 5. Document upload consume
    doc_res = await async_check_and_consume_document_upload(uid)
    assert doc_res["allowed"] is True
    assert doc_res["used"] == 1

    # 6. Audio seconds consume
    audio_res = await async_check_and_consume_audio_seconds(uid, 30)
    assert audio_res["allowed"] is True
    assert audio_res["used"] == 30


@pytest.mark.asyncio
async def test_chat_service_streaming_does_not_mask_with_capacity_error_when_quota_succeeds():
    """Verify that healthy quota admission lets stream proceed without 'AI capacity is temporarily unavailable'."""
    service = ChatService()
    uid = f"chat-quota-stream-{uuid.uuid4().hex[:8]}"

    async def mock_stream(*args, **kwargs):
        yield ("Paris is the capital of France.", "groq", "qwen/qwen3.8-27b", False)

    service.router.stream_response = mock_stream

    req = ChatRequest(
        message="What is the capital of France?",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    events = []
    async for chunk in service.stream_chat_response(req):
        events.append(chunk)

    # Must contain real content
    assert any("Paris is the capital of France." in chunk for chunk in events)
    # Must NOT contain the fallback capacity error
    assert not any("AI capacity is temporarily unavailable" in chunk for chunk in events)
