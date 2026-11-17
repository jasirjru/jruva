"""Regression test suite for daily usage PostgreSQL timestamptz datatype safety.

Guarantees that user_daily_usage atomic increments, updates, and refunds
use SQL-native CURRENT_TIMESTAMP instead of binding ISO format strings,
preventing asyncpg DataError: invalid input for query argument $3 (expected datetime).
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.quota.daily_usage_service import (
    async_check_and_consume_chat_prompt,
    async_refund_chat_prompt,
    async_check_and_consume_web_search,
    async_check_and_consume_document_upload,
    async_check_and_consume_audio_seconds,
    async_get_daily_usage,
    _daily_usage_transaction,
)


@pytest.mark.asyncio
async def test_all_daily_usage_queries_bind_zero_str_timestamps():
    """Verify that all daily usage admission functions execute cleanly without type mismatch."""
    uid = f"regression-user-{uuid.uuid4().hex[:8]}"

    # 1. Chat prompt admission
    chat_adm = await async_check_and_consume_chat_prompt(uid)
    assert chat_adm["allowed"] is True
    assert chat_adm["used"] == 1

    # 2. Web search admission
    search_adm = await async_check_and_consume_web_search(uid)
    assert search_adm["allowed"] is True
    assert search_adm["used"] == 1

    # 3. Document upload admission
    upload_adm = await async_check_and_consume_document_upload(uid)
    assert upload_adm["allowed"] is True
    assert upload_adm["used"] == 1

    # 4. Audio seconds admission
    audio_adm = await async_check_and_consume_audio_seconds(uid, 30)
    assert audio_adm["allowed"] is True
    assert audio_adm["used"] == 30

    # 5. Chat prompt refund
    await async_refund_chat_prompt(uid)

    # 6. Verify reconciled state
    usage = await async_get_daily_usage(uid)
    assert usage["chat_prompts"] == 0
    assert usage["web_searches"] == 1
    assert usage["document_uploads"] == 1
    assert usage["audio_seconds"] == 30


@pytest.mark.asyncio
async def test_daily_usage_last_updated_is_persisted():
    """Verify that last_updated column is properly populated by CURRENT_TIMESTAMP."""
    uid = f"regression-ts-user-{uuid.uuid4().hex[:8]}"
    await async_check_and_consume_chat_prompt(uid)

    async with _daily_usage_transaction() as session:
        res = await session.execute(
            text("SELECT last_updated FROM user_daily_usage WHERE user_id=:uid"),
            {"uid": uid},
        )
        row = res.fetchone()
        assert row is not None
        assert row[0] is not None
