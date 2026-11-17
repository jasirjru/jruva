"""Comprehensive test suite for Phase 2 (Storage Quotas & Data Reliability) and Phase 3 (Daily Usage Governance)."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from app.core.config import settings
from app.services.quota.daily_usage_service import (
    DailyQuotaExceededError,
    async_check_and_consume_audio_seconds,
    async_check_and_consume_chat_prompt,
    async_check_and_consume_document_upload,
    async_check_and_consume_web_search,
    async_get_daily_usage,
    async_refund_chat_prompt,
    get_seconds_until_utc_midnight,
)
from app.services.storage.document_quota import (
    DocumentQuotaError,
    async_commit_document_upload,
    async_get_document_storage_summary,
    async_release_committed_document,
    async_release_document_reservation,
    async_reserve_document_upload,
)
from app.services.storage.r2_backend import CloudflareR2StorageBackend


# ─────────────────────────────────────────────────────────────────────────────
# 1. Document Storage Quotas & Reservation Ledger Tests (Phase 2B)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_document_reservation_enforces_max_object_bytes():
    """Single file exceeding DOCUMENT_MAX_OBJECT_BYTES must be rejected immediately."""
    with patch.object(settings, "DOCUMENT_MAX_OBJECT_BYTES", 1024):
        with pytest.raises(DocumentQuotaError) as exc_info:
            await async_reserve_document_upload(
                user_id="test-user-1",
                file_bytes=2048,
                filename="large.pdf",
            )
        assert exc_info.value.code == "object_size_exceeded"


@pytest.mark.asyncio
async def test_document_reservation_and_commit_lifecycle():
    """Verify clean reservation -> commit -> usage tracking lifecycle."""
    uid = "lifecycle-user-1"
    file_size = 50_000

    # 1. Reserve
    res = await async_reserve_document_upload(
        user_id=uid,
        file_bytes=file_size,
        filename="notes.pdf",
    )
    res_id = res["reservation_id"]
    assert res["bytes"] == file_size
    assert res["user_id"] == uid

    # 2. Commit
    storage_path = f"supabase://jruva-documents/{uid}/notes.pdf"
    await async_commit_document_upload(res_id, storage_path)

    # 3. Verify usage reflected
    summary = await async_get_document_storage_summary(user_id=uid)
    assert summary["user"]["total_bytes"] >= file_size
    assert summary["user"]["file_count"] >= 1
    assert summary["project"]["total_bytes"] >= file_size

    # 4. Release on deletion
    released = await async_release_committed_document(file_id=res["file_id"], user_id=uid)
    assert released is True

    # 5. Verify usage decremented
    summary_after = await async_get_document_storage_summary(user_id=uid)
    assert summary_after["user"]["total_bytes"] == summary["user"]["total_bytes"] - file_size


@pytest.mark.asyncio
async def test_document_reservation_enforces_user_byte_ceiling():
    """User byte quota exhaustion must fail closed with user_storage_quota_exceeded."""
    uid = "byte-ceiling-user"
    with patch.object(settings, "DOCUMENT_MAX_USER_BYTES", 100_000):
        # First 80 KB upload succeeds
        res1 = await async_reserve_document_upload(uid, 80_000, "doc1.pdf")
        await async_commit_document_upload(res1["reservation_id"], "path1")

        # Second 30 KB upload (80 KB + 30 KB > 100 KB) must be blocked
        with pytest.raises(DocumentQuotaError) as exc:
            await async_reserve_document_upload(uid, 30_000, "doc2.pdf")
        assert exc.value.code == "user_storage_quota_exceeded"

        # Cleanup
        await async_release_committed_document(res1["file_id"], uid)


@pytest.mark.asyncio
async def test_document_reservation_enforces_user_file_count():
    """Exceeding DOCUMENT_MAX_USER_COUNT must raise user_file_count_exceeded."""
    uid = "file-count-user"
    with patch.object(settings, "DOCUMENT_MAX_USER_COUNT", 2):
        res1 = await async_reserve_document_upload(uid, 1000, "doc1.pdf")
        await async_commit_document_upload(res1["reservation_id"], "path1")

        res2 = await async_reserve_document_upload(uid, 1000, "doc2.pdf")
        await async_commit_document_upload(res2["reservation_id"], "path2")

        # 3rd document must fail
        with pytest.raises(DocumentQuotaError) as exc:
            await async_reserve_document_upload(uid, 1000, "doc3.pdf")
        assert exc.value.code == "user_file_count_exceeded"

        # Cleanup
        await async_release_committed_document(res1["file_id"], uid)
        await async_release_committed_document(res2["file_id"], uid)


@pytest.mark.asyncio
async def test_document_uncommitted_reservation_release():
    """Uncommitted reservation release restores available headroom."""
    uid = "abort-user-1"
    res = await async_reserve_document_upload(uid, 20_000, "aborted.pdf")
    res_id = res["reservation_id"]

    # Release without commit
    await async_release_document_reservation(res_id)

    # Next upload succeeds without lingering reservation hold
    res2 = await async_reserve_document_upload(uid, 20_000, "retry.pdf")
    assert res2["bytes"] == 20_000
    await async_release_document_reservation(res2["reservation_id"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Daily User Quotas & Free-Tier Cost Protection Tests (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

import uuid


@pytest.mark.asyncio
async def test_daily_chat_prompt_quota_enforcement():
    """Enforce daily limit on chat prompts and verify 429 quota exception."""
    uid = f"daily-chat-tester-{uuid.uuid4().hex[:8]}"
    with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", 3):
        # Prompts 1, 2, 3 succeed
        r1 = await async_check_and_consume_chat_prompt(uid)
        assert r1["used"] == 1
        assert r1["remaining"] == 2

        r2 = await async_check_and_consume_chat_prompt(uid)
        assert r2["used"] == 2

        r3 = await async_check_and_consume_chat_prompt(uid)
        assert r3["used"] == 3
        assert r3["remaining"] == 0

        # Prompt 4 fails with DailyQuotaExceededError
        with pytest.raises(DailyQuotaExceededError) as exc:
            await async_check_and_consume_chat_prompt(uid)
        err = exc.value
        assert err.quota_type == "chat_prompts"
        assert err.limit == 3
        assert err.used == 3
        assert err.retry_after_seconds > 0

        # Refund restores 1 prompt
        await async_refund_chat_prompt(uid)
        r_refunded = await async_check_and_consume_chat_prompt(uid)
        assert r_refunded["used"] == 3


@pytest.mark.asyncio
async def test_daily_web_search_quota_enforcement():
    """Enforce daily web search allowance (default 3 searches / user)."""
    uid = f"daily-search-tester-{uuid.uuid4().hex[:8]}"
    with patch.object(settings, "DAILY_SEARCHES_PER_USER", 2):
        s1 = await async_check_and_consume_web_search(uid)
        assert s1["used"] == 1

        s2 = await async_check_and_consume_web_search(uid)
        assert s2["used"] == 2

        with pytest.raises(DailyQuotaExceededError) as exc:
            await async_check_and_consume_web_search(uid)
        assert exc.value.quota_type == "web_searches"
        assert exc.value.limit == 2


@pytest.mark.asyncio
async def test_daily_audio_seconds_quota_enforcement():
    """Enforce daily audio transcription duration limit."""
    uid = f"daily-audio-tester-{uuid.uuid4().hex[:8]}"
    with patch.object(settings, "DAILY_AUDIO_SECONDS_PER_USER", 60):
        # 40 seconds succeeds
        a1 = await async_check_and_consume_audio_seconds(uid, 40)
        assert a1["used"] == 40
        assert a1["remaining"] == 20

        # Another 30 seconds exceeds 60s limit
        with pytest.raises(DailyQuotaExceededError) as exc:
            await async_check_and_consume_audio_seconds(uid, 30)
        assert exc.value.quota_type == "audio_seconds"


def test_seconds_until_utc_midnight_calculation():
    """Verify UTC midnight calculation returns positive bounded seconds."""
    secs = get_seconds_until_utc_midnight()
    assert 0 < secs <= 86400


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cloudflare R2 Storage Adapter Tests (Phase 2F)
# ─────────────────────────────────────────────────────────────────────────────

def test_r2_adapter_disabled_by_default():
    """R2 adapter must fail closed when ENABLE_R2_STORAGE is False."""
    backend = CloudflareR2StorageBackend()
    assert backend.is_enabled is False
    assert backend.is_configured is False
    with pytest.raises(RuntimeError) as exc:
        backend.put("test.pdf", b"sample content", "application/pdf")
    assert "disabled by configuration" in str(exc.value)


def test_r2_adapter_validates_path_traversal():
    """R2 adapter must block path traversal in object keys."""
    with patch.object(settings, "ENABLE_R2_STORAGE", True):
        backend = CloudflareR2StorageBackend(
            account_id="acc123",
            access_key_id="key123",
            secret_access_key="sec123",
            bucket_name="test-bucket",
            endpoint_url="https://acc123.r2.cloudflarestorage.com",
        )
        with pytest.raises(ValueError) as exc:
            backend._object_key("r2://test-bucket/../secret.env")
        assert "Path traversal" in str(exc.value)
