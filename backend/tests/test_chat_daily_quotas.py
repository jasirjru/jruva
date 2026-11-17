"""Tests verifying daily quota enforcement inside chat and search services."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.quota.daily_usage_service import (
    DailyQuotaExceededError,
    _daily_usage_transaction,
    async_get_daily_usage,
    get_current_utc_date_str,
)
from sqlalchemy import text


async def read_usage(uid):
    """Usage writes run in the background when limits are off; flush before reading."""
    from app.services.chat_service import flush_background_usage_writes

    await flush_background_usage_writes()
    return await async_get_daily_usage(uid)


@pytest.fixture
def chat_service():
    """Create ChatService instance with mocked router."""
    service = ChatService()
    service.router = MagicMock()
    return service


@pytest.mark.asyncio
async def test_non_streaming_chat_enforces_daily_prompt_quota(chat_service):
    """Verify non-streaming chat consumes prompt quota and raises 429 when exhausted."""
    uid = f"chat-quota-user-{uuid.uuid4().hex[:8]}"

    # Mock router response
    mock_res = ("Hello, human!", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    req = ChatRequest(
        message="What is the speed of light in vacuum?",
        model="auto",
        stream=False,
    )
    req.user_id = uid

    with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", 2):
        # 1st prompt succeeds
        res1 = await chat_service.generate_chat_response(req)
        assert res1.content == "Hello, human!"
        usage = await read_usage(uid)
        assert usage["chat_prompts"] == 1

        # 2nd prompt succeeds
        res2 = await chat_service.generate_chat_response(req)
        assert res2.content == "Hello, human!"
        usage = await read_usage(uid)
        assert usage["chat_prompts"] == 2

        # 3rd prompt fails with DailyQuotaExceededError (status 429)
        with pytest.raises(DailyQuotaExceededError) as exc_info:
            await chat_service.generate_chat_response(req)
        assert exc_info.value.quota_type == "chat_prompts"
        assert exc_info.value.limit == 2
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_streaming_chat_enforces_daily_prompt_quota(chat_service):
    """Verify streaming chat emits error SSE event when prompt quota is exhausted."""
    uid = f"stream-quota-user-{uuid.uuid4().hex[:8]}"

    async def mock_stream(*args, **kwargs):
        yield ("Streamed", "groq", "openai/gpt-oss-120b", False)
        yield (" response", "groq", "openai/gpt-oss-120b", False)

    chat_service.router.stream_response = mock_stream

    req = ChatRequest(
        message="Explain general relativity briefly.",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", 1):
        # 1st stream succeeds
        events_1 = []
        async for chunk in chat_service.stream_chat_response(req):
            events_1.append(chunk)
        assert any("Streamed" in c for c in events_1)
        usage = await read_usage(uid)
        assert usage["chat_prompts"] == 1

        # 2nd stream fails with error event
        events_2 = []
        async for chunk in chat_service.stream_chat_response(req):
            events_2.append(chunk)
        assert any("event: error" in c for c in events_2)
        assert any("Daily chat prompt limit reached" in c for c in events_2)


@pytest.mark.asyncio
async def test_web_search_quota_exhaustion_falls_back_gracefully(chat_service):
    """When the daily paid-search allowance is used up, chat still searches the free sources."""
    uid = f"search-quota-user-{uuid.uuid4().hex[:8]}"

    mock_res = ("Direct answer without web grounding", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    req = ChatRequest(
        message="What is the latest news today?",
        model="auto",
        stream=False,
        web_search=True,
    )
    req.user_id = uid

    from app.schemas.chat import SourceCitation

    with patch.object(settings, "DAILY_SEARCHES_PER_USER", 1):
        with patch("app.services.search.web_search.web_search_service.search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [SourceCitation(id="src_1", title="News 1", url="https://news.org", snippet="Text")]

            # 1st search may use the paid provider
            res1 = await chat_service.generate_chat_response(req)
            assert res1.content is not None
            assert mock_search.call_count == 1
            assert mock_search.call_args.kwargs.get("allow_paid", True) is True
            usage = await read_usage(uid)
            assert usage["web_searches"] == 1

            # 2nd search is over the allowance: still searches, free sources only
            res2 = await chat_service.generate_chat_response(req)
            assert res2.content == "Direct answer without web grounding"
            assert mock_search.call_count == 2
            assert mock_search.call_args.kwargs.get("allow_paid") is False
            assert (await read_usage(uid))["web_searches"] == 1


@pytest.mark.asyncio
async def test_streaming_search_over_quota_still_grounds_on_free_sources(chat_service):
    """Guests share one quota row; once it is used up, streaming chat must still search (free sources)."""
    from app.schemas.chat import SourceCitation

    async def fake_stream(*args, **kwargs):
        yield ("Grounded answer.", "groq", "qwen/qwen3.8-27b", False)

    chat_service.router.stream_response = fake_stream
    req = ChatRequest(message="What is the latest ISRO launch news?", model="auto", stream=True, web_search=True)
    with patch.object(settings, "DAILY_GUEST_SEARCH_POOL", 0):
        with patch("app.services.search.web_search.web_search_service.search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [SourceCitation(id="src_1", title="ISRO", url="https://isro.gov.in", snippet="Launch")]
            events = [chunk async for chunk in chat_service.stream_chat_response(req)]
    assert mock_search.call_count == 1
    assert mock_search.call_args.kwargs.get("allow_paid") is False
    assert any("Searching the web" in e for e in events)
    assert any('"citation"' in e for e in events)


@pytest.mark.asyncio
async def test_non_streaming_chat_refunds_quota_on_failure(chat_service):
    """Verify non-streaming chat admits prompt quota but refunds it if generation fails."""
    uid = f"chat-refund-user-{uuid.uuid4().hex[:8]}"

    # Mock router to raise an error during generation
    chat_service.router.generate_response = AsyncMock(
        side_effect=RuntimeError("Provider exploded unexpectedly")
    )

    req = ChatRequest(
        message="Calculate this complex equation.",
        model="auto",
        stream=False,
    )
    req.user_id = uid

    with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", 1):
        # Initial usage is 0
        usage_before = await read_usage(uid)
        assert usage_before["chat_prompts"] == 0

        # A non-capacity bug (RuntimeError) gets the generic internal-error message,
        # not the misleading "all models busy" text.
        res = await chat_service.generate_chat_response(req)
        assert "Something went wrong on our side" in res.content

        # Quota MUST be refunded to 0 (defect: prompt_quota_admitted was not set)
        usage_after = await read_usage(uid)
        assert usage_after["chat_prompts"] == 0

        # User is able to retry because quota was refunded
        chat_service.router.generate_response = AsyncMock(
            return_value=("Success on retry!", "groq", "openai/gpt-oss-120b", False)
        )
        res_retry = await chat_service.generate_chat_response(req)
        assert res_retry.content == "Success on retry!"
        usage_retry = await read_usage(uid)
        assert usage_retry["chat_prompts"] == 1


@pytest.mark.asyncio
async def test_streaming_chat_refunds_quota_on_timeout(chat_service):
    """Verify streaming chat refunds prompt quota if request deadline / timeout expires."""
    uid = f"stream-refund-user-{uuid.uuid4().hex[:8]}"

    async def hanging_stream(*args, **kwargs):
        raise TimeoutError("Simulated streaming timeout before tokens")
        yield  # noqa: unreachable

    chat_service.router.stream_response = hanging_stream

    req = ChatRequest(
        message="Explain quantum entanglement.",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", 1):
        usage_before = await read_usage(uid)
        assert usage_before["chat_prompts"] == 0

        events = []
        async for chunk in chat_service.stream_chat_response(req):
            events.append(chunk)

        assert any("event: error" in c for c in events)
        # Quota must be refunded
        usage_after = await read_usage(uid)
        assert usage_after["chat_prompts"] == 0


@pytest.mark.asyncio
async def test_public_beta_default_allows_prompts_beyond_twenty(chat_service):
    """Verify that by default in public beta (ENABLE_DAILY_PROMPT_LIMIT=False),

    users can submit more than 20 prompts without being rejected by an application-level cap.
    """
    uid = f"public-beta-user-{uuid.uuid4().hex[:8]}"
    mock_res = ("Response text", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    req = ChatRequest(
        message="Test prompt",
        model="auto",
        stream=False,
    )
    req.user_id = uid

    # Send 25 prompts consecutively under default settings
    for i in range(25):
        res = await chat_service.generate_chat_response(req)
        assert res.content == "Response text", f"Prompt {i+1} failed"

    # Verify usage was tracked without blocking
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 25


@pytest.mark.asyncio
async def test_previously_exhausted_users_unblocked_under_default_config(chat_service):
    uid = f"exhausted-user-{uuid.uuid4().hex[:8]}"
    # Pre-seed user with 20 prompts already consumed today
    dt = get_current_utc_date_str()
    async with _daily_usage_transaction() as session:
        await session.execute(
            text(
                "INSERT INTO user_daily_usage (user_id, usage_date, chat_prompts, provider_calls, last_updated) "
                "VALUES (:uid, :dt, 20, 20, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id, usage_date) DO UPDATE SET chat_prompts = 20"
            ),
            {"uid": uid, "dt": dt},
        )

    mock_res = ("Welcome back!", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    req = ChatRequest(
        message="Can I chat now?",
        model="auto",
        stream=False,
    )
    req.user_id = uid

    # Prompt succeeds immediately despite pre-existing 20/20 count
    res = await chat_service.generate_chat_response(req)
    assert res.content == "Welcome back!"
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 21

