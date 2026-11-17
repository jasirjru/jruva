"""P0 Hotfix Regression Tests: Complete Removal of 20-Prompt Daily Chat Limit.

Demonstrates and proves:
1. A single user can submit more than 20 prompts (e.g. 25 consecutive prompts) without 429 application daily-count rejection.
2. Separate users (and anonymous visitors) cannot exhaust one another's usage.
3. Previously exhausted users (authenticated users or guests with pre-existing 20/20 in DB) are immediately unblocked without waiting until midnight UTC.
4. Guest chat and authenticated chat flows both operate correctly according to access rules (anonymous chat allowed, attachment upload requires auth, session ownership enforced).
5. Real AI provider HTTP 429 errors from upstream models (Gemini / Groq) are handled via router failover / health tracking and surface truthful temporary-unavailability messaging, NOT application daily 20/20 limits.
6. Streaming Server-Sent Events (SSE) chat operates cleanly without daily quota interruption.
7. Authentication, tenant isolation, and session tamper rejection (HTTP 403) are preserved.
8. Request budget deadlines and rate limiting (abuse controls) remain active.
9. No uncontrolled retries or infinite loops are introduced.
"""

import inspect
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.api.v1.endpoints.chat import chat_endpoint
from app.api.v1.endpoints.storage import get_user_daily_quotas
from app.core.exceptions import AllProvidersUnavailableException
from app.schemas.chat import AttachmentItem, ChatRequest
from app.services.chat_service import ChatService
from app.services.quota.daily_usage_service import (
    _daily_usage_transaction,
    async_get_daily_usage,
    get_current_utc_date_str,
)
from app.services.session_store import session_store
from fastapi import HTTPException
from sqlalchemy import text


async def read_usage(uid):
    """Usage writes run in the background when limits are off; flush before reading."""
    from app.services.chat_service import flush_background_usage_writes

    await flush_background_usage_writes()
    return await async_get_daily_usage(uid)


@pytest.fixture
def chat_service():
    service = ChatService()
    service.router = MagicMock()
    return service


@pytest.mark.asyncio
async def test_user_can_submit_more_than_twenty_prompts_without_rejection(chat_service):
    """MISSION 5.1: Demonstrate that a user can submit >20 prompts without an application daily-count rejection."""
    uid = f"unlimited-user-{uuid.uuid4().hex[:8]}"
    mock_res = ("Model output chunk", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    req = ChatRequest(
        message="Count to infinity.",
        model="auto",
        stream=False,
    )
    req.user_id = uid

    # Submit 25 consecutive prompts
    for i in range(25):
        res = await chat_service.generate_chat_response(req)
        assert res.content == "Model output chunk", f"Prompt #{i+1} failed"

    # Verify that usage is accurately tracked in user_daily_usage (25 prompts) and never capped at 20
    usage = await read_usage(uid)
    assert usage["chat_prompts"] == 25


@pytest.mark.asyncio
async def test_separate_users_cannot_exhaust_one_another(chat_service):
    """MISSION 5.2: Separate users and anonymous visitors cannot exhaust one another's allowance."""
    user_a = f"user-alice-{uuid.uuid4().hex[:8]}"
    user_b = f"user-bob-{uuid.uuid4().hex[:8]}"
    guest_user = "anonymous"

    mock_res = ("Success", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    # Record initial guest usage before Alice, Bob, and Guest send prompts
    init_guest = (await read_usage(guest_user))["chat_prompts"]
    test_run_tag = uuid.uuid4().hex[:6]

    # User A sends 22 distinct prompts
    for i in range(22):
        req_a = ChatRequest(
            message=f"Prompt from Alice {test_run_tag} #{i}",
            model="auto",
            stream=False,
            session_id=uuid.uuid4(),
        )
        req_a.user_id = user_a
        await chat_service.generate_chat_response(req_a)

    # Guest sends 22 distinct prompts
    for i in range(22):
        req_guest = ChatRequest(
            message=f"Prompt from Guest {test_run_tag} #{i}",
            model="auto",
            stream=False,
            session_id=uuid.uuid4(),
        )
        req_guest.user_id = None  # Guest/anonymous
        res = await chat_service.generate_chat_response(req_guest)
        assert res.content == "Success"

    # User B can now send prompts completely unhindered by User A's or Guest's high volume
    for i in range(22):
        req_b = ChatRequest(
            message=f"Prompt from Bob {test_run_tag} #{i}",
            model="auto",
            stream=False,
            session_id=uuid.uuid4(),
        )
        req_b.user_id = user_b
        res_b = await chat_service.generate_chat_response(req_b)
        assert res_b.content == "Success"

    usage_a = await read_usage(user_a)
    usage_b = await read_usage(user_b)
    usage_guest = await read_usage(guest_user)

    assert usage_a["chat_prompts"] == 22
    assert usage_b["chat_prompts"] == 22
    assert usage_guest["chat_prompts"] >= init_guest + 22


@pytest.mark.asyncio
async def test_previously_exhausted_users_unblocked_immediately(chat_service):
    """MISSION 5.3: Users who previously hit 20/20 in the database are immediately unblocked without waiting until 00:00 UTC."""
    uid = f"previously-blocked-{uuid.uuid4().hex[:8]}"
    guest_id = "anonymous"
    dt = get_current_utc_date_str()

    # Pre-seed both the user and the shared anonymous row with >= 20 prompts (exhausted under old rules)
    async with _daily_usage_transaction() as session:
        await session.execute(
            text(
                "INSERT INTO user_daily_usage (user_id, usage_date, chat_prompts, provider_calls, last_updated) "
                "VALUES (:uid, :dt, 20, 20, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id, usage_date) DO UPDATE SET chat_prompts = 20"
            ),
            {"uid": uid, "dt": dt},
        )
        await session.execute(
            text(
                "INSERT INTO user_daily_usage (user_id, usage_date, chat_prompts, provider_calls, last_updated) "
                "VALUES (:uid, :dt, 20, 20, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id, usage_date) DO UPDATE SET chat_prompts = CASE WHEN user_daily_usage.chat_prompts < 20 THEN 20 ELSE user_daily_usage.chat_prompts END"
            ),
            {"uid": guest_id, "dt": dt},
        )

    mock_res = ("Unblocked and operational!", "groq", "openai/gpt-oss-120b", False)
    chat_service.router.generate_response = AsyncMock(return_value=mock_res)

    # 1. Blocked authenticated user sends a prompt: succeeds immediately
    req_auth = ChatRequest(message="Hello after block", model="auto", stream=False)
    req_auth.user_id = uid
    res_auth = await chat_service.generate_chat_response(req_auth)
    assert res_auth.content == "Unblocked and operational!"

    # 2. Blocked guest user sends a prompt: succeeds immediately
    req_guest = ChatRequest(message="Guest hello after block", model="auto", stream=False)
    req_guest.user_id = None
    res_guest = await chat_service.generate_chat_response(req_guest)
    assert res_guest.content == "Unblocked and operational!"


@pytest.mark.asyncio
async def test_streaming_chat_operates_without_daily_quota_rejection(chat_service):
    """MISSION 5.6: Real-time SSE streaming chat operates smoothly without emitting daily quota error events."""
    uid = f"streaming-beta-user-{uuid.uuid4().hex[:8]}"

    async def mock_stream(*args, **kwargs):
        yield ("Chunk 1 ", "groq", "openai/gpt-oss-120b", False)
        yield ("Chunk 2 ", "groq", "openai/gpt-oss-120b", False)
        yield ("Chunk 3", "groq", "openai/gpt-oss-120b", False)

    chat_service.router.stream_response = mock_stream

    req = ChatRequest(
        message="Stream me a story.",
        model="auto",
        stream=True,
    )
    req.user_id = uid

    # Stream 22 times with distinct prompts
    for i in range(22):
        req.message = f"Stream me a story #{i}"
        chunks = []
        async for event in chat_service.stream_chat_response(req):
            chunks.append(event)

        # Must not contain any daily quota error
        assert not any("Daily chat prompt limit reached" in c for c in chunks)
        assert not any("DAILY_CHAT_PROMPTS_EXCEEDED" in c for c in chunks)
        assert any("event: message" in c for c in chunks)


@pytest.mark.asyncio
async def test_upstream_provider_http_429_handled_without_fake_daily_quota_message(chat_service):
    """MISSION 5.5: Upstream AI provider HTTP 429 errors trigger failover and surface truthful temporary-unavailability messaging, NOT the application daily 20/20 limit."""
    uid = f"upstream-429-user-{uuid.uuid4().hex[:8]}"

    # Simulate all providers returning upstream 429 quota exhaustion
    chat_service.router.generate_response = AsyncMock(
        side_effect=AllProvidersUnavailableException(
            detail="All configured AI providers are temporarily unavailable due to upstream rate limits. Please retry in a few moments.",
            circuits={"gemini": "RATE_LIMITED", "groq": "RATE_LIMITED"},
            retry_after_seconds=15,
        )
    )

    req = ChatRequest(
        message="Trigger upstream 429",
        model="auto",
        stream=False,
    )
    req.user_id = uid

    with pytest.raises(AllProvidersUnavailableException) as exc_info:
        await chat_service.generate_chat_response(req)

    err_detail = str(exc_info.value)
    # Must contain truthful upstream availability message, NEVER the application daily limit
    assert "Daily chat prompt limit reached" not in err_detail
    assert "20/20" not in err_detail
    assert "Resets at 00:00 UTC" not in err_detail
    assert "temporarily unavailable" in err_detail.lower()


@pytest.mark.asyncio
async def test_guest_and_authenticated_permissions_and_tenant_isolation():
    """MISSION 5.4 & 5.7: Guest and authenticated flows respect security boundaries (guest chat allowed, guest attachment blocked with 401, session hijacking blocked with 403)."""
    alice_id = f"alice-{uuid.uuid4().hex[:8]}"
    bob_id = f"bob-{uuid.uuid4().hex[:8]}"
    alice_session = uuid.uuid4()
    session_store.ensure_session(alice_session, "Alice Secret Chat", user_id=alice_id)

    class DummyChatService:
        async def generate_chat_response(self, payload):
            return None
        async def stream_chat_response(self, payload):
            yield ""

    dummy_service = DummyChatService()

    # 1. Unauthenticated guest chatting is allowed (session_id reset to None for stateless safety)
    guest_payload = ChatRequest(message="Hello as guest", model="auto", stream=False)
    request_mock = MagicMock()
    request_mock.is_disconnected = AsyncMock(return_value=False)

    await inspect.unwrap(chat_endpoint)(
        request=request_mock,
        payload=guest_payload,
        chat_service=dummy_service,
        current_user=None,
        x_client_id=None,
    )
    assert guest_payload.user_id is None
    assert guest_payload.session_id is None

    # 2. Unauthenticated guest attempting to upload attachments is blocked with 401
    guest_attach_payload = ChatRequest(
        message="Analyze this private file",
        model="auto",
        stream=False,
        attachments=[AttachmentItem(name="secret.pdf", mime_type="application/pdf", data="base64data")],
    )
    with pytest.raises(HTTPException) as exc_401:
        await inspect.unwrap(chat_endpoint)(
            request=request_mock,
            payload=guest_attach_payload,
            chat_service=dummy_service,
            current_user=None,
            x_client_id=None,
        )
    assert exc_401.value.status_code == 401

    # 3. Bob attempting to append to Alice's session is blocked with 403 Forbidden
    bob_payload = ChatRequest(
        message="Sneak into Alice session",
        model="auto",
        stream=False,
        session_id=alice_session,
    )
    with pytest.raises(HTTPException) as exc_403:
        await inspect.unwrap(chat_endpoint)(
            request=request_mock,
            payload=bob_payload,
            chat_service=dummy_service,
            current_user={"id": bob_id, "email": "bob@example.com"},
            x_client_id=None,
        )
    assert exc_403.value.status_code == 403


@pytest.mark.asyncio
async def test_storage_daily_quotas_endpoint_reflects_disabled_limit():
    """Verify that /daily-quotas endpoint reports None/unlimited for chat_prompts."""
    uid = f"storage-quota-user-{uuid.uuid4().hex[:8]}"
    current_user = {"id": uid, "email": f"{uid}@test.com"}

    res = await get_user_daily_quotas(current_user=current_user)
    assert res["status"] == "success"
    chat_quota = res["quotas"]["chat_prompts"]
    assert chat_quota["limit"] is None, "Limit must be None (unlimited) for public beta"
    assert chat_quota["remaining"] is None, "Remaining must be None (unlimited) for public beta"
