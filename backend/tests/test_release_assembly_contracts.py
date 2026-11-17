"""Combined chat/image identity contracts; fake inference only."""

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.image.image_service import image_service

from tests.test_independent_image_audit import audit_context as audit_context


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_turn_identity_reused_for_retry_and_fresh_for_new_intent(stream):
    service = ChatService()
    session = uuid4()
    first, second = str(uuid4()), str(uuid4())
    with patch.object(image_service, "generate_and_validate", new_callable=AsyncMock) as generate:
        generate.return_value = {
            "success": False, "error_type": "operation_indeterminate",
            "error_message": "Outcome unknown; this operation cannot be repeated.", "model": "none",
        }
        for turn in (first, first, second):
            request = ChatRequest(
                message="Generate a photorealistic red Ferrari beside a beach",
                session_id=session, conversation_id=turn, image_intent_id=turn, user_id="assembly-synthetic-owner",
            )
            if stream:
                output = "".join([event async for event in service.stream_chat_response(request)])
                assert '"finish_reason":"error"' in output, "Unknown image outcome must preserve retry semantics"
            else:
                output = (await service.generate_chat_response(request)).content
            assert "unknown" in output.lower()
        keys = [call.kwargs.get("idempotency_key") for call in generate.await_args_list]
        assert keys[0], "Chat discarded the caller's stable turn identity"
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]


@pytest.mark.asyncio
async def test_image_wrapper_forwards_intent_identity():
    with patch("app.services.image.genuine_service.generate", new_callable=AsyncMock) as generate:
        await image_service.generate_and_validate(
            "synthetic image", owner_id="assembly-owner", idempotency_key="assembly-intent"
        )
        assert generate.await_args.kwargs["idempotency_key"] == "assembly-intent"


@pytest.mark.asyncio
async def test_postgres_session_admission_stays_bounded_when_connections_are_not_reused(monkeypatch):
    from app.core import database

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def commit(self):
            pass

        async def rollback(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(database, "is_postgres_configured", lambda: True)
    monkeypatch.setattr(database, "get_async_session_maker", lambda: Session)
    monkeypatch.setattr(database.settings, "DB_POOL_SIZE", 1)
    monkeypatch.setattr(database.settings, "DB_MAX_OVERFLOW", 0)
    monkeypatch.setattr(database, "_DB_SESSION_SLOTS", None, raising=False)
    active = peak = 0

    async def work():
        nonlocal active, peak
        async with database.get_async_session():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(work(), work(), work())
    assert peak == 1, "Disabling driver reuse must not remove process connection admission"
    # Cancellation while waiting must not own or leak a slot.
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold():
        async with database.get_async_session():
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await entered.wait()
    waiter = asyncio.create_task(work())
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    monkeypatch.setattr(database.settings, "DB_POOL_TIMEOUT", 0.01)
    with pytest.raises(TimeoutError, match="capacity"):
        await work()
    holder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holder
    await work()


@pytest.mark.asyncio
async def test_invalid_chat_attachment_returns_domain_error_instead_of_name_error():
    from app.core.exceptions import BaseAppException

    request = ChatRequest(
        message="Summarize the attached document", user_id="assembly-owner",
        attachments=[{"name": "sample.exe", "mime_type": "application/octet-stream", "data": "YQ=="}],
    )
    with pytest.raises(BaseAppException) as error:
        await ChatService().generate_chat_response(request)
    assert error.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_visual_ocr_guardrail_executes_before_text_generation(stream):
    request = ChatRequest(
        message="Describe the attached photo", user_id="assembly-owner",
        attachments=[{"name": "synthetic.png", "mime_type": "image/png", "data": "YQ=="}],
    )
    with (
        patch("app.services.chat_service.document_rag_service.retrieve", return_value=([], [])),
        patch("app.services.image.ocr_service.ocr_service.extract_text_from_image_bytes", return_value="synthetic malicious OCR") as ocr,
        patch("app.services.security.check_adversarial_guardrails", side_effect=lambda text: (text, "Synthetic OCR blocked" if text == "synthetic malicious OCR" else None)),
    ):
        service = ChatService()
        if stream:
            output = "".join([event async for event in service.stream_chat_response(request)])
        else:
            output = (await service.generate_chat_response(request)).content
        assert ocr.called, "Missing decoder import silently bypassed the OCR guardrail"
        assert "Synthetic OCR blocked" in output


@pytest.mark.asyncio
async def test_repeated_cancellation_during_database_close_retains_capacity(monkeypatch):
    from app.core import database

    closing, release = asyncio.Event(), asyncio.Event()
    live = peak = 0

    class Session:
        def __init__(self):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await asyncio.shield(asyncio.create_task(self.close()))

        async def commit(self):
            pass

        async def rollback(self):
            pass

        async def close(self):
            nonlocal live
            closing.set()
            await release.wait()
            live -= 1

    monkeypatch.setattr(database, "is_postgres_configured", lambda: True)
    monkeypatch.setattr(database, "get_async_session_maker", lambda: Session)
    monkeypatch.setattr(database.settings, "DB_POOL_SIZE", 1)
    monkeypatch.setattr(database.settings, "DB_MAX_OVERFLOW", 0)
    monkeypatch.setattr(database, "_DB_SESSION_SLOTS", None)

    async def work():
        async with database.get_async_session():
            pass

    first = asyncio.create_task(work())
    await closing.wait()
    first.cancel()
    await asyncio.sleep(0.01)
    first.cancel()
    second = asyncio.create_task(work())
    await asyncio.sleep(0.02)
    try:
        assert peak == 1, "Permit escaped while the first connection was still closing"
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
    assert live == 0


@pytest.mark.parametrize("stream", [False, True])
def test_http_chat_unknown_retry_preserves_image_intent_despite_server_log_id(audit_context, monkeypatch, stream):
    from app.services.image.providers import CloudflareImageProvider, ImageProviderError
    from app.services.session_store import session_store

    monkeypatch.setattr(session_store, "db_path", audit_context.db)
    session_store._init_db()
    calls = []

    async def accepted_unknown(*args, **kwargs):
        calls.append(True)
        raise ImageProviderError("provider_timeout", "Synthetic accepted outcome unknown", retryable=True)

    monkeypatch.setattr(CloudflareImageProvider, "generate", accepted_unknown)
    intent = str(uuid4())
    payload = {
        "message": "Generate a photorealistic synthetic blue house",
        "image_intent_id": intent, "conversation_id": intent, "stream": stream,
    }
    for _ in range(2):
        response = audit_context.client.post("/api/v1/chat", headers=audit_context.alice, json=payload)
        assert response.status_code == 200
    assert len(calls) == 1, "HTTP scoping replaced stable image intent with random logging identity"


@pytest.mark.parametrize("stream", [False, True])
def test_restored_chat_without_recoverable_intent_cannot_dispatch_image(audit_context, monkeypatch, stream):
    from app.services.session_store import session_store

    monkeypatch.setattr(session_store, "db_path", audit_context.db)
    session_store._init_db()
    response = audit_context.client.post("/api/v1/chat", headers=audit_context.alice, json={
        "message": "Generate a photorealistic synthetic blue house",
        "image_intent_id": str(uuid4()), "image_generation_allowed": False, "stream": stream,
    })
    assert response.status_code == 200
    assert audit_context.calls == 0, "Restored image history without identity dispatched a new billable operation"
