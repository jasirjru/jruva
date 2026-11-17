"""Deterministic end-to-end tests for Provider Failover Streaming covering Scenarios A through F."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.exceptions import (
    AllProvidersUnavailableException,
    LLMProviderException,
    QuotaExhaustedException,
)
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.llm.base import BaseLLMProvider


class MockFailingGemini(BaseLLMProvider):
    def __init__(self, failure_type="error"):
        self.failure_type = failure_type
        self.model = "gemini-3.6-flash"

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, *args, **kwargs) -> str:
        raise QuotaExhaustedException("Gemini quota exhausted")

    async def stream_response(self, *args, **kwargs):
        if self.failure_type == "429":
            raise QuotaExhaustedException("Gemini quota exhausted (returned 429)")
        elif self.failure_type == "404":
            raise LLMProviderException("Gemini model not found (404)")
        elif self.failure_type == "mid_stream":
            yield "Partial Gemini text before crash."
            raise LLMProviderException("Network cut mid-stream")
        else:
            raise LLMProviderException("Gemini connection failed before emission")


class MockSuccessfulGroq(BaseLLMProvider):
    def __init__(self):
        self.model = "openai/gpt-oss-120b"

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, *args, **kwargs) -> str:
        return "Groq fallback successful answer."

    async def stream_response(self, *args, **kwargs):
        tokens = ["Groq ", "fallback ", "successful ", "answer."]
        for t in tokens:
            yield t


class MockFailingGroq(BaseLLMProvider):
    def __init__(self):
        self.model = "openai/gpt-oss-120b"

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, *args, **kwargs) -> str:
        raise LLMProviderException("Groq also unavailable")

    async def stream_response(self, *args, **kwargs):
        raise LLMProviderException("Groq stream failed before emission")
        if False: yield "never"


def parse_sse_events(raw_events):
    parsed = []
    for raw in raw_events:
        for block in raw.strip().split("\n\n"):
            if not block.strip():
                continue
            event_type = "message"
            data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:].strip())
            if data:
                parsed.append({"event": event_type, "data": data})
    return parsed


@pytest.mark.asyncio
async def test_scenario_a_gemini_fails_before_emission_groq_recovers():
    """Scenario A: Gemini fails BEFORE first token -> router selects Groq -> Groq streams valid content."""
    service = ChatService()
    failing_gemini = MockFailingGemini(failure_type="error")
    successful_groq = MockSuccessfulGroq()

    with patch.object(service.router, "gemini_provider", failing_gemini), \
         patch.object(service.router, "groq_provider", successful_groq):
        req = ChatRequest(message="Summarize where Kuttippuram is", stream=True)
        raw_events = []
        async for chunk in service.stream_chat_response(req):
            raw_events.append(chunk)

    events = parse_sse_events(raw_events)
    assert len(events) >= 2

    # Check tokens
    token_events = [e for e in events if e["data"].get("type") == "token"]
    assert len(token_events) > 0
    full_text = "".join(t["data"].get("content", "") for t in token_events)
    assert "Groq fallback successful answer." in full_text

    # Verify provider metadata shows failover to groq
    for t in token_events:
        if "Groq" in t["data"].get("content", ""):
            assert t["data"].get("provider") == "groq"
            assert t["data"].get("failover") is True

    # Check terminal finish event
    finish_events = [e for e in events if e["data"].get("type") == "finish"]
    assert len(finish_events) == 1
    assert finish_events[0]["data"].get("provider") == "groq"
    assert finish_events[0]["data"].get("failover") is True
    assert finish_events[0]["data"].get("finish_reason") == "stop"

    # Confirm NO error frames were emitted
    error_events = [e for e in events if e["event"] == "error" or e["data"].get("type") == "error"]
    assert len(error_events) == 0


@pytest.mark.asyncio
async def test_scenario_b_gemini_429_quota_exhausted_fallback():
    """Scenario B: Gemini returns 429 before emission -> router seamlessly falls over to Groq."""
    service = ChatService()
    gemini_429 = MockFailingGemini(failure_type="429")
    successful_groq = MockSuccessfulGroq()

    with patch.object(service.router, "gemini_provider", gemini_429), \
         patch.object(service.router, "groq_provider", successful_groq):
        req = ChatRequest(message="Summarize facts about the capital of France", stream=True)
        raw_events = [chunk async for chunk in service.stream_chat_response(req)]

    events = parse_sse_events(raw_events)
    token_events = [e for e in events if e["data"].get("type") == "token"]
    full_text = "".join(t["data"].get("content", "") for t in token_events)
    assert "Groq fallback successful answer." in full_text

    finish_events = [e for e in events if e["data"].get("type") == "finish"]
    assert len(finish_events) == 1
    assert finish_events[0]["data"].get("provider") == "groq"
    assert finish_events[0]["data"].get("failover") is True


@pytest.mark.asyncio
async def test_scenario_c_gemini_404_not_found_fallback():
    """Scenario C: Gemini returns 404 model not found -> router seamlessly falls over to Groq."""
    service = ChatService()
    gemini_404 = MockFailingGemini(failure_type="404")
    successful_groq = MockSuccessfulGroq()

    with patch.object(service.router, "gemini_provider", gemini_404), \
         patch.object(service.router, "groq_provider", successful_groq):
        req = ChatRequest(message="Summarize how you are doing today", stream=True)
        raw_events = [chunk async for chunk in service.stream_chat_response(req)]

    events = parse_sse_events(raw_events)
    finish_events = [e for e in events if e["data"].get("type") == "finish"]
    assert len(finish_events) == 1
    assert finish_events[0]["data"].get("provider") == "groq"
    assert finish_events[0]["data"].get("failover") is True


@pytest.mark.asyncio
async def test_scenario_d_gemini_mid_stream_failure_no_concatenation_corruption():
    """Scenario D: Gemini emits partial output and then fails. Ensure the system does NOT concatenate
    a duplicate or corrupted new provider answer into the stream."""
    service = ChatService()
    gemini_mid = MockFailingGemini(failure_type="mid_stream")
    successful_groq = MockSuccessfulGroq()

    with patch.object(service.router, "gemini_provider", gemini_mid), \
         patch.object(service.router, "groq_provider", successful_groq):
        req = ChatRequest(message="Summarize a short joke", stream=True)
        raw_events = [chunk async for chunk in service.stream_chat_response(req)]

    events = parse_sse_events(raw_events)
    token_events = [e for e in events if e["data"].get("type") == "token"]
    full_text = "".join(t["data"].get("content", "") for t in token_events)

    # Initial partial text was received
    assert "Partial Gemini text before crash." in full_text

    # CRITICAL: Groq fallback must NOT have been concatenated mid-stream
    assert "Groq fallback successful answer." not in full_text

    # An error event terminates the stream cleanly
    error_events = [e for e in events if e["event"] == "error" or e["data"].get("finish_reason") == "error"]
    assert len(error_events) >= 1


@pytest.mark.asyncio
async def test_scenario_e_cascade_failure_all_providers_down():
    """Scenario E: Both Gemini and Groq fail before emission -> terminal error frame is sent cleanly, no hang."""
    service = ChatService()
    gemini_fail = MockFailingGemini(failure_type="error")
    groq_fail = MockFailingGroq()

    with patch.object(service.router, "gemini_provider", gemini_fail), \
         patch.object(service.router, "groq_provider", groq_fail), \
         patch.object(service.router, "openrouter_provider", MagicMock(is_available=False)):
        req = ChatRequest(message="test cascade failure", stream=True)
        raw_events = [chunk async for chunk in service.stream_chat_response(req)]

    events = parse_sse_events(raw_events)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) >= 1
    err_text = error_events[0]["data"].get("content", "").lower()
    assert "all free ai models are busy right now" in err_text or "connection issue" in err_text


@pytest.mark.asyncio
async def test_scenario_f_client_disconnect_cancellation():
    """Scenario F: Client disconnects/cancels early -> resources and generator close cleanly."""
    service = ChatService()
    successful_groq = MockSuccessfulGroq()

    with patch.object(service.router, "groq_provider", successful_groq):
        req = ChatRequest(message="tell me a long story", stream=True)
        gen = service.stream_chat_response(req)

        # Read first event
        first_chunk = await gen.__anext__()
        assert first_chunk is not None

        # Client disconnects: close the generator
        await gen.aclose()

        # Verify generator is closed
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
