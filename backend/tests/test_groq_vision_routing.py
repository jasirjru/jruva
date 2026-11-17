"""Images go to Groq's vision model first (fast, 1,000 RPD free) with Gemini as backup; PDFs stay on Gemini."""
import json
from unittest.mock import patch

import httpx
import pytest
from app.core.exceptions import LLMProviderException
from app.schemas.chat import AttachmentItem
from app.services.llm.base import BaseLLMProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import LLMRouter

PNG = AttachmentItem(name="invoice.png", mime_type="image/png", data="iVBORw0KGgo=")
PDF = AttachmentItem(name="scan.pdf", mime_type="application/pdf", data="JVBERi0=")


class RecordingProvider(BaseLLMProvider):
    def __init__(self, name, *, fails=False, vision=False):
        self.name = name
        self.fails = fails
        self.vision = vision
        self.calls = []
        self.model = "gemini-3.6-flash" if name == "gemini" else "text-model"
        self.last_active_model = self.model

    @property
    def provider_name(self):
        return self.name

    @property
    def is_available(self):
        return True

    def supports_vision_attachments(self, attachments):
        return self.vision

    async def generate_response(self, prompt, **kwargs):
        self.calls.append(kwargs.get("attachments"))
        if self.fails:
            raise LLMProviderException("synthetic upstream failure")
        return f"answer from {self.name}"

    async def stream_response(self, prompt, **kwargs):
        yield await self.generate_response(prompt, **kwargs)


def test_groq_accepts_only_images_within_limits():
    assert GroqProvider.supports_vision_attachments([PNG])
    assert GroqProvider.supports_vision_attachments([PNG.model_dump()])
    assert not GroqProvider.supports_vision_attachments([PDF])
    assert not GroqProvider.supports_vision_attachments([PNG, PDF])
    assert not GroqProvider.supports_vision_attachments([PNG] * (GroqProvider.MAX_VISION_IMAGES + 1))
    huge = AttachmentItem(name="big.png", mime_type="image/png", data="A" * (GroqProvider.MAX_VISION_IMAGE_B64_CHARS + 1))
    assert not GroqProvider.supports_vision_attachments([huge])
    assert not GroqProvider.supports_vision_attachments([])


def test_groq_vision_messages_carry_the_image():
    messages = GroqProvider(api_key="k")._build_messages("What is the total?", attachments=[PNG], vision=True)
    content = messages[-1]["content"]
    assert {"type": "text", "text": "What is the total?"} in content
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}} in content


@pytest.mark.parametrize("stream", [False, True])
async def test_image_goes_to_groq_vision_first(stream):
    gemini, groq = RecordingProvider("gemini"), RecordingProvider("groq", vision=True)
    router = LLMRouter(gemini_provider=gemini, groq_provider=groq, openrouter_provider=RecordingProvider("openrouter"))
    if stream:
        result = "".join([t[0] async for t in router.stream_response("Read it", attachments=[PNG])])
    else:
        result = await router.generate_response("Read it", attachments=[PNG])
    assert "groq" in str(result)
    assert groq.calls and not gemini.calls


@pytest.mark.parametrize("stream", [False, True])
async def test_image_falls_back_to_gemini_when_groq_fails(stream):
    gemini, groq = RecordingProvider("gemini"), RecordingProvider("groq", vision=True, fails=True)
    openrouter = RecordingProvider("openrouter")
    router = LLMRouter(gemini_provider=gemini, groq_provider=groq, openrouter_provider=openrouter)
    if stream:
        result = "".join([t[0] async for t in router.stream_response("Read it", attachments=[PNG])])
    else:
        result = await router.generate_response("Read it", attachments=[PNG])
    assert "gemini" in str(result)
    assert groq.calls and gemini.calls and not openrouter.calls


@pytest.mark.parametrize("stream", [False, True])
async def test_pdf_stays_on_gemini(stream):
    gemini, groq = RecordingProvider("gemini"), RecordingProvider("groq", vision=False)
    router = LLMRouter(gemini_provider=gemini, groq_provider=groq, openrouter_provider=RecordingProvider("openrouter"))
    if stream:
        "".join([t[0] async for t in router.stream_response("Summarize", attachments=[PDF])])
    else:
        await router.generate_response("Summarize", attachments=[PDF])
    assert gemini.calls and not groq.calls


async def test_real_groq_stream_sends_image_to_vision_model():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        body = "data: " + json.dumps({"choices": [{"delta": {"content": "Total Rs 4,250."}}]}) + "\n\ndata: [DONE]\n\n"
        return httpx.Response(200, text=body)

    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    with patch("httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
        provider = GroqProvider(api_key="synthetic-audit")
        result = "".join([t async for t in provider.stream_response("What is the total?", attachments=[PNG])])
    assert result == "Total Rs 4,250."
    assert seen["model"] in GroqProvider.VISION_MODELS
    parts = seen["messages"][-1]["content"]
    assert any(p.get("type") == "image_url" for p in parts)


@pytest.mark.parametrize("stream", [False, True])
async def test_gemini_gives_attachment_requests_time_to_answer(stream):
    from app.services.llm.gemini_provider import GeminiProvider

    timeouts = []
    client_class = httpx.AsyncClient
    payload = {"candidates": [{"content": {"parts": [{"text": "Total Rs 4,250."}]}}]}

    def handler(request):
        if stream:
            return httpx.Response(200, text="data: " + json.dumps(payload) + "\n\n")
        return httpx.Response(200, json=payload)

    def make_client(**kw):
        timeouts.append(kw.get("timeout"))
        return client_class(transport=httpx.MockTransport(handler), **kw)

    provider = GeminiProvider(api_key="synthetic-audit")
    with patch("httpx.AsyncClient", side_effect=make_client):
        if stream:
            "".join([t async for t in provider.stream_response("Read it", attachments=[PNG])])
            "".join([t async for t in provider.stream_response("Hello")])
        else:
            await provider.generate_response("Read it", attachments=[PNG])
            await provider.generate_response("Hello")
    with_attachment, text_only = (httpx.Timeout(t) for t in timeouts[:2])
    assert with_attachment.read >= 20 and with_attachment.connect <= 8
    assert text_only.read == 8
