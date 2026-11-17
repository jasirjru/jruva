"""Catch visual metadata bypasses and accidental text-only fallback."""

import base64
import io

import pytest
from app.core.config import settings
from app.core.exceptions import AllProvidersUnavailableException, LLMProviderException
from app.schemas.chat import AttachmentItem, ChatRequest
from app.services.chat_service import ChatService
from app.services.image import ocr_service as ocr_module
from app.services.image.visual_analysis_service import VisualAnalysisService
from app.services.llm.base import BaseLLMProvider
from app.services.llm.router import LLMRouter
from PIL import Image


class RecordingProvider(BaseLLMProvider):
    def __init__(self, name, *, available=True, fails=False):
        self.name = name
        self.available = available
        self.fails = fails
        self.calls = []
        self.model = "gemini-3.6-flash" if name == "gemini" else "text-model"
        self.last_active_model = self.model
        # An unverified attribute must not widen the current Gemini-only policy.
        self.has_vision = True

    @property
    def provider_name(self):
        return self.name

    @property
    def is_available(self):
        return self.available

    async def generate_response(self, prompt, **kwargs):
        self.calls.append(kwargs.get("attachments"))
        if self.fails:
            raise LLMProviderException("synthetic upstream failure")
        return "A synthetic complete inspection of the supplied attachment."

    async def stream_response(self, prompt, **kwargs):
        yield await self.generate_response(prompt, **kwargs)


VISUAL_METADATA = [
    ("photo.jpg", "application/octet-stream"),
    ("photo.JPEG", ""),
    ("photo.jpg", " IMAGE/JPEG "),
    ("scan.PDF", "application/octet-stream"),
    ("scan.pdf", ""),
    ("unknown", "Application/PDF; charset=binary"),
]


@pytest.mark.parametrize(("name", "mime"), VISUAL_METADATA)
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("available", [False, True])
async def test_visual_failure_never_invokes_text_provider(name, mime, as_dict, stream, available):
    gemini = RecordingProvider("gemini", available=available, fails=True)
    groq = RecordingProvider("groq")
    openrouter = RecordingProvider("openrouter")
    router = LLMRouter(gemini_provider=gemini, groq_provider=groq, openrouter_provider=openrouter)
    attachment = AttachmentItem(name=name, mime_type=mime, data="ZmFrZQ==")
    if as_dict:
        attachment = attachment.model_dump()
    with pytest.raises((LLMProviderException, AllProvidersUnavailableException)):
        if stream:
            async for _ in router.stream_response("Inspect the attachment", attachments=[attachment]):
                pass
        else:
            await router.generate_response("Inspect the attachment", attachments=[attachment])
    assert not groq.calls
    assert not openrouter.calls
    assert bool(gemini.calls) == available


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_forwards_generic_mime_visual_attachment(stream, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ENABLE_LOCAL_OCR", False)
    image = Image.new("RGB", (32, 32), (30, 40, 50))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    attachment = AttachmentItem(
        name="photo.jpg", mime_type="application/octet-stream",
        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    gemini = RecordingProvider("gemini")
    service = ChatService()
    service.router = LLMRouter(
        gemini_provider=gemini, groq_provider=RecordingProvider("groq"),
        openrouter_provider=RecordingProvider("openrouter"),
    )
    request = ChatRequest(message="Describe attached file", attachments=[attachment], stream=stream)
    if stream:
        async for _ in service.stream_chat_response(request):
            pass
    else:
        await service.generate_chat_response(request)
    assert gemini.calls
    assert gemini.calls[0] == [attachment]


@pytest.mark.parametrize("operation", ["enhance_image", "classify_visual_type", "generate_multimodal_cot_context"])
def test_production_optional_cv_never_opens_pixels(operation, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    opened = []

    def reject_open(*args, **kwargs):
        opened.append(True)
        raise AssertionError("production must delegate visual decoding to cloud vision")

    monkeypatch.setattr(Image, "open", reject_open)
    result = getattr(VisualAnalysisService, operation)(b"untrusted image payload")
    assert not opened
    if operation == "generate_multimodal_cot_context":
        assert result == ""


def test_production_disabled_ocr_does_not_reuse_cached_engine(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ENABLE_LOCAL_OCR", False)
    monkeypatch.setattr(ocr_module, "_OCR_ENGINE", object())
    assert ocr_module.get_ocr_engine() is None
