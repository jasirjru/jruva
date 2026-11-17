"""Tests for Multimodal Attachment handling in GeminiProvider, GroqProvider, and LLMRouter."""
import base64
import pytest
from app.schemas.chat import AttachmentItem
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.router import LLMRouter


def test_gemini_payload_multimodal_inline_data():
    """Verify GeminiProvider formats attachments into inline_data parts."""
    provider = GeminiProvider(api_key="mock_key", model="gemini-3.6-flash")
    raw_bytes = b"fake_png_binary_data"
    b64_data = base64.b64encode(raw_bytes).decode("utf-8")

    attachment = AttachmentItem(
        name="chart.png",
        mime_type="image/png",
        data=b64_data,
    )

    payload = provider._build_payload(
        prompt="Analyze this chart",
        attachments=[attachment],
    )

    contents = payload.get("contents", [])
    assert len(contents) == 1
    parts = contents[0].get("parts", [])
    assert len(parts) == 2

    # First part is inline_data
    assert "inline_data" in parts[0]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert parts[0]["inline_data"]["data"] == b64_data

    # Second part is text prompt
    assert parts[1] == {"text": "Analyze this chart"}


def test_groq_messages_decoded_document():
    """Verify GroqProvider decodes base64 text attachments and appends to prompt."""
    provider = GroqProvider(api_key="mock_key", model="openai/gpt-oss-120b")
    code_text = "def add(a, b):\n    return a + b\n"
    b64_code = base64.b64encode(code_text.encode("utf-8")).decode("utf-8")

    attachment = AttachmentItem(
        name="math.py",
        mime_type="text/x-python",
        data=b64_code,
    )

    messages = provider._build_messages(
        prompt="Review this code",
        attachments=[attachment],
    )

    assert len(messages) == 1
    user_msg = messages[0]["content"]
    assert "[Attached Document: math.py]" in user_msg
    assert "def add(a, b):" in user_msg
    assert "Review this code" in user_msg


def test_router_vision_awareness():
    """Verify router routes images to Groq's vision model first, with Gemini as backup, in auto mode."""
    gemini = GeminiProvider(api_key="mock_key", model="gemini-3.6-flash")
    groq = GroqProvider(api_key="mock_key", model="openai/gpt-oss-120b")
    router = LLMRouter(gemini_provider=gemini, groq_provider=groq)

    attachment = AttachmentItem(
        name="screenshot.png",
        mime_type="image/png",
        data="mock_base64_data",
    )

    primary, backup, model_id = router._resolve_providers(
        requested_model="auto",
        attachments=[attachment],
    )

    assert primary.provider_name == "groq"
    assert model_id in GroqProvider.VISION_MODELS
    assert backup.provider_name == "gemini"
