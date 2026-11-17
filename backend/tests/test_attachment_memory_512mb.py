"""Real attachment/chat orchestration with mocked inference under a Linux cgroup.

This is a resource regression, not a model accuracy evaluation. Local runs record
unbounded measurements; release CI requires a 512 MiB cgroup and zero swap.
"""

import asyncio
import base64
import gc
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.main import app
from app.schemas.chat import AttachmentItem, ChatRequest
from app.services.chat_service import ChatService
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.openrouter_provider import OpenRouterProvider
from app.services.llm.router import LLMRouter
from PIL import Image

MIB = 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]


def cgroup_directory():
    if sys.platform != "linux":
        return None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            candidate = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            if (candidate / "memory.max").exists():
                return candidate
    return None


def cgroup_events(directory):
    return {key: int(value) for key, value in (line.split() for line in (directory / "memory.events").read_text().splitlines())}


async def test_large_attachments_and_concurrency_under_memory_limit(monkeypatch):
    assert app.routes  # Include full API initialization in the warm-process budget.
    directory = cgroup_directory()
    required = os.getenv("REQUIRE_MEMORY_CGROUP") == "1"
    if required:
        assert directory is not None, "Release probe must have a Linux cgroup v2 memory limit"
        assert (directory / "memory.max").read_text().strip() == str(512 * MIB)
        assert (directory / "memory.swap.max").read_text().strip() == "0"
    before = cgroup_events(directory) if directory else {}
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ENABLE_LOCAL_OCR", False)
    monkeypatch.setattr(settings, "ENABLE_LOCAL_VISUAL_ANALYSIS", False)

    attachments = []
    for mode, fmt, suffix in [("RGB", "JPEG", "jpg"), ("RGB", "PNG", "png"), ("RGBA", "PNG", "png")]:
        image = Image.new(mode, (5000, 4000), color=(40, 80, 120, 128) if mode == "RGBA" else (40, 80, 120))
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        image.close()
        attachments.append(AttachmentItem(name=f"near-limit-{mode}.{suffix}", mime_type="application/octet-stream", data=base64.b64encode(buffer.getvalue()).decode("ascii")))
        buffer.close()
    for name in ("test_c_text_document.pdf", "test_d_scanned_document.pdf"):
        attachments.append(AttachmentItem(name=name, mime_type="application/octet-stream", data=base64.b64encode((ROOT / "fixtures" / name).read_bytes()).decode("ascii")))
    gc.collect()

    # Stress remaining headroom after a realistic warm-process baseline rather
    # than claiming a tiny isolated decoder process represents production.
    reserve = None
    if directory:
        reserve = bytearray(max(0, 300 * MIB - int((directory / "memory.current").read_text())))
        for offset in range(0, len(reserve), 4096):
            reserve[offset] = 1
    baseline = int((directory / "memory.current").read_text()) if directory else None

    provider = GeminiProvider(api_key="synthetic-key")
    observed = []

    async def infer(prompt, **kwargs):
        # Real Gemini payload construction still exercises serialization/copies.
        payload = provider._build_payload(prompt=prompt, attachments=kwargs.get("attachments"))
        encoded = json.dumps(payload)
        observed.append(len(payload["contents"][-1]["parts"]) - 1)
        await asyncio.sleep(0.01)
        assert encoded
        return "Synthetic upstream completion for attachment memory verification only."

    async def infer_stream(prompt, **kwargs):
        yield await infer(prompt, **kwargs)

    async def request(items, streaming):
        service = ChatService()
        service.router = LLMRouter(gemini_provider=provider, groq_provider=GroqProvider(api_key=""), openrouter_provider=OpenRouterProvider(api_key=""))
        request = ChatRequest(message="Analyze attached file(s)", attachments=items, stream=streaming)
        if streaming:
            events = [event async for event in service.stream_chat_response(request)]
            assert not any("event: error" in event for event in events)
            assert any('"finish_reason":"stop"' in event for event in events)
        else:
            result = await service.generate_chat_response(request)
            assert result.provider == "gemini"

    with patch.object(provider, "generate_response", side_effect=infer), patch.object(provider, "stream_response", side_effect=infer_stream), patch.object(Image, "open", side_effect=AssertionError("optional local pixels must not be decoded")):
        for attachment in attachments:
            await request([attachment], False)
            await request([attachment], True)
        await request(attachments, False)
        await request(attachments, True)
        await asyncio.gather(*(request(attachments, bool(index % 2)) for index in range(4)))
    after = cgroup_events(directory) if directory else {}
    peak = int((directory / "memory.peak").read_text()) if directory else None
    report = {
        "commit": os.getenv("RELEASE_COMMIT_SHA"), "bounded_linux": required,
        "memory_limit_bytes": 512 * MIB if required else None,
        "baseline_bytes": baseline, "cgroup_peak_bytes": peak,
        "memory_events_before": before, "memory_events_after": after,
        "completed_requests": len(observed), "forwarded_attachment_counts": observed,
        "baseline_reserve_bytes": len(reserve) if reserve is not None else 0,
        "inference": "mocked; real chat preprocessing, routing and Gemini payload serialization",
    }
    path = ROOT / "scratch/attachment-qa/linux-memory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert len(observed) == 16
    assert observed.count(5) == 6
    if required:
        assert after.get("oom", 0) == before.get("oom", 0)
        assert after.get("oom_kill", 0) == before.get("oom_kill", 0)
        assert peak < 380 * MIB, report
