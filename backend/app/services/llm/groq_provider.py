"""Groq Cloud AI Provider with high-speed streaming (~300 tok/s) and OpenAI compatibility."""
import asyncio
import base64
import contextvars
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.core.attachments import attachment_mime_type, is_visual_attachment
from app.core.config import settings
from app.core.exceptions import (
    LLMProviderException,
    QuotaExhaustedException,
)
from app.services.llm.base import BaseLLMProvider
from app.services.llm.rate_limit import cooldown_from_response, parse_model_list
from app.services.llm.stream_filter import ThinkingFilter

logger = logging.getLogger(__name__)

_groq_active_model_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_groq_active_model_var", default=None
)


class GroqProvider(BaseLLMProvider):
    """Groq Cloud LPU AI provider offering ultra-fast inference on open models.
    
    Supports Llama 3.3 70B, DeepSeek R1 Distill 70B, and Llama 3.1 8B on the free tier.
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    # Groq rate limits are per model (30 RPM / 1K RPD / 8K TPM / 200K TPD each on the free plan),
    # so every model in this list is an extra free bucket. Override with GROQ_FALLBACK_MODELS.
    FALLBACK_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]

    # Image input limits per Groq's vision docs (base64 payloads up to 4 MB, 5 images per request).
    VISION_MODELS = ["qwen/qwen3.8-27b"]
    MAX_VISION_IMAGES = 5
    MAX_VISION_IMAGE_B64_CHARS = 4 * 1024 * 1024

    @staticmethod
    def _attachment_base64(attachment: Any) -> str:
        data = attachment.get("data") if isinstance(attachment, dict) else getattr(attachment, "data", "")
        data = data or ""
        return data.split(",", 1)[1] if data.startswith("data:") and "," in data else data

    @classmethod
    def supports_vision_attachments(cls, attachments: list[Any] | None) -> bool:
        """True when every visual attachment is an image Groq's vision model can take directly (PDFs stay on Gemini)."""
        visual = [att for att in (attachments or []) if is_visual_attachment(att)]
        if not visual or len(visual) > cls.MAX_VISION_IMAGES:
            return False
        for att in visual:
            data = cls._attachment_base64(att)
            if not attachment_mime_type(att).startswith("image/") or not data or len(data) > cls.MAX_VISION_IMAGE_B64_CHARS:
                return False
        return True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        self._api_key = api_key
        raw = model or settings.GROQ_MODEL or "openai/gpt-oss-120b"
        legacy_groq = {
            "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
            "llama-3.1-8b-instant": "openai/gpt-oss-120b",
            "llama3-70b-8192": "openai/gpt-oss-120b",
            "llama3-8b-8192": "openai/gpt-oss-120b",
            "deepseek-r1-distill-llama-70b": "openai/gpt-oss-120b",
            "mixtral-8x7b-32768": "openai/gpt-oss-120b",
            "gemma2-9b-it": "openai/gpt-oss-120b",
        }
        if raw in legacy_groq:
            raw = legacy_groq[raw]
        self.model = raw
        self._last_active_model: Optional[str] = self.model
        self.FALLBACK_MODELS = parse_model_list(
            getattr(settings, "GROQ_FALLBACK_MODELS", ""), type(self).FALLBACK_MODELS
        )

    @property
    def last_active_model(self) -> str:
        scoped = _groq_active_model_var.get()
        if scoped is not None:
            return scoped
        return getattr(self, "_last_active_model", self.model)

    @last_active_model.setter
    def last_active_model(self, val: str) -> None:
        self._last_active_model = val
        _groq_active_model_var.set(val)

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    @property
    def api_key(self) -> str:
        import os
        return self._api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        attachments: Optional[List[Any]] = None,
        vision: bool = False,
    ) -> list[dict[str, Any]]:
        """Construct the OpenAI-compatible messages array with decoded attachments."""
        messages: list[dict[str, Any]] = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if history:
            for turn in history:
                role = turn.get("role", "user")
                # Normalize role
                if role in ["model", "assistant"]:
                    role = "assistant"
                else:
                    role = "user"
                messages.append({"role": role, "content": turn.get("content", "")})

        file_sections: List[str] = []
        image_parts: list[dict[str, Any]] = []
        if attachments:
            for att in attachments:
                name = getattr(att, "name", None) or (att.get("name") if isinstance(att, dict) else "attachment")
                mime_type = getattr(att, "mime_type", None) or (att.get("mime_type") if isinstance(att, dict) else "")
                data = getattr(att, "data", None) or (att.get("data") if isinstance(att, dict) else "")
                if vision and is_visual_attachment(att):
                    url = f"data:{attachment_mime_type(att)};base64,{self._attachment_base64(att)}"
                    image_parts.append({"type": "image_url", "image_url": {"url": url}})
                    continue
                if data:
                    try:
                        from app.services.rag.document_rag import document_rag_service
                        pages = document_rag_service.extract_document_pages(att)
                        if pages:
                            extracted_text = "\n\n".join(
                                f"--- Page {p.get('page', 1)} ---\n{p['text']}" if len(pages) > 1 else p["text"]
                                for p in pages if p.get("text")
                            )
                            if len(extracted_text) > 16000:
                                extracted_text = extracted_text[:16000] + "\n...[Content truncated for length]..."
                            file_type_label = "Image (OCR Extracted)" if mime_type.startswith("image/") else "Document"
                            file_sections.append(f"[Attached {file_type_label}: {name}]\n```\n{extracted_text}\n```")
                        elif mime_type.startswith("image/"):
                            file_sections.append(f"[Attached Image: {name} ({mime_type})]")
                        else:
                            decoded = base64.b64decode(data).decode("utf-8", errors="replace")
                            if len(decoded) > 16000:
                                decoded = decoded[:16000] + "\n...[Content truncated for length]..."
                            file_sections.append(f"[Attached Document: {name}]\n```\n{decoded}\n```")
                    except Exception:
                        file_sections.append(f"[Attached Binary File: {name} ({mime_type})]")

        user_content = "\n\n".join(file_sections + [prompt]) if file_sections else prompt
        if image_parts:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_content}, *image_parts]})
        else:
            messages.append({"role": "user", "content": user_content})
        return messages

    def _candidate_models(self, model_override: str | None, vision: bool) -> list[str]:
        if vision:
            return list(self.VISION_MODELS)
        candidate_models = [model_override or self.model]
        for alt in [self.model, *self.FALLBACK_MODELS]:
            if alt not in candidate_models:
                candidate_models.append(alt)
        return candidate_models

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 2,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Execute non-streaming inference request to Groq with candidate failover."""
        if not self.is_available:
            raise LLMProviderException("GROQ_API_KEY is not configured in backend environment.")

        vision = self.supports_vision_attachments(attachments)
        messages = self._build_messages(prompt, system_instruction, history, attachments, vision=vision)
        effective_tokens = min(max_tokens, 4096)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        candidate_models = self._candidate_models(model_override, vision)

        for active_model in candidate_models:
            from app.services.llm.circuit_breaker import model_health_tracker
            if not model_health_tracker.is_eligible("groq", active_model):
                logger.info(f"[GroqProvider] Fast-skipping {active_model} ({model_health_tracker.get_health('groq', active_model).value}); trying next candidate.")
                continue

            payload: Dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_tokens,
                "stream": False,
            }

            if "gpt-oss" in active_model.lower() or "reason" in active_model.lower():
                payload["reasoning_effort"] = "low"
                payload["reasoning_format"] = "hidden"

            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        response = await client.post(self.BASE_URL, json=payload, headers=headers)

                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        wait_sec = float(retry_after) if retry_after and retry_after.replace('.', '', 1).isdigit() else backoff
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure(
                            "groq", active_model, "quota_429", status_code=429,
                            cooldown_seconds=cooldown_from_response(response.headers),
                        )
                        if wait_sec > 2.0:
                            logger.warning(f"Groq {active_model} rate limit requires {wait_sec}s wait; trying next candidate model...")
                            break
                        if attempt < max_retries - 1:
                            logger.warning(f"Groq {active_model} 429 received. Backing off for {wait_sec}s...")
                            await asyncio.sleep(wait_sec)
                            backoff *= 2.0
                            continue
                        logger.warning(f"Groq model {active_model} rate limited; trying next candidate...")
                        break

                    if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
                        logger.warning(f"Groq HTTP {response.status_code} on {active_model}. Retrying in {backoff}s...")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("groq", active_model, f"http_{response.status_code}", status_code=response.status_code)
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue

                    if response.status_code != 200:
                        logger.error(f"Groq API error {response.status_code} on {active_model}: {response.text}")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("groq", active_model, f"http_{response.status_code}", status_code=response.status_code)
                        break

                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        logger.warning(f"Groq API returned empty choices for {active_model}; trying next candidate...")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("groq", active_model, "empty_choices")
                        break



                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if not content and message.get("reasoning"):
                        content = message.get("reasoning", "")
                    import re
                    clean_content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    from app.services.llm.circuit_breaker import model_health_tracker
                    model_health_tracker.record_success("groq", active_model)
                    self.last_active_model = active_model
                    return clean_content or content

                except httpx.RequestError as exc:
                    if attempt < max_retries - 1:
                        logger.warning(f"Groq connection error ({active_model}): {exc}. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    break

        raise QuotaExhaustedException("All Groq models exhausted quota or rate limits.")

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 2,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token chunks asynchronously via Groq OpenAI-compatible SSE endpoint with candidate failover."""
        if not self.is_available:
            yield "Backend configuration alert: `GROQ_API_KEY` is not set in `.env`. Obtain a free key from https://console.groq.com/keys."
            return

        vision = self.supports_vision_attachments(attachments)
        messages = self._build_messages(prompt, system_instruction, history, attachments, vision=vision)
        effective_tokens = min(max_tokens, 4096)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        candidate_models = self._candidate_models(model_override, vision)

        for active_model in candidate_models:
            from app.services.llm.circuit_breaker import model_health_tracker
            if not model_health_tracker.is_eligible("groq", active_model):
                logger.info(f"[GroqProvider] Stream fast-skipping {active_model} ({model_health_tracker.get_health('groq', active_model).value}); trying next candidate.")
                continue

            payload: Dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_tokens,
                "stream": True,
            }

            if "gpt-oss" in active_model.lower() or "reason" in active_model.lower():
                payload["reasoning_effort"] = "low"
                payload["reasoning_format"] = "hidden"

            backoff = 1.0
            for attempt in range(max_retries):
                has_yielded_token = False
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        async with client.stream("POST", self.BASE_URL, json=payload, headers=headers) as response:
                            if response.status_code == 429:
                                retry_after = response.headers.get("retry-after")
                                wait_sec = float(retry_after) if retry_after and retry_after.replace('.', '', 1).isdigit() else backoff
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure(
                                    "groq", active_model, "quota_429", status_code=429,
                                    cooldown_seconds=cooldown_from_response(response.headers),
                                )
                                if wait_sec > 2.0:
                                    logger.warning(f"Groq stream rate limit on {active_model} requires {wait_sec}s wait; trying next candidate model...")
                                    break
                                if attempt < max_retries - 1:
                                    logger.warning(f"Groq stream 429 on {active_model}. Backing off for {wait_sec}s...")
                                    await asyncio.sleep(wait_sec)
                                    backoff *= 2.0
                                    continue
                                logger.warning(f"Groq stream rate limit on {active_model}; attempting next Groq model...")
                                break

                            if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
                                logger.warning(f"Groq stream HTTP {response.status_code} on {active_model}. Retrying in {backoff}s...")
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("groq", active_model, f"http_{response.status_code}", status_code=response.status_code)
                                await asyncio.sleep(backoff)
                                backoff *= 2.0
                                continue

                            if response.status_code != 200:
                                error_body = await response.aread()
                                logger.error(f"Groq stream error {response.status_code} on {active_model}: {error_body.decode('utf-8', errors='ignore')}")
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("groq", active_model, f"http_{response.status_code}", status_code=response.status_code)
                                break

                            self.last_active_model = active_model
                            thinking_filter = ThinkingFilter()
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    raw_data = line[6:].strip()
                                    if raw_data == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(raw_data)
                                        choices = chunk_json.get("choices", [])
                                        if not choices:
                                            continue
                                        delta = choices[0].get("delta", {})

                                        text = delta.get("content")
                                        if not text and delta.get("reasoning"):
                                            # If content is absent but reasoning delta is present and not hidden
                                            text = delta.get("reasoning")

                                        if text:
                                            visible = thinking_filter.feed(text)
                                            if visible:
                                                has_yielded_token = True
                                                yield visible

                                    except json.JSONDecodeError:
                                        continue

                            if has_yielded_token:
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_success("groq", active_model)
                                self.last_active_model = active_model
                                return

                except httpx.RequestError as exc:
                    if has_yielded_token:
                        raise LLMProviderException(
                            f"Stream interrupted after yielding tokens for {active_model}: {exc}"
                        ) from exc
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error during Groq stream ({active_model}): {exc}. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    break

        raise QuotaExhaustedException("All Groq candidate models exhausted quota or rate limits.")
