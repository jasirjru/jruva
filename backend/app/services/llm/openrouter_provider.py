"""OpenRouter AI Provider supporting free frontier models (DeepSeek R1, Qwen 2.5 72B, Llama 3.3 70B)."""
import asyncio
import contextvars
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.services.llm.base import BaseLLMProvider
from app.services.llm.stream_filter import ThinkingFilter
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

_openrouter_active_model_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_openrouter_active_model_var", default=None
)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter Cloud Provider implementing OpenAI-compatible completions for free models."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    FALLBACK_MODELS = [
        "google/gemma-4-31b-it:free",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        raw_model = model or "google/gemma-4-31b-it:free"
        if raw_model.startswith("openrouter/"):
            raw_model = raw_model[len("openrouter/"):]
        # Map discontinued or unavailable free coding models to active free models
        if raw_model in ("qwen-2.5-coder-32b", "qwen-coder", "qwen-2.5-coder-32b:free", "qwen/qwen-2.5-coder-32b-instruct:free", "liquid/lfm-2.5-2.6b:free"):
            raw_model = "google/gemma-4-31b-it:free"
        elif raw_model in ("deepseek-r1", "deepseek-r1:free", "r1"):
            raw_model = "deepseek/deepseek-r1:free"
        self.model = raw_model
        self._last_active_model: Optional[str] = self.model

    @property
    def last_active_model(self) -> str:
        scoped = _openrouter_active_model_var.get()
        if scoped is not None:
            return scoped
        return getattr(self, "_last_active_model", self.model)

    @last_active_model.setter
    def last_active_model(self, val: str) -> None:
        self._last_active_model = val
        _openrouter_active_model_var.set(val)

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def api_key(self) -> str:
        import os
        from app.core.config import settings
        return self._api_key or getattr(settings, "OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        attachments: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if history:
            for turn in history:
                role = "assistant" if turn.get("role") in ["assistant", "model"] else "user"
                messages.append({"role": role, "content": turn.get("content", "")})

        if attachments:
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for att in attachments:
                mime_type = getattr(att, "mime_type", None) or (att.get("mime_type") if isinstance(att, dict) else "application/octet-stream")
                data = getattr(att, "data", None) or (att.get("data") if isinstance(att, dict) else "")
                if data:
                    clean_data = data.split(",", 1)[1] if ("," in data and "base64" in data) else data
                    if mime_type.startswith("image/"):
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{clean_data}"}
                        })
                    else:
                        import base64
                        import binascii
                        try:
                            decoded = base64.b64decode(clean_data).decode("utf-8", errors="replace")
                            user_content.append({
                                "type": "text",
                                "text": f"[Attached Document Content]\n{decoded[:8000]}"
                            })
                        except (binascii.Error, ValueError) as decode_err:
                            logger.debug("[OpenRouter] Non-critical attachment decode failure: %s", decode_err)
                        except Exception as exc:
                            logger.warning("[SilentFailureGuard] Unexpected error decoding document attachment in OpenRouter: %s", exc)
                            telemetry_service.record_silent_failure("llm_provider", str(exc))
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        if not self.is_available:
            raise LLMProviderException("OPENROUTER_API_KEY is not configured in backend environment.")

        messages = self._build_messages(prompt, system_instruction, history, attachments)
        effective_tokens = min(max_tokens, 4096)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://jruva.in",
            "X-Title": "Production AI Chatbot",
            "Content-Type": "application/json",
        }

        target_model = model_override or self.model
        candidate_models = [target_model]
        if self.model not in candidate_models:
            candidate_models.append(self.model)
        for alt in self.FALLBACK_MODELS:
            if alt not in candidate_models:
                candidate_models.append(alt)

        for active_model in candidate_models:
            from app.services.llm.circuit_breaker import model_health_tracker
            if not model_health_tracker.is_eligible("openrouter", active_model):
                logger.info(f"[OpenRouter] Fast-skipping {active_model} ({model_health_tracker.get_health('openrouter', active_model).value}); trying next candidate.")
                continue

            payload = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_tokens,
                "include_reasoning": False,
            }

            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        response = await client.post(self.BASE_URL, json=payload, headers=headers)

                    if response.status_code == 429:
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("openrouter", active_model, "quota_429", status_code=429)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                            continue
                        logger.warning(f"OpenRouter {active_model} rate limited; trying next candidate...")
                        break

                    if response.status_code == 404:
                        logger.warning(f"OpenRouter model {active_model} not found (404); trying next candidate...")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("openrouter", active_model, "not_found_404", status_code=404)
                        break

                    if response.status_code != 200:
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("openrouter", active_model, f"http_{response.status_code}", status_code=response.status_code)
                        break

                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_success("openrouter", active_model)
                        self.last_active_model = active_model
                        if content:
                            import re
                            clean = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                            return clean or content
                        reasoning = msg.get("reasoning")
                        if reasoning:
                            import re
                            clean = re.sub(r"<think>[\s\S]*?</think>", "", reasoning).strip()
                            return clean or reasoning.strip()
                    return ""

                except httpx.RequestError as exc:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    break

        raise QuotaExhaustedException("All OpenRouter models exhausted quota or rate limits.")

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available:
            yield "OpenRouter alert: `OPENROUTER_API_KEY` is not set in `.env`."
            return

        messages = self._build_messages(prompt, system_instruction, history, attachments)
        effective_tokens = min(max_tokens, 4096)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://jruva.in",
            "X-Title": "Production AI Chatbot",
            "Content-Type": "application/json",
        }

        target_model = model_override or self.model
        candidate_models = [target_model]
        if self.model not in candidate_models:
            candidate_models.append(self.model)
        for alt in self.FALLBACK_MODELS:
            if alt not in candidate_models:
                candidate_models.append(alt)

        for active_model in candidate_models:
            from app.services.llm.circuit_breaker import model_health_tracker
            if not model_health_tracker.is_eligible("openrouter", active_model):
                logger.info(f"[OpenRouter] Stream fast-skipping {active_model} ({model_health_tracker.get_health('openrouter', active_model).value}); trying next candidate.")
                continue

            payload = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_tokens,
                "stream": True,
                "include_reasoning": False,
            }

            backoff = 1.0
            for attempt in range(max_retries):
                has_yielded = False
                thinking_filter = ThinkingFilter()
                reasoning_fallback: List[str] = []
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        async with client.stream("POST", self.BASE_URL, json=payload, headers=headers) as response:
                            if response.status_code == 429:
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("openrouter", active_model, "quota_429", status_code=429)
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(backoff)
                                    backoff *= 2.0
                                    continue
                                break

                            if response.status_code in (404, 500, 502, 503):
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("openrouter", active_model, f"http_{response.status_code}", status_code=response.status_code)
                                break

                            if response.status_code != 200:
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("openrouter", active_model, f"http_{response.status_code}", status_code=response.status_code)
                                break

                            self.last_active_model = active_model
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    raw = line[6:].strip()
                                    if raw == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(raw)
                                        choices = chunk.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            text = delta.get("content")
                                            if not text and delta.get("reasoning"):
                                                reasoning_fallback.append(delta.get("reasoning"))

                                            if text:
                                                visible = thinking_filter.feed(text)
                                                if visible:
                                                    has_yielded = True
                                                    yield visible
                                    except Exception:
                                        continue

                            if not has_yielded and reasoning_fallback:
                                import re
                                full_fallback = "".join(reasoning_fallback)
                                clean = re.sub(r"<think>[\s\S]*?</think>", "", full_fallback).strip()
                                if clean:
                                    has_yielded = True
                                    yield clean

                            if has_yielded:
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_success("openrouter", active_model)
                                self.last_active_model = active_model
                                return

                except httpx.RequestError as exc:
                    if has_yielded:
                        raise LLMProviderException(
                            f"Stream interrupted after yielding tokens for {active_model}: {exc}"
                        ) from exc
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    break

        raise QuotaExhaustedException("All OpenRouter retries exhausted.")
