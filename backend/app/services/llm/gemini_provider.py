"""Google Gemini AI Provider with asynchronous streaming and 429 rate-limit resilience."""
import asyncio
import contextvars
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.core.attachments import attachment_mime_type
from app.core.config import settings
from app.core.exceptions import (
    LLMProviderException,
    QuotaExhaustedException,
)
from app.services.llm.base import BaseLLMProvider
from app.services.llm.rate_limit import cooldown_from_response, parse_model_list
from app.services.llm.stream_filter import ThinkingFilter

logger = logging.getLogger(__name__)

_gemini_active_model_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_gemini_active_model_var", default=None
)


def _warn_if_served_model_differs(requested: str, served: str | None) -> None:
    """Log when Google answers with a different model than requested.

    Google retires Gemini IDs by routing them to a successor (gemini-3.5-flash is
    served by gemini-3.6-flash). The registry should then be updated so the model
    Jruva reports is the model that answered.
    """
    if served and served.strip().lower() != requested.strip().lower():
        logger.warning(f"Gemini requested {requested} but Google served {served}; update the model registry.")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI implementation using asynchronous HTTP client.
    
    Adheres strictly to free-tier quotas and handles status 429 quota exhaustion.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        self._api_key = api_key
        target_model = model or settings.GEMINI_MODEL or "gemini-3.6-flash"
        # Map legacy, discontinued, or non-existent models to verified active Google Gemini models
        legacy_models = {
            "gemini-2.0-flash": "gemini-3.6-flash",
            "gemini-2.5-flash": "gemini-3.6-flash",
            "gemini-1.5-flash": "gemini-3.6-flash",
            "gemini-1.5-pro": "gemini-3.6-flash",
            "gemini-flash": "gemini-3.6-flash",
            "gemini-flash-latest": "gemini-3.6-flash",
            "gemini-3.5-flash": "gemini-3.6-flash",  # Google serves 3.5 requests with 3.6
        }
        self.model = legacy_models.get(target_model, target_model)
        self._last_active_model: Optional[str] = self.model
        self.FALLBACK_MODELS = parse_model_list(
            getattr(settings, "GEMINI_FALLBACK_MODELS", ""), type(self).FALLBACK_MODELS
        )

    @property
    def last_active_model(self) -> str:
        scoped = _gemini_active_model_var.get()
        if scoped is not None:
            return scoped
        return getattr(self, "_last_active_model", self.model)

    @last_active_model.setter
    def last_active_model(self, val: str) -> None:
        self._last_active_model = val
        _gemini_active_model_var.set(val)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    @property
    def api_key(self) -> str:
        import os
        if self._api_key is not None:
            return self._api_key
        return settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

    def _build_payload(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: Optional[List[Any]] = None,
        model: str | None = None,
        is_deep_thinking: bool = False,
    ) -> Dict:
        """Construct the Gemini API contents and generationConfig payload."""
        contents = []

        # Append historical messages if available
        if history:
            for msg in history:
                role = "model" if msg.get("role") in ["assistant", "model"] else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.get("content", "")}]
                })

        user_parts: List[Dict[str, Any]] = []

        # Add visual or document attachment parts (Gemini inline_data)
        if attachments:
            for att in attachments:
                raw_mime = attachment_mime_type(att)
                data = getattr(att, "data", None) or (att.get("data") if isinstance(att, dict) else "")

                if data:
                    clean_data = data.split(",", 1)[1] if ("," in data and "base64" in data) else data
                    clean_data = clean_data.strip().replace("\n", "").replace("\r", "")

                    # Normalize MIME type for Gemini API compliance
                    mime_type = (raw_mime or "").lower().strip()
                    if mime_type in ("image/jpg", "jpg"):
                        mime_type = "image/jpeg"
                    elif mime_type in ("image/png", "png"):
                        mime_type = "image/png"
                    elif mime_type in ("image/webp", "webp"):
                        mime_type = "image/webp"
                    elif "pdf" in mime_type:
                        mime_type = "application/pdf"
                    elif not mime_type or mime_type == "application/octet-stream":
                        name = (getattr(att, "name", None) or (att.get("name") if isinstance(att, dict) else "") or "").lower()
                        if name.endswith((".jpg", ".jpeg")):
                            mime_type = "image/jpeg"
                        elif name.endswith(".png"):
                            mime_type = "image/png"
                        elif name.endswith(".webp"):
                            mime_type = "image/webp"
                        elif name.endswith(".pdf"):
                            mime_type = "application/pdf"
                        else:
                            mime_type = "image/jpeg"

                    user_parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": clean_data
                        }
                    })

        user_parts.append({"text": prompt})

        # Append current user prompt
        contents.append({
            "role": "user",
            "parts": user_parts
        })

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }

        target_model = model or self.model
        thinking_cfg = self._build_thinking_config(target_model, is_deep_thinking=is_deep_thinking)
        if thinking_cfg:
            generation_config["thinkingConfig"] = thinking_cfg

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        return payload

    @staticmethod
    def _timeout_for(attachments: list[Any] | None) -> httpx.Timeout | float:
        """Images and PDFs take Gemini 4-8 s to start answering; text requests keep the tight 8 s limit."""
        return httpx.Timeout(8.0, read=20.0) if attachments else 8.0

    @staticmethod
    def _build_thinking_config(model: str, is_deep_thinking: bool = False) -> dict[str, Any] | None:
        """Construct model-aware thinkingConfig following official Google Gemini specifications.

        - Gemini 3.x (e.g. gemini-3.6-flash, gemini-3.5-flash-lite):
          Uses `thinkingLevel` ("MINIMAL", "LOW", "MEDIUM", "HIGH"). Never uses `thinkingBudget`.
        - Gemini 2.5 (e.g. gemini-2.5-flash, gemini-2.5-pro):
          Uses `thinkingBudget` (integer, e.g. 0 to disable thinking delay).
        - Gemini 1.5 and earlier:
          Does not support thinkingConfig; returns None.
        """
        m = model.lower().strip()
        # 1. Gemini 3.x models
        if "3." in m or "gemini-3" in m:
            level = "MEDIUM" if is_deep_thinking else "LOW"
            return {"thinkingLevel": level}
        # 2. Gemini 2.5 / 2.0 reasoning
        if "2.5" in m or "gemini-2.0-flash-thinking" in m:
            budget = 2048 if is_deep_thinking else 0
            return {"thinkingBudget": budget}
        return None

    # Each Gemini model has its own free-tier quota; the Flash-Lite variants add capacity on the same key.
    # Override with GEMINI_FALLBACK_MODELS.
    FALLBACK_MODELS = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

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
        """Execute synchronous / non-streaming inference request with multi-model retry resilience."""
        if not self.api_key:
            raise LLMProviderException("GEMINI_API_KEY is not configured in backend environment.")

        target_model = model_override or self.model
        candidate_models = [target_model]
        if self.model not in candidate_models:
            candidate_models.append(self.model)
        for alt in self.FALLBACK_MODELS:
            if alt not in candidate_models:
                candidate_models.append(alt)

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        for active_model in candidate_models:
            from app.services.llm.circuit_breaker import model_health_tracker
            if not model_health_tracker.is_eligible("gemini", active_model):
                logger.info(f"[GeminiProvider] Fast-skipping {active_model} ({model_health_tracker.get_health('gemini', active_model).value}); trying next candidate.")
                continue

            payload = self._build_payload(
                prompt, system_instruction, history, temperature, max_tokens, attachments, model=active_model
            )
            url = f"{self.BASE_URL}/{active_model}:generateContent"
            backoff = 1.0

            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=self._timeout_for(attachments)) as client:
                        response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 429:
                        cooldown = cooldown_from_response(response.headers, response.text)
                        logger.warning(f"Gemini {active_model} returned 429 quota exhausted (cooldown {cooldown:.1f}s). Trying next candidate model...")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("gemini", active_model, "quota_429", status_code=429, cooldown_seconds=cooldown)
                        break

                    if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
                        logger.warning(f"Gemini HTTP {response.status_code} on {active_model}. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue

                    if response.status_code == 400 and "thinkingConfig" in payload.get("generationConfig", {}):
                        logger.warning(f"Gemini {active_model} returned 400 with thinkingConfig; retrying without thinkingConfig...")
                        payload["generationConfig"].pop("thinkingConfig", None)
                        continue

                    if response.status_code == 404:
                        logger.warning(f"Gemini model {active_model} not found (404); trying next candidate...")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("gemini", active_model, "not_found_404", status_code=404)
                        break

                    if response.status_code != 200:
                        safe_body = response.text[:200]
                        if self.api_key:
                            safe_body = safe_body.replace(self.api_key, "[REDACTED]")
                        logger.error(f"Gemini API error {response.status_code} on {active_model}: {safe_body}")
                        from app.services.llm.circuit_breaker import model_health_tracker
                        model_health_tracker.record_failure("gemini", active_model, f"http_{response.status_code}", status_code=response.status_code)
                        break

                    data = response.json()
                    _warn_if_served_model_differs(active_model, data.get("modelVersion"))
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return "I could not generate a response. The prompt may have triggered safety filters."
                    parts = candidates[0].get("content", {}).get("parts", [])
                    from app.services.llm.circuit_breaker import model_health_tracker
                    model_health_tracker.record_success("gemini", active_model)
                    self.last_active_model = active_model
                    return ThinkingFilter().feed("".join(part.get("text", "") for part in parts))

                except httpx.RequestError as exc:
                    safe_exc = str(exc)
                    if self.api_key:
                        safe_exc = safe_exc.replace(self.api_key, "[REDACTED]")
                    if attempt < max_retries - 1:
                        logger.warning(f"Gemini connection error ({active_model}): {safe_exc}. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    break
                except Exception as exc:
                    if isinstance(exc, (LLMProviderException, QuotaExhaustedException)):
                        raise
                    safe_exc = str(exc)
                    if self.api_key:
                        safe_exc = safe_exc.replace(self.api_key, "[REDACTED]")
                    logger.error(f"Error parsing Gemini response ({active_model}): {safe_exc}")
                    break

        raise QuotaExhaustedException("All Gemini candidate models exhausted quota or rate limits.")

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
        """Stream token chunks asynchronously via Gemini SSE stream endpoint with retry resilience."""
        if not self.api_key:
            # Emits helpful setup warning in dev if key is missing
            yield "Backend configuration alert: `GEMINI_API_KEY` is not set in `.env`."
            return

        target_model = model_override or self.model
        candidate_models = [target_model]
        if self.model not in candidate_models:
            candidate_models.append(self.model)
        for alt in self.FALLBACK_MODELS:
            if alt not in candidate_models:
                candidate_models.append(alt)

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        for active_model in candidate_models:
            from app.services.llm.circuit_breaker import model_health_tracker
            if not model_health_tracker.is_eligible("gemini", active_model):
                logger.info(f"[GeminiProvider] Stream fast-skipping {active_model} ({model_health_tracker.get_health('gemini', active_model).value}); trying next candidate.")
                continue

            payload = self._build_payload(
                prompt, system_instruction, history, temperature, max_tokens, attachments, model=active_model
            )
            url = f"{self.BASE_URL}/{active_model}:streamGenerateContent?alt=sse"
            backoff = 1.0

            for attempt in range(max_retries):
                has_yielded_token = False
                try:
                    async with httpx.AsyncClient(timeout=self._timeout_for(attachments)) as client:
                        async with client.stream("POST", url, json=payload, headers=headers) as response:
                            if response.status_code == 400 and "thinkingConfig" in payload.get("generationConfig", {}):
                                logger.warning(f"Gemini {active_model} returned 400 with thinkingConfig; retrying without thinkingConfig...")
                                payload["generationConfig"].pop("thinkingConfig", None)
                                continue

                            if response.status_code == 429:
                                error_body = (await response.aread()).decode("utf-8", errors="ignore")
                                cooldown = cooldown_from_response(response.headers, error_body)
                                logger.warning(f"Gemini {active_model} returned 429 quota exhausted (cooldown {cooldown:.1f}s). Trying next candidate model...")
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("gemini", active_model, "quota_429", status_code=429, cooldown_seconds=cooldown)
                                break  # Move to next candidate in candidate_models

                            if response.status_code in (404, 500, 502, 503, 504):
                                error_body = await response.aread()
                                safe_err = error_body.decode("utf-8", errors="ignore")[:80]
                                if self.api_key:
                                    safe_err = safe_err.replace(self.api_key, "[REDACTED]")
                                logger.warning(
                                    f"Gemini model {active_model} returned {response.status_code}: {safe_err}. "
                                    f"Attempting retry..."
                                )
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("gemini", active_model, f"http_{response.status_code}", status_code=response.status_code)
                                if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
                                    await asyncio.sleep(0.5)
                                    continue
                                break  # Break attempt loop to try next candidate model

                            if response.status_code != 200:
                                error_body = await response.aread()
                                safe_err = error_body.decode("utf-8", errors="ignore")[:200]
                                if self.api_key:
                                    safe_err = safe_err.replace(self.api_key, "[REDACTED]")
                                logger.error(f"Gemini stream error {response.status_code}: {safe_err}")
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_failure("gemini", active_model, f"stream_error_{response.status_code}", status_code=response.status_code)
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
                                        if not has_yielded_token:
                                            _warn_if_served_model_differs(active_model, chunk_json.get("modelVersion"))
                                        candidates = chunk_json.get("candidates", [])
                                        if candidates:
                                            parts = candidates[0].get("content", {}).get("parts", [])
                                            for part in parts:
                                                text = part.get("text")
                                                visible = thinking_filter.feed(text) if text else ""
                                                if visible:
                                                    has_yielded_token = True
                                                    yield visible
                                    except json.JSONDecodeError:
                                        continue

                            if has_yielded_token:
                                from app.services.llm.circuit_breaker import model_health_tracker
                                model_health_tracker.record_success("gemini", active_model)
                                self.last_active_model = active_model
                                return  # Successfully finished stream!

                except httpx.RequestError as exc:
                    safe_exc = str(exc)
                    if self.api_key:
                        safe_exc = safe_exc.replace(self.api_key, "[REDACTED]")
                    if has_yielded_token:
                        raise LLMProviderException(
                            f"Stream interrupted after yielding tokens for {active_model}: {safe_exc}"
                        ) from exc
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error during Gemini stream ({active_model}): {safe_exc}. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    logger.warning(f"Failed candidate {active_model}: {safe_exc}")
                    break

        # If all candidates fail
        raise QuotaExhaustedException("All Gemini models exhausted quota or rate limits.")
