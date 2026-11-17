"""Generic OpenAI-compatible chat provider, enabled per vendor by environment variables only.

Cerebras, Mistral, SambaNova, NVIDIA and Cloudflare Workers AI all expose
``POST {base_url}/chat/completions`` in the OpenAI format. Each vendor becomes an
extra free-capacity fallback when its ``<VENDOR>_API_KEY`` is set; without a key
the vendor is skipped entirely. Base URL and model list have defaults and can be
overridden with ``<VENDOR>_BASE_URL`` and ``<VENDOR>_MODELS`` (comma-separated).
"""
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import LLMProviderException, QuotaExhaustedException
from app.services.llm.base import BaseLLMProvider
from app.services.llm.circuit_breaker import model_health_tracker
from app.services.llm.rate_limit import cooldown_from_response, parse_model_list
from app.services.llm.stream_filter import ThinkingFilter

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15.0
MAX_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class CompatVendor:
    """Static defaults for one OpenAI-compatible vendor. Model IDs are checked against docs on 10 Nov 2026."""

    name: str
    default_base_url: str
    default_models: tuple[str, ...]


# Order = fallback order after Groq and Gemini. Larger free daily capacity first.
COMPAT_VENDORS: tuple[CompatVendor, ...] = (
    CompatVendor("mistral", "https://api.mistral.ai/v1", ("mistral-small-latest",)),
    CompatVendor(
        "cloudflare",
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        ("@cf/openai/gpt-oss-120b", "@cf/openai/gpt-oss-20b"),
    ),
    CompatVendor("cerebras", "https://api.cerebras.ai/v1", ("gpt-oss-120b",)),
    CompatVendor("nvidia", "https://integrate.api.nvidia.com/v1", ("openai/gpt-oss-120b",)),
    CompatVendor("sambanova", "https://api.sambanova.ai/v1", ("gpt-oss-120b", "Meta-Llama-3.3-70B-Instruct")),
)


def _setting(name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


class OpenAICompatibleProvider(BaseLLMProvider):
    """One OpenAI-compatible vendor with its own model cascade and per-model cooldowns."""

    def __init__(self, name: str, base_url: str, api_key: str, models: list[str]) -> None:
        self._name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.models = list(models)
        self.model = self.models[0] if self.models else ""
        self.last_active_model: str | None = self.model

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self.base_url and self.models)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _build_messages(
        prompt: str,
        system_instruction: str | None,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        for turn in history or []:
            role = "assistant" if turn.get("role") in ("model", "assistant") else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _candidates(self, model_override: str | None) -> list[str]:
        ordered = [model_override] if model_override in self.models else []
        return ordered + [m for m in self.models if m not in ordered]

    def _payload(self, model: str, messages: list, temperature: float, max_tokens: int, stream: bool) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
            "stream": stream,
        }

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _record_http_failure(self, model: str, status_code: int, headers: Any, body: str) -> None:
        if status_code == 429:
            cooldown = cooldown_from_response(headers, body)
            logger.warning(f"[{self._name}] {model} rate limited; cooling down {cooldown:.0f}s.")
            model_health_tracker.record_failure(
                self._name, model, "quota_429", status_code=429, cooldown_seconds=cooldown
            )
            return
        logger.warning(f"[{self._name}] {model} returned HTTP {status_code}.")
        model_health_tracker.record_failure(self._name, model, f"http_{status_code}", status_code=status_code)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: list[Any] | None = None,
        model_override: str | None = None,
    ) -> str:
        if not self.is_available:
            raise LLMProviderException(f"{self._name} provider is not configured.")
        messages = self._build_messages(prompt, system_instruction, history)
        for model in self._candidates(model_override):
            if not model_health_tracker.is_eligible(self._name, model):
                continue
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        self.chat_url,
                        json=self._payload(model, messages, temperature, max_tokens, stream=False),
                        headers=self._headers(),
                    )
            except httpx.RequestError as exc:
                model_health_tracker.record_failure(self._name, model, f"ConnectError: {type(exc).__name__}")
                continue
            if response.status_code != 200:
                self._record_http_failure(model, response.status_code, response.headers, response.text)
                continue
            choices = response.json().get("choices") or []
            content = (choices[0].get("message", {}).get("content") or "") if choices else ""
            visible = ThinkingFilter().feed(content).strip()
            if not visible:
                model_health_tracker.record_failure(self._name, model, "empty_choices")
                continue
            model_health_tracker.record_success(self._name, model)
            self.last_active_model = model
            return visible
        raise QuotaExhaustedException(f"All {self._name} models are rate limited or unavailable.")

    async def stream_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: list[Any] | None = None,
        model_override: str | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available:
            raise LLMProviderException(f"{self._name} provider is not configured.")
        messages = self._build_messages(prompt, system_instruction, history)
        for model in self._candidates(model_override):
            if not model_health_tracker.is_eligible(self._name, model):
                continue
            yielded = False
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                    async with client.stream(
                        "POST",
                        self.chat_url,
                        json=self._payload(model, messages, temperature, max_tokens, stream=True),
                        headers=self._headers(),
                    ) as response:
                        if response.status_code != 200:
                            body = (await response.aread()).decode("utf-8", errors="ignore")
                            self._record_http_failure(model, response.status_code, response.headers, body)
                            continue
                        self.last_active_model = model
                        async for text in self._iter_sse_text(response):
                            yielded = True
                            yield text
            except httpx.RequestError as exc:
                if yielded:
                    raise LLMProviderException(f"{self._name} stream interrupted.") from exc
                model_health_tracker.record_failure(self._name, model, f"ConnectError: {type(exc).__name__}")
                continue
            if yielded:
                model_health_tracker.record_success(self._name, model)
                return
            model_health_tracker.record_failure(self._name, model, "empty_choices")
        raise QuotaExhaustedException(f"All {self._name} models are rate limited or unavailable.")

    @staticmethod
    async def _iter_sse_text(response: httpx.Response) -> AsyncGenerator[str, None]:
        thinking_filter = ThinkingFilter()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            text = (choices[0].get("delta", {}).get("content") or "") if choices else ""
            visible = thinking_filter.feed(text) if text else ""
            if visible:
                yield visible


def _base_url_for(vendor: CompatVendor) -> str:
    override = _setting(f"{vendor.name.upper()}_BASE_URL")
    if override:
        return override
    if "{account_id}" in vendor.default_base_url:
        account_id = _setting("CLOUDFLARE_ACCOUNT_ID")
        return vendor.default_base_url.format(account_id=account_id) if account_id else ""
    return vendor.default_base_url


def build_openai_compatible_providers() -> list[OpenAICompatibleProvider]:
    """Create one provider per vendor whose API key is set. Vendors without a key are disabled."""
    providers: list[OpenAICompatibleProvider] = []
    for vendor in COMPAT_VENDORS:
        prefix = vendor.name.upper()
        api_key = _setting(f"{prefix}_API_KEY")
        if not api_key:
            continue
        base_url = _base_url_for(vendor)
        models = parse_model_list(_setting(f"{prefix}_MODELS"), list(vendor.default_models))
        provider = OpenAICompatibleProvider(vendor.name, base_url, api_key, models)
        if provider.is_available:
            providers.append(provider)
        else:
            logger.warning(f"[OpenAICompat] {vendor.name} has a key but no base URL; disabled.")
    return providers
