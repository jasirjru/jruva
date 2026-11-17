import contextvars
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.core.exceptions import LLMProviderException
from app.services.llm.base import BaseLLMProvider
from app.services.llm.stream_filter import ThinkingFilter
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

_ollama_active_model_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_ollama_active_model_var", default=None
)


class OllamaProvider(BaseLLMProvider):
    """Local Ollama AI Provider supporting quantized fine-tuned GGUF models on local GPU/CPU."""

    BASE_URL = "http://127.0.0.1:11434/api/chat"
    TAGS_URL = "http://127.0.0.1:11434/api/tags"

    STOP_TOKENS = [
        "### Instruction:",
        "### Response:",
        "### System:",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "</s>",
        "<unstructured>",
        "</unstructured>",
        "<structured>",
        "</structured>",
    ]

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        self.model = model or "jru-specialist:v6"
        self.base_url = base_url or self.BASE_URL
        self._last_active_model: Optional[str] = self.model
        self._installed_models: Optional[set[str]] = None

    @property
    def last_active_model(self) -> str:
        scoped = _ollama_active_model_var.get()
        if scoped is not None:
            return scoped
        return getattr(self, "_last_active_model", self.model)

    @last_active_model.setter
    def last_active_model(self, val: str) -> None:
        self._last_active_model = val
        _ollama_active_model_var.set(val)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def is_available(self) -> bool:
        """Check if local Ollama daemon is active and responding."""
        try:
            with httpx.Client(timeout=0.3) as client:
                r = client.get(self.TAGS_URL)
                if r.status_code == 200:
                    data = r.json()
                    self._installed_models = {
                        item["name"] for item in data.get("models", [])
                        if isinstance(item, dict) and isinstance(item.get("name"), str)
                    }
                return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception as exc:
            logger.warning(f"[SilentFailureGuard] Unexpected error in Ollama is_available: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("router_shadow", str(exc))
            return False

    def is_model_available(self, model_id: str) -> bool:
        """Use the inventory fetched by the availability probe; no extra HTTP calls."""
        target = model_id.removeprefix("ollama/")
        return self._installed_models is None or target in self._installed_models

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        attachments: Optional[List[Any]] = None,
    ) -> List[Dict[str, str]]:
        """Construct standard messages payload for Ollama with decoded attachments & OCR transcripts."""
        messages: List[Dict[str, str]] = []

        # Use the full system instruction passed from ChatService (same prompt as cloud providers)
        # Only fall back to a minimal prompt if absolutely nothing is provided
        effective_system = system_instruction or (
            "You are jru, a helpful AI engineer and assistant.\n"
            "- Code formatting: Always wrap code inside markdown triple backticks, e.g. ```html ... ``` or ```python ... ```.\n"
            "- Web components: Deliver complete, self-contained HTML/CSS/JS inside a ```html code block.\n"
            "- Never output meta text like 'Artifact Banner' or 'html code block'.\n"
            "- Be direct, accurate, and concise."
        )
        messages.append({"role": "system", "content": effective_system})

        if history:
            for turn in history:
                role = "assistant" if turn.get("role") in ["assistant", "model"] else "user"
                content = turn.get("content", "").strip()
                if content:
                    messages.append({"role": role, "content": content})

        file_sections: List[str] = []
        if attachments:
            for att in attachments:
                name = getattr(att, "name", None) or (att.get("name") if isinstance(att, dict) else "attachment")
                mime_type = getattr(att, "mime_type", None) or (att.get("mime_type") if isinstance(att, dict) else "")
                data = getattr(att, "data", None) or (att.get("data") if isinstance(att, dict) else "")
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
                    except Exception:
                        file_sections.append(f"[Attached File: {name}]")

        user_content = "\n\n".join(file_sections + [prompt]) if file_sections else prompt
        messages.append({"role": "user", "content": user_content})
        return messages

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Generate a complete text response synchronously from local Ollama."""
        effective_model_str = model_override or self.model
        self.last_active_model = effective_model_str
        messages = self._build_messages(prompt, system_instruction, history, attachments=attachments)
        payload = {
            "model": effective_model_str.replace("ollama/", ""),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "repeat_penalty": 1.18,
                "repeat_last_n": 64,
                "num_predict": max_tokens,
                "stop": self.STOP_TOKENS,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.base_url, json=payload)
                if response.status_code != 200:
                    raise LLMProviderException(
                        f"Local Ollama returned error: {response.text}",
                        status_code=response.status_code,
                    )
                data = response.json()
                raw_text = data.get("message", {}).get("content", "")
                import re
                clean_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text)
                clean_text = re.sub(r"</?(?:unstructured|structured)>", "", clean_text).strip()
                return clean_text
        except httpx.ConnectError:
            raise LLMProviderException("Local Ollama daemon is not running. Start Ollama with `ollama serve`.")

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream token chunks via Ollama's native streaming endpoint with repeat suppression."""
        effective_model_str = model_override or self.model
        self.last_active_model = effective_model_str
        messages = self._build_messages(prompt, system_instruction, history, attachments=attachments)
        payload = {
            "model": effective_model_str.replace("ollama/", ""),
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "repeat_penalty": 1.18,
                "repeat_last_n": 64,
                "num_predict": max_tokens,
                "stop": self.STOP_TOKENS,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", self.base_url, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        raise LLMProviderException(
                            f"Ollama stream error: {error_body.decode('utf-8')}",
                            status_code=response.status_code,
                        )

                    thinking_filter = ThinkingFilter()
                    import re
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            if chunk.get("done", False):
                                break
                            raw_token = chunk.get("message", {}).get("content")
                            if raw_token:
                                # Strip structured/unstructured markup
                                token = re.sub(r"</?(?:unstructured|structured)>", "", raw_token)
                                if not token:
                                    continue
                                visible = thinking_filter.feed(token)
                                if visible:
                                    yield visible
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise LLMProviderException("Local Ollama daemon is not running on port 11434.")
