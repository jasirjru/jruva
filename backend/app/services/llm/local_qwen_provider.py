"""Local Qwen3-8B Provider running 4-bit NF4 on NVIDIA GeForce RTX 4060 Laptop GPU."""
from typing import Optional
from app.services.llm.ollama_provider import OllamaProvider


class LocalQwenProvider(OllamaProvider):
    """Local Qwen3-8B Provider using custom streaming engine on port 11435."""

    BASE_URL = "http://127.0.0.1:11435/api/chat"
    TAGS_URL = "http://127.0.0.1:11435/api/tags"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super().__init__(model=model or "qwen3-8b-local", base_url=base_url or self.BASE_URL)

    @property
    def provider_name(self) -> str:
        return "local_qwen"
