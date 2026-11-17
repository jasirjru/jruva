"""Abstract Base Class (Interface) for LLM Providers."""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional


class BaseLLMProvider(ABC):
    """Provider contract to ensure modularity and multi-provider fallback."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g. 'gemini', 'groq')."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Indicates if provider has necessary API credentials configured."""
        pass

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Generate a complete text response synchronously."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        attachments: Optional[List[Any]] = None,
        model_override: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as an asynchronous generator."""
        pass
