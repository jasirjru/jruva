"""FastAPI Dependency Injection Providers."""
from functools import lru_cache
from app.core.config import Settings, settings
from app.services.chat_service import ChatService
from app.services.llm.gemini_provider import GeminiProvider


from app.services.llm.router import LLMRouter
from app.api.v1.endpoints.auth import get_current_user, get_current_user_optional

@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return settings


# Global singleton instances for in-memory session persistence in free-tier deployment
_router_instance = LLMRouter()
_chat_service_instance = ChatService(router=_router_instance)


def get_llm_router() -> LLMRouter:
    """Provide LLMRouter singleton instance."""
    return _router_instance


def get_chat_service() -> ChatService:
    """Provide ChatService instance."""
    return _chat_service_instance
