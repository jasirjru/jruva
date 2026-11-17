"""Custom Exceptions and RFC 7807 Problem Details Handlers."""
from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ProblemDetails(BaseModel):
    """RFC 7807 compliant error payload."""
    type: str
    title: str
    status: int
    detail: str
    instance: Optional[str] = None
    code: Optional[str] = None


class BaseAppException(Exception):
    """Base domain exception with HTTP status mapping."""
    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_SERVER_ERROR",
        title: str = "Internal Server Error"
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.title = title
        super().__init__(detail)


class RateLimitExceededException(BaseAppException):
    def __init__(self, detail: str = "Free tier rate limit exceeded. Please wait a moment."):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            title="Rate Limit Exceeded"
        )


class LLMProviderException(BaseAppException):
    def __init__(self, detail: str = "AI inference provider failure.", status_code: int = status.HTTP_502_BAD_GATEWAY):
        super().__init__(
            detail=detail,
            status_code=status_code,
            error_code="LLM_PROVIDER_ERROR",
            title="AI Service Unavailable"
        )


class QuotaExhaustedException(BaseAppException):
    def __init__(self, detail: str = "Model quota exhausted on free tier. Retry later."):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="QUOTA_EXHAUSTED",
            title="Model Quota Exhausted"
        )


class AllProvidersUnavailableException(LLMProviderException):
    """Raised when both primary and backup LLM providers are down or their circuit breakers are OPEN."""
    def __init__(
        self,
        detail: str = "All upstream AI inference providers are currently unavailable. Circuit breakers tripped.",
        circuits: Optional[Dict[str, Any]] = None,
        retry_after_seconds: int = 15,
        trace_id: Optional[str] = None,
    ):
        super().__init__(detail=detail)
        self.status_code = status.HTTP_502_BAD_GATEWAY
        self.error_code = "LLM_PROVIDER_ERROR"
        self.title = "All AI Upstream Providers Unavailable"
        self.circuits = circuits or {}
        self.retry_after_seconds = retry_after_seconds
        self.trace_id = trace_id


from app.core.logging_config import sanitize_sensitive_data


async def app_exception_handler(request: Request, exc: BaseAppException) -> JSONResponse:
    """Serialize BaseAppException into RFC 7807 Problem Details with diagnostic headers."""
    content: Dict[str, Any] = {
        "type": f"https://api.chatbot.dev/errors/{exc.error_code.lower().replace('_', '-')}",
        "title": exc.title,
        "status": exc.status_code,
        "detail": sanitize_sensitive_data(exc.detail),
        "instance": str(request.url.path),
        "code": exc.error_code,
    }
    if getattr(exc, "trace_id", None):
        content["trace_id"] = exc.trace_id
    if getattr(exc, "circuits", None):
        content["circuits"] = exc.circuits
    if getattr(exc, "retry_after_seconds", None):
        content["retry_after_seconds"] = exc.retry_after_seconds

    headers = {}
    if getattr(exc, "retry_after_seconds", None):
        headers["Retry-After"] = str(exc.retry_after_seconds)

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers,
        media_type="application/problem+json",
    )
