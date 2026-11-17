"""Security Utilities and Rate Limiter Setup."""
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings


import uuid


def get_user_or_ip_identifier(request: Request) -> str:
    """Use verified authentication or client IP; benchmark bypass is opt-in."""
    bypass_token = request.headers.get("x-benchmark-bypass")
    if settings.BENCHMARK_MODE and settings.BENCHMARK_BYPASS_SECRET and bypass_token == settings.BENCHMARK_BYPASS_SECRET:
        return f"bypass:{uuid.uuid4().hex}"

    # Lazy import avoids the API/router initialization cycle. Use the same
    # cookie/Bearer verification and account lookup as endpoint authorization.
    from app.api.v1.endpoints.auth import get_current_user_optional
    user = get_current_user_optional(request, request.headers.get("authorization"))
    if user and user.get("id"):
        return f"user:{user['id']}"

    # Fallback to client remote IP
    return f"ip:{get_remote_address(request)}"


# In-memory rate limiter configured for free-tier protection per user / IP
limiter = Limiter(
    key_func=get_user_or_ip_identifier,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri="memory://",
)

# Stricter per-user/IP limit for endpoints that spend LLM, OCR, embedding or CPU budget.
EXPENSIVE_ENDPOINT_LIMIT = "10/minute"
