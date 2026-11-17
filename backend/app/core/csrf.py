"""CSRF Origin Verification Middleware.

Protects state-changing endpoints (POST, PUT, PATCH, DELETE) against
cross-site request forgery by verifying the request's Origin (or Referer
fallback) header against the configured ALLOWED_ORIGINS list.

This is the OWASP-recommended approach for API-only backends consumed
by single-page applications:
https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html#verifying-origin-with-standard-headers

Design decisions:
  - SAFE METHODS (GET, HEAD, OPTIONS) are always allowed through.
  - When both Origin and Referer are absent on a mutating request,
    the request is REJECTED (403). This is a deliberate choice — some
    privacy-focused browsers or proxies strip these headers, but silently
    allowing requests without origin verification would defeat the entire
    purpose of this middleware. Users in those edge environments will see
    a clear 403 error rather than a silent security bypass.
  - The /auth/google endpoint is exempt because Google's OAuth redirect
    POST may arrive without a matching Origin header.
"""
import logging
from urllib.parse import urlparse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("chatbot.csrf")

# HTTP methods that never cause state changes — always safe to pass through
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Paths exempt from CSRF origin checks (e.g., OAuth callbacks where
# the Origin may not match our frontend domain)
_EXEMPT_PATHS = frozenset()


class CsrfOriginMiddleware(BaseHTTPMiddleware):
    """Verify Origin/Referer header on mutating requests against allowed origins.

    Args:
        app: The ASGI application to wrap.
        allowed_origins: List of allowed origin strings
            (e.g., ["https://my-app.vercel.app", "http://localhost:3000"]).
    """

    def __init__(self, app, allowed_origins: list[str] | None = None):
        super().__init__(app)
        # Normalize origins: lowercase, no trailing slash
        self._allowed_origins: set[str] = set()
        for origin in (allowed_origins or []):
            cleaned = origin.strip().rstrip("/").lower()
            if cleaned:
                self._allowed_origins.add(cleaned)

    def _extract_origin(self, request: Request) -> str | None:
        """Extract origin from the Origin header, falling back to Referer.

        Returns the origin string (scheme + host + port) or None if both
        headers are missing/unparseable.
        """
        # Primary: the Origin header (set by browsers on cross-origin requests)
        origin = request.headers.get("origin")
        if origin and origin.lower() != "null":
            return origin.strip().rstrip("/").lower()

        # Fallback: extract origin from the Referer header
        referer = request.headers.get("referer")
        if referer:
            parsed = urlparse(referer.strip())
            if parsed.scheme and parsed.netloc:
                port_suffix = ""
                if parsed.port and parsed.port not in (80, 443):
                    port_suffix = f":{parsed.port}"
                # Reconstruct origin as scheme://host[:port]
                host_no_port = parsed.hostname or parsed.netloc.split(":")[0]
                return f"{parsed.scheme}://{host_no_port}{port_suffix}".lower()

        return None

    async def dispatch(self, request: Request, call_next):
        # Safe methods are always allowed
        if request.method in _SAFE_METHODS:
            return await call_next(request)

        # Exempt paths (e.g., OAuth callback)
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Non-production environments: skip CSRF checks for local development
        from app.core.config import settings
        if settings.ENVIRONMENT.lower() != "production":
            return await call_next(request)

        # If no allowed origins are configured, we can't verify — reject
        if not self._allowed_origins:
            logger.warning(
                "CSRF middleware active but ALLOWED_ORIGINS is empty — "
                "rejecting mutating request to %s", request.url.path
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF check failed: no allowed origins configured."},
            )

        request_origin = self._extract_origin(request)

        # DELIBERATE SAFE DEFAULT: If both Origin and Referer are absent,
        # reject the request. Some privacy proxies/browsers strip these
        # headers, but allowing requests without origin verification would
        # silently defeat CSRF protection entirely.
        if request_origin is None:
            logger.warning(
                "CSRF rejected %s %s — no Origin or Referer header present",
                request.method, request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "CSRF check failed: Origin header is required for "
                              "state-changing requests. If you are using a privacy "
                              "proxy that strips Origin headers, please allowlist "
                              "this site.",
                },
            )

        # Check if the request origin matches any allowed origin
        if request_origin not in self._allowed_origins:
            logger.warning(
                "CSRF rejected %s %s — origin '%s' not in allowed list %s",
                request.method, request.url.path,
                request_origin, self._allowed_origins,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": f"CSRF check failed: origin '{request_origin}' is not allowed."},
            )

        return await call_next(request)
