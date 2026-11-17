"""Cryptographic Google identity verification with an explicit application audience."""

import os
import re

from fastapi import HTTPException
from google.auth.exceptions import TransportError
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

CLIENT_PATTERN = re.compile(r"[A-Za-z0-9_-]+\.apps\.googleusercontent\.com\Z")


def validate_production_auth(settings) -> None:
    if settings.ENVIRONMENT.lower() != "production":
        return
    if not CLIENT_PATTERN.fullmatch(settings.GOOGLE_CLIENT_ID or ""):
        raise RuntimeError("Production requires a valid GOOGLE_CLIENT_ID application audience.")
    if len(os.environ.get("JWT_SECRET_KEY", "").strip()) < 32:
        raise RuntimeError("Production requires JWT_SECRET_KEY of at least 32 characters.")


def verify_google_identity(credential: str, audience: str) -> dict:
    if not CLIENT_PATTERN.fullmatch(audience or ""):
        raise HTTPException(503, "Authentication is not configured. Please try again later.")
    transport = google_requests.Request()

    def bounded_request(url, method="GET", **kwargs):
        kwargs.pop("timeout", None)
        return transport(url=url, method=method, timeout=5, **kwargs)

    # Google-auth verifies the RSA signature, audience, iat/expiry and Google issuer.
    try:
        data = id_token.verify_oauth2_token(credential, bounded_request, audience=audience)
    except TransportError:
        raise HTTPException(503, "Authentication verification is temporarily unavailable.") from None
    if not isinstance(data.get("sub"), str) or not data["sub"] or len(data["sub"]) > 255:
        raise ValueError("Missing stable Google subject")
    if not data.get("email") or str(data.get("email_verified", "")).lower() != "true":
        raise ValueError("Unverified account email")
    return data
