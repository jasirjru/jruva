import logging
import sqlite3
import traceback
from typing import Optional

try:
    from asyncpg import PostgresError
except ImportError:  # pragma: no cover
    class PostgresError(Exception):  # type: ignore
        pass
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from google.auth.exceptions import GoogleAuthError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.google_identity import verify_google_identity
from app.schemas.auth import AuthResponse, GoogleAuthRequest, UserResponse
from app.services.auth_service import AccountDeletionPending, GoogleAccountConflict, auth_service
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _log_login_failure(stage: str, exc: Exception) -> None:
    """Log diagnostic locations without tokens, SQL parameters or exception text."""
    frames = traceback.extract_tb(exc.__traceback__)
    locations = ",".join(f"{frame.name}:{frame.lineno}" for frame in frames[-8:])
    logger.error("Google login failed stage=%s error_type=%s locations=%s", stage, type(exc).__name__, locations)


def _resolve_user(
    request: Request,
    authorization: Optional[str] = None,
    allow_pending: bool = False,
) -> Optional[dict]:
    """Extract and verify JWT from httpOnly secure cookie or Authorization header."""
    token = request.cookies.get("access_token")
    if not token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    if not token:
        return None
    try:
        claims = auth_service.verify_jwt_token(token)
        if claims and "sub" in claims:
            user = auth_service.get_user_by_id(claims["sub"])
            if user and (allow_pending or not user.get("deletion_pending")):
                return user
    except Exception as exc:
        logger.warning(f"[SilentFailureGuard] Optional auth extraction failed: {exc}", exc_info=True)
        telemetry_service.record_silent_failure("auth_service", str(exc))
    return None


def get_current_user_optional(request: Request, authorization: Optional[str] = Header(None)) -> Optional[dict]:
    return _resolve_user(request, authorization)


def get_erasure_user(request: Request, authorization: Optional[str] = Header(None)) -> dict:
    user = _resolve_user(request, authorization, allow_pending=True)
    if not user:
        raise HTTPException(401, "Sign in to delete your account.")
    return user


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> dict:
    """Require valid authenticated JWT user from httpOnly cookie or Authorization header."""
    user = get_current_user_optional(request, authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication token. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
def get_profile(current_user: dict = Depends(get_current_user)):
    """Fetch current user profile data from verified JWT."""
    return UserResponse(**current_user)


@router.delete(
    "/me",
    summary="Permanently delete user account and associated records",
)
def delete_account(response: Response, current_user: dict = Depends(get_erasure_user)):
    """Permanently delete user account and associated memories using real SQL DELETE."""
    user_id = current_user["id"]
    try:
        success = auth_service.delete_user(user_id)
    except Exception:
        logger.warning("Account erasure remains incomplete")
        raise HTTPException(503, "Account deletion is incomplete. Retry this request to finish deletion.")
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account could not be found or deleted.",
        )
    is_production = getattr(settings, "ENVIRONMENT", "development").lower() == "production"
    # NOTE: In production, SameSite=None + Secure=True is required for
    # cross-origin cookies (e.g., Vercel frontend → Railway backend).
    # Secure=True REQUIRES HTTPS — cookies will silently fail to set over HTTP.
    # Both Railway and Vercel serve HTTPS by default, so this is safe.
    response.delete_cookie(
        key="access_token", path="/", httponly=True,
        secure=is_production,
        samesite="none" if is_production else "lax",
    )
    return {"status": "success", "message": "Owned application records and document storage deleted. Backup and provider retention follow the privacy policy."}


@router.post(
    "/logout",
    summary="Invalidate client session / log out",
)
def logout(response: Response):
    """Client-side logout: clear httpOnly secure cookie."""
    is_production = getattr(settings, "ENVIRONMENT", "development").lower() == "production"
    response.delete_cookie(
        key="access_token", path="/", httponly=True,
        secure=is_production,
        samesite="none" if is_production else "lax",
    )
    return {"status": "success", "message": "Successfully logged out."}


@router.post(
    "/google",
    response_model=AuthResponse,
    summary="Authenticate or register user via Google OAuth 2.0 credential",
)
def google_auth(payload: GoogleAuthRequest, response: Response):
    """Verify Google OAuth credential, set httpOnly secure cookie, and return authenticated session."""
    try:
        data = verify_google_identity(payload.credential, settings.GOOGLE_CLIENT_ID)
    except HTTPException:
        raise
    except (ValueError, GoogleAuthError):
        logger.warning("Google login rejected stage=verification")
        raise HTTPException(400, "Failed to verify Google credentials. Please try again.") from None
    except Exception as exc:
        _log_login_failure("verification", exc)
        raise HTTPException(500, "Authentication could not be completed. Please try again later.") from None

    stage = "account"
    try:
        email = data.get("email")
        if not email or str(data.get("email_verified", "")).lower() != "true":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google account email is not verified or token is invalid.",
            )

        full_name = data.get("name") or email.split("@")[0]
        google_sub = data.get("sub", "")
        avatar_url = data.get("picture")

        result = auth_service.authenticate_or_register_google_user(
            email=email,
            full_name=full_name,
            google_sub=google_sub,
            avatar_url=avatar_url,
        )

        # Validate the complete response before issuing a session cookie.
        stage = "response"
        auth_response = AuthResponse(
            access_token=result["token"],
            token_type="bearer",
            user=UserResponse(**result["user"]),
            message="Google authentication successful.",
        )

        # Set httpOnly secure cookie.
        # Production: SameSite=None + Secure=True for cross-origin (Vercel → Railway).
        # Development: SameSite=Lax + Secure=False for localhost without HTTPS.
        # WARNING: Secure=True cookies ONLY work over HTTPS. If testing against
        # a non-HTTPS Railway URL or local prod-mode, auth cookies will silently
        # fail to set — this is the most common cause of "auth randomly broke."
        is_production = getattr(settings, "ENVIRONMENT", "development").lower() == "production"
        stage = "cookie"
        response.set_cookie(
            key="access_token",
            value=result["token"],
            httponly=True,
            secure=is_production,
            samesite="none" if is_production else "lax",
            max_age=7 * 24 * 3600,
            path="/",
        )

        return auth_response
    except GoogleAccountConflict:
        logger.warning("Google login rejected stage=account reason=identity_conflict")
        raise HTTPException(403, "This account requires account recovery before Google sign-in can be used. Contact support.") from None
    except AccountDeletionPending:
        logger.warning("Google login rejected stage=account reason=deletion_pending")
        raise HTTPException(403, "Account deletion is pending. Sign-in is unavailable.") from None
    except (SQLAlchemyError, sqlite3.Error, PostgresError, OSError, TimeoutError) as exc:
        _log_login_failure(stage, exc)
        raise HTTPException(
            503, "Authentication storage is temporarily unavailable. Please try again later.",
            headers={"Retry-After": "5"},
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        _log_login_failure(stage, exc)
        raise HTTPException(500, "Authentication could not be completed. Please try again later.") from None
