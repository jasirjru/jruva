"""Server-side authorization shared by every operational route."""

from typing import Annotated

from fastapi import Depends, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.api.v1.endpoints.auth import get_current_user_optional
from app.core import config


class OperationsRoute(APIRoute):
    """Bound validation errors without echoing submitted diagnostic filter values."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request):
            try:
                return await original(request)
            except RequestValidationError:
                return JSONResponse(
                    status_code=422,
                    content={"detail": "Invalid operational query."},
                    headers={"Cache-Control": "no-store"},
                )

        return handle


def require_operations_admin(
    response: Response,
    current_user: Annotated[dict | None, Depends(get_current_user_optional)],
) -> dict:
    """Trust the account repository's role, never JWT roles or caller headers."""
    response.headers["Cache-Control"] = "no-store"
    if current_user is None:
        raise HTTPException(
            401,
            "Missing or invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
        )
    if not current_user.get("id") or current_user.get("role") != "admin":
        raise HTTPException(
            403, "Administrator access required.", headers={"Cache-Control": "no-store"}
        )
    return current_user


def require_operations_reset(
    current_user: Annotated[dict, Depends(require_operations_admin)],
) -> dict:
    """Reset is opt-in and only allowed in explicitly named local environments."""
    settings = config.settings
    if (
        settings.ENVIRONMENT.strip().lower() not in {"development", "test"}
        or not settings.OPERATIONS_RESET_ENABLED
    ):
        raise HTTPException(
            403, "Operational reset is disabled.", headers={"Cache-Control": "no-store"}
        )
    return current_user
