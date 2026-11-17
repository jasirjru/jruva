"""Authentication Schemas for User Registration and Login."""
from typing import Optional
from pydantic import BaseModel, Field




class UserResponse(BaseModel):
    """Public user profile data."""
    id: str
    username: str
    email: str
    full_name: str
    role: str
    created_at: Optional[str] = None


class AuthResponse(BaseModel):
    """Successful authentication payload returning JWT token and profile."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    message: str = "Authentication successful"


class GoogleAuthRequest(BaseModel):
    """Payload containing Google ID token / credential JWT."""
    credential: str = Field(..., description="Google OAuth 2.0 ID token / credential JWT")
