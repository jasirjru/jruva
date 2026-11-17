"""RFC 7807 Standard Error Schemas."""
from typing import Optional
from pydantic import BaseModel, Field


class ProblemDetails(BaseModel):
    """RFC 7807 Problem Details representation."""
    type: str = Field(..., examples=["https://api.chatbot.dev/errors/rate-limit-exceeded"])
    title: str = Field(..., examples=["Rate Limit Exceeded"])
    status: int = Field(..., examples=[429])
    detail: str = Field(..., examples=["Free tier threshold reached. Please retry in 15 seconds."])
    instance: Optional[str] = Field(None, examples=["/api/v1/chat"])
    code: Optional[str] = Field(None, examples=["RATE_LIMIT_EXCEEDED"])
