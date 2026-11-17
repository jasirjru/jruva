"""Pydantic schemas for User Memory Card management."""
from typing import List, Optional
from pydantic import BaseModel, Field


class UserMemoryItem(BaseModel):
    """Single fact stored in user's Memory Card."""
    id: str
    userId: str
    category: str = Field(..., description="Category: academic, goal, preference, interest, general")
    fact: str = Field(..., description="The remembered fact string")
    source: str = Field(default="auto", description="Source: auto or manual")
    confidence: float = Field(default=1.0, description="Confidence score 0.0 to 1.0")
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class MemoryListResponse(BaseModel):
    """List of all memories for the authenticated user."""
    memories: List[UserMemoryItem]
    count: int
    user_id: str


class MemoryCreateRequest(BaseModel):
    """Payload to add a manual memory to the card."""
    fact: str = Field(..., min_length=2, max_length=200, description="The memory content to store")
    category: str = Field(default="general", description="Category: academic, goal, preference, interest, general")


class MemoryUpdateRequest(BaseModel):
    """Payload to update an existing memory."""
    fact: str = Field(..., min_length=2, max_length=200)
    category: Optional[str] = Field(default=None)
