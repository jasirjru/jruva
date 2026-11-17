"""Chat Data Transfer Objects (Pydantic Models) adhering to OpenAPI 3.1.0 Contract."""
from datetime import datetime, timezone
from typing import List, Literal, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class SourceCitation(BaseModel):
    """Source context citation used for RAG grounding."""
    id: str = Field(..., description="Unique document / chunk identifier")
    title: str = Field(..., description="Source title or heading")
    url: Optional[str] = Field(None, description="Direct URL if available")
    snippet: Optional[str] = Field(None, description="Relevance excerpt")
    file_id: Optional[str] = Field(None, description="Associated document file ID")
    page: Optional[int] = Field(None, description="Document page number")
    score: Optional[float] = Field(None, description="Retrieval similarity or rerank score")
    source_type: Optional[str] = Field("document", description="Source classification: document, web, or domain")


class UsageMetadata(BaseModel):
    """Token consumption metadata."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AttachmentItem(BaseModel):
    """Multimodal attachment payload (image, document, audio)."""
    name: str = Field(..., description="File name with extension")
    mime_type: str = Field(..., description="MIME type, e.g. image/png, image/jpeg, application/pdf")
    data: str = Field(..., description="Base64-encoded data string without data URL prefix")


MAX_HISTORY_TURNS = 10
# Bounds per-request cost for guest history; the frontend sends at most 10 x 2000 chars.
MAX_HISTORY_TOTAL_CHARS = 20_000


class HistoryTurn(BaseModel):
    """One prior turn resent by a guest client; guests have no server-side history."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """Primary chat request model."""
    history: list[HistoryTurn] | None = Field(
        default=None,
        max_length=MAX_HISTORY_TURNS,
        description="Recent prior turns, used only for guests. Signed-in users always get server-stored history.",
    )

    @field_validator("history")
    @classmethod
    def _cap_history_size(cls, turns: list[HistoryTurn] | None) -> list[HistoryTurn] | None:
        if turns and sum(len(t.content) for t in turns) > MAX_HISTORY_TOTAL_CHARS:
            raise ValueError(f"history exceeds {MAX_HISTORY_TOTAL_CHARS} characters in total")
        return turns
    session_id: Optional[UUID] = Field(
        default=None,
        description="Existing conversation UUID. If omitted, a new UUID is generated."
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional turn/message tracking UUID. If omitted, session_id or generated UUID is used."
    )
    image_intent_id: UUID | None = Field(
        default=None,
        description="Owned image generation intent: reuse for retries; use a fresh UUID for a new image. Independent of server logging IDs.",
    )
    image_generation_allowed: bool = Field(
        default=True,
        description="Set false when replaying history without a recoverable image intent; text chat remains available.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Authenticated user ID for personal memory and session persistence."
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="User text prompt",
        examples=["Explain quantum computing in simple terms."]
    )
    attachments: Optional[List[AttachmentItem]] = Field(
        default=None,
        description="Optional multimodal attachments (images, documents)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model override (e.g. gemini-3.5-flash-lite)"
    )
    web_search: Optional[bool] = Field(
        default=None,
        description="Optional flag to force real-time web search grounding on or off"
    )
    stream: bool = Field(
        default=True,
        description="Flag indicating if token streaming (SSE) is requested"
    )
    system_instruction: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional system prompt directive"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    tier: Literal["B", "A", "S", "S++", "S+++"] = Field(
        default="S+++",
        description="Execution compute tier: Tier B (Standard), Tier A (Professional), Tier S (Dedicated Specialist), Tier S++ (Supreme Zero-Lag), Tier S+++ (Supreme Frontier)"
    )


class ModelInfoItem(BaseModel):
    """Information for a supported LLM in the catalog."""
    id: str
    name: str
    provider: str
    badge: str
    description: str
    icon: Optional[str] = None
    speed: Optional[str] = None
    recommended: bool = False
    available: bool = True
    hidden: Optional[bool] = False
    health: Optional[str] = "healthy"
    runtime: Optional[str] = "cloud"
    actual_backend_model: Optional[str] = None
    fallback_capable: Optional[bool] = True


class ModelListResponse(BaseModel):
    """Response containing list of available LLM options."""
    models: List[ModelInfoItem]
    default_provider: str
    active_count: int


class ChatResponse(BaseModel):
    """Synchronous (non-streaming) chat response."""
    session_id: UUID
    message_id: UUID = Field(default_factory=uuid4)
    conversation_id: Optional[str] = Field(default=None, description="Unique log ID for feedback correlation")
    role: Literal["assistant"] = "assistant"
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None
    tier: Optional[str] = "B"
    cached: Optional[bool] = False
    failover: Optional[bool] = False
    requested_model: Optional[str] = None
    requested_model_label: Optional[str] = None
    actual_model: Optional[str] = None
    actual_provider: Optional[str] = None
    fallback_used: Optional[bool] = False
    fallback_reason: Optional[str] = None
    sources: List[SourceCitation] = Field(default_factory=list)
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatStreamEvent(BaseModel):
    """Server-Sent Event (SSE) payload format for token streaming."""
    type: Literal["token", "citation", "finish", "error", "metadata", "status"]
    session_id: Optional[UUID] = None
    conversation_id: Optional[str] = None
    content: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tier: Optional[str] = None
    cached: Optional[bool] = None
    failover: Optional[bool] = None
    requested_model: Optional[str] = None
    requested_model_label: Optional[str] = None
    actual_model: Optional[str] = None
    actual_provider: Optional[str] = None
    fallback_used: Optional[bool] = None
    fallback_reason: Optional[str] = None
    sources: Optional[List[SourceCitation]] = None
    finish_reason: Literal["stop", "length", "safety", "error", "capacity"] | None = None
    total_tokens: Optional[int] = None


class FeedbackRequest(BaseModel):
    """Payload for submitting feedback on a conversation log."""
    conversation_id: str = Field(..., description="The conversation/turn UUID to rate")
    signal: Literal["up", "down", "regenerated"] = Field(..., description="Feedback signal")

