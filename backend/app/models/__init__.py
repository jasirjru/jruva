"""Relational Database ORM Models for JRUVA.

Maps all 11 audited durable subsystems:
1. User (users table)
2. Session (sessions table)
3. Message (messages table)
4. ConversationLog (conversation_logs table)
5. UserMemory (user_memories table)
6. Project (projects table)
7. LibraryFile (library_files table)
8. ImageRecord (image_records table)
9. ScheduledTask (scheduled_tasks table)
10. TaskExecution (task_executions table)
11. UserIntegration (user_integrations table)
"""
from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from app.core.database import Base


class User(Base):
    """User account entity with PBKDF2 hash and Google OAuth support."""
    __tablename__ = "users"

    id = Column(String(128), primary_key=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    role = Column(String(64), default="student")
    google_sub = Column(String(255), nullable=True, unique=True)
    deletion_pending = Column(Boolean, nullable=False, default=False, server_default="false")
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())
    last_login = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())


class Session(Base):
    """Conversational session metadata.
    
    CRITICAL AUDIT CONSTRAINT: user_id is deliberately unconstrained by a Foreign Key
    to support anonymous device IDs ('anon_*') and legacy unauthenticated sessions (NULL).
    """
    __tablename__ = "sessions"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    session_id = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(String(128), nullable=True, index=True)
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("idx_sessions_user_updated", "user_id", "updated_at"),
    )


class Message(Base):
    """Individual conversational turns (user, assistant, system)."""
    __tablename__ = "messages"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    __table_args__ = (
        Index("idx_messages_session_id", "session_id", "id"),
    )


class ConversationLog(Base):
    """Interaction diagnostic logs, model telemetry, shadow comparisons, and feedback."""
    __tablename__ = "conversation_logs"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    user_query = Column(Text, nullable=False)
    retrieved_chunks = Column(Text, nullable=False, default="[]")
    provider_used = Column(String(64), nullable=False)
    model_response = Column(Text, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    feedback = Column(String(32), nullable=True)
    shadow_response = Column(Text, nullable=True)
    shadow_model_id = Column(String(128), nullable=True)
    served_model_version = Column(String(64), nullable=True)
    ab_test_group = Column(String(64), nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)


class UserMemory(Base):
    """Long-term personal memory cards.
    
    user_id supports both registered user accounts and anonymous device identifiers.
    """
    __tablename__ = "user_memories"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True)
    category = Column(String(64), nullable=False, default="general", index=True)
    fact = Column(Text, nullable=False)
    source = Column(String(32), default="auto")
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("idx_memories_user_cat", "user_id", "category"),
    )


class Project(Base):
    """Workspace projects container."""
    __tablename__ = "projects"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), default="guest", index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)
    preferred_model = Column(String(128), default="auto")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class LibraryFile(Base):
    """Knowledge library uploaded document metadata."""
    __tablename__ = "library_files"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), default="guest", index=True)
    project_id = Column(String(128), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(64), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    storage_path = Column(Text, nullable=False)
    status = Column(String(32), default="indexed")
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())


class ImageRecord(Base):
    """Synthesized image records and prompt history."""
    __tablename__ = "image_records"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), default="guest", index=True)
    prompt = Column(Text, nullable=False)
    enhanced_prompt = Column(Text, nullable=True)
    model = Column(String(64), nullable=False)
    tier = Column(String(16), default="B")
    style = Column(String(64), default="photorealistic")
    width = Column(Integer, default=1024)
    height = Column(Integer, default=1024)
    file_url = Column(Text, nullable=False)
    bytes = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now(), index=True)


class ImageStorageQuota(Base):
    """Retained storage quota tracking per user and project-wide."""
    __tablename__ = "image_storage_quotas"

    user_id = Column(String(128), primary_key=True)
    image_count = Column(Integer, default=0, nullable=False)
    total_bytes = Column(BigInteger, default=0, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())


class ImageStorageReservation(Base):
    """Durable receipt connecting reserved capacity to a possible object and image."""
    __tablename__ = "image_storage_reservations"
    reservation_id = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False)
    bytes = Column(BigInteger, nullable=False)
    state = Column(String(32), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    object_path = Column(Text)
    image_id = Column(String(128), unique=True)
    operation_id = Column(String(128))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        CheckConstraint("bytes>0", name="ck_storage_reservation_bytes"),
        CheckConstraint("state IN ('active','writing','cleanup','committed','released','abandoned')", name="ck_storage_reservation_state"),
        Index("ix_storage_reservation_owner_state", "user_id", "state"),
    )


class ImageIdempotency(Base):
    """Fenced generation ownership and external dispatch uncertainty."""
    __tablename__ = "image_idempotency"
    owner_id = Column(String(128), primary_key=True)
    idempotency_key = Column(String(128), primary_key=True)
    request_hash = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)
    response_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False)
    worker_id = Column(String(128))
    lease_expires_at = Column(DateTime(timezone=True))
    fencing_token = Column(BigInteger, nullable=False, server_default="1")
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    operation_id = Column(String(128))
    external_state = Column(String(32), nullable=False, server_default="unstarted")
    provider_model = Column(String(128))
    provider_attempts = Column(Integer, nullable=False, server_default="0")


class ScheduledTask(Base):
    """Scheduled automated jobs."""
    __tablename__ = "scheduled_tasks"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), default="guest", index=True)
    name = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    cron_expression = Column(String(64), default="0 8 * * *")
    timezone = Column(String(64), default="UTC")
    status = Column(String(32), default="active")
    last_run = Column(DateTime(timezone=True), nullable=True)
    next_run = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class TaskExecution(Base):
    """Execution history of scheduled tasks.
    
    CRITICAL AUDIT CONSTRAINT: task_id is NOT a rigid Foreign Key to allow historical
    orphaned records from testing without failing integrity checks.
    """
    __tablename__ = "task_executions"

    id = Column(String(128), primary_key=True)
    task_id = Column(String(128), nullable=False, index=True)
    status = Column(String(32), nullable=False)
    output = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    executed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now(), index=True)


class UserIntegration(Base):
    """User-level tool integration and plugin enable/disable state."""
    __tablename__ = "user_integrations"

    id = Column(String(128), primary_key=True)
    user_id = Column(String(128), default="guest")
    plugin_id = Column(String(128), nullable=False)
    is_enabled = Column(Integer, default=1)
    config_encrypted = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "plugin_id", name="uq_user_plugin"),
    )
