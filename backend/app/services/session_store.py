"""Persistent Conversation & Memory Store for JRUVA.

Supports dual persistence:
1. PostgreSQL via SQLAlchemy 2.0 AsyncSession + asyncpg (Production target)
   - Multi-writer MVCC with connection pooling
   - Atomic transactions for session deletion and user migration
2. SQLite embedded WAL via sqlite3 standard library (Local dev / testing fallback)
   - Preserves compatibility with existing test suites and offline workflows
"""
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, desc, select, update

from app.core.config import settings
from app.core.database import is_postgres_configured, get_async_session, run_sync_db
from app.models import Message, Session, User
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chatbot.db")


class SessionStore:
    """Async-safe session repository for conversational turns, session metadata, and user memory."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    async def _async_private_write(self, session_id, user_id, title=None, role=None, content=None):
        async with get_async_session() as db:
            async with db.begin():
                user = await db.scalar(select(User).where(User.id == user_id).with_for_update())
                if not user or user.deletion_pending:
                    return
                record = await db.scalar(select(Session).where(Session.session_id == str(session_id)).with_for_update())
                if record and record.user_id != user_id:
                    raise PermissionError("Session ownership mismatch")
                if not record:
                    db.add(Session(session_id=str(session_id), user_id=user_id, title=title or "New Conversation"))
                if role and content:
                    db.add(Message(session_id=str(session_id), role=role, content=content))

    def _private_write(self, session_id, user_id, title=None, role=None, content=None):
        if not user_id:
            return
        if is_postgres_configured():
            return run_sync_db(self._async_private_write, session_id, user_id, title, role, content)
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            user = conn.execute("SELECT deletion_pending FROM users WHERE id = ?", (user_id,)).fetchone()
            if not user or user["deletion_pending"]:
                return
            record = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (str(session_id),)).fetchone()
            if record and record["user_id"] != user_id:
                raise PermissionError("Session ownership mismatch")
            now = datetime.now(timezone.utc).isoformat()
            if not record:
                conn.execute("INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                             (str(session_id), user_id, title or "New Conversation", now, now))
            if role and content:
                conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (str(session_id), role, content))

    def ensure_private_session(self, session_id, title=None, user_id=None):
        return self._private_write(session_id, user_id, title=title)

    def record_private_turn(self, session_id, role, content, user_id):
        return self._private_write(session_id, user_id, role=role, content=content)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_db(self) -> None:
        """Initialize database tables for SQLite fallback mode."""
        if is_postgres_configured():
            return

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL UNIQUE,
                        user_id TEXT DEFAULT NULL,
                        title TEXT NOT NULL DEFAULT 'New Conversation',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                try:
                    conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT NULL")
                except sqlite3.OperationalError:
                    pass
                except Exception as alter_err:
                    logger.warning(
                        f"[SilentFailureGuard] Column user_id alter failed unexpectedly: {alter_err}",
                        exc_info=True,
                    )
                    telemetry_service.record_silent_failure("session_store", str(alter_err))

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions (updated_at DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id, updated_at DESC)
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE,
                        email TEXT UNIQUE,
                        full_name TEXT,
                        role TEXT NOT NULL DEFAULT 'student',
                        password_hash TEXT,
                        salt TEXT,
                        hashed_password TEXT,
                        google_sub TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        deletion_pending BOOLEAN DEFAULT 0
                    )
                """)
                conn.commit()
            logger.info(f"Initialized persistent session database at {self.db_path}")
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Failed to initialize SQLite session database: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))

    # ─── Async PostgreSQL Operations ───────────────────────────────────────────

    async def async_list_sessions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return sessions ordered by most recently updated scoped to user_id (PostgreSQL)."""
        if not user_id:
            return []
        async with get_async_session() as session:
            stmt = (
                select(Session)
                .where(Session.user_id == user_id)
                .order_by(desc(Session.updated_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.session_id,
                    "userId": r.user_id,
                    "title": r.title,
                    "createdAt": str(r.created_at) if r.created_at else "",
                    "updatedAt": str(r.updated_at) if r.updated_at else "",
                }
                for r in rows
            ]

    async def async_get_session(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific session (PostgreSQL)."""
        sess_str = str(session_id)
        async with get_async_session() as session:
            stmt = select(Session).where(Session.session_id == sess_str)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                return {
                    "id": row.session_id,
                    "userId": row.user_id,
                    "title": row.title,
                    "createdAt": str(row.created_at) if row.created_at else "",
                    "updatedAt": str(row.updated_at) if row.updated_at else "",
                }
            return None

    async def async_ensure_session(
        self, session_id: UUID, title: Optional[str] = None, user_id: Optional[str] = None
    ) -> None:
        """Create session metadata if it doesn't exist, or update timestamp (PostgreSQL)."""
        sess_str = str(session_id)
        async with get_async_session() as session:
            stmt = select(Session).where(Session.session_id == sess_str)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            now = datetime.now(timezone.utc)
            if existing:
                if user_id and not existing.user_id:
                    existing.user_id = user_id
                existing.updated_at = now
            else:
                new_session = Session(
                    session_id=sess_str,
                    user_id=user_id,
                    title=title or "New Conversation",
                    created_at=now,
                    updated_at=now,
                )
                session.add(new_session)

    async def async_rename_session(self, session_id: UUID, title: str) -> bool:
        """Rename a session's title (PostgreSQL)."""
        sess_str = str(session_id)
        now = datetime.now(timezone.utc)
        async with get_async_session() as session:
            stmt = (
                update(Session)
                .where(Session.session_id == sess_str)
                .values(title=title.strip(), updated_at=now)
            )
            result = await session.execute(stmt)
            return result.rowcount > 0

    async def async_delete_session(self, session_id: UUID, user_id: Optional[str] = None) -> bool:
        """Delete a session and all its messages in a single atomic transaction (PostgreSQL)."""
        sess_str = str(session_id)
        async with get_async_session() as session:
            stmt = select(Session).where(Session.session_id == sess_str)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if not existing:
                return False
            if user_id and existing.user_id and existing.user_id != user_id:
                return False

            # Delete messages first, then session
            await session.execute(delete(Message).where(Message.session_id == sess_str))
            res = await session.execute(delete(Session).where(Session.session_id == sess_str))
            return res.rowcount > 0

    async def async_migrate_anonymous_sessions(self, anon_user_id: str, authenticated_user_id: str) -> int:
        """Transfer sessions owned by an anonymous client ID to a registered user account (PostgreSQL)."""
        if not anon_user_id or not authenticated_user_id or anon_user_id == authenticated_user_id:
            return 0
        async with get_async_session() as session:
            stmt = (
                update(Session)
                .where(Session.user_id == anon_user_id)
                .values(user_id=authenticated_user_id)
            )
            result = await session.execute(stmt)
            return result.rowcount

    async def async_get_session_history(self, session_id: UUID, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve recent turns chronologically for LLM context (PostgreSQL)."""
        sess_str = str(session_id)
        async with get_async_session() as session:
            # Query recent limit turns ordered by ID desc, then reverse to chronological
            stmt = (
                select(Message.role, Message.content)
                .where(Message.session_id == sess_str)
                .order_by(desc(Message.id))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.all()
            return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    async def async_get_full_history(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Retrieve ALL messages for session UI restoration (PostgreSQL)."""
        sess_str = str(session_id)
        async with get_async_session() as session:
            stmt = (
                select(Message.role, Message.content, Message.created_at)
                .where(Message.session_id == sess_str)
                .order_by(Message.id.asc())
            )
            result = await session.execute(stmt)
            rows = result.all()
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "createdAt": str(r.created_at) if r.created_at else "",
                }
                for r in rows
            ]

    async def async_record_turn(self, session_id: UUID, role: str, content: str) -> None:
        """Store a conversational turn permanently in PostgreSQL."""
        sess_str = str(session_id)
        content_clean = content.strip()
        if not content_clean:
            return
        async with get_async_session() as session:
            msg = Message(
                session_id=sess_str,
                role=role,
                content=content_clean,
                created_at=datetime.now(timezone.utc),
            )
            session.add(msg)

    async def async_clear_session(self, session_id: UUID) -> None:
        """Clear message history for session in PostgreSQL."""
        sess_str = str(session_id)
        async with get_async_session() as session:
            await session.execute(delete(Message).where(Message.session_id == sess_str))

    # ─── Dual-Backend Synchronous Public Contracts ─────────────────────────────

    def list_sessions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return sessions ordered by most recently updated. Scoped strictly to user_id."""
        if not user_id:
            return []
        if is_postgres_configured():
            return run_sync_db(self.async_list_sessions, user_id, limit)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, user_id, title, created_at, updated_at
                    FROM sessions
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": row["session_id"],
                        "userId": row["user_id"],
                        "title": row["title"],
                        "createdAt": row["created_at"],
                        "updatedAt": row["updated_at"],
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error listing sessions: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))
            return []

    def get_session(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific session."""
        if is_postgres_configured():
            return run_sync_db(self.async_get_session, session_id)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT session_id, user_id, title, created_at, updated_at FROM sessions WHERE session_id = ?",
                    (sess_str,),
                ).fetchone()
                if row:
                    return {
                        "id": row["session_id"],
                        "userId": row["user_id"],
                        "title": row["title"],
                        "createdAt": row["created_at"],
                        "updatedAt": row["updated_at"],
                    }
                return None
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error getting session {sess_str}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))
            return None

    def ensure_session(self, session_id: UUID, title: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """Create session metadata if it doesn't exist, or update timestamp and link user_id."""
        if is_postgres_configured():
            return run_sync_db(self.async_ensure_session, session_id, title, user_id)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                existing = conn.execute(
                    "SELECT id, user_id FROM sessions WHERE session_id = ?", (sess_str,)
                ).fetchone()

                now = datetime.now(timezone.utc).isoformat()

                if existing:
                    if user_id and not existing["user_id"]:
                        conn.execute(
                            "UPDATE sessions SET updated_at = ?, user_id = ? WHERE session_id = ?",
                            (now, user_id, sess_str),
                        )
                    else:
                        conn.execute(
                            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                            (now, sess_str),
                        )
                else:
                    conn.execute(
                        "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (sess_str, user_id, title or "New Conversation", now, now),
                    )
                conn.commit()
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error ensuring session {sess_str}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))

    def rename_session(self, session_id: UUID, title: str) -> bool:
        """Rename a session's title."""
        if is_postgres_configured():
            return run_sync_db(self.async_rename_session, session_id, title)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title.strip(), datetime.now(timezone.utc).isoformat(), sess_str),
                )
                conn.commit()
                return result.rowcount > 0
        except Exception as exc:
            logger.error(f"Error renaming session {sess_str}: {exc}")
            return False

    def delete_session(self, session_id: UUID, user_id: Optional[str] = None) -> bool:
        """Delete a session and all its messages permanently, checking user ownership if provided."""
        if is_postgres_configured():
            return run_sync_db(self.async_delete_session, session_id, user_id)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT user_id FROM sessions WHERE session_id = ?", (sess_str,)
                ).fetchone()
                if not row:
                    return False
                if user_id and row["user_id"] and row["user_id"] != user_id:
                    return False

                conn.execute("DELETE FROM messages WHERE session_id = ?", (sess_str,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (sess_str,))
                conn.commit()
                return True
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error deleting session {sess_str}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))
            return False

    def migrate_anonymous_sessions(self, anon_user_id: str, authenticated_user_id: str) -> int:
        """Transfer sessions owned by an anonymous client ID to a registered user account."""
        if not anon_user_id or not authenticated_user_id or anon_user_id == authenticated_user_id:
            return 0
        if is_postgres_configured():
            return run_sync_db(self.async_migrate_anonymous_sessions, anon_user_id, authenticated_user_id)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sessions SET user_id = ? WHERE user_id = ?",
                    (authenticated_user_id, anon_user_id),
                )
                conn.commit()
                return cursor.rowcount
        except Exception as exc:
            logger.error(
                f"[SilentFailureGuard] Error migrating sessions from {anon_user_id} to {authenticated_user_id}: {exc}",
                exc_info=True,
            )
            telemetry_service.record_silent_failure("session_store", str(exc))
            return 0

    def get_session_history(self, session_id: UUID, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve recent conversation turns for a session in chronological order."""
        if is_postgres_configured():
            return run_sync_db(self.async_get_session_history, session_id, limit)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content FROM (
                        SELECT id, role, content FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (sess_str, limit),
                )
                rows = cursor.fetchall()
                return [{"role": row["role"], "content": row["content"]} for row in rows]
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error fetching session history for {sess_str}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))
            return []

    def get_full_history(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Retrieve ALL messages for a session (for frontend session restore)."""
        if is_postgres_configured():
            return run_sync_db(self.async_get_full_history, session_id)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content, created_at FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (sess_str,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "createdAt": row["created_at"],
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error fetching full history for {sess_str}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))
            return []

    def record_turn(self, session_id: UUID, role: str, content: str) -> None:
        """Store a conversation message turn permanently."""
        content_clean = content.strip()
        if not content_clean:
            return

        if is_postgres_configured():
            return run_sync_db(self.async_record_turn, session_id, role, content_clean)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (sess_str, role, content_clean),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error recording turn for session {sess_str}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("session_store", str(exc))

    append_turn = record_turn

    def clear_session(self, session_id: UUID) -> None:
        """Clear all message history for a specific session."""
        if is_postgres_configured():
            return run_sync_db(self.async_clear_session, session_id)

        sess_str = str(session_id)
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sess_str,))
                conn.commit()
        except Exception as exc:
            logger.error(f"Error clearing session {sess_str}: {exc}")

    def generate_title_from_message(self, message: str) -> str:
        """Generate a short conversation title from the first user message."""
        clean = message.strip()
        if not clean:
            return "New Conversation"
        if len(clean) <= 40:
            return clean
        truncated = clean[:40]
        last_space = truncated.rfind(" ")
        if last_space > 20:
            return truncated[:last_space] + "..."
        return truncated + "..."


session_store = SessionStore()
