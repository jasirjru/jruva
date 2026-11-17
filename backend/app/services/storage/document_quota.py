"""Durable concurrency-safe document storage quotas and reservation ledger.

Provides PostgreSQL and SQLite backed accounting to enforce per-user byte limits,
per-user file counts, and project-wide storage capacity ceilings without relying
on vulnerable in-memory dictionaries.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.services.workspace_store import workspace_store
from sqlalchemy import text

logger = logging.getLogger("chatbot.document_quota")

DOCUMENT_QUOTA_DDL = (
    "CREATE TABLE IF NOT EXISTS document_storage_quotas ("
    "user_id VARCHAR(128) PRIMARY KEY, "
    "file_count INTEGER NOT NULL DEFAULT 0, "
    "total_bytes BIGINT NOT NULL DEFAULT 0, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ")"
)

DOCUMENT_RESERVATION_DDL = (
    "CREATE TABLE IF NOT EXISTS document_storage_reservations ("
    "reservation_id VARCHAR(128) PRIMARY KEY, "
    "user_id VARCHAR(128) NOT NULL, "
    "bytes BIGINT NOT NULL CHECK(bytes > 0), "
    "state VARCHAR(32) NOT NULL CHECK(state IN ('active','writing','cleanup','committed','released','abandoned')), "
    "expires_at TIMESTAMP NOT NULL, "
    "storage_path TEXT, "
    "file_id VARCHAR(128) UNIQUE, "
    "filename TEXT, "
    "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
    ")"
)

_CURRENT_DOC_RESERVATION: ContextVar[Optional[str]] = ContextVar("doc_storage_reservation", default=None)


class DocumentQuotaError(Exception):
    """Raised when an upload violates durable storage quota constraints."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _get_document_limits(
    max_user_count: Optional[int] = None,
    max_user_bytes: Optional[int] = None,
    max_project_bytes: Optional[int] = None,
    max_object_bytes: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    return (
        max_user_count if max_user_count is not None else settings.DOCUMENT_MAX_USER_COUNT,
        max_user_bytes if max_user_bytes is not None else settings.DOCUMENT_MAX_USER_BYTES,
        max_project_bytes if max_project_bytes is not None else settings.DOCUMENT_MAX_PROJECT_BYTES,
        max_object_bytes if max_object_bytes is not None else settings.DOCUMENT_MAX_OBJECT_BYTES,
    )


def ensure_sqlite_document_quota_schema(conn) -> None:
    conn.execute(DOCUMENT_QUOTA_DDL)
    conn.execute(DOCUMENT_RESERVATION_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_storage_reservation_owner_state "
        "ON document_storage_reservations(user_id, state)"
    )


class _SQLiteDocSession:
    def __init__(self, conn):
        self.conn = conn

    async def execute(self, statement, params=None):
        sql = (
            str(statement)
            .replace("expires_at<=:now", "julianday(expires_at)<=julianday(:now)")
            .replace("expires_at>:now", "julianday(expires_at)>julianday(:now)")
        )
        values = {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in (params or {}).items()
        }
        return self.conn.execute(sql, values)


@asynccontextmanager
async def _doc_quota_transaction():
    if is_postgres_configured():
        async with get_async_session() as session, session.begin():
            yield session
    else:
        conn = workspace_store._get_sqlite_connection()
        try:
            ensure_sqlite_document_quota_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            yield _SQLiteDocSession(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


async def lock_document_project(session) -> None:
    """Serialize all quota/reservation writers on __project__ row."""
    await session.execute(
        text(
            "INSERT INTO document_storage_quotas(user_id, file_count, total_bytes) "
            "VALUES ('__project__', 0, 0) ON CONFLICT(user_id) DO NOTHING"
        )
    )
    suffix = "" if isinstance(session, _SQLiteDocSession) else " FOR UPDATE"
    await session.execute(
        text("SELECT total_bytes FROM document_storage_quotas WHERE user_id='__project__'" + suffix)
    )


async def async_reserve_document_upload(
    user_id: str,
    file_bytes: int,
    filename: str,
    file_id: Optional[str] = None,
    lease_seconds: int = 300,
) -> Dict[str, Any]:
    """Atomically check capacity ceilings and reserve bytes before storing a document blob."""
    max_user_count, max_user_bytes, max_project_bytes, max_object_bytes = _get_document_limits()

    if file_bytes <= 0:
        raise DocumentQuotaError("invalid_size", "Document cannot be empty.")
    if file_bytes > max_object_bytes:
        raise DocumentQuotaError(
            "object_size_exceeded",
            f"Document size ({file_bytes} bytes) exceeds maximum single file limit ({max_object_bytes} bytes).",
            {"file_bytes": file_bytes, "max_object_bytes": max_object_bytes},
        )

    reservation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=lease_seconds)
    fid = str(uuid.UUID(file_id)) if file_id else str(uuid.uuid4())

    async with _doc_quota_transaction() as session:
        await lock_document_project(session)

        # 1. Expire stale uncommitted reservations and release their hold
        await session.execute(
            text(
                "UPDATE document_storage_reservations SET state='abandoned' "
                "WHERE state IN ('active','writing') AND expires_at<=:now"
            ),
            {"now": now},
        )

        # 2. Query project-wide consumption
        proj_res = await session.execute(
            text("SELECT total_bytes FROM document_storage_quotas WHERE user_id='__project__'")
        )
        row = proj_res.fetchone()
        project_committed_bytes = int(row[0]) if row and row[0] is not None else 0

        # Sum active reservations
        proj_res_active = await session.execute(
            text(
                "SELECT COALESCE(SUM(bytes), 0) FROM document_storage_reservations "
                "WHERE state IN ('active','writing') AND expires_at>:now"
            ),
            {"now": now},
        )
        row_act = proj_res_active.fetchone()
        project_reserved_bytes = int(row_act[0]) if row_act and row_act[0] is not None else 0
        total_project_bytes = project_committed_bytes + project_reserved_bytes

        if total_project_bytes + file_bytes > max_project_bytes:
            raise DocumentQuotaError(
                "project_storage_quota_exceeded",
                "Service storage capacity hard ceiling reached. Uploads are temporarily paused.",
                {
                    "project_bytes": total_project_bytes,
                    "requested_bytes": file_bytes,
                    "max_project_bytes": max_project_bytes,
                },
            )

        # 3. Query user consumption
        user_res = await session.execute(
            text(
                "SELECT file_count, total_bytes FROM document_storage_quotas WHERE user_id=:user_id"
            ),
            {"user_id": user_id},
        )
        row_user = user_res.fetchone()
        user_committed_count = int(row_user[0]) if row_user and row_user[0] is not None else 0
        user_committed_bytes = int(row_user[1]) if row_user and row_user[1] is not None else 0

        user_res_active = await session.execute(
            text(
                "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM document_storage_reservations "
                "WHERE user_id=:user_id AND state IN ('active','writing') AND expires_at>:now"
            ),
            {"user_id": user_id, "now": now},
        )
        row_uact = user_res_active.fetchone()
        user_reserved_count = int(row_uact[0]) if row_uact and row_uact[0] is not None else 0
        user_reserved_bytes = int(row_uact[1]) if row_uact and row_uact[1] is not None else 0

        total_user_count = user_committed_count + user_reserved_count
        total_user_bytes = user_committed_bytes + user_reserved_bytes

        if total_user_count + 1 > max_user_count:
            raise DocumentQuotaError(
                "user_file_count_exceeded",
                f"You have reached the maximum allowed document count ({max_user_count}). Delete unused documents to upload more.",
                {"current_count": total_user_count, "max_user_count": max_user_count},
            )

        if total_user_bytes + file_bytes > max_user_bytes:
            raise DocumentQuotaError(
                "user_storage_quota_exceeded",
                f"Document exceeds your available personal storage quota ({max_user_bytes // (1024 * 1024)} MB).",
                {
                    "current_user_bytes": total_user_bytes,
                    "requested_bytes": file_bytes,
                    "max_user_bytes": max_user_bytes,
                },
            )

        # 4. Record the durable reservation
        await session.execute(
            text(
                "INSERT INTO document_storage_reservations ("
                "reservation_id, user_id, bytes, state, expires_at, file_id, filename"
                ") VALUES (:res_id, :uid, :b, 'active', :exp, :fid, :fn)"
            ),
            {
                "res_id": reservation_id,
                "uid": user_id,
                "b": file_bytes,
                "exp": expires_at,
                "fid": fid,
                "fn": filename,
            },
        )

    _CURRENT_DOC_RESERVATION.set(reservation_id)
    return {
        "reservation_id": reservation_id,
        "file_id": fid,
        "bytes": file_bytes,
        "expires_at": expires_at.isoformat(),
        "user_id": user_id,
    }


async def async_commit_document_upload(
    reservation_id: str,
    storage_path: str,
) -> None:
    """Atomically commit a verified document upload: changes state to 'committed' and updates durable quotas."""
    now = datetime.now(timezone.utc)
    async with _doc_quota_transaction() as session:
        await lock_document_project(session)

        res = await session.execute(
            text(
                "SELECT user_id, bytes, state FROM document_storage_reservations "
                "WHERE reservation_id=:res_id"
            ),
            {"res_id": reservation_id},
        )
        row = res.fetchone()
        if not row:
            raise RuntimeError(f"Storage reservation {reservation_id} not found.")

        user_id, file_bytes, state = str(row[0]), int(row[1]), str(row[2])
        if state == "committed":
            return  # Idempotent success

        if state not in ("active", "writing"):
            raise RuntimeError(f"Cannot commit reservation {reservation_id} in state '{state}'.")

        # Mark committed
        await session.execute(
            text(
                "UPDATE document_storage_reservations "
                "SET state='committed', storage_path=:sp WHERE reservation_id=:res_id"
            ),
            {"sp": storage_path, "res_id": reservation_id},
        )

        # Increment user quota
        await session.execute(
            text(
                "INSERT INTO document_storage_quotas(user_id, file_count, total_bytes, updated_at) "
                "VALUES (:uid, 1, :b, :now) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "file_count = document_storage_quotas.file_count + 1, "
                "total_bytes = document_storage_quotas.total_bytes + :b, "
                "updated_at = :now"
            ),
            {"uid": user_id, "b": file_bytes, "now": now},
        )

        # Increment project quota
        await session.execute(
            text(
                "UPDATE document_storage_quotas SET "
                "file_count = file_count + 1, "
                "total_bytes = total_bytes + :b, "
                "updated_at = :now "
                "WHERE user_id='__project__'"
            ),
            {"b": file_bytes, "now": now},
        )


async def async_release_document_reservation(reservation_id: str) -> None:
    """Release an uncommitted or failed reservation without consuming permanent quota."""
    async with _doc_quota_transaction() as session:
        await session.execute(
            text(
                "UPDATE document_storage_reservations SET state='released' "
                "WHERE reservation_id=:res_id AND state IN ('active','writing')"
            ),
            {"res_id": reservation_id},
        )


async def async_release_committed_document(file_id: str, user_id: str) -> bool:
    """Release quota after verified document deletion from object storage."""
    now = datetime.now(timezone.utc)
    async with _doc_quota_transaction() as session:
        await lock_document_project(session)

        res = await session.execute(
            text(
                "SELECT reservation_id, bytes, state FROM document_storage_reservations "
                "WHERE file_id=:fid AND user_id=:uid"
            ),
            {"fid": file_id, "uid": user_id},
        )
        row = res.fetchone()
        if not row:
            # If not tracked in reservations, query library_files for size
            res_lib = await session.execute(
                text("SELECT file_size FROM library_files WHERE id=:fid AND user_id=:uid"),
                {"fid": file_id, "uid": user_id},
            )
            row_lib = res_lib.fetchone()
            if not row_lib:
                return False
            bytes_freed = int(row_lib[0]) if row_lib[0] else 0
        else:
            res_id, bytes_freed, state = str(row[0]), int(row[1]), str(row[2])
            await session.execute(
                text(
                    "UPDATE document_storage_reservations SET state='released' "
                    "WHERE reservation_id=:res_id"
                ),
                {"res_id": res_id},
            )

        if bytes_freed > 0:
            # Decrement user quota (floor at 0)
            await session.execute(
                text(
                    "UPDATE document_storage_quotas SET "
                    "file_count = CASE WHEN file_count > 0 THEN file_count - 1 ELSE 0 END, "
                    "total_bytes = CASE WHEN total_bytes >= :b THEN total_bytes - :b ELSE 0 END, "
                    "updated_at = :now "
                    "WHERE user_id=:uid"
                ),
                {"uid": user_id, "b": bytes_freed, "now": now},
            )

            # Decrement project quota
            await session.execute(
                text(
                    "UPDATE document_storage_quotas SET "
                    "file_count = CASE WHEN file_count > 0 THEN file_count - 1 ELSE 0 END, "
                    "total_bytes = CASE WHEN total_bytes >= :b THEN total_bytes - :b ELSE 0 END, "
                    "updated_at = :now "
                    "WHERE user_id='__project__'"
                ),
                {"b": bytes_freed, "now": now},
            )
        return True


async def async_get_document_storage_summary(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Return high-level storage utilization, limits, warnings, and throttle state."""
    max_user_count, max_user_bytes, max_project_bytes, max_object_bytes = _get_document_limits()
    warn_pct = getattr(settings, "STORAGE_CAPACITY_WARN_PCT", 0.80)
    throttle_pct = getattr(settings, "STORAGE_CAPACITY_THROTTLE_PCT", 0.90)

    now = datetime.now(timezone.utc)
    async with _doc_quota_transaction() as session:
        proj_res = await session.execute(
            text("SELECT file_count, total_bytes FROM document_storage_quotas WHERE user_id='__project__'")
        )
        row_p = proj_res.fetchone()
        proj_count = int(row_p[0]) if row_p and row_p[0] is not None else 0
        proj_bytes = int(row_p[1]) if row_p and row_p[1] is not None else 0

        user_info = None
        if user_id:
            user_res = await session.execute(
                text("SELECT file_count, total_bytes FROM document_storage_quotas WHERE user_id=:uid"),
                {"uid": user_id},
            )
            row_u = user_res.fetchone()
            u_count = int(row_u[0]) if row_u and row_u[0] is not None else 0
            u_bytes = int(row_u[1]) if row_u and row_u[1] is not None else 0
            user_info = {
                "user_id": user_id,
                "file_count": u_count,
                "total_bytes": u_bytes,
                "max_files": max_user_count,
                "max_bytes": max_user_bytes,
                "bytes_remaining": max(0, max_user_bytes - u_bytes),
                "pct_consumed": round((u_bytes / max_user_bytes) * 100, 2) if max_user_bytes else 0.0,
            }

    project_pct = round((proj_bytes / max_project_bytes) * 100, 2) if max_project_bytes else 0.0
    is_warning = project_pct >= (warn_pct * 100)
    is_throttled = project_pct >= (throttle_pct * 100)
    is_hard_ceiling = proj_bytes >= max_project_bytes

    return {
        "project": {
            "file_count": proj_count,
            "total_bytes": proj_bytes,
            "max_project_bytes": max_project_bytes,
            "bytes_remaining": max(0, max_project_bytes - proj_bytes),
            "pct_consumed": project_pct,
            "status": "hard_ceiling" if is_hard_ceiling else ("throttled" if is_throttled else ("warning" if is_warning else "healthy")),
            "warning": is_warning,
            "throttled": is_throttled,
        },
        "user": user_info,
        "limits": {
            "max_object_bytes": max_object_bytes,
            "max_user_bytes": max_user_bytes,
            "max_user_count": max_user_count,
            "max_project_bytes": max_project_bytes,
        },
    }


# Synchronous convenience wrappers
def reserve_document_upload(user_id: str, file_bytes: int, filename: str, file_id: Optional[str] = None) -> Dict[str, Any]:
    return run_sync_db(async_reserve_document_upload, user_id, file_bytes, filename, file_id)


def commit_document_upload(reservation_id: str, storage_path: str) -> None:
    return run_sync_db(async_commit_document_upload, reservation_id, storage_path)


def release_document_reservation(reservation_id: str) -> None:
    return run_sync_db(async_release_document_reservation, reservation_id)


def release_committed_document(file_id: str, user_id: str) -> bool:
    return run_sync_db(async_release_committed_document, file_id, user_id)


def get_document_storage_summary(user_id: Optional[str] = None) -> Dict[str, Any]:
    return run_sync_db(async_get_document_storage_summary, user_id)
