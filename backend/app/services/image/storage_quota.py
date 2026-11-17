"""Durable storage reservation receipts and serialized online reconciliation."""

import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings
from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import workspace_store
from sqlalchemy import text

logger = logging.getLogger("chatbot.storage_quota")

QUOTA_DDL = (
    "CREATE TABLE IF NOT EXISTS image_storage_quotas (user_id VARCHAR(128) PRIMARY KEY, "
    "image_count INTEGER NOT NULL DEFAULT 0, total_bytes BIGINT NOT NULL DEFAULT 0, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
RESERVATION_DDL = (
    "CREATE TABLE IF NOT EXISTS image_storage_reservations (reservation_id VARCHAR(128) PRIMARY KEY, "
    "user_id VARCHAR(128) NOT NULL, bytes BIGINT NOT NULL CHECK(bytes>0), "
    "state VARCHAR(32) NOT NULL CHECK(state IN ('active','writing','cleanup','committed','released','abandoned')), "
    "expires_at TIMESTAMP NOT NULL, object_path TEXT, image_id VARCHAR(128) UNIQUE, "
    "operation_id VARCHAR(128), "
    "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
)
_CURRENT_RESERVATION: ContextVar[tuple[str, int, str, str] | None] = ContextVar("storage_reservation", default=None)


def _get_limits(max_user_count=None, max_user_bytes=None, max_project_bytes=None, max_object_bytes=None):
    return (
        max_user_count if max_user_count is not None else settings.IMAGE_MAX_USER_COUNT,
        max_user_bytes if max_user_bytes is not None else settings.IMAGE_MAX_USER_BYTES,
        max_project_bytes if max_project_bytes is not None else settings.IMAGE_MAX_PROJECT_BYTES,
        max_object_bytes if max_object_bytes is not None else settings.IMAGE_MAX_OBJECT_BYTES,
    )


def ensure_sqlite_quota_schema(conn):
    conn.execute(QUOTA_DDL)
    conn.execute(RESERVATION_DDL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(image_storage_reservations)")}
    if "operation_id" not in columns:
        conn.execute("ALTER TABLE image_storage_reservations ADD COLUMN operation_id VARCHAR(128)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_storage_reservation_owner_state ON image_storage_reservations(user_id,state)"
    )


class _SQLiteSession:
    def __init__(self, conn):
        self.conn = conn

    async def execute(self, statement, params=None):
        sql = (
            str(statement)
            .replace("expires_at<=:now", "julianday(expires_at)<=julianday(:now)")
            .replace("expires_at>:now", "julianday(expires_at)>julianday(:now)")
        )
        values = {
            key: value.isoformat() if isinstance(value, datetime) else value for key, value in (params or {}).items()
        }
        return self.conn.execute(sql, values)


@asynccontextmanager
async def _quota_transaction():
    if is_postgres_configured():
        async with get_async_session() as session, session.begin():
            yield session
    else:
        conn = workspace_store._get_sqlite_connection()
        try:
            ensure_sqlite_quota_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            yield _SQLiteSession(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


async def lock_storage_project(session):
    """All quota/record writers serialize here; PostgreSQL holds it until commit."""
    await session.execute(
        text(
            "INSERT INTO image_storage_quotas(user_id,image_count,total_bytes) "
            "VALUES ('__project__',0,0) ON CONFLICT(user_id) DO NOTHING"
        )
    )
    suffix = "" if isinstance(session, _SQLiteSession) else " FOR UPDATE"
    await session.execute(text("SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__'" + suffix))


async def _database_now(session):
    if isinstance(session, _SQLiteSession):
        return datetime.now(timezone.utc)
    return (await session.execute(text("SELECT clock_timestamp()"))).fetchone()[0]


async def _check_owner(session, owner, require_exists=False):
    row = (
        await session.execute(text("SELECT deletion_pending FROM users WHERE id=:owner"), {"owner": owner})
    ).fetchone()
    if (require_exists and row is None) or (row and row[0]):
        raise ImageProviderError("authentication_required", "Your account is no longer available.", 401)


async def _rebuild(session):
    now = await _database_now(session)
    await session.execute(
        text("UPDATE image_storage_reservations SET state='abandoned' WHERE state='active' AND expires_at<=:now"),
        {"now": now},
    )
    retained = (
        await session.execute(
            text("SELECT user_id,COUNT(*),COALESCE(SUM(bytes),0) FROM image_records GROUP BY user_id")
        )
    ).fetchall()
    pending = (
        await session.execute(
            text(
                "SELECT user_id,COUNT(*),COALESCE(SUM(bytes),0) "
                "FROM image_storage_reservations WHERE state IN ('active','writing','cleanup') "
                "GROUP BY user_id"
            )
        )
    ).fetchall()
    owners = {
        row[0]: [0, 0]
        for row in (
            await session.execute(text("SELECT user_id FROM image_storage_quotas WHERE user_id!='__project__'"))
        ).fetchall()
    }
    for owner, count, size in [*retained, *pending]:
        if owner == "__project__":
            raise RuntimeError("Reserved quota scope used as an account identity")
        totals = owners.setdefault(owner, [0, 0])
        totals[0] += count
        totals[1] += size
    for owner, (count, size) in owners.items():
        await session.execute(
            text(
                "INSERT INTO image_storage_quotas(user_id,image_count,total_bytes) VALUES(:u,:c,:b) "
                "ON CONFLICT(user_id) DO UPDATE SET image_count=excluded.image_count, "
                "total_bytes=excluded.total_bytes,updated_at=CURRENT_TIMESTAMP"
            ),
            {"u": owner, "c": count, "b": size},
        )
    await session.execute(
        text("UPDATE image_storage_quotas SET total_bytes=:b,updated_at=CURRENT_TIMESTAMP WHERE user_id='__project__'"),
        {"b": sum(value[1] for value in owners.values())},
    )


async def _reserve(
    owner, size, count_limit, byte_limit, project_limit, reservation_id, operation_id, require_active_owner
):
    async with _quota_transaction() as session:
        await lock_storage_project(session)
        await _check_owner(session, owner, require_exists=require_active_owner)
        await _rebuild(session)
        row = (
            await session.execute(
                text("SELECT image_count,total_bytes FROM image_storage_quotas WHERE user_id=:u"), {"u": owner}
            )
        ).fetchone()
        project = (
            await session.execute(text("SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__'"))
        ).fetchone()[0]
        count, used = tuple(row) if row else (0, 0)
        if count + 1 > count_limit:
            raise ImageProviderError(
                "storage_quota_exhausted",
                f"Maximum retained image count ({count_limit}) reached. Delete older images to generate new ones.",
                429,
            )
        if used + size > byte_limit:
            raise ImageProviderError(
                "storage_quota_exhausted",
                "User retained storage quota exceeded. Delete older images to free space.",
                429,
            )
        if project + size > project_limit:
            raise ImageProviderError(
                "storage_quota_exhausted", "Total project image storage capacity reached. Please contact support.", 429
            )
        now = await _database_now(session)
        await session.execute(
            text(
                "INSERT INTO image_storage_reservations(reservation_id,user_id,bytes,state,expires_at,operation_id) "
                "VALUES(:r,:u,:b,'active',:expires,:operation)"
            ),
            {
                "r": reservation_id,
                "u": owner,
                "b": size,
                "expires": now + timedelta(seconds=300),
                "operation": operation_id,
            },
        )
        await session.execute(
            text(
                "INSERT INTO image_storage_quotas(user_id,image_count,total_bytes) VALUES(:u,1,:b) "
                "ON CONFLICT(user_id) DO UPDATE SET image_count=image_storage_quotas.image_count+1, "
                "total_bytes=image_storage_quotas.total_bytes+:b,updated_at=CURRENT_TIMESTAMP"
            ),
            {"u": owner, "b": size},
        )
        await session.execute(
            text("UPDATE image_storage_quotas SET total_bytes=total_bytes+:b WHERE user_id='__project__'"), {"b": size}
        )
    return reservation_id


def reserve_storage_quota(
    owner_id,
    new_bytes,
    max_user_count=None,
    max_user_bytes=None,
    max_project_bytes=None,
    max_object_bytes=None,
    *,
    operation_id=None,
    require_active_owner=False,
) -> str:
    limits = _get_limits(max_user_count, max_user_bytes, max_project_bytes, max_object_bytes)
    if not owner_id or owner_id == "__project__" or new_bytes <= 0 or any(limit <= 0 for limit in limits):
        raise ImageProviderError("quota_configuration", "Storage quota parameters are invalid.", 400)
    if new_bytes > limits[3]:
        raise ImageProviderError("image_too_large", "Image size exceeds the maximum allowed limit.", 400)
    identifier = run_sync_db(
        _reserve(owner_id, new_bytes, *limits[:3], str(uuid.uuid4()), operation_id, require_active_owner)
    )
    _CURRENT_RESERVATION.set((owner_id, new_bytes, identifier, workspace_store.db_path))
    return identifier


def implicit_reservation(owner, size):
    value = _CURRENT_RESERVATION.get()
    if value and value[0] == owner and value[1] == size and value[3] == workspace_store.db_path:
        return value[2]
    return None


def clear_reservation_context():
    _CURRENT_RESERVATION.set(None)


async def _begin_write(owner, reservation_id, path, require_active_owner):
    async with _quota_transaction() as session:
        await lock_storage_project(session)
        await _check_owner(session, owner, require_exists=require_active_owner)
        now = await _database_now(session)
        result = await session.execute(
            text(
                "UPDATE image_storage_reservations SET state='writing',object_path=:path "
                "WHERE reservation_id=:r AND user_id=:u AND state='active' AND expires_at>:now"
            ),
            {"path": path, "r": reservation_id, "u": owner, "now": now},
        )
        if result.rowcount != 1:
            raise ImageProviderError(
                "reservation_expired", "Storage reservation is unavailable; no write is authorized.", 409
            )


def begin_storage_write(owner, reservation_id, object_path, *, require_active_owner=False):
    return run_sync_db(_begin_write(owner, reservation_id, object_path, require_active_owner))


async def commit_storage_reservation(session, owner, size, reservation_id, image_id, path):
    now = await _database_now(session)
    result = await session.execute(
        text(
            "UPDATE image_storage_reservations SET state='committed',image_id=:id,object_path=:path "
            "WHERE reservation_id=:r AND user_id=:u AND bytes=:b AND state IN ('active','writing') "
            "AND (state='writing' OR expires_at>:now) AND (object_path IS NULL OR object_path=:path)"
        ),
        {"id": image_id, "path": path, "r": reservation_id, "u": owner, "b": size, "now": now},
    )
    if result.rowcount != 1:
        raise ImageProviderError("reservation_expired", "Image reservation no longer permits publication.", 409)


def commit_storage_reservation_sqlite(conn, owner, size, reservation_id, image_id, path):
    result = conn.execute(
        "UPDATE image_storage_reservations SET state='committed',image_id=:id,object_path=:path "
        "WHERE reservation_id=:r AND user_id=:u AND bytes=:b AND state IN ('active','writing') "
        "AND (state='writing' OR julianday(expires_at)>julianday(:now)) AND (object_path IS NULL OR object_path=:path)",
        {
            "id": image_id,
            "path": path,
            "r": reservation_id,
            "u": owner,
            "b": size,
            "now": datetime.now(timezone.utc).isoformat(),
        },
    )
    if result.rowcount != 1:
        raise ImageProviderError("reservation_expired", "Image reservation no longer permits publication.", 409)


async def _release_storage_quota_in_session(session, owner_id, released_bytes, image_id=None):
    """Caller already holds project lock and removes the image in this transaction."""
    if image_id:
        await session.execute(
            text("UPDATE image_storage_reservations SET state='released' WHERE image_id=:id AND user_id=:u"),
            {"id": image_id, "u": owner_id},
        )
    await session.execute(
        text(
            "UPDATE image_storage_quotas SET image_count=CASE WHEN image_count>0 THEN image_count-1 ELSE 0 END, "
            "total_bytes=CASE WHEN total_bytes>=:b THEN total_bytes-:b ELSE 0 END WHERE user_id=:u"
        ),
        {"u": owner_id, "b": released_bytes},
    )
    await session.execute(
        text(
            "UPDATE image_storage_quotas SET total_bytes=CASE WHEN total_bytes>=:b THEN total_bytes-:b ELSE 0 END "
            "WHERE user_id='__project__'"
        ),
        {"b": released_bytes},
    )


async def _release(owner, size, reservation_id, verified_absent):
    async with _quota_transaction() as session:
        await lock_storage_project(session)
        if reservation_id is None:
            # Legacy callers may release a pre-write reservation, never another
            # worker's writing receipt. New production callers pass exact IDs.
            row = (
                await session.execute(
                    text(
                        "SELECT reservation_id FROM image_storage_reservations "
                        "WHERE user_id=:u AND bytes=:b AND state='active' ORDER BY created_at LIMIT 1"
                    ),
                    {"u": owner, "b": size},
                )
            ).fetchone()
            reservation_id = row[0] if row else None
        if reservation_id is None:
            return
        row = (
            await session.execute(
                text("SELECT user_id,bytes,state FROM image_storage_reservations WHERE reservation_id=:r"),
                {"r": reservation_id},
            )
        ).fetchone()
        if not row:
            return
        if row[0] != owner or row[1] != size:
            raise ImageProviderError(
                "reservation_conflict", "Storage reservation does not belong to this request.", 409
            )
        if row[2] in {"released", "abandoned"}:
            return
        if row[2] == "committed":
            raise ImageProviderError(
                "reservation_conflict", "Delete the retained image to release committed storage.", 409
            )
        if row[2] in {"writing", "cleanup"} and not verified_absent:
            raise ImageProviderError(
                "storage_cleanup_required", "Verify object cleanup before releasing storage capacity.", 503
            )
        await session.execute(
            text("UPDATE image_storage_reservations SET state='released' WHERE reservation_id=:r"),
            {"r": reservation_id},
        )
        await _release_storage_quota_in_session(session, owner, size)


def release_storage_quota(owner_id, released_bytes, reservation_id=None, *, verified_absent=False):
    result = run_sync_db(_release(owner_id, released_bytes, reservation_id, verified_absent))
    clear_reservation_context()
    return result


async def _reconcile():
    async with _quota_transaction() as session:
        await lock_storage_project(session)
        await _rebuild(session)


def reconcile_storage_quota(owner_id=None, force: bool = False):
    """Rebuild every counter from committed rows plus outstanding receipts under a project lock."""
    from app.core.maintenance import is_maintenance_mode
    if is_maintenance_mode() and not force:
        logger.info("[MAINTENANCE] Skipping storage quota reconciliation during maintenance mode.")
        return None
    return run_sync_db(_reconcile())


async def _usage(owner):
    async with _quota_transaction() as session:
        u = (
            await session.execute(
                text("SELECT image_count,total_bytes FROM image_storage_quotas WHERE user_id=:u"), {"u": owner}
            )
        ).fetchone()
        p = (
            await session.execute(text("SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__'"))
        ).fetchone()
        return (u[0] if u else 0, u[1] if u else 0, p[0] if p else 0)


def get_storage_quota_usage(owner_id) -> dict[str, Any]:
    count, size, project = run_sync_db(_usage(owner_id))
    limits = _get_limits()
    return {
        "user_id": owner_id,
        "image_count": count,
        "total_bytes": size,
        "project_bytes": project,
        "max_user_count": limits[0],
        "max_user_bytes": limits[1],
        "max_project_bytes": limits[2],
        "max_object_bytes": limits[3],
    }


async def _cleanup_receipts(owner):
    async with _quota_transaction() as session:
        await lock_storage_project(session)
        rows = (
            await session.execute(
                text(
                    "SELECT reservation_id,bytes,state,object_path FROM image_storage_reservations "
                    "WHERE user_id=:u AND state IN ('active','writing','cleanup')"
                ),
                {"u": owner},
            )
        ).fetchall()
        return [tuple(row) for row in rows]


def cancel_owner_reservations(owner):
    """Deletion-pending owner cannot start/commit writes. Do not erase live write receipts."""
    rows = run_sync_db(_cleanup_receipts(owner))
    if any(row[2] != "active" for row in rows):
        raise RuntimeError("In-flight image writes require verified cleanup before account erasure")
    for identifier, size, _state, _path in rows:
        release_storage_quota(owner, size, identifier)


def mark_storage_cleanup(owner, reservation_id):
    async def _mark():
        async with _quota_transaction() as session:
            await lock_storage_project(session)
            await session.execute(
                text(
                    "UPDATE image_storage_reservations SET state='cleanup' "
                    "WHERE reservation_id=:r AND user_id=:u AND state='writing'"
                ),
                {"r": reservation_id, "u": owner},
            )

    return run_sync_db(_mark())


def recover_storage_reservation(
    owner, reservation_id, *, confirmed_writer_stopped=False, confirmed_remote_write_settled=False
):
    """Explicit crash recovery: stop the originating writer, delete known object, verify absence."""
    if not confirmed_writer_stopped:
        raise ValueError("Recovery requires confirmation that the originating writer is stopped")
    from app.services.storage import storage_service
    from app.services.storage.local import LocalStorageBackend

    if not isinstance(storage_service, LocalStorageBackend) and not confirmed_remote_write_settled:
        raise ValueError("Remote recovery also requires evidence that the accepted upload has settled")

    async def _prepare():
        async with _quota_transaction() as session:
            await lock_storage_project(session)
            row = (
                await session.execute(
                    text(
                        "SELECT reservation_id,bytes,state,object_path FROM image_storage_reservations "
                        "WHERE reservation_id=:r AND user_id=:u AND state IN ('active','writing','cleanup')"
                    ),
                    {"r": reservation_id, "u": owner},
                )
            ).fetchone()
            if row:
                # Fence publication before touching object storage. Failed
                # cleanup keeps both the receipt and its capacity reservation.
                await session.execute(
                    text("UPDATE image_storage_reservations SET state='cleanup' WHERE reservation_id=:r"),
                    {"r": reservation_id},
                )
            return tuple(row) if row else None

    row = run_sync_db(_prepare())
    if not row:
        return
    identifier, size, state, path = row
    if state != "active" and not path:
        raise RuntimeError("Missing durable object reference; recovery cannot safely refund capacity")
    if path:
        storage_service.delete_verified(path)
    release_storage_quota(owner, size, identifier, verified_absent=True)
