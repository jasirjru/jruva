"""Durable Budget Enforcement and Distributed Image Generation Idempotency.

Supports dual persistence backends:
1. PostgreSQL (production deployment via Supabase, Neon, Cloud Run)
   - Atomic row-level locking with FOR UPDATE
   - Distributed leasing with worker_id, lease_expires_at, and fencing_token
   - Rejection of delayed completions from stale workers (fencing violation guards)
2. SQLite (local development and offline testing fallback)
   - Atomic transactions with BEGIN IMMEDIATE
   - Full column-level parity with PostgreSQL schema
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.services.image.providers import ImageProviderError
from app.services.workspace_store import workspace_store
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Process-level default worker identifier
PROCESS_WORKER_ID = f"worker_{platform.node()}_{os.getpid()}_{uuid.uuid4().hex[:6]}"

_ACTIVE_IN_FLIGHT_KEYS: set[tuple[str, str]] = set()
_IN_FLIGHT_LOCK = threading.Lock()
_GENERATION_LEASE: ContextVar[tuple[str, str, str, int] | None] = ContextVar("image_generation_lease", default=None)


@contextmanager
def generation_operation(owner: str, key: str, worker: str, fence: int):
    token = _GENERATION_LEASE.set((owner, key, worker, fence))
    try:
        yield
    finally:
        _GENERATION_LEASE.reset(token)


def current_generation_lease():
    return _GENERATION_LEASE.get()


@contextmanager
def active_in_flight_generation(owner_id: str, idempotency_key: str):
    """Context manager registering an active in-flight generation task lease in local process memory."""
    key = (owner_id, idempotency_key)
    with _IN_FLIGHT_LOCK:
        _ACTIVE_IN_FLIGHT_KEYS.add(key)
    try:
        yield
    finally:
        with _IN_FLIGHT_LOCK:
            _ACTIVE_IN_FLIGHT_KEYS.discard(key)


def is_in_flight(owner_id: str, idempotency_key: str) -> bool:
    """Return True if a generation task for this key is currently executing in this process."""
    with _IN_FLIGHT_LOCK:
        return (owner_id, idempotency_key) in _ACTIVE_IN_FLIGHT_KEYS


# ─── Daily Budget DDL and Allocation ─────────────────────────────────────────

DDL = (
    "CREATE TABLE IF NOT EXISTS image_generation_budgets ("
    "scope VARCHAR(128) NOT NULL, "
    "day VARCHAR(10) NOT NULL, "
    "used INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY(scope, day))"
)


def charges(account: str, owner: str, neurons: int, daily_limit: int, owner_limit: int):
    if not owner or neurons <= 0 or not 0 < daily_limit <= 10000 or owner_limit <= 0:
        raise ImageProviderError(
            "quota_configuration", "Image quota policy is not safely configured."
        )
    day = datetime.now(timezone.utc).date().isoformat()
    keys = [
        ("account:" + account, neurons, daily_limit),
        ("owner:" + owner, 1, owner_limit),
    ]
    return [
        (hashlib.sha256(k.encode()).hexdigest(), day, cost, limit)
        for k, cost, limit in keys
    ]


async def _reserve_postgres(rows):
    async with get_async_session() as session, session.begin():
        for scope, day, cost, limit in rows:
            await session.execute(
                text(
                    "INSERT INTO image_generation_budgets(scope,day,used) "
                    "VALUES(:scope,:day,0) ON CONFLICT(scope,day) DO NOTHING"
                ),
                {"scope": scope, "day": day},
            )
            result = await session.execute(
                text(
                    "UPDATE image_generation_budgets "
                    "SET used=used+:cost "
                    "WHERE scope=:scope AND day=:day AND used+:cost<=:limit "
                    "RETURNING used"
                ),
                {"scope": scope, "day": day, "cost": cost, "limit": limit},
            )
            if result.scalar_one_or_none() is None:
                raise ImageProviderError(
                    "quota_exhausted",
                    "Daily image attempt quota is exhausted. Retry after 00:00 UTC.",
                    429,
                )


def reserve_image_budget(account: str, owner: str, neurons: int, daily_limit: int, owner_limit: int):
    rows = charges(account, owner, neurons, daily_limit, owner_limit)
    if is_postgres_configured():
        return run_sync_db(_reserve_postgres(rows))
    with workspace_store._get_sqlite_connection() as connection:
        connection.execute(DDL)
        connection.execute("BEGIN IMMEDIATE")
        for scope, day, cost, limit in rows:
            connection.execute(
                "INSERT OR IGNORE INTO image_generation_budgets(scope,day,used) VALUES(?,?,0)",
                (scope, day),
            )
            result = connection.execute(
                "UPDATE image_generation_budgets SET used=used+? WHERE scope=? AND day=? AND used+?<=?",
                (cost, scope, day, cost, limit),
            )
            if result.rowcount != 1:
                raise ImageProviderError(
                    "quota_exhausted",
                    "Daily image attempt quota is exhausted. Retry after 00:00 UTC.",
                    429,
                )


def reserve_media_write_budget(owner: str, owner_limit: int = 5) -> None:
    """Reserve an atomic daily media-write attempt for an owner without consuming Cloudflare neurons."""
    if not owner or owner_limit <= 0:
        raise ImageProviderError(
            "quota_configuration", "Media write quota policy is not safely configured."
        )
    day = datetime.now(timezone.utc).date().isoformat()
    key = "owner_media_writes:" + owner
    scope = hashlib.sha256(key.encode()).hexdigest()
    rows = [(scope, day, 1, owner_limit)]
    if is_postgres_configured():
        return run_sync_db(_reserve_postgres(rows))
    with workspace_store._get_sqlite_connection() as connection:
        connection.execute(DDL)
        connection.execute("BEGIN IMMEDIATE")
        for sc, dy, cost, limit in rows:
            connection.execute(
                "INSERT OR IGNORE INTO image_generation_budgets(scope,day,used) VALUES(?,?,0)",
                (sc, dy),
            )
            result = connection.execute(
                "UPDATE image_generation_budgets SET used=used+? WHERE scope=? AND day=? AND used+?<=?",
                (cost, sc, dy, cost, limit),
            )
            if result.rowcount != 1:
                raise ImageProviderError(
                    "quota_exhausted",
                    "Daily media write quota is exhausted. Retry after 00:00 UTC.",
                    429,
                )


# ─── Distributed Idempotency DDL & Parity ────────────────────────────────────

IDEMPOTENCY_DDL = (
    "CREATE TABLE IF NOT EXISTS image_idempotency ("
    "owner_id VARCHAR(128) NOT NULL, "
    "idempotency_key VARCHAR(128) NOT NULL, "
    "request_hash VARCHAR(64) NOT NULL, "
    "status VARCHAR(32) NOT NULL, "
    "response_json TEXT, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "worker_id VARCHAR(128), "
    "lease_expires_at TIMESTAMP, "
    "fencing_token INTEGER DEFAULT 1, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "operation_id VARCHAR(128), external_state VARCHAR(32) NOT NULL DEFAULT 'unstarted', "
    "provider_model VARCHAR(128), provider_attempts INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY(owner_id, idempotency_key))"
)


def _ensure_sqlite_idempotency_schema(conn) -> None:
    """Ensure SQLite table exists and includes distributed leasing columns."""
    conn.execute(IDEMPOTENCY_DDL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(image_idempotency)").fetchall()}
    if "worker_id" not in cols:
        conn.execute("ALTER TABLE image_idempotency ADD COLUMN worker_id VARCHAR(128)")
    if "lease_expires_at" not in cols:
        conn.execute("ALTER TABLE image_idempotency ADD COLUMN lease_expires_at TIMESTAMP")
    if "fencing_token" not in cols:
        conn.execute("ALTER TABLE image_idempotency ADD COLUMN fencing_token INTEGER DEFAULT 1")
    if "updated_at" not in cols:
        # SQLite cannot ALTER a populated table with a non-constant default.
        # Existing rows may remain NULL until their next lifecycle update.
        conn.execute("ALTER TABLE image_idempotency ADD COLUMN updated_at TIMESTAMP")
    for name, definition in (
        ("operation_id", "VARCHAR(128)"),
        ("external_state", "VARCHAR(32) NOT NULL DEFAULT 'unstarted'"),
        ("provider_model", "VARCHAR(128)"),
        ("provider_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE image_idempotency ADD COLUMN {name} {definition}")


# ─── PostgreSQL Distributed Leasing & Fencing ────────────────────────────────

def _lease_is_stale(now: datetime, lease_expires_at, created_at, lease_seconds: int = 180) -> bool:
    """A renewable lease is authoritative; creation age is only for pre-006 rows."""
    value = lease_expires_at if lease_expires_at is not None else created_at
    if value is None:
        return False
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if lease_expires_at is not None:
            return now >= stamp
        return (now - stamp).total_seconds() > lease_seconds
    except (TypeError, ValueError):
        # Do not revoke ownership based on an unreadable timestamp.
        return False


async def _claim_idempotency_lease_postgres(
    owner_id: str,
    idempotency_key: str,
    request_hash: str,
    worker_id: str | None = None,
    lease_seconds: int = 180,
) -> tuple[dict | None, int | None, str]:
    from sqlalchemy.exc import IntegrityError

    effective_worker = worker_id or PROCESS_WORKER_ID
    reconciliation_needed = False
    async with get_async_session() as session:
        async with session.begin():
            res = await session.execute(
                text(
                    "SELECT request_hash, status, response_json, created_at, worker_id, lease_expires_at, fencing_token "
                    "FROM image_idempotency WHERE owner_id=:o AND idempotency_key=:k FOR UPDATE"
                ),
                {"o": owner_id, "k": idempotency_key},
            )
            row = res.fetchone()
            now = datetime.now(timezone.utc)
            lease_expires = now + timedelta(seconds=lease_seconds)

            if row:
                stored_hash, status, resp_json, created_at, cur_worker, cur_lease_exp, cur_fence = (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6] or 1
                )
                # Parameter conflict check: must reject altered request payload
                if stored_hash != request_hash:
                    raise ImageProviderError(
                        "idempotency_conflict",
                        "Idempotency key was used with different request parameters.",
                        409,
                    )

                if status == "indeterminate":
                    raise ImageProviderError("operation_indeterminate", "Previous provider outcome is unknown or awaiting local recovery. No generation was repeated.")

                # Cached completed result replay
                if status == "completed" and resp_json:
                    cached_data = json.loads(resp_json)
                    img_id = cached_data.get("image_id")
                    image_exists = True
                    if img_id:
                        image_exists = (await session.execute(
                            text("SELECT id FROM image_records WHERE id=:id AND user_id=:owner"),
                            {"id": img_id, "owner": owner_id},
                        )).scalar_one_or_none() is not None
                    if not image_exists:
                        raise ImageProviderError(
                            "image_deleted",
                            "The image referenced by this idempotency key has been deleted.",
                            410,
                        )
                    return cached_data, cur_fence, cur_worker or effective_worker

                if status == "pending":
                    # Check lease expiration
                    is_stale = _lease_is_stale(now, cur_lease_exp, created_at, lease_seconds)

                    if is_stale:
                        # Stale pending lock detected: reconcile deterministically with incremented fencing token
                        new_fence = cur_fence + 1
                        await session.execute(
                            text(
                                "UPDATE image_idempotency "
                                "SET status='stale_expired', fencing_token=:f, updated_at=:now "
                                "WHERE owner_id=:o AND idempotency_key=:k"
                            ),
                            {"f": new_fence, "now": now, "o": owner_id, "k": idempotency_key},
                        )
                        reconciliation_needed = True
                    else:
                        # Active lease held by another or current worker
                        raise ImageProviderError(
                            "idempotency_in_progress",
                            f"A request with this idempotency key is currently processing by worker {cur_worker or 'active'}.",
                            409,
                        )

                if not reconciliation_needed:
                    # Status in ('stale_expired', 'failed'): atomically claim lease with incremented fencing token
                    new_fence = cur_fence + 1
                    await session.execute(
                        text(
                            "UPDATE image_idempotency "
                            "SET status='pending', request_hash=:h, worker_id=:w, "
                            "    lease_expires_at=:exp, fencing_token=:f, updated_at=:now "
                            "WHERE owner_id=:o AND idempotency_key=:k"
                        ),
                        {
                            "h": request_hash,
                            "w": effective_worker,
                            "exp": lease_expires,
                            "f": new_fence,
                            "now": now,
                            "o": owner_id,
                            "k": idempotency_key,
                        },
                    )
                    return None, new_fence, effective_worker

        if reconciliation_needed:
            raise ImageProviderError(
                "idempotency_needs_reconciliation",
                "Previous generation attempt timed out and stale lock has been reconciled. Retry to initiate a fresh request.",
                409,
            )

        # Row does not exist: create fresh claim with fencing_token = 1
        try:
            await session.execute(
                text(
                    "INSERT INTO image_idempotency ("
                    "    owner_id, idempotency_key, request_hash, status, response_json, "
                    "    created_at, worker_id, lease_expires_at, fencing_token, updated_at"
                    ") VALUES (:o, :k, :h, 'pending', NULL, :now, :w, :exp, 1, :now)"
                ),
                {
                    "o": owner_id,
                    "k": idempotency_key,
                    "h": request_hash,
                    "now": now,
                    "w": effective_worker,
                    "exp": lease_expires,
                },
            )
            return None, 1, effective_worker
        except IntegrityError:
            raise ImageProviderError(
                "idempotency_in_progress",
                "A request with this idempotency key is currently processing.",
                409,
            )


async def _check_or_claim_idempotency_postgres(
    owner_id: str,
    idempotency_key: str,
    request_hash: str,
    worker_id: str | None = None,
    lease_seconds: int = 180,
) -> dict | None:
    """Backward-compatible internal helper for test harnesses expecting dictionary or None."""
    cached_data, _fence, _worker = await _claim_idempotency_lease_postgres(
        owner_id, idempotency_key, request_hash, worker_id, lease_seconds
    )
    return cached_data



async def _complete_idempotency_postgres(
    owner_id: str,
    idempotency_key: str,
    response_json_str: str,
    worker_id: str | None = None,
    fencing_token: int | None = None,
) -> bool:
    async with get_async_session() as session, session.begin():
        query = (
            "UPDATE image_idempotency "
            "SET status='completed', response_json=:r, updated_at=NOW() "
            "WHERE owner_id=:o AND idempotency_key=:k AND status IN ('pending','indeterminate')"
        )
        params: dict[str, Any] = {"r": response_json_str, "o": owner_id, "k": idempotency_key}
        if worker_id is not None:
            query += " AND worker_id=:w"
            params["w"] = worker_id
        if fencing_token is not None:
            query += " AND fencing_token=:f"
            params["f"] = fencing_token

        res = await session.execute(text(query), params)
        if res.rowcount == 0:
            logger.warning(
                f"[FencingViolationGuard] Completion rejected for key {idempotency_key}: "
                f"worker_id={worker_id}, fencing_token={fencing_token}. "
                "Lease was revoked or ownership was reclaimed by a newer worker."
            )
            return False
        return True


async def _fail_idempotency_postgres(
    owner_id: str,
    idempotency_key: str,
    worker_id: str | None = None,
    fencing_token: int | None = None,
) -> bool:
    async with get_async_session() as session, session.begin():
        query = (
            "UPDATE image_idempotency "
            "SET status='failed', updated_at=NOW() "
            "WHERE owner_id=:o AND idempotency_key=:k AND status='pending'"
        )
        params: dict[str, Any] = {"o": owner_id, "k": idempotency_key}
        if worker_id is not None:
            query += " AND worker_id=:w"
            params["w"] = worker_id
        if fencing_token is not None:
            query += " AND fencing_token=:f"
            params["f"] = fencing_token

        res = await session.execute(text(query), params)
        return res.rowcount > 0


async def _heartbeat_idempotency_lease_postgres(
    owner_id: str,
    idempotency_key: str,
    worker_id: str,
    fencing_token: int,
    extension_seconds: int = 180,
) -> bool:
    async with get_async_session() as session, session.begin():
        now = datetime.now(timezone.utc)
        new_exp = now + timedelta(seconds=extension_seconds)
        res = await session.execute(
            text(
                "UPDATE image_idempotency "
                "SET lease_expires_at=:exp, updated_at=:now "
                "WHERE owner_id=:o AND idempotency_key=:k AND status IN ('pending','indeterminate') "
                "  AND worker_id=:w AND fencing_token=:f"
            ),
            {"exp": new_exp, "now": now, "o": owner_id, "k": idempotency_key, "w": worker_id, "f": fencing_token},
        )
        return res.rowcount > 0


async def _reconcile_stale_idempotency_postgres(owner_id: str, idempotency_key: str) -> dict[str, Any]:
    async with get_async_session() as session, session.begin():
        res = await session.execute(
            text(
                "SELECT status, response_json, fencing_token, lease_expires_at, created_at FROM image_idempotency "
                "WHERE owner_id=:o AND idempotency_key=:k FOR UPDATE"
            ),
            {"o": owner_id, "k": idempotency_key},
        )
        row = res.fetchone()
        if not row:
            return {"status": "not_found", "reconciled": False}
        status, resp_json, token = row[0], row[1], row[2] or 1
        if status == "completed":
            return {"status": "completed", "reconciled": False, "response": resp_json, "fencing_token": token}
        if status != "pending" or not _lease_is_stale(datetime.now(timezone.utc), row[3], row[4]):
            return {"status": status, "reconciled": False, "fencing_token": token}
        new_token = token + 1
        await session.execute(
            text(
                "UPDATE image_idempotency "
                "SET status='stale_expired', fencing_token=:f, updated_at=NOW() "
                "WHERE owner_id=:o AND idempotency_key=:k"
            ),
            {"f": new_token, "o": owner_id, "k": idempotency_key},
        )
        return {"status": "stale_expired", "reconciled": True, "fencing_token": new_token}


# ─── SQLite Distributed Parity Implementation ────────────────────────────────

def _claim_idempotency_lease_sqlite(
    owner_id: str,
    idempotency_key: str,
    request_hash: str,
    worker_id: str,
    lease_seconds: int = 180,
) -> tuple[dict | None, int | None, str]:
    now = datetime.now(timezone.utc)
    lease_expires = now + timedelta(seconds=lease_seconds)
    lease_exp_str = lease_expires.strftime("%Y-%m-%d %H:%M:%S")

    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "SELECT request_hash, status, response_json, created_at, worker_id, lease_expires_at, fencing_token "
            "FROM image_idempotency WHERE owner_id=? AND idempotency_key=?",
            (owner_id, idempotency_key),
        )
        row = cursor.fetchone()
        if row:
            stored_hash, status, resp_json, created_at, cur_worker, cur_lease_exp, cur_fence = (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6] or 1
            )
            if stored_hash != request_hash:
                raise ImageProviderError(
                    "idempotency_conflict",
                    "Idempotency key was used with different request parameters.",
                    409,
                )
            if status == "indeterminate":
                raise ImageProviderError("operation_indeterminate", "Previous provider outcome is unknown or awaiting local recovery. No generation was repeated.")
            if status == "completed" and resp_json:
                cached_data = json.loads(resp_json)
                img_id = cached_data.get("image_id")
                if img_id and not workspace_store.get_image(img_id, owner_id):
                    raise ImageProviderError(
                        "image_deleted",
                        "The image referenced by this idempotency key has been deleted.",
                        410,
                    )
                return cached_data, cur_fence, cur_worker or worker_id

            if status == "pending":
                if is_in_flight(owner_id, idempotency_key) and cur_worker == worker_id:
                    raise ImageProviderError(
                        "idempotency_in_progress",
                        "A request with this idempotency key is currently processing.",
                        409,
                    )
                is_stale = _lease_is_stale(now, cur_lease_exp, created_at, lease_seconds)

                if is_stale:
                    new_fence = cur_fence + 1
                    conn.execute(
                        "UPDATE image_idempotency "
                        "SET status='stale_expired', fencing_token=?, updated_at=CURRENT_TIMESTAMP "
                        "WHERE owner_id=? AND idempotency_key=?",
                        (new_fence, owner_id, idempotency_key),
                    )
                    conn.commit()
                    raise ImageProviderError(
                        "idempotency_needs_reconciliation",
                        "Previous generation attempt timed out and stale lock has been reconciled. Retry to initiate a fresh request.",
                        409,
                    )

                raise ImageProviderError(
                    "idempotency_in_progress",
                    f"A request with this idempotency key is currently processing by worker {cur_worker or 'active'}.",
                    409,
                )

            new_fence = cur_fence + 1
            conn.execute(
                "UPDATE image_idempotency "
                "SET status='pending', request_hash=?, worker_id=?, lease_expires_at=?, fencing_token=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE owner_id=? AND idempotency_key=?",
                (request_hash, worker_id, lease_exp_str, new_fence, owner_id, idempotency_key),
            )
            conn.commit()
            return None, new_fence, worker_id

        conn.execute(
            "INSERT INTO image_idempotency ("
            "    owner_id, idempotency_key, request_hash, status, response_json, "
            "    created_at, worker_id, lease_expires_at, fencing_token, updated_at"
            ") VALUES (?, ?, ?, 'pending', NULL, CURRENT_TIMESTAMP, ?, ?, 1, CURRENT_TIMESTAMP)",
            (owner_id, idempotency_key, request_hash, worker_id, lease_exp_str),
        )
        conn.commit()
        return None, 1, worker_id


def _complete_idempotency_sqlite(
    owner_id: str,
    idempotency_key: str,
    response_json_str: str,
    worker_id: str | None = None,
    fencing_token: int | None = None,
) -> bool:
    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        query = (
            "UPDATE image_idempotency "
            "SET status='completed', response_json=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE owner_id=? AND idempotency_key=? AND status IN ('pending','indeterminate')"
        )
        params = [response_json_str, owner_id, idempotency_key]
        if worker_id is not None:
            query += " AND worker_id=?"
            params.append(worker_id)
        if fencing_token is not None:
            query += " AND fencing_token=?"
            params.append(fencing_token)

        cursor = conn.execute(query, tuple(params))
        conn.commit()
        if cursor.rowcount == 0:
            logger.warning(
                f"[FencingViolationGuard] Completion rejected for key {idempotency_key}: "
                f"worker_id={worker_id}, fencing_token={fencing_token}."
            )
            return False
        return True


def _fail_idempotency_sqlite(
    owner_id: str,
    idempotency_key: str,
    worker_id: str | None = None,
    fencing_token: int | None = None,
) -> bool:
    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        query = (
            "UPDATE image_idempotency "
            "SET status='failed', updated_at=CURRENT_TIMESTAMP "
            "WHERE owner_id=? AND idempotency_key=? AND status='pending'"
        )
        params = [owner_id, idempotency_key]
        if worker_id is not None:
            query += " AND worker_id=?"
            params.append(worker_id)
        if fencing_token is not None:
            query += " AND fencing_token=?"
            params.append(fencing_token)

        cursor = conn.execute(query, tuple(params))
        conn.commit()
        return cursor.rowcount > 0


# ─── Public Distributed Idempotency API ──────────────────────────────────────

def claim_idempotency_lease(
    owner_id: str,
    idempotency_key: str,
    request_hash: str,
    worker_id: str | None = None,
    lease_seconds: int = 180,
) -> tuple[dict | None, int | None, str]:
    """Atomically claim a distributed generation lease with fencing tokens.

    Returns:
        (cached_response, fencing_token, active_worker_id)
        If cached_response is not None, a previously completed generation was replayed.
        If cached_response is None, a new lease was successfully claimed with fencing_token.
    """
    effective_worker = worker_id or PROCESS_WORKER_ID
    from app.core.maintenance import is_maintenance_mode, MaintenanceModeError
    if is_maintenance_mode():
        record = get_idempotency_record(owner_id, idempotency_key)
        if record and record.get("status") == "completed":
            resp_str = record.get("response_json") or "{}"
            resp = json.loads(resp_str) if isinstance(resp_str, str) else resp_str
            return resp, record.get("fencing_token", 1), record.get("worker_id") or PROCESS_WORKER_ID
        raise MaintenanceModeError("System is in maintenance mode: new generation leases cannot be claimed.")

    if is_postgres_configured():
        return run_sync_db(
            _claim_idempotency_lease_postgres(
                owner_id, idempotency_key, request_hash, effective_worker, lease_seconds
            )
        )
    return _claim_idempotency_lease_sqlite(
        owner_id, idempotency_key, request_hash, effective_worker, lease_seconds
    )


def check_or_claim_idempotency(
    owner_id: str,
    idempotency_key: str,
    request_hash: str,
    worker_id: str | None = None,
    lease_seconds: int = 180,
) -> dict | None:
    """Backward-compatible wrapper returning cached dictionary or None upon successful claim."""
    cached_data, _fence, _worker = claim_idempotency_lease(
        owner_id, idempotency_key, request_hash, worker_id, lease_seconds
    )
    return cached_data


def complete_idempotency(
    owner_id: str,
    idempotency_key: str,
    response_data: dict,
    worker_id: str | None = None,
    fencing_token: int | None = None,
) -> bool:
    """Record completed response with conditional fencing validation.

    Ensures old workers whose leases expired or were superseded cannot overwrite newer claims.
    """
    json_str = json.dumps(response_data)
    if is_postgres_configured():
        return run_sync_db(
            _complete_idempotency_postgres(
                owner_id, idempotency_key, json_str, worker_id, fencing_token
            )
        )
    return _complete_idempotency_sqlite(
        owner_id, idempotency_key, json_str, worker_id, fencing_token
    )


def fail_idempotency(
    owner_id: str,
    idempotency_key: str,
    worker_id: str | None = None,
    fencing_token: int | None = None,
) -> bool:
    """Mark an idempotency claim as failed with conditional fencing validation."""
    if is_postgres_configured():
        return run_sync_db(
            _fail_idempotency_postgres(owner_id, idempotency_key, worker_id, fencing_token)
        )
    return _fail_idempotency_sqlite(owner_id, idempotency_key, worker_id, fencing_token)


def heartbeat_idempotency_lease(
    owner_id: str,
    idempotency_key: str,
    worker_id: str,
    fencing_token: int,
    extension_seconds: int = 180,
) -> bool:
    """Extend lease expiration for long-running generation tasks holding a valid fencing token."""
    if is_postgres_configured():
        return run_sync_db(
            _heartbeat_idempotency_lease_postgres(
                owner_id, idempotency_key, worker_id, fencing_token, extension_seconds
            )
        )
    now = datetime.now(timezone.utc)
    new_exp = now + timedelta(seconds=extension_seconds)
    exp_str = new_exp.strftime("%Y-%m-%d %H:%M:%S")
    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        res = conn.execute(
            "UPDATE image_idempotency "
            "SET lease_expires_at=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE owner_id=? AND idempotency_key=? AND status IN ('pending','indeterminate') "
            "  AND worker_id=? AND fencing_token=?",
            (exp_str, owner_id, idempotency_key, worker_id, fencing_token),
        )
        conn.commit()
        return res.rowcount > 0


def reconcile_stale_idempotency(owner_id: str, idempotency_key: str, force: bool = False) -> dict[str, Any]:
    """Reconcile or release an orphaned/stale idempotency key deterministically."""
    from app.core.maintenance import is_maintenance_mode
    if is_maintenance_mode() and not force:
        logger.info("[MAINTENANCE] Skipping stale idempotency reconciliation during maintenance mode.")
        return {"status": "maintenance_blocked", "reconciled": False}

    if is_postgres_configured():
        return run_sync_db(_reconcile_stale_idempotency_postgres(owner_id, idempotency_key))

    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, response_json, fencing_token, lease_expires_at, created_at FROM image_idempotency WHERE owner_id=? AND idempotency_key=?",
            (owner_id, idempotency_key),
        ).fetchone()
        if not row:
            return {"status": "not_found", "reconciled": False}
        status, resp_json, cur_fence = row[0], row[1], row[2] or 1
        if status == "completed":
            return {"status": "completed", "reconciled": False, "response": resp_json, "fencing_token": cur_fence}
        if status != "pending" or not _lease_is_stale(datetime.now(timezone.utc), row[3], row[4]):
            return {"status": status, "reconciled": False, "fencing_token": cur_fence}
        new_fence = cur_fence + 1
        conn.execute(
            "UPDATE image_idempotency "
            "SET status='stale_expired', fencing_token=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE owner_id=? AND idempotency_key=?",
            (new_fence, owner_id, idempotency_key),
        )
        conn.commit()
        return {"status": "stale_expired", "reconciled": True, "fencing_token": new_fence}


def get_idempotency_record(owner_id: str, idempotency_key: str) -> dict[str, Any] | None:
    """Retrieve diagnostic idempotency record details."""
    if is_postgres_configured():
        async def _query():
            async with get_async_session() as session:
                res = await session.execute(
                    text(
                        "SELECT owner_id, idempotency_key, request_hash, status, response_json, "
                        "       created_at, worker_id, lease_expires_at, fencing_token, updated_at, operation_id, external_state, provider_model, provider_attempts "
                        "FROM image_idempotency WHERE owner_id=:o AND idempotency_key=:k"
                    ),
                    {"o": owner_id, "k": idempotency_key},
                )
                row = res.fetchone()
                if not row:
                    return None
                return dict(row._mapping)
        return run_sync_db(_query)

    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        row = conn.execute(
            "SELECT owner_id, idempotency_key, request_hash, status, response_json, "
            "       created_at, worker_id, lease_expires_at, fencing_token, updated_at, operation_id, external_state, provider_model, provider_attempts "
            "FROM image_idempotency WHERE owner_id=? AND idempotency_key=?",
            (owner_id, idempotency_key),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def begin_external_dispatch(owner: str, key: str, worker: str, fence: int, model: str) -> str:
    """Commit uncertainty before any potentially billable call. Never replay unknown work."""
    operation_id = str(uuid.uuid4())
    query = (
        "UPDATE image_idempotency SET status='indeterminate', external_state='dispatched', "
        "operation_id=COALESCE(operation_id,:operation), provider_model=:model, provider_attempts=provider_attempts+1 "
        "WHERE owner_id=:owner AND idempotency_key=:key AND worker_id=:worker AND fencing_token=:fence "
        "AND status='pending' AND lease_expires_at > :now"
    )
    params = {"owner": owner, "key": key, "worker": worker, "fence": fence, "operation": operation_id,
              "model": model, "now": datetime.now(timezone.utc)}
    if is_postgres_configured():
        async def _dispatch():
            async with get_async_session() as session, session.begin():
                result = await session.execute(text(query), params)
                if result.rowcount != 1:
                    raise ImageProviderError("operation_indeterminate", "Dispatch permission is unavailable. No generation was repeated.")
                return (await session.execute(text("SELECT operation_id FROM image_idempotency WHERE owner_id=:owner AND idempotency_key=:key"), params)).scalar_one()
        return run_sync_db(_dispatch())
    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        # SQLite timestamps are stored without a UTC offset by lease helpers.
        params["now"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        query = query.replace("lease_expires_at > :now", "julianday(lease_expires_at) > julianday(:now)")
        if conn.execute(query, params).rowcount != 1:
            raise ImageProviderError("operation_indeterminate", "Dispatch permission is unavailable. No generation was repeated.")
        return conn.execute("SELECT operation_id FROM image_idempotency WHERE owner_id=:owner AND idempotency_key=:key", params).fetchone()[0]


def record_external_outcome(owner: str, key: str, worker: str, fence: int, outcome: str) -> bool:
    if outcome not in {"confirmed_failure", "succeeded", "indeterminate"}:
        raise ValueError("Invalid external outcome")
    query = (
        "UPDATE image_idempotency SET external_state=:outcome, status=:status "
        "WHERE owner_id=:owner AND idempotency_key=:key AND worker_id=:worker AND fencing_token=:fence "
        "AND status='indeterminate'"
    )
    params = {"outcome": outcome, "status": "pending" if outcome == "confirmed_failure" else "indeterminate",
              "owner": owner, "key": key, "worker": worker, "fence": fence}
    if is_postgres_configured():
        async def _record():
            async with get_async_session() as session, session.begin():
                return (await session.execute(text(query), params)).rowcount == 1
        return run_sync_db(_record())
    with workspace_store._get_sqlite_connection() as conn:
        _ensure_sqlite_idempotency_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        return conn.execute(query, params).rowcount == 1
