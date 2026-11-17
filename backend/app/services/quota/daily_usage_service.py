"""Durable daily usage tracking and free-tier cost protection.

Tracks daily resource consumption per authenticated user per UTC day in SQLite and PostgreSQL.
Enforces beta limits:
  - 20 chat prompts / user / UTC day
  - 3 web searches / user / UTC day
  - 5 document uploads / user / UTC day
  - 120 audio seconds / user / UTC day
  - Global project-wide daily prompt cap
Provides atomic admission checks and fail-closed protection against upstream billing spikes.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.services.workspace_store import workspace_store
from sqlalchemy import text

from app.core.exceptions import BaseAppException

logger = logging.getLogger("chatbot.daily_quota")

DAILY_USAGE_DDL = (
    "CREATE TABLE IF NOT EXISTS user_daily_usage ("
    "user_id VARCHAR(128) NOT NULL, "
    "usage_date VARCHAR(10) NOT NULL, "
    "chat_prompts INTEGER NOT NULL DEFAULT 0, "
    "web_searches INTEGER NOT NULL DEFAULT 0, "
    "document_uploads INTEGER NOT NULL DEFAULT 0, "
    "audio_seconds INTEGER NOT NULL DEFAULT 0, "
    "provider_calls INTEGER NOT NULL DEFAULT 0, "
    "provider_errors INTEGER NOT NULL DEFAULT 0, "
    "last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (user_id, usage_date)"
    ")"
)


class DailyQuotaExceededError(BaseAppException):
    """Raised when an operation exceeds daily free-tier user or project limits."""

    def __init__(
        self,
        quota_type: str,
        limit: int,
        used: int,
        retry_after_seconds: int,
        message: Optional[str] = None,
    ):
        msg = message or f"Daily {quota_type} limit reached ({used}/{limit}). Resets at 00:00 UTC."
        super().__init__(
            detail=msg,
            status_code=429,
            error_code=f"DAILY_{quota_type.upper()}_EXCEEDED",
            title="Daily Usage Quota Exceeded",
        )
        self.quota_type = quota_type
        self.limit = limit
        self.used = used
        self.retry_after_seconds = retry_after_seconds
        self.message = msg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "daily_quota_exceeded",
            "quota_type": self.quota_type,
            "limit": self.limit,
            "used": self.used,
            "retry_after_seconds": self.retry_after_seconds,
            "message": self.message,
        }


def get_current_utc_date_str() -> str:
    """Return current UTC date in YYYY-MM-DD format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_seconds_until_utc_midnight() -> int:
    """Calculate remaining seconds until 00:00:00 UTC."""
    now = datetime.now(timezone.utc)
    # Target next midnight
    from datetime import time, timedelta
    tomorrow = (now + timedelta(days=1)).date()
    midnight = datetime.combine(tomorrow, time.min, tzinfo=timezone.utc)
    diff = int((midnight - now).total_seconds())
    return max(1, diff)


def ensure_sqlite_daily_usage_schema(conn) -> None:
    conn.execute(DAILY_USAGE_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_daily_usage_date ON user_daily_usage(usage_date)"
    )


class _SQLiteDailySession:
    def __init__(self, conn):
        self.conn = conn

    async def execute(self, statement, params=None):
        values = {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in (params or {}).items()
        }
        return self.conn.execute(str(statement), values)


@asynccontextmanager
async def _daily_usage_transaction():
    if is_postgres_configured():
        async with get_async_session() as session, session.begin():
            yield session
    else:
        conn = workspace_store._get_sqlite_connection()
        try:
            ensure_sqlite_daily_usage_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            yield _SQLiteDailySession(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


async def async_get_daily_usage(user_id: str, date_str: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve usage numbers for a given user and UTC date."""
    target_date = date_str or get_current_utc_date_str()
    async with _daily_usage_transaction() as session:
        res = await session.execute(
            text(
                "SELECT chat_prompts, web_searches, document_uploads, audio_seconds, "
                "provider_calls, provider_errors "
                "FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )
        row = res.fetchone()
        if not row:
            return {
                "user_id": user_id,
                "usage_date": target_date,
                "chat_prompts": 0,
                "web_searches": 0,
                "document_uploads": 0,
                "audio_seconds": 0,
                "provider_calls": 0,
                "provider_errors": 0,
            }
        return {
            "user_id": user_id,
            "usage_date": target_date,
            "chat_prompts": int(row[0]),
            "web_searches": int(row[1]),
            "document_uploads": int(row[2]),
            "audio_seconds": int(row[3]),
            "provider_calls": int(row[4]),
            "provider_errors": int(row[5]),
        }


async def lock_daily_usage_global(session, target_date: str) -> None:
    """Serialize all quota admission writers for target_date on __global__ sentinel row."""
    await session.execute(
        text(
            "INSERT INTO user_daily_usage("
            "user_id, usage_date, chat_prompts, web_searches, document_uploads, audio_seconds, provider_calls, provider_errors"
            ") VALUES ('__global__', :dt, 0, 0, 0, 0, 0, 0) "
            "ON CONFLICT(user_id, usage_date) DO NOTHING"
        ),
        {"dt": target_date},
    )
    suffix = "" if isinstance(session, _SQLiteDailySession) else " FOR UPDATE"
    await session.execute(
        text("SELECT chat_prompts FROM user_daily_usage WHERE user_id='__global__' AND usage_date=:dt" + suffix),
        {"dt": target_date},
    )


async def lock_daily_usage_user(session, user_id: str, target_date: str) -> None:
    """Serialize per-user quota admission writers on the user's row."""
    await session.execute(
        text(
            "INSERT INTO user_daily_usage("
            "user_id, usage_date, chat_prompts, web_searches, document_uploads, audio_seconds, provider_calls, provider_errors"
            ") VALUES (:uid, :dt, 0, 0, 0, 0, 0, 0) "
            "ON CONFLICT(user_id, usage_date) DO NOTHING"
        ),
        {"uid": user_id, "dt": target_date},
    )
    suffix = "" if isinstance(session, _SQLiteDailySession) else " FOR UPDATE"
    await session.execute(
        text("SELECT chat_prompts FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt" + suffix),
        {"uid": user_id, "dt": target_date},
    )


async def async_check_and_consume_chat_prompt(user_id: str) -> Dict[str, Any]:
    """Atomically record chat prompt admission with concurrency protection."""
    target_date = get_current_utc_date_str()
    retry_after = get_seconds_until_utc_midnight()
    enable_prompt_limit = getattr(settings, "ENABLE_DAILY_PROMPT_LIMIT", False)

    if not enable_prompt_limit:
        # Fixed application-level daily chat prompt-count limit is disabled for public beta.
        # Record usage atomically for auditing and telemetry without global locks, blocking, or daily caps.
        used_count = 1
        try:
            async with _daily_usage_transaction() as session:
                await session.execute(
                    text(
                        "INSERT INTO user_daily_usage ("
                        "user_id, usage_date, chat_prompts, web_searches, document_uploads, "
                        "audio_seconds, provider_calls, provider_errors, last_updated"
                        ") VALUES (:uid, :dt, 1, 0, 0, 0, 1, 0, CURRENT_TIMESTAMP) "
                        "ON CONFLICT(user_id, usage_date) DO UPDATE SET "
                        "chat_prompts = user_daily_usage.chat_prompts + 1, "
                        "provider_calls = user_daily_usage.provider_calls + 1, "
                        "last_updated = CURRENT_TIMESTAMP"
                    ),
                    {"uid": user_id, "dt": target_date},
                )
                res = await session.execute(
                    text("SELECT chat_prompts FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt"),
                    {"uid": user_id, "dt": target_date},
                )
                row = res.fetchone()
                if row and row[0] is not None:
                    used_count = int(row[0])
        except Exception as track_err:
            logger.warning(f"Non-blocking daily usage tracking failed for {user_id}: {track_err}")

        return {
            "allowed": True,
            "quota_type": "chat_prompts",
            "used": used_count,
            "limit": None,
            "remaining": None,
            "retry_after_seconds": 0,
        }

    user_limit = getattr(settings, "DAILY_PROMPTS_PER_USER", 20)
    global_limit = getattr(settings, "DAILY_GLOBAL_PROMPT_BUDGET", 500)

    async with _daily_usage_transaction() as session:
        # Lock global sentinel first, then user row (strict lock ordering prevents deadlocks)
        await lock_daily_usage_global(session, target_date)
        await lock_daily_usage_user(session, user_id, target_date)

        # 1. Project-wide global daily cap check (excluding sentinel row)
        proj_res = await session.execute(
            text(
                "SELECT COALESCE(SUM(chat_prompts), 0) FROM user_daily_usage "
                "WHERE usage_date=:dt AND user_id != '__global__'"
            ),
            {"dt": target_date},
        )
        proj_row = proj_res.fetchone()
        proj_total = int(proj_row[0]) if proj_row and proj_row[0] is not None else 0
        if proj_total >= global_limit:
            raise DailyQuotaExceededError(
                quota_type="global_prompt_budget",
                limit=global_limit,
                used=proj_total,
                retry_after_seconds=retry_after,
                message=f"Global service daily prompt capacity reached ({proj_total}/{global_limit}). Resets at 00:00 UTC.",
            )

        # 2. Per-user atomic check & increment
        res = await session.execute(
            text(
                "SELECT chat_prompts FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )
        row = res.fetchone()
        current_used = int(row[0]) if row and row[0] is not None else 0

        if current_used >= user_limit:
            raise DailyQuotaExceededError(
                quota_type="chat_prompts",
                limit=user_limit,
                used=current_used,
                retry_after_seconds=retry_after,
                message=f"Daily chat prompt limit reached ({current_used}/{user_limit}). Resets at 00:00 UTC.",
            )

        # Increment user usage row
        await session.execute(
            text(
                "UPDATE user_daily_usage SET "
                "chat_prompts = user_daily_usage.chat_prompts + 1, "
                "provider_calls = user_daily_usage.provider_calls + 1, "
                "last_updated = CURRENT_TIMESTAMP "
                "WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )

        return {
            "allowed": True,
            "quota_type": "chat_prompts",
            "used": current_used + 1,
            "limit": user_limit,
            "remaining": max(0, user_limit - (current_used + 1)),
            "retry_after_seconds": retry_after,
        }


async def async_refund_chat_prompt(user_id: str) -> None:
    """Refund prompt reservation if operation failed without calling external model."""
    target_date = get_current_utc_date_str()
    async with _daily_usage_transaction() as session:
        await session.execute(
            text(
                "UPDATE user_daily_usage SET "
                "chat_prompts = CASE WHEN chat_prompts > 0 THEN chat_prompts - 1 ELSE 0 END, "
                "provider_calls = CASE WHEN provider_calls > 0 THEN provider_calls - 1 ELSE 0 END, "
                "last_updated = CURRENT_TIMESTAMP "
                "WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )


GUEST_SEARCH_BUCKET = "anonymous"


async def async_check_and_consume_web_search(user_id: str) -> Dict[str, Any]:
    """Atomically verify and record web search admission."""
    target_date = get_current_utc_date_str()
    user_limit = (
        getattr(settings, "DAILY_GUEST_SEARCH_POOL", 20)
        if user_id == GUEST_SEARCH_BUCKET
        else getattr(settings, "DAILY_SEARCHES_PER_USER", 3)
    )
    retry_after = get_seconds_until_utc_midnight()

    async with _daily_usage_transaction() as session:
        await lock_daily_usage_user(session, user_id, target_date)
        res = await session.execute(
            text(
                "SELECT web_searches FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )
        row = res.fetchone()
        current_used = int(row[0]) if row and row[0] is not None else 0

        if current_used >= user_limit:
            raise DailyQuotaExceededError(
                quota_type="web_searches",
                limit=user_limit,
                used=current_used,
                retry_after_seconds=retry_after,
                message=f"Daily web search limit reached ({current_used}/{user_limit}). Resets at 00:00 UTC.",
            )

        await session.execute(
            text(
                "UPDATE user_daily_usage SET "
                "web_searches = user_daily_usage.web_searches + 1, "
                "last_updated = CURRENT_TIMESTAMP "
                "WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )

        return {
            "allowed": True,
            "quota_type": "web_searches",
            "used": current_used + 1,
            "limit": user_limit,
            "remaining": max(0, user_limit - (current_used + 1)),
            "retry_after_seconds": retry_after,
        }


async def async_check_and_consume_document_upload(user_id: str) -> Dict[str, Any]:
    """Atomically verify and record daily document upload count."""
    target_date = get_current_utc_date_str()
    user_limit = getattr(settings, "DAILY_UPLOADS_PER_USER", 5)
    retry_after = get_seconds_until_utc_midnight()

    async with _daily_usage_transaction() as session:
        await lock_daily_usage_user(session, user_id, target_date)
        res = await session.execute(
            text(
                "SELECT document_uploads FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )
        row = res.fetchone()
        current_used = int(row[0]) if row and row[0] is not None else 0

        if current_used >= user_limit:
            raise DailyQuotaExceededError(
                quota_type="document_uploads",
                limit=user_limit,
                used=current_used,
                retry_after_seconds=retry_after,
                message=f"Daily document upload limit reached ({current_used}/{user_limit}). Resets at 00:00 UTC.",
            )

        await session.execute(
            text(
                "UPDATE user_daily_usage SET "
                "document_uploads = user_daily_usage.document_uploads + 1, "
                "last_updated = CURRENT_TIMESTAMP "
                "WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )

        return {
            "allowed": True,
            "quota_type": "document_uploads",
            "used": current_used + 1,
            "limit": user_limit,
            "remaining": max(0, user_limit - (current_used + 1)),
            "retry_after_seconds": retry_after,
        }


async def async_check_and_consume_audio_seconds(user_id: str, seconds: int, limit: int | None = None) -> dict[str, Any]:
    """Atomically verify and record daily audio transcription duration (limit overrides the per-user default)."""
    target_date = get_current_utc_date_str()
    user_limit = limit if limit is not None else getattr(settings, "DAILY_AUDIO_SECONDS_PER_USER", 120)
    retry_after = get_seconds_until_utc_midnight()

    async with _daily_usage_transaction() as session:
        await lock_daily_usage_user(session, user_id, target_date)
        res = await session.execute(
            text(
                "SELECT audio_seconds FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date},
        )
        row = res.fetchone()
        current_used = int(row[0]) if row and row[0] is not None else 0

        if current_used + seconds > user_limit:
            raise DailyQuotaExceededError(
                quota_type="audio_seconds",
                limit=user_limit,
                used=current_used,
                retry_after_seconds=retry_after,
                message=f"Daily voice transcription limit reached ({current_used}s/{user_limit}s). Resets at 00:00 UTC.",
            )

        await session.execute(
            text(
                "UPDATE user_daily_usage SET "
                "audio_seconds = user_daily_usage.audio_seconds + :sec, "
                "last_updated = CURRENT_TIMESTAMP "
                "WHERE user_id=:uid AND usage_date=:dt"
            ),
            {"uid": user_id, "dt": target_date, "sec": seconds},
        )

        return {
            "allowed": True,
            "quota_type": "audio_seconds",
            "used": current_used + seconds,
            "limit": user_limit,
            "remaining": max(0, user_limit - (current_used + seconds)),
            "retry_after_seconds": retry_after,
        }


# Synchronous convenience wrappers
def get_daily_usage(user_id: str, date_str: Optional[str] = None) -> Dict[str, Any]:
    return run_sync_db(async_get_daily_usage, user_id, date_str)


def check_and_consume_chat_prompt(user_id: str) -> Dict[str, Any]:
    return run_sync_db(async_check_and_consume_chat_prompt, user_id)


def refund_chat_prompt(user_id: str) -> None:
    return run_sync_db(async_refund_chat_prompt, user_id)


def check_and_consume_web_search(user_id: str) -> Dict[str, Any]:
    return run_sync_db(async_check_and_consume_web_search, user_id)


def check_and_consume_document_upload(user_id: str) -> Dict[str, Any]:
    return run_sync_db(async_check_and_consume_document_upload, user_id)


def check_and_consume_audio_seconds(user_id: str, seconds: int) -> Dict[str, Any]:
    return run_sync_db(async_check_and_consume_audio_seconds, user_id, seconds)
