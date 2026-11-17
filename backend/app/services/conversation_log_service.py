"""Asynchronous Conversation Logging & Feedback Repository.

Logs all chat interactions (query, retrieved RAG chunks, provider, response, latency)
to PostgreSQL (production) or SQLite (development/fallback).
Provides feedback capture (up/down/regenerated) and shadow evaluation telemetry.
"""
import json
import logging
import os
import queue
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import delete, func, select, update

from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.models import ConversationLog, User

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "chatbot.db"
)


class ConversationLogService:
    """Asynchronous conversation logger and feedback tracker with PostgreSQL & SQLite support."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._queue: queue.Queue = queue.Queue()
        if not is_postgres_configured():
            self._init_sqlite_db()
        self._start_worker()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_sqlite_db(self) -> None:
        """Create conversation_logs SQLite table if it does not exist and ensure shadow/AB test columns."""
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_logs (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        user_query TEXT NOT NULL,
                        retrieved_chunks TEXT NOT NULL DEFAULT '[]',
                        provider_used TEXT NOT NULL,
                        model_response TEXT NOT NULL,
                        response_time_ms REAL NOT NULL,
                        feedback TEXT DEFAULT NULL CHECK (feedback IN ('up', 'down', 'regenerated') OR feedback IS NULL),
                        shadow_response TEXT DEFAULT NULL,
                        shadow_model_id TEXT DEFAULT NULL,
                        served_model_version TEXT DEFAULT NULL,
                        ab_test_group TEXT DEFAULT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conv_logs_timestamp ON conversation_logs (timestamp)
                """)

                cursor = conn.execute("PRAGMA table_info(conversation_logs)")
                existing_cols = {row["name"] for row in cursor.fetchall()}
                new_cols = [
                    ("shadow_response", "TEXT DEFAULT NULL"),
                    ("shadow_model_id", "TEXT DEFAULT NULL"),
                    ("served_model_version", "TEXT DEFAULT NULL"),
                    ("ab_test_group", "TEXT DEFAULT NULL"),
                    ("prompt_tokens", "INTEGER DEFAULT NULL"),
                    ("completion_tokens", "INTEGER DEFAULT NULL"),
                    ("cost_usd", "REAL DEFAULT NULL"),
                    ("confidence_score", "REAL DEFAULT NULL"),
                    ("user_id", "TEXT DEFAULT NULL"),
                ]
                for col_name, col_def in new_cols:
                    if col_name not in existing_cols:
                        conn.execute(f"ALTER TABLE conversation_logs ADD COLUMN {col_name} {col_def}")
                        logger.info(f"Migrated conversation_logs schema: added {col_name}")

                if "user_id" not in existing_cols:
                    conn.execute("UPDATE conversation_logs SET user_query = '', model_response = '', retrieved_chunks = '[]', shadow_response = NULL")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_logs_user_id ON conversation_logs(user_id)")
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to initialize SQLite conversation_logs table: {exc}")

    def _start_worker(self) -> None:
        """Daemon worker processing log insertions asynchronously."""
        def worker():
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    action = item.get("action", "insert")
                    if action == "insert":
                        if is_postgres_configured():
                            run_sync_db(self._async_insert_log_postgres(item))
                        else:
                            self._insert_log_sqlite(item)
                    elif action == "update_shadow":
                        if is_postgres_configured():
                            run_sync_db(self.async_update_shadow(
                                item["id"], item["shadow_response"], item["shadow_model_id"]
                            ))
                        else:
                            self._update_shadow_sqlite(item)
                except Exception as exc:
                    logger.error(f"Error processing conversation log queue: {exc}")
                finally:
                    self._queue.task_done()

        thread = threading.Thread(target=worker, daemon=True, name="conv-log-worker")
        thread.start()

    # ─── SQLite Implementations ──────────────────────────────────────────────

    def _insert_log_sqlite(self, data: Dict[str, Any]) -> None:
        if not data.get("user_id"):
            return
        with self._get_sqlite_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            user = conn.execute("SELECT deletion_pending FROM users WHERE id = ?", (data["user_id"],)).fetchone()
            if not user or user["deletion_pending"]:
                return
            existing = conn.execute("SELECT user_id FROM conversation_logs WHERE id = ?", (data["id"],)).fetchone()
            if existing and existing["user_id"] != data["user_id"]:
                return
            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_logs (
                    id, user_id, timestamp, user_query, retrieved_chunks,
                    provider_used, model_response, response_time_ms, feedback,
                    shadow_response, shadow_model_id, served_model_version, ab_test_group,
                    prompt_tokens, completion_tokens, cost_usd, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["user_id"],
                    data["timestamp"],
                    data["user_query"],
                    json.dumps(data["retrieved_chunks"], ensure_ascii=False) if isinstance(data["retrieved_chunks"], (list, dict)) else str(data["retrieved_chunks"]),
                    data["provider_used"],
                    data["model_response"],
                    data["response_time_ms"],
                    data.get("feedback"),
                    data.get("shadow_response"),
                    data.get("shadow_model_id"),
                    data.get("served_model_version"),
                    data.get("ab_test_group"),
                    data.get("prompt_tokens"),
                    data.get("completion_tokens"),
                    data.get("cost_usd"),
                    data.get("confidence_score"),
                ),
            )
            conn.commit()

    def _update_shadow_sqlite(self, data: Dict[str, Any]) -> None:
        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                UPDATE conversation_logs
                SET shadow_response = ?, shadow_model_id = ?
                WHERE id = ?
                """,
                (None, data["shadow_model_id"], data["id"]),
            )
            conn.commit()

    # ─── PostgreSQL Async Implementations ────────────────────────────────────

    async def _async_insert_log_postgres(self, data: Dict[str, Any]) -> None:
        """Insert or replace a conversation log into PostgreSQL."""
        ts_val = data.get("timestamp")
        if isinstance(ts_val, str):
            try:
                dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now(timezone.utc)
        elif isinstance(ts_val, datetime):
            dt = ts_val
        else:
            dt = datetime.now(timezone.utc)

        chunks = data.get("retrieved_chunks")
        if isinstance(chunks, (list, dict)):
            chunks_str = json.dumps(chunks, ensure_ascii=False)
        else:
            chunks_str = str(chunks or "[]")

        async with get_async_session() as session:
            async with session.begin():
                if not data.get("user_id"):
                    return
                user = await session.scalar(select(User).where(User.id == data["user_id"]).with_for_update())
                if not user or user.deletion_pending:
                    return
                stmt = select(ConversationLog).where(ConversationLog.id == data["id"])
                res = await session.execute(stmt)
                record = res.scalar_one_or_none()
                if record and record.user_id != data["user_id"]:
                    return
                if record:
                    record.timestamp = dt
                    record.user_query = data["user_query"]
                    record.retrieved_chunks = chunks_str
                    record.provider_used = data["provider_used"]
                    record.model_response = data["model_response"]
                    record.response_time_ms = data["response_time_ms"]
                    record.feedback = data.get("feedback")
                    record.shadow_response = data.get("shadow_response")
                    record.shadow_model_id = data.get("shadow_model_id")
                    record.served_model_version = data.get("served_model_version")
                    record.ab_test_group = data.get("ab_test_group")
                    record.prompt_tokens = data.get("prompt_tokens")
                    record.completion_tokens = data.get("completion_tokens")
                    record.cost_usd = data.get("cost_usd")
                    record.confidence_score = data.get("confidence_score")
                else:
                    new_log = ConversationLog(
                        id=data["id"],
                        user_id=data["user_id"],
                        timestamp=dt,
                        user_query=data["user_query"],
                        retrieved_chunks=chunks_str,
                        provider_used=data["provider_used"],
                        model_response=data["model_response"],
                        response_time_ms=data["response_time_ms"],
                        feedback=data.get("feedback"),
                        shadow_response=data.get("shadow_response"),
                        shadow_model_id=data.get("shadow_model_id"),
                        served_model_version=data.get("served_model_version"),
                        ab_test_group=data.get("ab_test_group"),
                        prompt_tokens=data.get("prompt_tokens"),
                        completion_tokens=data.get("completion_tokens"),
                        cost_usd=data.get("cost_usd"),
                        confidence_score=data.get("confidence_score"),
                    )
                    session.add(new_log)

    async def async_log(
        self,
        conversation_id: Optional[str],
        user_query: str,
        retrieved_chunks: List[Any],
        provider_used: str,
        model_response: str,
        response_time_ms: float,
        feedback: Optional[str] = None,
        shadow_response: Optional[str] = None,
        shadow_model_id: Optional[str] = None,
        served_model_version: Optional[str] = None,
        ab_test_group: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        confidence_score: Optional[float] = None,
        owner_id: Optional[str] = None,
    ) -> str:
        """Directly insert conversation log into PostgreSQL without relying on in-memory queues."""
        log_id = str(conversation_id or uuid4())
        normalized_chunks = []
        for c in retrieved_chunks:
            if hasattr(c, "model_dump"):
                normalized_chunks.append(c.model_dump())
            elif hasattr(c, "__dict__"):
                normalized_chunks.append(c.__dict__)
            elif isinstance(c, dict):
                normalized_chunks.append(c)
            else:
                normalized_chunks.append({"text": str(c)})

        data = {
            "id": log_id,
            "timestamp": datetime.now(timezone.utc),
            "user_query": "",
            "user_id": owner_id,
            "retrieved_chunks": [],
            "provider_used": provider_used or "unknown",
            "model_response": "",
            "response_time_ms": round(float(response_time_ms), 2),
            "feedback": feedback,
            "shadow_response": None,
            "shadow_model_id": shadow_model_id,
            "served_model_version": served_model_version,
            "ab_test_group": ab_test_group,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "confidence_score": confidence_score,
        }
        await self._async_insert_log_postgres(data)
        return log_id

    async def async_update_shadow(self, conversation_id: str, shadow_response: str, shadow_model_id: str) -> None:
        """Update shadow response in PostgreSQL."""
        async with get_async_session() as session:
            async with session.begin():
                stmt = (
                    update(ConversationLog)
                    .where(ConversationLog.id == conversation_id)
                    .values(shadow_response=None, shadow_model_id=shadow_model_id)
                )
                await session.execute(stmt)

    async def async_update_feedback(self, conversation_id: str, signal: str, owner_id: str) -> bool:
        """Update feedback signal in PostgreSQL."""
        if not owner_id or signal not in ("up", "down", "regenerated"):
            return False
        async with get_async_session() as session:
            async with session.begin():
                stmt = (
                    update(ConversationLog)
                    .where(ConversationLog.id == conversation_id, ConversationLog.user_id == owner_id)
                    .values(feedback=signal)
                )
                res = await session.execute(stmt)
                return res.rowcount > 0

    async def async_fetch_shadow_comparisons(self) -> List[Dict[str, Any]]:
        """Retrieve shadow comparison logs from PostgreSQL."""
        async with get_async_session() as session:
            stmt = (
                select(ConversationLog)
                .where(ConversationLog.shadow_response.isnot(None), ConversationLog.shadow_response != "")
                .order_by(ConversationLog.timestamp.desc())
            )
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "user_query": r.user_query,
                    "provider_used": r.provider_used,
                    "model_response": r.model_response,
                    "response_time_ms": r.response_time_ms,
                    "feedback": r.feedback,
                    "shadow_response": r.shadow_response,
                    "shadow_model_id": r.shadow_model_id,
                    "served_model_version": r.served_model_version,
                    "ab_test_group": r.ab_test_group,
                }
                for r in rows
            ]

    async def async_fetch_version_feedback_stats(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate feedback ratings grouped by model version from PostgreSQL."""
        async with get_async_session() as session:
            version_col = func.coalesce(ConversationLog.served_model_version, "baseline_v0")
            stmt = (
                select(
                    version_col.label("version"),
                    func.count().label("total_requests"),
                    func.sum(func.case((ConversationLog.feedback == "up", 1), else_=0)).label("positive_feedback"),
                    func.sum(func.case((ConversationLog.feedback == "down", 1), else_=0)).label("negative_feedback"),
                    func.sum(func.case((ConversationLog.feedback == "regenerated", 1), else_=0)).label("regenerations"),
                    func.avg(ConversationLog.response_time_ms).label("avg_latency_ms"),
                )
                .group_by(version_col)
            )
            res = await session.execute(stmt)
            stats = {}
            for row in res.all():
                tot = row.total_requests or 1
                pos = row.positive_feedback or 0
                stats[row.version] = {
                    "version": row.version,
                    "total_requests": row.total_requests,
                    "positive_feedback": pos,
                    "negative_feedback": row.negative_feedback or 0,
                    "regenerations": row.regenerations or 0,
                    "avg_latency_ms": float(row.avg_latency_ms or 0.0),
                    "satisfaction_rate": round(pos / tot, 4),
                }
            return stats

    async def async_get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent conversation logs from PostgreSQL."""
        async with get_async_session() as session:
            stmt = (
                select(ConversationLog)
                .order_by(ConversationLog.timestamp.desc())
                .limit(limit)
            )
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "user_query": r.user_query,
                    "retrieved_chunks": r.retrieved_chunks,
                    "provider_used": r.provider_used,
                    "model_response": r.model_response,
                    "response_time_ms": r.response_time_ms,
                    "feedback": r.feedback,
                    "shadow_response": r.shadow_response,
                    "shadow_model_id": r.shadow_model_id,
                    "served_model_version": r.served_model_version,
                    "ab_test_group": r.ab_test_group,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "cost_usd": r.cost_usd,
                    "confidence_score": r.confidence_score,
                }
                for r in rows
            ]

    # ─── Public API (Sync facade routing to PostgreSQL or SQLite) ───────────

    def log_async(
        self,
        conversation_id: Optional[str],
        user_query: str,
        retrieved_chunks: List[Any],
        provider_used: str,
        model_response: str,
        response_time_ms: float,
        feedback: Optional[str] = None,
        shadow_response: Optional[str] = None,
        shadow_model_id: Optional[str] = None,
        served_model_version: Optional[str] = None,
        ab_test_group: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        confidence_score: Optional[float] = None,
        owner_id: Optional[str] = None,
    ) -> str:
        """Enqueue conversation log entry asynchronously."""
        log_id = str(conversation_id or uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        normalized_chunks = []
        for c in retrieved_chunks:
            if hasattr(c, "model_dump"):
                normalized_chunks.append(c.model_dump())
            elif hasattr(c, "__dict__"):
                normalized_chunks.append(c.__dict__)
            elif isinstance(c, dict):
                normalized_chunks.append(c)
            else:
                normalized_chunks.append({"text": str(c)})

        item = {
            "action": "insert",
            "id": log_id,
            "timestamp": timestamp,
            "user_query": "",
            "user_id": owner_id,
            "retrieved_chunks": [],
            "provider_used": provider_used or "unknown",
            "model_response": "",
            "response_time_ms": round(float(response_time_ms), 2),
            "feedback": feedback,
            "shadow_response": None,
            "shadow_model_id": shadow_model_id,
            "served_model_version": served_model_version,
            "ab_test_group": ab_test_group,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "confidence_score": confidence_score,
        }

        from app.services.tenant_access import active_account
        if active_account(owner_id):
            self._queue.put(item)
        return log_id

    def update_shadow_async(self, conversation_id: str, shadow_response: str, shadow_model_id: str) -> None:
        """Enqueue asynchronous update for shadow mode response."""
        self._queue.put({
            "action": "update_shadow",
            "id": conversation_id,
            "shadow_response": None,
            "shadow_model_id": shadow_model_id,
        })

    def update_feedback(self, conversation_id: str, signal: str, owner_id: Optional[str] = None) -> bool:
        """Update feedback signal ('up', 'down', 'regenerated') for a conversation log."""
        if not owner_id or signal not in ("up", "down", "regenerated"):
            return False

        if is_postgres_configured():
            return run_sync_db(self.async_update_feedback(conversation_id, signal, owner_id))

        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.execute(
                    "UPDATE conversation_logs SET feedback = ? WHERE id = ? AND user_id = ? AND EXISTS (SELECT 1 FROM users WHERE id = ? AND deletion_pending = 0)",
                    (signal, conversation_id, owner_id, owner_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as exc:
            logger.error(f"Failed to update feedback for {conversation_id}: {exc}")
            return False

    def fetch_shadow_comparisons(self) -> List[Dict[str, Any]]:
        """Retrieve all records with valid shadow evaluations for comparison analysis."""
        if is_postgres_configured():
            return run_sync_db(self.async_fetch_shadow_comparisons())

        with self._get_sqlite_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, user_query, provider_used, model_response,
                       response_time_ms, feedback, shadow_response, shadow_model_id,
                       served_model_version, ab_test_group
                FROM conversation_logs
                WHERE shadow_response IS NOT NULL AND shadow_response != ''
                ORDER BY timestamp DESC
                """
            )
            return [dict(r) for r in cursor.fetchall()]

    def fetch_version_feedback_stats(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate feedback ratings grouped by served model version for regression checks."""
        if is_postgres_configured():
            return run_sync_db(self.async_fetch_version_feedback_stats())

        with self._get_sqlite_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COALESCE(served_model_version, 'baseline_v0') as version,
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END) as positive_feedback,
                    SUM(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END) as negative_feedback,
                    SUM(CASE WHEN feedback = 'regenerated' THEN 1 ELSE 0 END) as regenerations,
                    AVG(response_time_ms) as avg_latency_ms
                FROM conversation_logs
                GROUP BY COALESCE(served_model_version, 'baseline_v0')
                """
            )
            stats = {}
            for row in cursor.fetchall():
                r = dict(row)
                tot = r["total_requests"] or 1
                pos = r["positive_feedback"] or 0
                r["satisfaction_rate"] = round(pos / tot, 4)
                stats[r["version"]] = r
            return stats

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent conversation logs ordered by timestamp descending."""
        if is_postgres_configured():
            return run_sync_db(self.async_get_recent_logs(limit))

        with self._get_sqlite_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, user_query, retrieved_chunks, provider_used,
                       model_response, response_time_ms, feedback, shadow_response,
                       shadow_model_id, served_model_version, ab_test_group,
                       prompt_tokens, completion_tokens, cost_usd, confidence_score
                FROM conversation_logs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]


conversation_log_service = ConversationLogService()
