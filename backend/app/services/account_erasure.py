"""Retryable account erasure: revoke access before clearing owned stores.

No cross-store atomicity is claimed. Failed cleanup keeps a deletion-pending
account inaccessible except to its authenticated erasure retry endpoint.
"""

import os
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import delete, select

from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.models import (
    ConversationLog,
    ImageRecord,
    LibraryFile,
    Message,
    Project,
    ScheduledTask,
    Session,
    TaskExecution,
    User,
    UserIntegration,
    UserMemory,
)


async def _mark_pending_postgres(user_id):
    async with get_async_session() as session, session.begin():
        user = await session.scalar(select(User).where(User.id == user_id).with_for_update())
        if not user:
            return False
        user.deletion_pending = True
    return True


async def _erase_postgres(user_id):
    async with get_async_session() as session, session.begin():
        from app.services.image.storage_quota import lock_storage_project
        await lock_storage_project(session)
        user = await session.scalar(select(User).where(User.id == user_id).with_for_update())
        if not user or not user.deletion_pending:
            raise RuntimeError("Missing deletion-pending account")
        sessions = select(Session.session_id).where(Session.user_id == user_id)
        tasks = select(ScheduledTask.id).where(ScheduledTask.user_id == user_id)
        await session.execute(delete(Message).where(Message.session_id.in_(sessions)))
        await session.execute(delete(TaskExecution).where(TaskExecution.task_id.in_(tasks)))
        try:
            from sqlalchemy import text
            u_bytes_res = await session.execute(
                text("SELECT COALESCE(SUM(bytes), 0) FROM image_records WHERE user_id = :u"),
                {"u": user_id},
            )
            u_bytes = u_bytes_res.scalar() or 0
            if u_bytes > 0:
                await session.execute(
                    text(
                        "UPDATE image_storage_quotas "
                        "SET total_bytes = GREATEST(0, total_bytes - :b), updated_at = NOW() "
                        "WHERE user_id = '__project__'"
                    ),
                    {"b": u_bytes},
                )
            await session.execute(text("DELETE FROM image_storage_quotas WHERE user_id = :u"), {"u": user_id})
            await session.execute(text("DELETE FROM image_storage_reservations WHERE user_id = :u"), {"u": user_id})
        except Exception:
            pass
        for model in (ConversationLog, LibraryFile, UserMemory, UserIntegration, ImageRecord, ScheduledTask, Project, Session):
            await session.execute(delete(model).where(model.user_id == user_id))
        try:
            from sqlalchemy import text
            await session.execute(text("DELETE FROM image_idempotency WHERE owner_id = :owner_id"), {"owner_id": user_id})
        except Exception:
            pass
        await session.delete(user)


async def _owned_assets_postgres(user_id):
    async with get_async_session() as session:
        files = (await session.scalars(select(LibraryFile).where(LibraryFile.user_id == user_id))).all()
        images = (await session.scalars(select(ImageRecord).where(ImageRecord.user_id == user_id))).all()
        return ([{"id": f.id, "storage_path": f.storage_path, "status": f.status} for f in files], [i.file_url for i in images])


def _owned_assets_sqlite(workspace, user_id):
    with workspace._get_sqlite_connection() as conn:
        files = [dict(row) for row in conn.execute("SELECT id, storage_path, status FROM library_files WHERE user_id = ?", (user_id,))]
        images = [row[0] for row in conn.execute("SELECT file_url FROM image_records WHERE user_id = ?", (user_id,))]
        return files, images


def _erase_local_image(storage, url):
    if url.startswith("/static/generated/"):
        base = os.path.realpath(Path(__file__).resolve().parents[1] / "static" / "generated")
        target = os.path.realpath(os.path.join(base, url.removeprefix("/static/generated/")))
        if os.path.commonpath([base, target]) != base or target == base:
            raise PermissionError("Image path escapes generated storage")
        if os.path.exists(target):
            os.remove(target)
        if os.path.exists(target):
            raise RuntimeError("Generated image remains after erasure")
    elif os.path.isabs(url) or url.startswith("supabase://") or not urlsplit(url).scheme:
        storage.delete_verified(url)
    # Externally hosted public media is subject to provider retention, not an
    # application-owned blob. Do not send deletion requests to arbitrary URLs.


def _erase_sqlite_database(path, user_id, delete_account=False):
    import sqlite3
    with sqlite3.connect(path, timeout=15) as conn:
        conn.execute("BEGIN IMMEDIATE")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "sessions" in tables and "messages" in tables:
            conn.execute("DELETE FROM messages WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)", (user_id,))
        if "scheduled_tasks" in tables and "task_executions" in tables:
            conn.execute("DELETE FROM task_executions WHERE task_id IN (SELECT id FROM scheduled_tasks WHERE user_id = ?)", (user_id,))
        if "image_records" in tables and "image_storage_quotas" in tables:
            u_bytes_row = conn.execute("SELECT COALESCE(SUM(bytes), 0) FROM image_records WHERE user_id = ?", (user_id,)).fetchone()
            u_bytes = u_bytes_row[0] if u_bytes_row else 0
            if "image_storage_reservations" in tables:
                u_bytes += conn.execute("SELECT COALESCE(SUM(bytes),0) FROM image_storage_reservations WHERE user_id=? AND state IN ('active','writing','cleanup')", (user_id,)).fetchone()[0]
            if u_bytes > 0:
                conn.execute(
                    "UPDATE image_storage_quotas SET total_bytes = MAX(0, total_bytes - ?), updated_at = CURRENT_TIMESTAMP WHERE user_id = '__project__'",
                    (u_bytes,),
                )
        for table in ("conversation_logs", "library_files", "user_memories", "user_integrations", "image_records", "scheduled_tasks", "projects", "sessions"):
            if table in tables:
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                if "user_id" in cols:
                    conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                    if conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)).fetchone()[0]:
                        raise RuntimeError("Owned records remain after erasure")
        if "image_storage_quotas" in tables:
            conn.execute("DELETE FROM image_storage_quotas WHERE user_id = ?", (user_id,))
            if conn.execute("SELECT COUNT(*) FROM image_storage_quotas WHERE user_id = ?", (user_id,)).fetchone()[0]:
                raise RuntimeError("Quota records remain after erasure")
        if "image_storage_reservations" in tables:
            conn.execute("DELETE FROM image_storage_reservations WHERE user_id=?", (user_id,))
        if "image_idempotency" in tables:
            conn.execute("DELETE FROM image_idempotency WHERE owner_id = ?", (user_id,))
            if conn.execute("SELECT COUNT(*) FROM image_idempotency WHERE owner_id = ?", (user_id,)).fetchone()[0]:
                raise RuntimeError("Idempotency records remain after erasure")
        if delete_account:
            conn.execute("DELETE FROM users WHERE id = ? AND deletion_pending = 1", (user_id,))


def erase_account(user_id, auth_repo):
    from app.services.cache.response_cache import response_cache
    from app.services.conversation_log_service import conversation_log_service
    from app.services.memory_service import memory_service
    from app.services.rag.qdrant_vector_store import (
        _BM25_COLLECTION_CACHE,
        QdrantVectorStore,
        query_embedding_cache,
    )
    from app.services.session_store import session_store
    from app.services.storage import storage_service
    from app.services.workspace_store import workspace_store

    if is_postgres_configured():
        found = run_sync_db(_mark_pending_postgres, user_id)
    else:
        with auth_repo._get_connection() as conn:
            found = conn.execute("UPDATE users SET deletion_pending = 1 WHERE id = ?", (user_id,)).rowcount > 0
    if not found:
        return False

    # Clear unscoped caches rather than guessing which entry belongs to the user.
    response_cache.clear()
    _BM25_COLLECTION_CACHE.clear()
    query_embedding_cache.clear()
    from app.services.image.storage_quota import cancel_owner_reservations
    cancel_owner_reservations(user_id)
    files, images = (run_sync_db(_owned_assets_postgres, user_id) if is_postgres_configured()
                     else _owned_assets_sqlite(workspace_store, user_id))
    store = QdrantVectorStore()
    for file in files:
        if file.get("status") in ("uploading", "processing"):
            raise RuntimeError("An active or uncertain document operation requires confirmed worker recovery before erasure")
        if not workspace_store.update_library_file_status(file["id"], "deleting", user_id=user_id):
            raise RuntimeError("Cannot mark document for erasure")
        store.delete_document_chunks("user_knowledge_base", file_id=file["id"], owner_id=user_id)
        if file.get("storage_path"):
            storage_service.delete_verified(file["storage_path"])
    for image in images:
        _erase_local_image(storage_service, image)
    # Also remove owner-tagged orphan vectors left by older partial failures.
    store.delete_owner_chunks("user_knowledge_base", user_id)

    if is_postgres_configured():
        run_sync_db(_erase_postgres, user_id)
    else:
        paths = {os.path.abspath(service.db_path) for service in (workspace_store, session_store, memory_service, conversation_log_service)}
        account_path = os.path.abspath(auth_repo.db_path)
        for path in paths - {account_path}:
            _erase_sqlite_database(path, user_id)
        _erase_sqlite_database(account_path, user_id, delete_account=True)
    if auth_repo.get_user_by_id(user_id) is not None:
        raise RuntimeError("Account remains after erasure")
    # Erasure already subtracts this owner's bytes transactionally. Recomputing
    # the project total from retained records here discards other owners'
    # reservations whose writes have not yet produced image_records.
    return True
