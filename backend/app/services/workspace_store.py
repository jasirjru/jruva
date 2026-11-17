"""Persistent Workspace Repository for Jruva AI OS.

Manages relational persistence for:
1. Projects (workspace containers, custom instructions, models)
2. Library Documents (uploaded knowledge files, chunking stats)
3. Image Gallery Records (generated visual assets and prompt history)
4. Scheduled Tasks & Automation Logs
5. Plugin & Tool Integrations
6. Global Workspace Search with Tenant Isolation

Supports both PostgreSQL (production) and SQLite (development/fallback).
"""
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, desc, select, update

from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.models import (
    ImageRecord,
    LibraryFile,
    Message,
    Project,
    ScheduledTask,
    Session,
    TaskExecution,
    UserIntegration,
)
from app.services.storage import storage_service
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chatbot.db")


class WorkspaceStore:
    """Async-safe workspace repository with PostgreSQL & SQLite support."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        if not is_postgres_configured():
            self._init_sqlite_db()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_sqlite_db(self) -> None:
        """Initialize relational workspace SQLite schema with proper indexing."""
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        name TEXT NOT NULL,
                        description TEXT,
                        instructions TEXT,
                        preferred_model TEXT DEFAULT 'auto',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_user ON projects (user_id);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS library_files (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        project_id TEXT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        storage_path TEXT NOT NULL,
                        status TEXT DEFAULT 'indexed',
                        chunk_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(project_id) REFERENCES projects(id)
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_library_user ON library_files (user_id);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS image_records (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        prompt TEXT NOT NULL,
                        enhanced_prompt TEXT,
                        model TEXT NOT NULL,
                        tier TEXT DEFAULT 'B',
                        style TEXT DEFAULT 'photorealistic',
                        width INTEGER DEFAULT 1024,
                        height INTEGER DEFAULT 1024,
                        file_url TEXT NOT NULL,
                        bytes INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                try:
                    conn.execute("ALTER TABLE image_records ADD COLUMN bytes INTEGER DEFAULT 0;")
                except Exception:
                    pass
                conn.execute("CREATE INDEX IF NOT EXISTS idx_images_user ON image_records (user_id, created_at DESC);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_tasks (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        name TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        cron_expression TEXT NOT NULL,
                        timezone TEXT DEFAULT 'UTC',
                        status TEXT DEFAULT 'active',
                        last_run TIMESTAMP,
                        next_run TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_user ON scheduled_tasks (user_id);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_executions (
                        id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        output TEXT,
                        error_message TEXT,
                        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_task_exec ON task_executions (task_id, executed_at DESC);")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_integrations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        plugin_id TEXT NOT NULL,
                        is_enabled INTEGER DEFAULT 1,
                        config_encrypted TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_plugin ON user_integrations (user_id, plugin_id);")

                conn.commit()
            logger.info("Initialized complete AI workspace schema in chatbot.db")
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Failed to initialize workspace schema: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("workspace_store", str(exc))

    # ─── Projects CRUD (PostgreSQL Async) ────────────────────────────────────

    async def async_list_projects(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(Project).where(Project.user_id == user_id).order_by(Project.updated_at.desc())
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": p.id,
                    "user_id": p.user_id,
                    "name": p.name,
                    "description": p.description,
                    "instructions": p.instructions,
                    "preferred_model": p.preferred_model,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                }
                for p in rows
            ]

    async def async_create_project(
        self,
        name: str,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        preferred_model: str = "auto",
        user_id: str = "guest",
    ) -> Dict[str, Any]:
        proj_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        proj = Project(
            id=proj_id,
            user_id=user_id,
            name=name.strip(),
            description=description,
            instructions=instructions,
            preferred_model=preferred_model,
            created_at=now,
            updated_at=now,
        )
        async with get_async_session() as session:
            async with session.begin():
                session.add(proj)
        return {
            "id": proj_id,
            "user_id": user_id,
            "name": name.strip(),
            "description": description,
            "instructions": instructions,
            "preferred_model": preferred_model,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

    async def async_get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(Project).where(Project.id == project_id)
            res = await session.execute(stmt)
            p = res.scalar_one_or_none()
            if not p:
                return None
            return {
                "id": p.id,
                "user_id": p.user_id,
                "name": p.name,
                "description": p.description,
                "instructions": p.instructions,
                "preferred_model": p.preferred_model,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            }

    async def async_update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> bool:
        values: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if name is not None:
            values["name"] = name.strip()
        if description is not None:
            values["description"] = description
        if instructions is not None:
            values["instructions"] = instructions
        if preferred_model is not None:
            values["preferred_model"] = preferred_model

        async with get_async_session() as session:
            async with session.begin():
                stmt = update(Project).where(Project.id == project_id).values(**values)
                res = await session.execute(stmt)
                return res.rowcount > 0

    async def async_delete_project(self, project_id: str) -> bool:
        async with get_async_session() as session:
            async with session.begin():
                stmt = delete(Project).where(Project.id == project_id)
                res = await session.execute(stmt)
                return res.rowcount > 0

    # ─── Library Files CRUD (PostgreSQL Async) ────────────────────────────────

    async def async_list_library_files(
        self, user_id: str = "guest", project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(LibraryFile).where(LibraryFile.user_id == user_id)
            if project_id:
                stmt = stmt.where(LibraryFile.project_id == project_id)
            stmt = stmt.order_by(LibraryFile.created_at.desc())
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": f.id,
                    "user_id": f.user_id,
                    "project_id": f.project_id,
                    "filename": f.filename,
                    "file_type": f.file_type,
                    "file_size": f.file_size,
                    "storage_path": f.storage_path,
                    "status": f.status,
                    "chunk_count": f.chunk_count,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in rows
            ]

    async def async_record_library_file(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        storage_path: str,
        chunk_count: int = 0,
        status: str = "indexed",
        project_id: Optional[str] = None,
        user_id: str = "guest",
        file_id: Optional[str] = None,
        require_active_owner: bool = False,
    ) -> Dict[str, Any]:
        file_id = str(uuid.UUID(file_id)) if file_id else str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        rec = LibraryFile(
            id=file_id,
            user_id=user_id,
            project_id=project_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            storage_path=storage_path,
            status=status,
            chunk_count=chunk_count,
            created_at=now,
        )
        async with get_async_session() as session:
            async with session.begin():
                if require_active_owner:
                    from sqlalchemy import text
                    owner = (await session.execute(text("SELECT deletion_pending FROM users WHERE id=:u FOR UPDATE"), {"u": user_id})).fetchone()
                    if not owner or owner[0]:
                        raise RuntimeError("Document owner is unavailable")
                session.add(rec)
        return {
            "id": file_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "storage_path": storage_path,
            "status": status,
            "chunk_count": chunk_count,
            "created_at": now.isoformat(),
        }

    async def async_get_library_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(LibraryFile).where(LibraryFile.id == file_id)
            res = await session.execute(stmt)
            f = res.scalar_one_or_none()
            if not f:
                return None
            return {
                "id": f.id,
                "user_id": f.user_id,
                "project_id": f.project_id,
                "filename": f.filename,
                "file_type": f.file_type,
                "file_size": f.file_size,
                "storage_path": f.storage_path,
                "status": f.status,
                "chunk_count": f.chunk_count,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }

    async def async_update_library_file_status(
        self,
        file_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        user_id: Optional[str] = None,
        expected_statuses: Optional[tuple[str, ...]] = None,
    ) -> bool:
        async with get_async_session() as session:
            async with session.begin():
                stmt = select(LibraryFile).where(LibraryFile.id == file_id).with_for_update()
                res = await session.execute(stmt)
                f = res.scalar_one_or_none()
                if not f:
                    return False
                if user_id and f.user_id != user_id:
                    return False
                if expected_statuses is not None and f.status not in expected_statuses:
                    return False
                if f.status == "deleting" and status != "deleting":
                    return False
                f.status = status
                if chunk_count is not None:
                    f.chunk_count = chunk_count
                return True

    async def async_delete_library_file(self, file_id: str, user_id: Optional[str] = None) -> bool:
        async with get_async_session() as session:
            async with session.begin():
                stmt = select(LibraryFile).where(LibraryFile.id == file_id)
                res = await session.execute(stmt)
                f = res.scalar_one_or_none()
                if not f:
                    return False
                if user_id and f.user_id != user_id:
                    logger.warning(
                        f"[SecurityWarning] User {user_id} attempted unauthorized deletion of file {file_id} owned by {f.user_id}"
                    )
                    return False
                if f.storage_path:
                    storage_service.delete_verified(f.storage_path)
                    if storage_service.exists(f.storage_path):
                        raise RuntimeError("Document blob remains after deletion")
                from app.services.storage.document_quota import async_release_committed_document
                await async_release_committed_document(file_id, f.user_id)
                await session.delete(f)
                return True

    # ─── Image Records CRUD (PostgreSQL Async) ────────────────────────────────

    async def async_list_images(self, user_id: str = "guest", limit: int = 50) -> List[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = (
                select(ImageRecord)
                .where(ImageRecord.user_id == user_id)
                .order_by(ImageRecord.created_at.desc())
                .limit(limit)
            )
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": img.id,
                    "user_id": img.user_id,
                    "prompt": img.prompt,
                    "enhanced_prompt": img.enhanced_prompt,
                    "model": img.model,
                    "tier": img.tier,
                    "style": img.style,
                    "width": img.width,
                    "height": img.height,
                    "file_url": img.file_url,
                    "created_at": img.created_at.isoformat() if img.created_at else None,
                }
                for img in rows
            ]

    async def async_record_image(
        self,
        prompt: str,
        file_url: str,
        model: str,
        enhanced_prompt: Optional[str] = None,
        tier: str = "B",
        style: str = "photorealistic",
        width: int = 1024,
        height: int = 1024,
        user_id: str = "guest",
        bytes: int = 0,
        reservation_id: str | None = None,
        require_active_owner: bool = False,
    ) -> Dict[str, Any]:
        img_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        img = ImageRecord(
            id=img_id,
            user_id=user_id,
            prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            model=model,
            tier=tier,
            style=style,
            width=width,
            height=height,
            file_url=file_url,
            bytes=bytes,
            created_at=now,
        )
        async with get_async_session() as session:
            async with session.begin():
                from app.services.image.storage_quota import (
                    _check_owner,
                    commit_storage_reservation,
                    lock_storage_project,
                )
                await lock_storage_project(session)
                if require_active_owner:
                    await _check_owner(session, user_id, require_exists=True)
                if reservation_id:
                    await commit_storage_reservation(session, user_id, bytes, reservation_id, img_id, file_url)
                session.add(img)
        return {
            "id": img_id,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "model": model,
            "tier": tier,
            "style": style,
            "width": width,
            "height": height,
            "file_url": file_url,
            "bytes": bytes,
            "created_at": now.isoformat(),
        }

    async def async_delete_image(self, image_id: str) -> bool:
        async with get_async_session() as session:
            async with session.begin():
                from app.services.image.storage_quota import lock_storage_project
                await lock_storage_project(session)
                stmt = select(ImageRecord).where(ImageRecord.id == image_id).with_for_update()
                res = await session.execute(stmt)
                record = res.scalar_one_or_none()
                if not record:
                    return False
                owner_id = record.user_id
                rel_bytes = record.bytes or 0
                await session.delete(record)
                from app.services.image.storage_quota import _release_storage_quota_in_session
                await _release_storage_quota_in_session(session, owner_id, rel_bytes, image_id=image_id)
                return True

    # ─── Scheduled Tasks CRUD (PostgreSQL Async) ──────────────────────────────

    async def async_list_scheduled_tasks(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = (
                select(ScheduledTask)
                .where(ScheduledTask.user_id == user_id)
                .order_by(ScheduledTask.created_at.desc())
            )
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": t.id,
                    "user_id": t.user_id,
                    "name": t.name,
                    "prompt": t.prompt,
                    "cron_expression": t.cron_expression,
                    "timezone": t.timezone,
                    "status": t.status,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                    "next_run": t.next_run.isoformat() if t.next_run else None,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                }
                for t in rows
            ]

    async def async_create_scheduled_task(
        self,
        name: str,
        prompt: str,
        cron_expression: str = "0 8 * * *",
        timezone_str: str = "UTC",
        user_id: str = "guest",
    ) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        t = ScheduledTask(
            id=task_id,
            user_id=user_id,
            name=name.strip(),
            prompt=prompt.strip(),
            cron_expression=cron_expression,
            timezone=timezone_str,
            status="active",
            created_at=now,
            updated_at=now,
        )
        async with get_async_session() as session:
            async with session.begin():
                session.add(t)
        return {
            "id": task_id,
            "name": name.strip(),
            "prompt": prompt.strip(),
            "cron_expression": cron_expression,
            "timezone": timezone_str,
            "status": "active",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

    async def async_toggle_scheduled_task(self, task_id: str) -> Optional[str]:
        async with get_async_session() as session:
            async with session.begin():
                stmt = select(ScheduledTask).where(ScheduledTask.id == task_id)
                res = await session.execute(stmt)
                t = res.scalar_one_or_none()
                if not t:
                    return None
                next_status = "paused" if t.status == "active" else "active"
                t.status = next_status
                t.updated_at = datetime.now(timezone.utc)
                return next_status

    async def async_delete_scheduled_task(self, task_id: str) -> bool:
        async with get_async_session() as session:
            async with session.begin():
                stmt = delete(ScheduledTask).where(ScheduledTask.id == task_id)
                res = await session.execute(stmt)
                return res.rowcount > 0

    async def async_record_execution(
        self,
        task_id: str,
        status: str,
        output: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        exec_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        te = TaskExecution(
            id=exec_id,
            task_id=task_id,
            status=status,
            output=output,
            error_message=error_message,
            executed_at=now,
        )
        async with get_async_session() as session:
            async with session.begin():
                session.add(te)
                stmt = update(ScheduledTask).where(ScheduledTask.id == task_id).values(last_run=now)
                await session.execute(stmt)
        return {
            "id": exec_id,
            "task_id": task_id,
            "status": status,
            "output": output,
            "error_message": error_message,
            "executed_at": now.isoformat(),
        }

    async def async_list_executions(self, task_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = (
                select(TaskExecution)
                .where(TaskExecution.task_id == task_id)
                .order_by(TaskExecution.executed_at.desc())
                .limit(limit)
            )
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": e.id,
                    "task_id": e.task_id,
                    "status": e.status,
                    "output": e.output,
                    "error_message": e.error_message,
                    "executed_at": e.executed_at.isoformat() if e.executed_at else None,
                }
                for e in rows
            ]

    # ─── User Integrations CRUD (PostgreSQL Async) ───────────────────────────

    async def async_list_integrations(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        async with get_async_session() as session:
            stmt = select(UserIntegration).where(UserIntegration.user_id == user_id)
            res = await session.execute(stmt)
            rows = res.scalars().all()
            return [
                {
                    "id": ui.id,
                    "user_id": ui.user_id,
                    "plugin_id": ui.plugin_id,
                    "is_enabled": ui.is_enabled,
                    "config_encrypted": ui.config_encrypted,
                    "updated_at": ui.updated_at.isoformat() if ui.updated_at else None,
                }
                for ui in rows
            ]

    async def async_toggle_integration(self, plugin_id: str, is_enabled: bool, user_id: str = "guest") -> bool:
        now = datetime.now(timezone.utc)
        async with get_async_session() as session:
            async with session.begin():
                stmt = select(UserIntegration).where(
                    UserIntegration.user_id == user_id,
                    UserIntegration.plugin_id == plugin_id,
                )
                res = await session.execute(stmt)
                ui = res.scalar_one_or_none()
                if ui:
                    ui.is_enabled = 1 if is_enabled else 0
                    ui.updated_at = now
                else:
                    new_ui = UserIntegration(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        plugin_id=plugin_id,
                        is_enabled=1 if is_enabled else 0,
                        updated_at=now,
                    )
                    session.add(new_ui)
                return True

    # ─── Global Workspace Search with Strict Tenant Isolation ────────────────

    async def async_global_search(self, query: str, user_id: str = "guest", limit: int = 10) -> Dict[str, Any]:
        term = f"%{query.strip()}%"
        results: Dict[str, List[Dict[str, Any]]] = {
            "chats": [],
            "projects": [],
            "files": [],
            "images": [],
            "tasks": [],
        }

        async with get_async_session() as session:
            # 1. Search Sessions & Messages with TENANT ISOLATION
            try:
                msg_subq = (
                    select(Message.content)
                    .where(Message.session_id == Session.session_id, Message.content.ilike(term))
                    .limit(1)
                    .scalar_subquery()
                )
                sess_query = select(Session.session_id, Session.title, Session.updated_at, msg_subq.label("snippet"))
                if user_id and user_id != "guest":
                    sess_query = sess_query.where(Session.user_id == user_id)
                else:
                    sess_query = sess_query.where((Session.user_id == "guest") | (Session.user_id.is_(None)))

                sess_query = sess_query.where(
                    Session.title.ilike(term)
                    | Session.session_id.in_(
                        select(Message.session_id).where(Message.content.ilike(term))
                    )
                ).order_by(Session.updated_at.desc()).limit(limit)

                s_res = await session.execute(sess_query)
                results["chats"] = [
                    {
                        "id": r.session_id,
                        "title": r.title,
                        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                        "snippet": r.snippet,
                    }
                    for r in s_res.all()
                ]
            except Exception as e:
                logger.warning(f"Error searching sessions in Postgres: {e}")

            # 2. Search Projects
            try:
                proj_stmt = (
                    select(Project)
                    .where(
                        Project.user_id == user_id,
                        Project.name.ilike(term) | Project.description.ilike(term) | Project.instructions.ilike(term),
                    )
                    .order_by(Project.updated_at.desc())
                    .limit(limit)
                )
                p_res = await session.execute(proj_stmt)
                results["projects"] = [
                    {
                        "id": p.id,
                        "name": p.name,
                        "description": p.description,
                        "instructions": p.instructions,
                        "preferred_model": p.preferred_model,
                        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                    }
                    for p in p_res.scalars().all()
                ]
            except Exception as e:
                logger.warning(f"Error searching projects in Postgres: {e}")

            # 3. Search Library Files
            try:
                file_stmt = (
                    select(LibraryFile)
                    .where(
                        LibraryFile.user_id == user_id,
                        LibraryFile.filename.ilike(term) | LibraryFile.file_type.ilike(term),
                    )
                    .order_by(LibraryFile.created_at.desc())
                    .limit(limit)
                )
                f_res = await session.execute(file_stmt)
                results["files"] = [
                    {
                        "id": f.id,
                        "filename": f.filename,
                        "file_type": f.file_type,
                        "file_size": f.file_size,
                        "chunk_count": f.chunk_count,
                        "created_at": f.created_at.isoformat() if f.created_at else None,
                    }
                    for f in f_res.scalars().all()
                ]
            except Exception as e:
                logger.warning(f"Error searching library_files in Postgres: {e}")

            # 4. Search Image Records
            try:
                img_stmt = (
                    select(ImageRecord)
                    .where(
                        ImageRecord.user_id == user_id,
                        ImageRecord.prompt.ilike(term)
                        | ImageRecord.enhanced_prompt.ilike(term)
                        | ImageRecord.style.ilike(term),
                    )
                    .order_by(ImageRecord.created_at.desc())
                    .limit(limit)
                )
                i_res = await session.execute(img_stmt)
                results["images"] = [
                    {
                        "id": i.id,
                        "prompt": i.prompt,
                        "enhanced_prompt": i.enhanced_prompt,
                        "model": i.model,
                        "style": i.style,
                        "file_url": i.file_url,
                        "created_at": i.created_at.isoformat() if i.created_at else None,
                    }
                    for i in i_res.scalars().all()
                ]
            except Exception as e:
                logger.warning(f"Error searching images in Postgres: {e}")

            # 5. Search Scheduled Tasks
            try:
                task_stmt = (
                    select(ScheduledTask)
                    .where(
                        ScheduledTask.user_id == user_id,
                        ScheduledTask.name.ilike(term) | ScheduledTask.prompt.ilike(term),
                    )
                    .order_by(ScheduledTask.updated_at.desc())
                    .limit(limit)
                )
                t_res = await session.execute(task_stmt)
                results["tasks"] = [
                    {
                        "id": t.id,
                        "name": t.name,
                        "prompt": t.prompt,
                        "cron_expression": t.cron_expression,
                        "status": t.status,
                        "last_run": t.last_run.isoformat() if t.last_run else None,
                        "next_run": t.next_run.isoformat() if t.next_run else None,
                    }
                    for t in t_res.scalars().all()
                ]
            except Exception as e:
                logger.warning(f"Error searching tasks in Postgres: {e}")

        total_matches = sum(len(items) for items in results.values())
        return {
            "query": query,
            "total": total_matches,
            "results": results,
        }

    # ─── Public API Sync Facades (Routing between PostgreSQL & SQLite) ───────

    def list_projects(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_list_projects(user_id))
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error listing projects: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("workspace_store", str(exc))
            return []

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        preferred_model: str = "auto",
        user_id: str = "guest",
    ) -> Dict[str, Any]:
        if is_postgres_configured():
            return run_sync_db(
                self.async_create_project(name, description, instructions, preferred_model, user_id)
            )
        proj_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                INSERT INTO projects (id, user_id, name, description, instructions, preferred_model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (proj_id, user_id, name.strip(), description, instructions, preferred_model, now, now),
            )
            conn.commit()
        return {
            "id": proj_id,
            "user_id": user_id,
            "name": name.strip(),
            "description": description,
            "instructions": instructions,
            "preferred_model": preferred_model,
            "created_at": now,
            "updated_at": now,
        }

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_get_project(project_id))
        with self._get_sqlite_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row) if row else None

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> bool:
        if is_postgres_configured():
            return run_sync_db(
                self.async_update_project(project_id, name, description, instructions, preferred_model)
            )
        now = datetime.now(timezone.utc).isoformat()
        fields = ["updated_at = ?"]
        values = [now]
        if name is not None:
            fields.append("name = ?")
            values.append(name.strip())
        if description is not None:
            fields.append("description = ?")
            values.append(description)
        if instructions is not None:
            fields.append("instructions = ?")
            values.append(instructions)
        if preferred_model is not None:
            fields.append("preferred_model = ?")
            values.append(preferred_model)
        values.append(project_id)

        with self._get_sqlite_connection() as conn:
            res = conn.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
            return res.rowcount > 0

    def delete_project(self, project_id: str) -> bool:
        if is_postgres_configured():
            return run_sync_db(self.async_delete_project(project_id))
        with self._get_sqlite_connection() as conn:
            res = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            return res.rowcount > 0

    def list_library_files(self, user_id: str = "guest", project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_list_library_files(user_id, project_id))
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                if project_id:
                    cursor.execute(
                        "SELECT * FROM library_files WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                        (user_id, project_id),
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM library_files WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error listing library files: {exc}")
            return []

    def record_library_file(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        storage_path: str,
        chunk_count: int = 0,
        status: str = "indexed",
        project_id: Optional[str] = None,
        user_id: str = "guest",
        file_id: Optional[str] = None,
        require_active_owner: bool = False,
    ) -> Dict[str, Any]:
        if is_postgres_configured():
            return run_sync_db(
                self.async_record_library_file(
                    filename, file_type, file_size, storage_path, chunk_count, status, project_id, user_id, file_id, require_active_owner
                )
            )
        file_id = str(uuid.UUID(file_id)) if file_id else str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_sqlite_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if require_active_owner:
                owner = conn.execute("SELECT deletion_pending FROM users WHERE id=?", (user_id,)).fetchone()
                if not owner or owner[0]:
                    raise RuntimeError("Document owner is unavailable")
            conn.execute(
                """
                INSERT INTO library_files (id, user_id, project_id, filename, file_type, file_size, storage_path, status, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, user_id, project_id, filename, file_type, file_size, storage_path, status, chunk_count, now),
            )
            conn.commit()
        return {
            "id": file_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "storage_path": storage_path,
            "status": status,
            "chunk_count": chunk_count,
            "created_at": now,
        }

    def get_library_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_get_library_file(file_id))
        try:
            with self._get_sqlite_connection() as conn:
                row = conn.execute("SELECT * FROM library_files WHERE id = ?", (file_id,)).fetchone()
                return dict(row) if row else None
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error getting library file {file_id}: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("workspace_store", str(exc))
            return None

    def update_library_file_status(
        self,
        file_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        user_id: Optional[str] = None,
        expected_statuses: Optional[tuple[str, ...]] = None,
    ) -> bool:
        if is_postgres_configured():
            return run_sync_db(self.async_update_library_file_status(file_id, status, chunk_count, user_id, expected_statuses))
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT * FROM library_files WHERE id = ?", (file_id,)).fetchone()
                if not row:
                    return False
                if user_id and row["user_id"] != user_id:
                    return False
                if expected_statuses is not None and row["status"] not in expected_statuses:
                    return False
                if row["status"] == "deleting" and status != "deleting":
                    return False
                if chunk_count is not None:
                    conn.execute(
                        "UPDATE library_files SET status = ?, chunk_count = ? WHERE id = ?",
                        (status, chunk_count, file_id),
                    )
                else:
                    conn.execute(
                        "UPDATE library_files SET status = ? WHERE id = ?",
                        (status, file_id),
                    )
                conn.commit()
                return True
        except Exception as exc:
            logger.error(f"[SilentFailureGuard] Error updating library file {file_id} status: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("workspace_store", str(exc))
            return False

    def delete_library_file(self, file_id: str, user_id: Optional[str] = None) -> bool:
        if is_postgres_configured():
            return run_sync_db(self.async_delete_library_file(file_id, user_id))
        with self._get_sqlite_connection() as conn:
            row = conn.execute("SELECT * FROM library_files WHERE id = ?", (file_id,)).fetchone()
            if not row:
                return False
            if user_id and row["user_id"] != user_id:
                logger.warning(
                    f"[SecurityWarning] User {user_id} attempted unauthorized deletion of file {file_id} owned by {row['user_id']}"
                )
                return False
            if row["storage_path"]:
                storage_service.delete_verified(row["storage_path"])
                if storage_service.exists(row["storage_path"]):
                    raise RuntimeError("Document blob remains after deletion")
            from app.services.storage.document_quota import release_committed_document
            release_committed_document(file_id, row["user_id"])
            res = conn.execute("DELETE FROM library_files WHERE id = ?", (file_id,))
            conn.commit()
            return res.rowcount > 0

    async def async_get_image(self, image_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        async with get_async_session() as session:
            result = await session.execute(select(ImageRecord).where(ImageRecord.id == image_id, ImageRecord.user_id == user_id))
            image = result.scalar_one_or_none()
            if image is None:
                return None
            return {column.name: getattr(image, column.name) for column in ImageRecord.__table__.columns}

    def get_image(self, image_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_get_image(image_id, user_id))
        with self._get_sqlite_connection() as conn:
            row = conn.execute("SELECT * FROM image_records WHERE id=? AND user_id=?", (image_id, user_id)).fetchone()
            return dict(row) if row else None

    def list_images(self, user_id: str = "guest", limit: int = 50) -> List[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_list_images(user_id, limit))
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM image_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"[DatabaseError] Error listing images: {exc}", exc_info=True)
            telemetry_service.record_silent_failure("workspace_store", str(exc))
            raise

    def record_image(
        self,
        prompt: str,
        file_url: str,
        model: str,
        enhanced_prompt: Optional[str] = None,
        tier: str = "B",
        style: str = "photorealistic",
        width: int = 1024,
        height: int = 1024,
        user_id: str = "guest",
        bytes: int = 0,
        reservation_id: str | None = None,
        require_active_owner: bool = False,
    ) -> Dict[str, Any]:
        from app.services.image.storage_quota import implicit_reservation
        reservation_id = reservation_id or implicit_reservation(user_id, bytes)
        if is_postgres_configured():
            return run_sync_db(
                self.async_record_image(
                    prompt, file_url, model, enhanced_prompt, tier, style, width, height, user_id, bytes=bytes,
                    reservation_id=reservation_id, require_active_owner=require_active_owner,
                )
            )
        img_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_sqlite_connection() as conn:
            from app.services.image.storage_quota import (
                clear_reservation_context,
                commit_storage_reservation_sqlite,
                ensure_sqlite_quota_schema,
            )
            ensure_sqlite_quota_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            if require_active_owner:
                owner = conn.execute("SELECT deletion_pending FROM users WHERE id=?", (user_id,)).fetchone()
                if not owner or owner[0]:
                    raise PermissionError("Image owner is no longer active")
            if reservation_id:
                commit_storage_reservation_sqlite(conn, user_id, bytes, reservation_id, img_id, file_url)
            conn.execute(
                """
                INSERT INTO image_records (id, user_id, prompt, enhanced_prompt, model, tier, style, width, height, file_url, bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (img_id, user_id, prompt, enhanced_prompt, model, tier, style, width, height, file_url, bytes, now),
            )
            conn.commit()
            clear_reservation_context()
        return {
            "id": img_id,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "model": model,
            "tier": tier,
            "style": style,
            "width": width,
            "height": height,
            "file_url": file_url,
            "bytes": bytes,
            "created_at": now,
        }

    def delete_image(self, image_id: str) -> bool:
        if is_postgres_configured():
            return run_sync_db(self.async_delete_image(image_id))
        with self._get_sqlite_connection() as conn:
            from app.services.image.storage_quota import ensure_sqlite_quota_schema
            ensure_sqlite_quota_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT user_id, bytes FROM image_records WHERE id = ?", (image_id,)).fetchone()
            if not row:
                return False
            owner_id = row[0]
            rel_bytes = row[1] if row[1] is not None else 0
            res = conn.execute("DELETE FROM image_records WHERE id = ?", (image_id,))
            conn.execute("UPDATE image_storage_reservations SET state='released' WHERE image_id=? AND user_id=?", (image_id, owner_id))
            conn.execute("UPDATE image_storage_quotas SET image_count=MAX(0,image_count-1),total_bytes=MAX(0,total_bytes-?) WHERE user_id=?", (rel_bytes, owner_id))
            conn.execute("UPDATE image_storage_quotas SET total_bytes=MAX(0,total_bytes-?) WHERE user_id='__project__'", (rel_bytes,))
            conn.commit()
            deleted = res.rowcount > 0
            return deleted

    def list_scheduled_tasks(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_list_scheduled_tasks(user_id))
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM scheduled_tasks WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error listing scheduled tasks: {exc}")
            return []

    def create_scheduled_task(
        self,
        name: str,
        prompt: str,
        cron_expression: str = "0 8 * * *",
        timezone_str: str = "UTC",
        user_id: str = "guest",
    ) -> Dict[str, Any]:
        if is_postgres_configured():
            return run_sync_db(
                self.async_create_scheduled_task(name, prompt, cron_expression, timezone_str, user_id)
            )
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_tasks (id, user_id, name, prompt, cron_expression, timezone, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (task_id, user_id, name.strip(), prompt.strip(), cron_expression, timezone_str, now, now),
            )
            conn.commit()
        return {
            "id": task_id,
            "name": name.strip(),
            "prompt": prompt.strip(),
            "cron_expression": cron_expression,
            "timezone": timezone_str,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }

    def toggle_scheduled_task(self, task_id: str) -> Optional[str]:
        if is_postgres_configured():
            return run_sync_db(self.async_toggle_scheduled_task(task_id))
        with self._get_sqlite_connection() as conn:
            row = conn.execute("SELECT status FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                return None
            next_status = "paused" if row["status"] == "active" else "active"
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE scheduled_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (next_status, now, task_id),
            )
            conn.commit()
            return next_status

    def delete_scheduled_task(self, task_id: str) -> bool:
        if is_postgres_configured():
            return run_sync_db(self.async_delete_scheduled_task(task_id))
        with self._get_sqlite_connection() as conn:
            res = conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
            conn.commit()
            return res.rowcount > 0

    def record_execution(
        self,
        task_id: str,
        status: str,
        output: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        from app.core.maintenance import is_maintenance_mode
        if is_maintenance_mode():
            logger.info("[MAINTENANCE] Skipping task execution record during maintenance mode.")
            return {"id": "maintenance_blocked", "status": "blocked", "output": None}

        if is_postgres_configured():
            return run_sync_db(self.async_record_execution(task_id, status, output, error_message))
        exec_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                INSERT INTO task_executions (id, task_id, status, output, error_message, executed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (exec_id, task_id, status, output, error_message, now),
            )
            conn.execute(
                "UPDATE scheduled_tasks SET last_run = ? WHERE id = ?",
                (now, task_id),
            )
            conn.commit()
        return {
            "id": exec_id,
            "task_id": task_id,
            "status": status,
            "output": output,
            "error_message": error_message,
            "executed_at": now,
        }

    def list_executions(self, task_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_list_executions(task_id, limit))
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM task_executions WHERE task_id = ? ORDER BY executed_at DESC LIMIT ?",
                (task_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_integrations(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        if is_postgres_configured():
            return run_sync_db(self.async_list_integrations(user_id))
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_integrations WHERE user_id = ?", (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    def toggle_integration(self, plugin_id: str, is_enabled: bool, user_id: str = "guest") -> bool:
        if is_postgres_configured():
            return run_sync_db(self.async_toggle_integration(plugin_id, is_enabled, user_id))
        int_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_sqlite_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_integrations (id, user_id, plugin_id, is_enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, plugin_id) DO UPDATE SET
                    is_enabled = excluded.is_enabled,
                    updated_at = excluded.updated_at
                """,
                (int_id, user_id, plugin_id, 1 if is_enabled else 0, now),
            )
            conn.commit()
            return True

    def global_search(self, query: str, user_id: str = "guest", limit: int = 10) -> Dict[str, Any]:
        if is_postgres_configured():
            return run_sync_db(self.async_global_search(query, user_id, limit))

        term = f"%{query.strip()}%"
        results: Dict[str, List[Dict[str, Any]]] = {
            "chats": [],
            "projects": [],
            "files": [],
            "images": [],
            "tasks": [],
        }
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()

            # 1. Search Sessions & Messages with Strict Tenant Isolation
            try:
                if user_id and user_id != "guest":
                    cursor.execute(
                        """
                        SELECT s.session_id as id, s.title, s.updated_at,
                               (SELECT content FROM messages WHERE session_id = s.session_id AND content LIKE ? LIMIT 1) as snippet
                        FROM sessions s
                        WHERE s.user_id = ? AND (s.title LIKE ? OR EXISTS (
                            SELECT 1 FROM messages m WHERE m.session_id = s.session_id AND m.content LIKE ?
                        ))
                        ORDER BY s.updated_at DESC LIMIT ?
                        """,
                        (term, user_id, term, term, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT s.session_id as id, s.title, s.updated_at,
                               (SELECT content FROM messages WHERE session_id = s.session_id AND content LIKE ? LIMIT 1) as snippet
                        FROM sessions s
                        WHERE (s.user_id = 'guest' OR s.user_id IS NULL) AND (s.title LIKE ? OR EXISTS (
                            SELECT 1 FROM messages m WHERE m.session_id = s.session_id AND m.content LIKE ?
                        ))
                        ORDER BY s.updated_at DESC LIMIT ?
                        """,
                        (term, term, term, limit),
                    )
                results["chats"] = [dict(r) for r in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"[SilentFailureGuard] Error searching sessions: {e}", exc_info=True)
                telemetry_service.record_silent_failure("workspace_store", str(e))

            # 2. Search Projects
            try:
                cursor.execute(
                    """
                    SELECT id, name, description, instructions, preferred_model, updated_at
                    FROM projects
                    WHERE user_id = ? AND (name LIKE ? OR description LIKE ? OR instructions LIKE ?)
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (user_id, term, term, term, limit),
                )
                results["projects"] = [dict(r) for r in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"[SilentFailureGuard] Error searching projects: {e}", exc_info=True)
                telemetry_service.record_silent_failure("workspace_store", str(e))

            # 3. Search Library Files
            try:
                cursor.execute(
                    """
                    SELECT id, filename, file_type, file_size, chunk_count, created_at
                    FROM library_files
                    WHERE user_id = ? AND (filename LIKE ? OR file_type LIKE ?)
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (user_id, term, term, limit),
                )
                results["files"] = [dict(r) for r in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"[SilentFailureGuard] Error searching library_files: {e}", exc_info=True)
                telemetry_service.record_silent_failure("workspace_store", str(e))

            # 4. Search Image Records
            try:
                cursor.execute(
                    """
                    SELECT id, prompt, enhanced_prompt, model, style, file_url, created_at
                    FROM image_records
                    WHERE user_id = ? AND (prompt LIKE ? OR enhanced_prompt LIKE ? OR style LIKE ?)
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (user_id, term, term, term, limit),
                )
                results["images"] = [dict(r) for r in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"[SilentFailureGuard] Error searching image_records: {e}", exc_info=True)
                telemetry_service.record_silent_failure("workspace_store", str(e))

            # 5. Search Scheduled Tasks
            try:
                cursor.execute(
                    """
                    SELECT id, name, prompt, cron_expression, status, last_run, next_run
                    FROM scheduled_tasks
                    WHERE user_id = ? AND (name LIKE ? OR prompt LIKE ?)
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (user_id, term, term, limit),
                )
                results["tasks"] = [dict(r) for r in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"[SilentFailureGuard] Error searching scheduled_tasks: {e}", exc_info=True)
                telemetry_service.record_silent_failure("workspace_store", str(e))

        total_matches = sum(len(items) for items in results.values())
        return {
            "query": query,
            "total": total_matches,
            "results": results,
        }


workspace_store = WorkspaceStore()
