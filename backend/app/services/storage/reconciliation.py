"""Read-only storage, document, and quota orphan reconciliation auditor.

CRITICAL SAFETY RULES:
1. 100% read-only inspection. Zero automated deletion, zero destructive repair.
2. Reports stranded reservations, abandoned uploads, missing objects, and quota drift.
3. Safe for production operation without blocking writers or invalidating receipts.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.database import is_postgres_configured, run_sync_db
from app.services.image.storage_quota import _quota_transaction
from app.services.storage import storage_service
from app.services.storage.local import LocalStorageBackend
from app.services.workspace_store import workspace_store
from sqlalchemy import text

logger = logging.getLogger("chatbot.storage_reconciliation")


@dataclass
class StorageAuditReport:
    audited_at: str
    threshold_seconds: int
    abandoned_document_uploads: List[Dict[str, Any]] = field(default_factory=list)
    stuck_document_deletions: List[Dict[str, Any]] = field(default_factory=list)
    incomplete_image_reservations: List[Dict[str, Any]] = field(default_factory=list)
    missing_storage_objects: List[Dict[str, Any]] = field(default_factory=list)
    untracked_storage_objects: List[Dict[str, Any]] = field(default_factory=list)
    stranded_storage_quotas: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.astimezone(timezone.utc) if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def audit_storage_state(
    threshold_seconds: int = 3600,
    check_object_existence: bool = True,
    scan_untracked_local_files: bool = True,
) -> StorageAuditReport:
    """Perform a comprehensive, strictly read-only audit of storage and persistence state."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=threshold_seconds)

    report = StorageAuditReport(
        audited_at=now.isoformat(),
        threshold_seconds=threshold_seconds,
    )

    # 1. Document Library Audit (Abandoned uploads & stuck deletions)
    all_files: List[Dict[str, Any]] = []
    if is_postgres_configured():
        async def _fetch_pg():
            from app.core.database import get_async_session
            async with get_async_session() as session:
                rows = (await session.execute(text(
                    "SELECT id, user_id, filename, file_size, storage_path, status, created_at "
                    "FROM library_files"
                ))).fetchall()
                return [dict(r._mapping) for r in rows]
        all_files = run_sync_db(_fetch_pg())
    else:
        with workspace_store._get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, filename, file_size, storage_path, status, created_at "
                "FROM library_files"
            )
            all_files = [dict(row) for row in cursor.fetchall()]

    tracked_paths = set()

    for f in all_files:
        path = f.get("storage_path")
        if path:
            tracked_paths.add(path)

        st = f.get("status")
        created_dt = _parse_ts(f.get("created_at"))

        # Check abandoned uploads
        if st in ("uploading", "processing") and (created_dt is None or created_dt <= cutoff):
            report.abandoned_document_uploads.append({
                "file_id": f.get("id"),
                "user_id": f.get("user_id"),
                "filename": f.get("filename"),
                "file_size": f.get("file_size"),
                "storage_path": path,
                "status": st,
                "created_at": f.get("created_at"),
                "age_seconds": int((now - created_dt).total_seconds()) if created_dt else None,
            })

        # Check stuck deletions
        elif st == "deleting" and (created_dt is None or created_dt <= cutoff):
            report.stuck_document_deletions.append({
                "file_id": f.get("id"),
                "user_id": f.get("user_id"),
                "filename": f.get("filename"),
                "storage_path": path,
                "created_at": f.get("created_at"),
                "age_seconds": int((now - created_dt).total_seconds()) if created_dt else None,
            })

        # Check missing storage objects for indexed files
        if check_object_existence and st == "indexed" and path:
            try:
                if not storage_service.exists(path):
                    report.missing_storage_objects.append({
                        "entity_type": "library_file",
                        "id": f.get("id"),
                        "user_id": f.get("user_id"),
                        "storage_path": path,
                        "status": st,
                    })
            except Exception as e:
                logger.warning(f"Storage existence check failed for {path}: {e}")

    # 2. Image Reservations & Records Audit
    async def _audit_images():
        incomplete_res = []
        missing_imgs = []
        stranded_q = []

        async with _quota_transaction() as session:
            # A. Incomplete Reservations
            res_rows = (await session.execute(text(
                "SELECT reservation_id, user_id, bytes, state, expires_at, object_path, image_id, created_at "
                "FROM image_storage_reservations "
                "WHERE state IN ('active', 'writing', 'cleanup')"
            ))).fetchall()

            for r in res_rows:
                rd = dict(r._mapping if hasattr(r, "_mapping") else dict(zip(
                    ["reservation_id", "user_id", "bytes", "state", "expires_at", "object_path", "image_id", "created_at"], r
                )))
                p = rd.get("object_path")
                if p:
                    tracked_paths.add(p)

                exp_dt = _parse_ts(rd.get("expires_at"))
                created_dt = _parse_ts(rd.get("created_at"))
                if (exp_dt and exp_dt <= now) or (created_dt and created_dt <= cutoff):
                    incomplete_res.append({
                        "reservation_id": rd.get("reservation_id"),
                        "user_id": rd.get("user_id"),
                        "bytes": rd.get("bytes"),
                        "state": rd.get("state"),
                        "expires_at": str(rd.get("expires_at")),
                        "created_at": str(rd.get("created_at")),
                        "object_path": p,
                        "image_id": rd.get("image_id"),
                        "is_expired": bool(exp_dt and exp_dt <= now),
                    })

            # B. Image Records Missing Blobs
            img_rows = (await session.execute(text(
                "SELECT id, user_id, file_url, bytes FROM image_records"
            ))).fetchall()

            for img in img_rows:
                imd = dict(img._mapping if hasattr(img, "_mapping") else dict(zip(["id", "user_id", "file_url", "bytes"], img)))
                u = imd.get("file_url")
                if u:
                    tracked_paths.add(u)
                    if check_object_existence:
                        try:
                            if not storage_service.exists(u):
                                missing_imgs.append({
                                    "entity_type": "image_record",
                                    "id": imd.get("id"),
                                    "user_id": imd.get("user_id"),
                                    "storage_path": u,
                                    "bytes": imd.get("bytes"),
                                })
                        except Exception as e:
                            logger.warning(f"Image storage existence check failed for {u}: {e}")

            # C. Stranded Quota Drift Check
            quota_rows = (await session.execute(text(
                "SELECT user_id, image_count, total_bytes FROM image_storage_quotas"
            ))).fetchall()

            for q in quota_rows:
                qd = dict(q._mapping if hasattr(q, "_mapping") else dict(zip(["user_id", "image_count", "total_bytes"], q)))
                uid = qd.get("user_id")
                if uid == "__project__":
                    continue

                actual_records = (await session.execute(text(
                    "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM image_records WHERE user_id=:u"
                ), {"u": uid})).fetchone()
                act_cnt = actual_records[0] if actual_records else 0
                act_bytes = actual_records[1] if actual_records else 0

                active_res = (await session.execute(text(
                    "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM image_storage_reservations "
                    "WHERE user_id=:u AND state IN ('active', 'writing', 'cleanup')"
                ), {"u": uid})).fetchone()
                res_cnt = active_res[0] if active_res else 0
                res_bytes = active_res[1] if active_res else 0

                expected_cnt = act_cnt + res_cnt
                expected_bytes = act_bytes + res_bytes

                rec_cnt = qd.get("image_count", 0)
                rec_bytes = qd.get("total_bytes", 0)

                if rec_bytes > expected_bytes or rec_cnt > expected_cnt:
                    stranded_q.append({
                        "user_id": uid,
                        "recorded_bytes": rec_bytes,
                        "expected_bytes": expected_bytes,
                        "stranded_bytes": rec_bytes - expected_bytes,
                        "recorded_count": rec_cnt,
                        "expected_count": expected_cnt,
                    })

        return incomplete_res, missing_imgs, stranded_q

    inc_res, miss_img, str_q = run_sync_db(_audit_images())
    report.incomplete_image_reservations.extend(inc_res)
    report.missing_storage_objects.extend(miss_img)
    report.stranded_storage_quotas.extend(str_q)

    # 3. Untracked Local Storage Files Check
    if scan_untracked_local_files and isinstance(storage_service, LocalStorageBackend):
        root_dir = Path(getattr(storage_service, "root_dir", None) or getattr(storage_service, "base_dir", None) or "media")
        if root_dir.exists() and root_dir.is_dir():
            for p in root_dir.rglob("*"):
                if p.is_file():
                    # Format as relative path or canonical string
                    rel_path = str(p.relative_to(root_dir)).replace("\\", "/")
                    full_str = str(p)
                    # If neither the full path nor relative path is tracked in DB
                    if rel_path not in tracked_paths and full_str not in tracked_paths and f"media/{rel_path}" not in tracked_paths:
                        report.untracked_storage_objects.append({
                            "path": rel_path,
                            "full_path": full_str,
                            "size_bytes": p.stat().st_size,
                        })

    report.summary = {
        "abandoned_document_uploads": len(report.abandoned_document_uploads),
        "stuck_document_deletions": len(report.stuck_document_deletions),
        "incomplete_image_reservations": len(report.incomplete_image_reservations),
        "missing_storage_objects": len(report.missing_storage_objects),
        "untracked_storage_objects": len(report.untracked_storage_objects),
        "stranded_storage_quotas": len(report.stranded_storage_quotas),
        "total_anomalies": (
            len(report.abandoned_document_uploads)
            + len(report.stuck_document_deletions)
            + len(report.incomplete_image_reservations)
            + len(report.missing_storage_objects)
            + len(report.untracked_storage_objects)
            + len(report.stranded_storage_quotas)
        ),
    }

    return report
