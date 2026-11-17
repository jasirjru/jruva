"""Synthetic storage, document, and quota orphan reconciliation regressions."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from app.services.storage import storage_service
from app.services.storage.local import LocalStorageBackend
from app.services.storage.reconciliation import audit_storage_state
from app.services.workspace_store import workspace_store
from tests.test_independent_image_audit import audit_context as audit_context


def test_audit_empty_database_reports_zero_anomalies(audit_context):
    report = audit_storage_state(threshold_seconds=0)
    assert report.summary["total_anomalies"] == 0
    assert report.abandoned_document_uploads == []
    assert report.stuck_document_deletions == []
    assert report.incomplete_image_reservations == []
    assert report.missing_storage_objects == []


def test_audit_detects_abandoned_document_upload(audit_context):
    # Record an upload that is stuck in uploading state
    rec = workspace_store.record_library_file(
        filename="stuck.pdf",
        file_type="PDF",
        file_size=1024,
        storage_path="documents/test/stuck.pdf",
        status="uploading",
        user_id="audit-alice",
    )
    # Threshold 0 treats current time as cutoff
    report = audit_storage_state(threshold_seconds=0)
    assert len(report.abandoned_document_uploads) == 1
    assert report.abandoned_document_uploads[0]["file_id"] == rec["id"]
    assert report.abandoned_document_uploads[0]["status"] == "uploading"
    assert report.summary["abandoned_document_uploads"] == 1


def test_audit_detects_stuck_document_deletion(audit_context):
    rec = workspace_store.record_library_file(
        filename="deleting.pdf",
        file_type="PDF",
        file_size=1024,
        storage_path="documents/test/deleting.pdf",
        status="deleting",
        user_id="audit-alice",
    )
    report = audit_storage_state(threshold_seconds=0)
    assert len(report.stuck_document_deletions) == 1
    assert report.stuck_document_deletions[0]["file_id"] == rec["id"]
    assert report.summary["stuck_document_deletions"] == 1


def test_audit_detects_incomplete_image_reservation(audit_context):
    from app.services.image.storage_quota import ensure_sqlite_quota_schema
    with workspace_store._get_sqlite_connection() as conn:
        ensure_sqlite_quota_schema(conn)
        now = datetime.now(timezone.utc)
        expired = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """
            INSERT INTO image_storage_reservations (reservation_id, user_id, bytes, state, expires_at, created_at)
            VALUES ('res-123', 'audit-alice', 4096, 'writing', ?, ?)
            """,
            (expired, expired),
        )
        conn.commit()

    report = audit_storage_state(threshold_seconds=0)
    assert len(report.incomplete_image_reservations) == 1
    assert report.incomplete_image_reservations[0]["reservation_id"] == "res-123"
    assert report.incomplete_image_reservations[0]["is_expired"] is True


def test_audit_detects_missing_storage_objects(audit_context):
    # Record an indexed document whose storage object does not exist on disk
    rec = workspace_store.record_library_file(
        filename="missing.pdf",
        file_type="PDF",
        file_size=1024,
        storage_path="documents/test/nonexistent_blob.pdf",
        status="indexed",
        user_id="audit-alice",
    )
    report = audit_storage_state(threshold_seconds=0, check_object_existence=True)
    missing = [m for m in report.missing_storage_objects if m["id"] == rec["id"]]
    assert len(missing) == 1
    assert missing[0]["storage_path"] == "documents/test/nonexistent_blob.pdf"


def test_audit_detects_untracked_local_files(audit_context):
    # Create an untracked raw file directly in storage directory
    media_dir = audit_context.media
    untracked_file = media_dir / "untracked_phantom.bin"
    untracked_file.write_bytes(b"phantom data")

    report = audit_storage_state(threshold_seconds=0, scan_untracked_local_files=True)
    untracked = [u for u in report.untracked_storage_objects if "untracked_phantom.bin" in u["path"]]
    assert len(untracked) == 1
    assert untracked[0]["size_bytes"] == len(b"phantom data")


def test_audit_detects_stranded_storage_quota_drift(audit_context):
    from app.services.image.storage_quota import ensure_sqlite_quota_schema
    # Inject quota records that have higher counts than actual records
    with workspace_store._get_sqlite_connection() as conn:
        ensure_sqlite_quota_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO image_storage_quotas (user_id, image_count, total_bytes)
            VALUES ('audit-alice', 5, 500000)
            """
        )
        conn.commit()

    report = audit_storage_state(threshold_seconds=0)
    assert len(report.stranded_storage_quotas) == 1
    assert report.stranded_storage_quotas[0]["user_id"] == "audit-alice"
    assert report.stranded_storage_quotas[0]["stranded_bytes"] == 500000


def test_audit_is_strictly_read_only_and_preserves_records(audit_context):
    # Insert multiple anomalies
    rec1 = workspace_store.record_library_file(
        filename="stuck.pdf", file_type="PDF", file_size=1024,
        storage_path="documents/test/stuck.pdf", status="uploading", user_id="audit-alice",
    )
    rec2 = workspace_store.record_library_file(
        filename="deleting.pdf", file_type="PDF", file_size=1024,
        storage_path="documents/test/deleting.pdf", status="deleting", user_id="audit-alice",
    )
    # Run audit multiple times
    r1 = audit_storage_state(threshold_seconds=0)
    r2 = audit_storage_state(threshold_seconds=0)
    assert r1.summary["total_anomalies"] == r2.summary["total_anomalies"]

    # Verify rows still exist with exact same statuses
    assert workspace_store.get_library_file(rec1["id"])["status"] == "uploading"
    assert workspace_store.get_library_file(rec2["id"])["status"] == "deleting"
