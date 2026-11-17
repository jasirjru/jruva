"""Durable planned document writes; uncertain outcomes require explicit recovery.

The existing library row is the write receipt. Never write a blob until its
owner and immutable target have committed. Never automatically redispatch or
delete a possibly active remote upload after losing its acknowledgement.
"""

import mimetypes
from uuid import uuid4

from app.services.storage.base import document_write_scope
from app.services.workspace_store import workspace_store


def persist_document(storage, filename, content, owner_id, project_id=None):
    from app.services.storage.document_quota import (
        reserve_document_upload,
        commit_document_upload,
        release_document_reservation,
    )

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    storage.validate_upload(filename, content, mime)

    # 1. Durable Quota Reservation Check
    reservation = reserve_document_upload(
        user_id=owner_id,
        file_bytes=len(content),
        filename=filename,
    )
    reservation_id = reservation["reservation_id"]
    identifier = reservation["file_id"]

    try:
        target = storage.reserved_document_path(owner_id, identifier, filename)
        record = workspace_store.record_library_file(
            filename=filename,
            file_type=filename.rsplit(".", 1)[-1].upper(),
            file_size=len(content),
            storage_path=target,
            status="uploading",
            chunk_count=0,
            project_id=project_id,
            user_id=owner_id,
            file_id=identifier,
            require_active_owner=True,
        )
        with document_write_scope(owner_id, identifier, filename):
            actual = storage.save_bytes(filename, content, content_type=mime)
        if actual != target:
            raise RuntimeError("Document upload target could not be verified")

        # 2. Commit Quota Reservation on Successful Write
        commit_document_upload(reservation_id, target)

        # Any failed or lost acknowledgement leaves the durable uploading receipt
        # blocking deletion. It cannot be mistaken for an indexed success.
        if not workspace_store.update_library_file_status(
            identifier, "processing", user_id=owner_id, expected_statuses=("uploading",),
        ):
            raise RuntimeError("Document upload completion could not be committed")
        record["status"] = "processing"
        return record
    except Exception:
        # Release uncommitted reservation hold immediately so quota is not locked
        try:
            release_document_reservation(reservation_id)
        except Exception:
            pass
        raise


def recover_document_upload(file_id, owner_id, *, confirmed_writer_stopped=False, confirmed_remote_upload_settled=False):
    """Fence an abandoned receipt only with explicit worker/provider evidence.

    Retain the object and metadata. The owner can then reindex a verified object
    or request deletion; no cleanup or redispatch happens in this function.
    """
    from app.services.storage import storage_service
    from app.services.storage.local import LocalStorageBackend

    if not confirmed_writer_stopped:
        raise ValueError("Writer termination must be confirmed")
    if not isinstance(storage_service, LocalStorageBackend) and not confirmed_remote_upload_settled:
        raise ValueError("Remote upload settlement must be confirmed")
    if not workspace_store.update_library_file_status(
        file_id, "failed", user_id=owner_id, expected_statuses=("uploading", "processing"),
    ):
        raise RuntimeError("No matching uncertain document upload")
