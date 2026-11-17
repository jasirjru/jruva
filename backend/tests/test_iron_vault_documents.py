"""Synthetic document durability and deletion fencing regressions."""

import base64

import pytest
from app.api.v1.endpoints import library
from app.services.document_state import commit_index
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store

from tests.test_independent_image_audit import audit_context as audit_context


def upload(ctx, name="synthetic.txt", data=b"Synthetic engineering document"):
    return ctx.client.post("/api/v1/library/upload-json", headers=ctx.alice,
                           json={"filename": name, "data": base64.b64encode(data).decode()})


def test_metadata_failure_never_creates_untracked_document_blob(audit_context, monkeypatch):
    def fail(**kwargs):
        raise OSError("Synthetic metadata failure")
    monkeypatch.setattr(workspace_store, "record_library_file", fail)
    try:
        upload(audit_context)
    except OSError:
        pass
    assert list(audit_context.media.rglob("*")) == [], "Blob written before durable metadata receipt"


def test_upload_lost_ack_retains_receipt_and_blocks_unsafe_deletion(audit_context, monkeypatch):
    from app.services.document_upload import recover_document_upload
    original = storage_service.put
    def lost_ack(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("Synthetic upload completed but response was lost")
    monkeypatch.setattr(storage_service, "put", lost_ack)
    response = upload(audit_context)
    assert response.status_code == 503
    records = workspace_store.list_library_files("audit-alice")
    assert len(records) == 1 and records[0]["status"] == "uploading"
    record = workspace_store.get_library_file(records[0]["id"])
    assert storage_service.exists(record["storage_path"])
    path = f'/api/v1/library/files/{record["id"]}'
    assert audit_context.client.delete(path, headers=audit_context.alice).status_code == 409
    assert audit_context.client.post(path + "/reindex", headers=audit_context.alice).status_code == 409
    with pytest.raises(ValueError):
        recover_document_upload(record["id"], "audit-alice")
    recover_document_upload(record["id"], "audit-alice", confirmed_writer_stopped=True)
    assert storage_service.exists(record["storage_path"]), "Recovery must not erase uncertain user data"
    monkeypatch.setattr(library.QdrantVectorStore, "delete_document_chunks", lambda *args, **kwargs: None)
    assert audit_context.client.delete(path, headers=audit_context.alice).status_code == 200
    assert audit_context.client.delete(path, headers=audit_context.alice).status_code == 404


def test_pending_account_rejects_admitted_upload_before_blob_write(audit_context):
    from app.services.document_upload import persist_document
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("UPDATE users SET deletion_pending=1 WHERE id='audit-alice'")
        conn.commit()
    with pytest.raises(RuntimeError):
        persist_document(storage_service, "synthetic.txt", b"synthetic", "audit-alice")
    assert list(audit_context.media.rglob("*")) == []


def test_upload_metadata_ack_loss_does_not_dispatch_object(audit_context, monkeypatch):
    original = workspace_store.record_library_file
    def lost_ack(**kwargs):
        original(**kwargs)
        raise OSError("Synthetic metadata commit acknowledgement lost")
    monkeypatch.setattr(workspace_store, "record_library_file", lost_ack)
    assert upload(audit_context).status_code == 503
    assert list(audit_context.media.rglob("*")) == []
    assert workspace_store.list_library_files("audit-alice")[0]["status"] == "uploading"


def test_successful_upload_private_restart_read_and_failed_delete(audit_context, monkeypatch):
    monkeypatch.setattr(library.document_rag_service, "index_library_document", lambda **kwargs: (1, []))
    monkeypatch.setattr(library.QdrantVectorStore, "delete_document_chunks", lambda *args, **kwargs: None)
    first, second = upload(audit_context), upload(audit_context)
    assert first.status_code == second.status_code == 200
    assert first.json()["file"]["id"] != second.json()["file"]["id"]
    identifier = first.json()["file"]["id"]
    url = f"/api/v1/library/files/{identifier}"
    assert audit_context.client.get(url + "/content", headers=audit_context.bob).status_code == 403
    from app.services.storage.local import LocalStorageBackend
    fresh = LocalStorageBackend(str(audit_context.media))
    assert fresh.get(workspace_store.get_library_file(identifier)["storage_path"]) == b"Synthetic engineering document"
    def fail(*args, **kwargs):
        raise OSError("Synthetic storage deletion outage")
    monkeypatch.setattr(storage_service, "delete_verified", fail)
    assert audit_context.client.delete(url, headers=audit_context.alice).status_code == 503
    assert workspace_store.get_library_file(identifier)["status"] == "deleting"
    assert audit_context.client.post(url + "/reindex", headers=audit_context.alice).status_code == 409


def test_late_index_commit_cannot_resurrect_deleting_document(audit_context):
    record = workspace_store.record_library_file("synthetic.txt", "TXT", 3, "synthetic-path", status="processing", user_id="audit-alice")
    assert workspace_store.update_library_file_status(record["id"], "deleting", user_id="audit-alice")
    with pytest.raises(RuntimeError):
        commit_index(record["id"], 1, "audit-alice")
    assert workspace_store.get_library_file(record["id"])["status"] == "deleting"
    assert not workspace_store.update_library_file_status(record["id"], "failed", user_id="audit-alice")


@pytest.mark.asyncio
async def test_multipart_read_is_bounded_before_admission(audit_context):
    class HugeFile:
        filename = "synthetic.txt"
        content_type = "text/plain"
        sizes = []
        async def read(self, size=-1):
            self.sizes.append(size)
            return b"x" * (library.MAX_ATTACHMENT_SIZE_BYTES + 1)
    file = HugeFile()
    with pytest.raises(library.HTTPException) as error:
        await library.upload_file(file, None, "audit-alice")
    assert error.value.status_code == 413
    assert file.sizes == [library.MAX_ATTACHMENT_SIZE_BYTES + 1]
