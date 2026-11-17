"""Commit the metadata boundary before claiming a private index is usable."""

from app.services.tenant_access import active_account
from app.services.workspace_store import workspace_store


def commit_index(file_id: str, count: int, owner_id: str) -> None:
    if count <= 0 or not active_account(owner_id):
        raise RuntimeError("Document indexing cannot be committed")
    if not workspace_store.update_library_file_status(
        file_id=file_id, status="indexed", chunk_count=count, user_id=owner_id,
        expected_statuses=("processing",),
    ):
        raise RuntimeError("Document metadata commit failed")
    record = workspace_store.get_library_file(file_id)
    if not record or record.get("user_id") != owner_id or record.get("status") != "indexed" or record.get("chunk_count") != count:
        raise RuntimeError("Document metadata verification failed")
