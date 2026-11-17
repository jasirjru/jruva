"""Fail-closed checks shared by private retrieval and background work."""


def active_account(user_id: str | None) -> bool:
    if not user_id or user_id in ("guest", "public") or user_id.startswith("anon_"):
        return False
    from app.services.auth_service import auth_service
    user = auth_service.get_user_by_id(user_id)
    return bool(user and user.get("id") == user_id and not user.get("deletion_pending"))


def private_chunk_visible(owner_id: str | None, chunk: dict) -> bool:
    try:
        if not active_account(owner_id) or chunk.get("owner_id") != owner_id or not chunk.get("file_id"):
            return False
        from app.services.workspace_store import workspace_store
        record = workspace_store.get_library_file(chunk["file_id"])
        return bool(record and record.get("user_id") == owner_id and record.get("status") == "indexed")
    except Exception:  # noqa: BLE001 -- metadata/auth failure must deny private retrieval
        return False
