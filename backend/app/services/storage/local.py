"""Local Filesystem Storage Backend for Development and Single-Instance Deployment."""

import logging
import hashlib
import mimetypes
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.services.storage.base import BaseStorageBackend

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "uploads",
)

# Whitelist safe extensions for document library
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".docx", ".json", ".tsv",
    ".png", ".jpg", ".jpeg", ".webp"
}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal, hidden files, and command injection."""
    clean = os.path.basename(filename).strip()
    # Strip null bytes and control chars
    clean = clean.replace("\x00", "").replace("\r", "").replace("\n", "")
    # Remove path traversal tokens
    clean = re.sub(r"\.\.+", ".", clean)
    # Replace non-safe chars with underscore
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
    if not clean or clean.startswith("."):
        clean = f"document_{secrets.token_hex(4)}.dat"
    return clean


class LocalStorageBackend(BaseStorageBackend):
    """Stores files on the local filesystem with path-traversal and tenant protection."""

    def __init__(self, base_dir: str = DEFAULT_UPLOAD_DIR):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _resolve_safe_path(self, storage_path: str) -> str:
        """Resolve path and verify it remains strictly within base_dir."""
        # storage_path may be a relative name or absolute path inside base_dir
        if not os.path.isabs(storage_path):
            candidate = os.path.abspath(os.path.join(self.base_dir, storage_path))
        else:
            candidate = os.path.abspath(storage_path)

        # Path traversal guard
        candidate = os.path.realpath(candidate)
        base = os.path.realpath(self.base_dir)
        try:
            contained = os.path.normcase(os.path.commonpath([base, candidate])) == os.path.normcase(base)
        except ValueError:
            contained = False
        if not contained or os.path.normcase(candidate) == os.path.normcase(base):
            raise PermissionError(f"[SecurityAlert] Path traversal attempt detected: {storage_path}")
        return candidate

    def put(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        from app.services.storage.base import current_document_write, current_image_reservation
        reservation_id = current_image_reservation()
        document = current_document_write()
        if reservation_id:
            target_path = self.reserved_image_path(reservation_id)
        elif document:
            target_path = self.reserved_document_path(*document)
        else:
            clean_name = sanitize_filename(filename)
            safe_name = f"{secrets.token_hex(8)}_{clean_name}"
            target_path = os.path.join(self.base_dir, safe_name)

        # Validate path
        resolved = self._resolve_safe_path(target_path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        # Exclusive creation prevents a retry or collision from replacing a
        # previously accepted object. A partial file remains at its receipt's
        # known path, where explicit recovery can inspect it after a crash.
        with open(resolved, "xb") as f:
            f.write(content)
        return resolved

    def reserved_document_path(self, owner_id: str, file_id: str, filename: str) -> str:
        from uuid import UUID
        tenant = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
        return self._resolve_safe_path(os.path.join(self.base_dir, "documents", tenant, f"{UUID(file_id)}_{sanitize_filename(filename)}"))

    def reserved_image_path(self, reservation_id: str) -> str:
        from uuid import UUID
        return self._resolve_safe_path(os.path.join(self.base_dir, f"image_{UUID(reservation_id)}.png"))

    def get(self, storage_path: str) -> bytes:
        resolved = self._resolve_safe_path(storage_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"File not found: {storage_path}")
        with open(resolved, "rb") as f:
            return f.read()

    def delete(self, storage_path: str) -> bool:
        resolved = self._resolve_safe_path(storage_path)
        try:
            if os.path.exists(resolved):
                os.remove(resolved)
                return True
        except Exception as e:
            logger.warning(f"Failed to remove local file {storage_path}: {e}")
            return False
        return False


    def exists(self, storage_path: str) -> bool:
        try:
            resolved = self._resolve_safe_path(storage_path)
            return os.path.exists(resolved)
        except Exception:
            return False

    def metadata(self, storage_path: str) -> Dict[str, Any]:
        resolved = self._resolve_safe_path(storage_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"File not found: {storage_path}")
        stat = os.stat(resolved)
        mime, _ = mimetypes.guess_type(resolved)
        return {
            "size_bytes": stat.st_size,
            "content_type": mime or "application/octet-stream",
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "storage_path": resolved,
        }
