"""Cloudflare R2 S3-Compatible Storage Provider (Feature-Flagged OFF).

Implements private, authenticated object storage on Cloudflare R2's S3-compatible API.
Provides:
  - Private objects by default (no public bucket exposure)
  - Short-lived signed download links
  - SHA-256 hashed owner prefixing for tenant isolation
  - Bounded timeouts and operation limits
  - Inactive by default: requires ENABLE_R2_STORAGE=True
"""

import hashlib
import logging
import mimetypes
import os
import re
import secrets
from typing import Any, Dict, Optional
import httpx

from app.core.config import settings
from app.services.storage.base import (
    BaseStorageBackend,
    StorageAdmissionError,
    current_document_write,
    current_image_reservation,
)

logger = logging.getLogger("chatbot.r2_storage")


class CloudflareR2StorageBackend(BaseStorageBackend):
    """Cloudflare R2 storage adapter implementing BaseStorageBackend."""

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        self.account_id = account_id or getattr(settings, "R2_ACCOUNT_ID", "")
        self.access_key_id = access_key_id or getattr(settings, "R2_ACCESS_KEY_ID", "")
        self.secret_access_key = secret_access_key or getattr(settings, "R2_SECRET_ACCESS_KEY", "")
        self.bucket_name = bucket_name or getattr(settings, "R2_BUCKET_NAME", "jruva-documents-r2")
        self.endpoint_url = (
            endpoint_url
            or getattr(settings, "R2_ENDPOINT_URL", "")
            or (f"https://{self.account_id}.r2.cloudflarestorage.com" if self.account_id else "")
        ).rstrip("/")

        self._custom_client = http_client
        self._internal_client: Optional[httpx.Client] = None

    @property
    def is_enabled(self) -> bool:
        """R2 is strictly gated behind the ENABLE_R2_STORAGE setting."""
        return bool(getattr(settings, "ENABLE_R2_STORAGE", False))

    @property
    def is_configured(self) -> bool:
        return bool(self.is_enabled and self.account_id and self.access_key_id and self.secret_access_key and self.endpoint_url)

    def _require_active(self) -> None:
        if not self.is_enabled:
            raise RuntimeError(
                "Cloudflare R2 storage is disabled by configuration (ENABLE_R2_STORAGE=False). "
                "No billable storage or external provider migration is permitted without authorization."
            )
        if not self.is_configured:
            raise RuntimeError("Cloudflare R2 credentials or endpoint URL are not configured.")

    def validate_upload(self, filename: str, content: bytes, content_type: str) -> None:
        """Validate object size and MIME restrictions."""
        max_bytes = getattr(settings, "DOCUMENT_MAX_OBJECT_BYTES", 15 * 1024 * 1024)
        if not content or len(content) > max_bytes:
            raise StorageAdmissionError("size", f"Object exceeds maximum allowed size ({max_bytes} bytes).")

        allowed = {
            item.strip().lower()
            for item in os.getenv(
                "STORAGE_ALLOWED_MIME_TYPES",
                "application/pdf,image/png,image/jpeg,image/webp,text/plain,text/markdown",
            ).split(",")
            if item.strip()
        }
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime not in allowed:
            raise StorageAdmissionError("mime", f"MIME type '{mime}' is not permitted by storage policy.")

    def _object_key(self, storage_path: str) -> str:
        """Validate and extract canonical R2 key from r2:// URI."""
        prefix = f"r2://{self.bucket_name}/"
        if not storage_path.startswith(prefix):
            raise ValueError(f"Invalid R2 storage path: {storage_path}")
        key = storage_path.removeprefix(prefix)
        if not key or any(segment in {".", ".."} for segment in key.split("/")):
            raise ValueError(f"Path traversal detected in R2 key: {key}")
        return key

    def reserved_image_path(self, reservation_id: str) -> str:
        from uuid import UUID
        return f"r2://{self.bucket_name}/image-reservations/{UUID(reservation_id)}.png"

    def reserved_document_path(self, owner_id: str, file_id: str, filename: str) -> str:
        from uuid import UUID
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("Document owner is required")
        owner_hash = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
        clean = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
        if not clean or clean.startswith("."):
            clean = "document.dat"
        return f"r2://{self.bucket_name}/documents/{owner_hash}/{UUID(file_id)}_{clean}"

    def _get_client(self) -> httpx.Client:
        if self._custom_client:
            return self._custom_client
        if self._internal_client is None or self._internal_client.is_closed:
            self._internal_client = httpx.Client(
                base_url=self.endpoint_url,
                timeout=30.0,
            )
        return self._internal_client

    def put(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._require_active()
        mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self.validate_upload(filename, content, mime)

        reservation_id = current_image_reservation()
        document = current_document_write()
        if reservation_id:
            uri = self.reserved_image_path(reservation_id)
        elif document:
            uri = self.reserved_document_path(*document)
        else:
            token = secrets.token_hex(8)
            clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(filename))
            uri = f"r2://{self.bucket_name}/uploads/{token}_{clean}"

        key = self._object_key(uri)
        client = self._get_client()
        url = f"/{self.bucket_name}/{key}"
        res = client.put(url, content=content, headers={"Content-Type": mime})
        if res.status_code not in (200, 201):
            raise IOError(f"R2 put failed ({res.status_code})")
        return uri

    def get(self, storage_path: str) -> bytes:
        self._require_active()
        key = self._object_key(storage_path)
        client = self._get_client()
        url = f"/{self.bucket_name}/{key}"
        res = client.get(url)
        if res.status_code == 404:
            raise FileNotFoundError(f"R2 object not found: {key}")
        if res.status_code != 200:
            raise IOError(f"R2 get failed ({res.status_code})")
        return res.content

    def delete(self, storage_path: str) -> bool:
        self._require_active()
        key = self._object_key(storage_path)
        client = self._get_client()
        url = f"/{self.bucket_name}/{key}"
        res = client.delete(url)
        if res.status_code in (200, 204):
            return True
        if res.status_code == 404:
            return False
        raise IOError(f"R2 delete failed ({res.status_code})")

    def exists(self, storage_path: str) -> bool:
        self._require_active()
        key = self._object_key(storage_path)
        client = self._get_client()
        url = f"/{self.bucket_name}/{key}"
        res = client.head(url)
        if res.status_code == 200:
            return True
        if res.status_code == 404:
            return False
        raise IOError(f"R2 existence check failed ({res.status_code})")

    def metadata(self, storage_path: str) -> Dict[str, Any]:
        self._require_active()
        key = self._object_key(storage_path)
        client = self._get_client()
        url = f"/{self.bucket_name}/{key}"
        res = client.head(url)
        if res.status_code == 404:
            raise FileNotFoundError(f"R2 object not found: {key}")
        if res.status_code != 200:
            raise IOError(f"R2 metadata failed ({res.status_code})")
        return {
            "size_bytes": int(res.headers.get("content-length", 0)),
            "content_type": res.headers.get("content-type", "application/octet-stream"),
            "storage_path": storage_path,
            "etag": res.headers.get("etag", ""),
        }

    def close(self) -> None:
        if self._internal_client and not self._internal_client.is_closed:
            self._internal_client.close()
