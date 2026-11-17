"""Supabase Storage Provider for Cloud-Native Document & Asset Durability.

Communicates with Supabase Storage REST API using httpx without requiring heavy external SDKs.
Enables horizontal scaling across Cloud Run container instances.
"""

import hashlib
import logging
import mimetypes
import os
import re
import secrets
from typing import Any, Dict, Optional
import httpx

from app.services.storage.base import BaseStorageBackend, StorageAdmissionError

logger = logging.getLogger(__name__)


def sanitize_storage_key(filename: str) -> str:
    """Sanitize filename into a clean cloud object key."""
    clean = os.path.basename(filename).strip()
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
    if not clean or clean.startswith("."):
        clean = f"document_{secrets.token_hex(4)}.dat"
    return f"{secrets.token_hex(8)}_{clean}"


class SupabaseStorageBackend(BaseStorageBackend):
    """Stores files in Supabase Storage buckets via REST API."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        service_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        from app.core.config import settings
        self.supabase_url = (
            supabase_url or os.getenv("SUPABASE_URL") or getattr(settings, "SUPABASE_URL", "")
        ).rstrip("/")
        self.service_key = (
            service_key
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "")
            or os.getenv("SUPABASE_KEY")
            or getattr(settings, "SUPABASE_KEY", "")
        )
        self.bucket_name = (
            bucket_name
            or os.getenv("STORAGE_BUCKET_NAME")
            or getattr(settings, "STORAGE_BUCKET_NAME", "jruva-documents")
        )
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.bucket_name):
            raise ValueError("Invalid Supabase storage bucket name")
        self._custom_client = http_client
        self._internal_client: Optional[httpx.Client] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.service_key)

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError("Supabase Storage is not configured.")

    def validate_upload(self, filename: str, content: bytes, content_type: str) -> None:
        """Mirror operator-configured bucket limits before any write is admitted."""
        try:
            max_bytes = int(os.getenv("STORAGE_MAX_OBJECT_BYTES", str(10 * 1024 * 1024)))
        except ValueError:
            raise RuntimeError("Invalid storage object size limit configuration.") from None
        if max_bytes <= 0:
            raise RuntimeError("Storage object size limit must be positive.")
        if not content or len(content) > max_bytes:
            raise StorageAdmissionError("size", "Object exceeds the storage size policy or is empty.")
        allowed = {
            item.strip().lower() for item in os.getenv(
                "STORAGE_ALLOWED_MIME_TYPES", "application/pdf,image/png,image/jpeg,image/webp"
            ).split(",") if item.strip()
        }
        if not allowed:
            raise RuntimeError("Storage MIME policy must not be empty.")
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime not in allowed:
            raise StorageAdmissionError("mime", "Object MIME type is not allowed by the storage policy.")

    def _object_key(self, storage_path: str) -> str:
        """Accept canonical references and legacy safe keys within this bucket."""
        if not isinstance(storage_path, str):
            raise ValueError("Invalid Supabase object reference")
        prefix = f"supabase://{self.bucket_name}/"
        key = storage_path.removeprefix(prefix)
        if not key or not re.fullmatch(r"[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*", key):
            raise ValueError("Invalid Supabase object reference")
        if any(segment in {".", ".."} for segment in key.split("/")):
            raise ValueError("Invalid Supabase object reference")
        return key

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Keep provider response bodies and transport diagnostics out of errors."""
        self._require_configured()
        try:
            return self._get_client().request(method, url, **kwargs)
        except httpx.HTTPError:
            raise IOError("Supabase Storage request failed.") from None

    @staticmethod
    def _response_error(operation: str, response: httpx.Response) -> IOError:
        return IOError(f"Supabase Storage {operation} failed ({response.status_code}).")

    def _get_client(self) -> httpx.Client:
        if self._custom_client:
            return self._custom_client
        if self._internal_client is None or self._internal_client.is_closed:
            self._internal_client = httpx.Client(
                base_url=f"{self.supabase_url}/storage/v1",
                headers={
                    "Authorization": f"Bearer {self.service_key}",
                    "apikey": self.service_key,
                },
                timeout=30.0,
            )
        return self._internal_client

    def close(self) -> None:
        """Explicitly close internally owned HTTP client connections."""
        if self._internal_client and not self._internal_client.is_closed:
            self._internal_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def put(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.is_configured:
            raise RuntimeError("Supabase Storage is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

        from app.services.storage.base import current_document_write, current_image_reservation
        reservation_id = current_image_reservation()
        document = current_document_write()
        if reservation_id:
            key = self._object_key(self.reserved_image_path(reservation_id))
        elif document:
            key = self._object_key(self.reserved_document_path(*document))
        else:
            key = sanitize_storage_key(filename)
        mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self.validate_upload(filename, content, mime)
        mime = mime.split(";", 1)[0].strip().lower()

        url = f"/object/{self.bucket_name}/{key}"
        res = self._request(
            "POST",
            url,
            content=content,
            headers={"Content-Type": mime, "x-upsert": "false"},
        )
        if res.status_code not in (200, 201):
            raise self._response_error("put", res)

        # Return the canonical object URI or storage key
        return f"supabase://{self.bucket_name}/{key}"

    def reserved_image_path(self, reservation_id: str) -> str:
        from uuid import UUID
        return f"supabase://{self.bucket_name}/image-reservations/{UUID(reservation_id)}.png"

    def reserved_document_path(self, owner_id: str, file_id: str, filename: str) -> str:
        from uuid import UUID

        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("Document owner is required")
        owner = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
        clean = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
        if not clean or clean.startswith("."):
            clean = "document.dat"
        return f"supabase://{self.bucket_name}/documents/{owner}/{UUID(file_id)}_{clean}"

    def get(self, storage_path: str) -> bytes:
        key = self._object_key(storage_path)
        url = f"/object/authenticated/{self.bucket_name}/{key}"
        res = self._request("GET", url)
        if res.status_code == 404:
            raise FileNotFoundError("Supabase object not found")
        if res.status_code != 200:
            raise self._response_error("get", res)
        return res.content

    def delete(self, storage_path: str) -> bool:
        key = self._object_key(storage_path)
        url = f"/object/{self.bucket_name}"
        res = self._request("DELETE", url, json={"prefixes": [key]})
        if res.status_code == 404:
            return False
        if res.status_code not in (200, 204):
            raise self._response_error("delete", res)
        return True

    def exists(self, storage_path: str) -> bool:
        key = self._object_key(storage_path)
        url = f"/object/info/authenticated/{self.bucket_name}/{key}"
        res = self._request("GET", url)
        if res.status_code == 404:
            return False
        if res.status_code != 200:
            raise self._response_error("existence check", res)
        return True

    def metadata(self, storage_path: str) -> Dict[str, Any]:
        key = self._object_key(storage_path)
        url = f"/object/info/authenticated/{self.bucket_name}/{key}"
        res = self._request("GET", url)
        if res.status_code == 404:
            raise FileNotFoundError("Supabase object not found")
        if res.status_code != 200:
            raise self._response_error("metadata request", res)
        try:
            data = res.json()
            if not isinstance(data, dict):
                raise ValueError("Invalid metadata")
        except ValueError:
            raise IOError("Supabase Storage returned invalid metadata.") from None
        return {
            "size_bytes": data.get("size", 0),
            "content_type": data.get("content_type") or data.get("mimetype") or "application/octet-stream",
            "storage_path": storage_path,
            "created_at": data.get("created_at"),
            "modified_at": data.get("last_modified") or data.get("updated_at"),
        }
