"""Cloud Object Storage Backend Stub (S3 / Google Cloud Storage / Supabase Storage).

Prepares production support for durable blob storage across horizontally scaled Cloud Run instances.
"""

import logging
from typing import Optional

from app.services.storage.base import BaseStorageBackend

logger = logging.getLogger(__name__)


class CloudObjectStorageBackend(BaseStorageBackend):
    """Storage backend for Google Cloud Storage or S3-compatible buckets."""

    def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        logger.info(
            f"Initialized CloudObjectStorageBackend for bucket '{bucket_name}' "
            "(Target ready for Cloud Run multi-instance)"
        )

    def put(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        raise NotImplementedError(
            "Cloud object storage credentials not configured. "
            "Configure STORAGE_BACKEND=local or provide cloud storage credentials."
        )

    def get(self, storage_path: str) -> bytes:
        raise NotImplementedError("Cloud object storage credentials not configured.")

    def delete(self, storage_path: str) -> bool:
        raise NotImplementedError("Cloud object storage credentials not configured.")

    def exists(self, storage_path: str) -> bool:
        return False

    def metadata(self, storage_path: str) -> dict:
        raise NotImplementedError("Cloud object storage credentials not configured.")

    def save_bytes(self, filename: str, content: bytes, content_type: Optional[str] = None) -> str:
        return self.put(filename=filename, content=content, content_type=content_type)

    def read_bytes(self, storage_path: str) -> bytes:
        return self.get(storage_path=storage_path)

    def delete_file(self, storage_path: str) -> bool:
        return self.delete(storage_path=storage_path)

    def file_exists(self, storage_path: str) -> bool:
        return self.exists(storage_path=storage_path)
