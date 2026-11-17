"""Storage package factory for JRUVA documents and assets."""

import os
from typing import Optional

from app.services.storage.base import BaseStorageBackend
from app.services.storage.local import LocalStorageBackend
from app.services.storage.s3_gcs import CloudObjectStorageBackend
from app.services.storage.supabase import SupabaseStorageBackend

_active_backend: Optional[BaseStorageBackend] = None


def get_storage_backend() -> BaseStorageBackend:
    """Retrieve the active storage backend based on environment configuration."""
    global _active_backend
    if _active_backend is None:
        from app.core.config import settings
        backend_type = (
            os.getenv("STORAGE_BACKEND") or getattr(settings, "STORAGE_BACKEND", "local")
        ).lower().strip()
        if backend_type == "local":
            _active_backend = LocalStorageBackend()
        elif backend_type == "supabase":
            bucket = os.getenv("STORAGE_BUCKET_NAME") or getattr(settings, "STORAGE_BUCKET_NAME", "jruva-documents")
            _active_backend = SupabaseStorageBackend(bucket_name=bucket)
        elif backend_type in ("gcs", "s3", "cloud"):
            bucket = os.getenv("STORAGE_BUCKET_NAME") or getattr(settings, "STORAGE_BUCKET_NAME", "jruva-documents")
            _active_backend = CloudObjectStorageBackend(bucket_name=bucket)
        else:
            raise ValueError(
                f"Unsupported storage backend: '{backend_type}'. "
                "Expected 'local', 'supabase', 'gcs', 's3', or 'cloud'."
            )
    return _active_backend


def reset_storage_backend_for_testing(backend: Optional[BaseStorageBackend] = None) -> None:
    """Helper to reset or inject a custom storage backend during testing."""
    global _active_backend
    _active_backend = backend


def close_storage_backend() -> None:
    """Close the active storage backend and any open HTTP sessions."""
    global _active_backend
    if _active_backend is not None and hasattr(_active_backend, "close"):
        try:
            _active_backend.close()
        except Exception:
            pass


storage_service = get_storage_backend()

__all__ = [
    "BaseStorageBackend",
    "LocalStorageBackend",
    "SupabaseStorageBackend",
    "CloudObjectStorageBackend",
    "get_storage_backend",
    "reset_storage_backend_for_testing",
    "close_storage_backend",
    "storage_service",
]
