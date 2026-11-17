"""Base Storage Abstraction for Jruva Document and Media Assets.

Decouples file metadata stored in PostgreSQL from underlying blob bytes
(local filesystem vs. Supabase Storage / S3 / Google Cloud Storage).
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional
from uuid import UUID

_IMAGE_RESERVATION = ContextVar("storage_image_reservation", default=None)
_DOCUMENT_WRITE = ContextVar("storage_document_write", default=None)


class StorageAdmissionError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@contextmanager
def document_write_scope(owner_id, file_id, filename):
    token = _DOCUMENT_WRITE.set((owner_id, str(UUID(file_id)), filename))
    try:
        yield
    finally:
        _DOCUMENT_WRITE.reset(token)


def current_document_write():
    return _DOCUMENT_WRITE.get()


@contextmanager
def image_write_scope(reservation_id):
    # Only server-created UUIDs can select deterministic image keys. Ordinary
    # document filenames cannot opt into this scope or overwrite an image key.
    identifier = str(UUID(reservation_id))
    token = _IMAGE_RESERVATION.set(identifier)
    try:
        yield
    finally:
        _IMAGE_RESERVATION.reset(token)


def current_image_reservation():
    return _IMAGE_RESERVATION.get()


class BaseStorageBackend(ABC):
    """Abstract contract for durable document and blob object storage."""

    def validate_upload(self, filename: str, content: bytes, content_type: str) -> None:
        """Optional backend policy admission, before creating a write receipt."""

    def reserved_image_path(self, reservation_id: str) -> str:
        raise NotImplementedError("This storage backend does not support durable image reservation keys")

    def reserved_document_path(self, owner_id: str, file_id: str, filename: str) -> str:
        raise NotImplementedError("This storage backend does not support durable document keys")

    @abstractmethod
    def put(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist raw bytes and return the durable storage path / URI."""
        pass

    @abstractmethod
    def get(self, storage_path: str) -> bytes:
        """Retrieve raw file content bytes from the storage path."""
        pass

    @abstractmethod
    def delete(self, storage_path: str) -> bool:
        """Delete file at the given storage path."""
        pass

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Check whether file exists at the given storage path."""
        pass

    @abstractmethod
    def metadata(self, storage_path: str) -> Dict[str, Any]:
        """Retrieve metadata for the stored object (size, content_type, created_at, etc.)."""
        pass

    # Backward-compatible convenience aliases
    def save_bytes(self, filename: str, content: bytes, content_type: Optional[str] = None) -> str:
        return self.put(filename=filename, content=content, content_type=content_type)

    def read_bytes(self, storage_path: str) -> bytes:
        return self.get(storage_path=storage_path)

    def delete_file(self, storage_path: str) -> bool:
        return self.delete(storage_path=storage_path)

    def delete_verified(self, storage_path: str) -> None:
        """Do not confuse a failed existence probe with confirmed absence."""
        self.delete_file(storage_path)
        try:
            self.metadata(storage_path)
        except FileNotFoundError:
            return
        raise RuntimeError("Blob remains after deletion")

    def file_exists(self, storage_path: str) -> bool:
        return self.exists(storage_path=storage_path)
