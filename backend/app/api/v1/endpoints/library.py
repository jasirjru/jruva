"""Library & Knowledge Document Management API Endpoints."""
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api.v1.endpoints.sessions import get_session_owner_id
from app.services.rag.document_rag import ALLOWED_EXTENSIONS, MAX_ATTACHMENT_SIZE_BYTES, document_rag_service
from app.services.rag.qdrant_vector_store import QdrantVectorStore
from app.services.storage import storage_service
from app.services.workspace_store import workspace_store
from app.services.document_state import commit_index
from app.services.document_upload import persist_document
from app.services.storage.base import StorageAdmissionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/library", tags=["Knowledge Library"])


@router.get("/files", summary="List library knowledge documents")
async def list_files(
    project_id: Optional[str] = None,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Retrieve catalog of uploaded library files with indexing status, isolated to owner.

    Also reports whether new uploads are accepted, so the Library can disable its
    upload control instead of offering an action that can only fail.
    """
    from app.core.config import settings

    effective_user = owner_id
    return {
        "files": workspace_store.list_library_files(user_id=effective_user, project_id=project_id),
        "uploads_enabled": bool(settings.effective_enable_document_uploads),
        "max_file_bytes": min(MAX_ATTACHMENT_SIZE_BYTES, settings.DOCUMENT_MAX_OBJECT_BYTES),
    }


class UploadJSONRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    data: str = Field(..., max_length=4 * ((MAX_ATTACHMENT_SIZE_BYTES + 2) // 3) + 1024)
    file_type: Optional[str] = None
    project_id: Optional[str] = None


def _validate_file_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if not ext or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Permitted: PDF, TXT, MD, CSV, TSV, JSON, DOCX, XLSX, Code.",
        )
    return ext


def _ensure_uploads_enabled() -> None:
    from app.core.config import settings
    if not settings.effective_enable_document_uploads:
        raise HTTPException(
            status_code=503,
            detail="Document uploads are currently disabled until durable cloud storage is configured.",
        )


# Quota outcomes are known before any blob is written, so they get precise
# statuses instead of the "upload could not be confirmed" recovery message.
_QUOTA_ERROR_STATUS = {
    "invalid_size": 400,
    "object_size_exceeded": 413,
    "user_storage_quota_exceeded": 413,
    "user_file_count_exceeded": 409,
    "project_storage_quota_exceeded": 503,
}


def _persist_upload(filename, content_bytes, owner_id, project_id):
    from app.services.storage.document_quota import DocumentQuotaError

    try:
        return persist_document(storage_service, filename, content_bytes, owner_id, project_id)
    except StorageAdmissionError as error:
        raise HTTPException(413 if error.kind == "size" else 415, str(error)) from None
    except DocumentQuotaError as error:
        raise HTTPException(_QUOTA_ERROR_STATUS.get(error.code, 409), error.message) from None
    except Exception:
        logger.warning("Document storage outcome requires recovery")
        raise HTTPException(503, "Document upload could not be confirmed. Refresh your library before retrying; an incomplete upload may require recovery.") from None


@router.post("/upload-json", summary="Upload document via JSON base64")
async def upload_file_json(
    payload: UploadJSONRequest,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Upload document as base64 string, extract text, and index chunks with truthful status transitions."""
    _ensure_uploads_enabled()
    filename = payload.filename or "uploaded_document"
    ext = _validate_file_extension(filename)

    try:
        raw_b64 = payload.data.split(",")[-1] if "," in payload.data else payload.data
        content_bytes = base64.b64decode(raw_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 document payload.")

    file_size = len(content_bytes)
    if file_size == 0:
        raise HTTPException(status_code=422, detail="Document file is empty (0 bytes).")
    if file_size > MAX_ATTACHMENT_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size limit of {MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    effective_user = owner_id

    if payload.project_id:
        project = workspace_store.get_project(payload.project_id)
        if not project or project.get("user_id") != effective_user:
            raise HTTPException(403, "Forbidden project ownership")

    record = _persist_upload(filename, content_bytes, effective_user, payload.project_id)

    file_id = record["id"]

    # Extract, chunk, embed & index
    try:
        mime = payload.file_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        indexed_count, _ = document_rag_service.index_library_document(
            file_id=file_id,
            filename=filename,
            content_bytes=content_bytes,
            mime_type=mime,
            owner_id=effective_user,
            collection_name="user_knowledge_base",
        )
        commit_index(file_id, indexed_count, effective_user)
        record["status"] = "indexed"
        record["chunk_count"] = indexed_count
        return {"success": True, "file": record}
    except Exception as exc:
        logger.warning("Library document indexing failed")
        workspace_store.update_library_file_status(
            file_id=file_id,
            status="failed",
            chunk_count=0,
            user_id=effective_user,
            expected_statuses=("processing",),
        )
        record["status"] = "failed"
        record["chunk_count"] = 0
        raise HTTPException(status_code=422, detail="Document indexing failed. Please retry or upload a supported readable file.")


@router.post("/upload", summary="Upload document to Knowledge Library")
async def upload_file(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(default=None),
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Upload PDF, TXT, MD, CSV, DOCX to Library, extract text, and index chunks with truthful status."""
    _ensure_uploads_enabled()
    filename = file.filename or "uploaded_document"
    ext = _validate_file_extension(filename)

    try:
        content_bytes = await file.read(MAX_ATTACHMENT_SIZE_BYTES + 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read document payload.")

    file_size = len(content_bytes)
    if file_size == 0:
        raise HTTPException(status_code=422, detail="Document file is empty (0 bytes).")
    if file_size > MAX_ATTACHMENT_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size limit of {MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    effective_user = owner_id

    if project_id:
        project = workspace_store.get_project(project_id)
        if not project or project.get("user_id") != effective_user:
            raise HTTPException(403, "Forbidden project ownership")

    record = _persist_upload(filename, content_bytes, effective_user, project_id)

    file_id = record["id"]

    try:
        mime = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        indexed_count, _ = document_rag_service.index_library_document(
            file_id=file_id,
            filename=filename,
            content_bytes=content_bytes,
            mime_type=mime,
            owner_id=effective_user,
            collection_name="user_knowledge_base",
        )
        commit_index(file_id, indexed_count, effective_user)
        record["status"] = "indexed"
        record["chunk_count"] = indexed_count
        return {"success": True, "file": record}
    except Exception as exc:
        logger.warning("Library document indexing failed")
        workspace_store.update_library_file_status(
            file_id=file_id,
            status="failed",
            chunk_count=0,
            user_id=effective_user,
            expected_statuses=("processing",),
        )
        record["status"] = "failed"
        record["chunk_count"] = 0
        raise HTTPException(status_code=422, detail="Document indexing failed. Please retry or upload a supported readable file.")


@router.post("/files/{file_id}/reindex", summary="Reindex document in Knowledge Library")
async def reindex_file(
    file_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Trigger re-extraction, chunking, and embedding for a previously uploaded document."""
    file_record = workspace_store.get_library_file(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    file_owner = file_record.get("user_id")
    effective_user = owner_id
    if file_owner != effective_user:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to reindex this document.",
        )
    if file_record.get("status") in ("deleting", "uploading", "processing"):
        raise HTTPException(409, "Document has an active or incomplete operation. Refresh the library or request recovery.")

    storage_path = file_record.get("storage_path")
    if not storage_path or not storage_service.exists(storage_path):
        raise HTTPException(status_code=404, detail="Underlying document storage file not found")

    content_bytes = storage_service.get(storage_path)
    filename = file_record.get("filename", "document")
    mime, _ = mimetypes.guess_type(filename)

    claimed = workspace_store.update_library_file_status(
        file_id=file_id,
        status="processing",
        chunk_count=0,
        user_id=effective_user,
        expected_statuses=("indexed", "failed"),
    )
    if not claimed:
        raise HTTPException(409, "Document operation changed. Refresh the library.")

    try:
        indexed_count, _ = document_rag_service.index_library_document(
            file_id=file_id,
            filename=filename,
            content_bytes=content_bytes,
            mime_type=mime or "application/octet-stream",
            owner_id=effective_user,
            collection_name="user_knowledge_base",
        )
        commit_index(file_id, indexed_count, effective_user)
        updated_record = workspace_store.get_library_file(file_id)
        return {"success": True, "file": updated_record}
    except Exception as exc:
        workspace_store.update_library_file_status(
            file_id=file_id,
            status="failed",
            chunk_count=0,
            user_id=effective_user,
            expected_statuses=("processing",),
        )
        raise HTTPException(status_code=422, detail="Document reindexing failed. Please retry later.")


@router.get("/search", summary="Search indexed library documents")
async def search_library(
    q: str = Query(..., min_length=1, max_length=200, description="Query string"),
    limit: int = Query(default=5, ge=1, le=20, description="Max passages to return"),
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Retrieve ranked semantic & lexical passages from authorized user's indexed library documents."""
    effective_user = owner_id
    store = QdrantVectorStore()
    results = store.search_hybrid("user_knowledge_base", query=q, limit=limit, owner_id=effective_user)
    return {
        "query": q,
        "user_id": effective_user,
        "results": results,
        "count": len(results),
    }


@router.get("/files/{file_id}/passages/{passage_id}", summary="Inspect grounded passage excerpt")
async def get_file_passage(
    file_id: str,
    passage_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Retrieve verified passage excerpt, page number, and metadata for Ground-Truth Studio inspection."""
    store = QdrantVectorStore()

    # 1. Institutional / Domain Knowledge Passages
    if file_id in ("domain", "institutional", "jru", "stem") or file_id.startswith("domain_"):
        passage = store.get_passage_by_id("jru_knowledge_base", passage_id=passage_id)
        if not passage:
            passage = store.get_passage_by_id("stem_knowledge_base", passage_id=passage_id)
        if not passage:
            raise HTTPException(status_code=404, detail="Passage not found")
        return {
            "passage_id": passage["passage_id"],
            "file_id": file_id,
            "doc_name": passage.get("doc_name", "Institutional Knowledge"),
            "page": passage.get("page", 1),
            "section": passage.get("section", ""),
            "breadcrumb": passage.get("breadcrumb", ""),
            "text": passage["text"],
            "char_count": len(passage["text"]),
            "download_url": None,
            "source_type": "domain",
            "verified_owner": True,
        }

    # 2. User Private Knowledge Document Passages
    file_record = workspace_store.get_library_file(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="Passage not found")

    file_owner = file_record.get("user_id")
    effective_user = owner_id
    if file_owner != effective_user:
        # Standard security rule: return 404 to avoid revealing whether file exists
        raise HTTPException(status_code=404, detail="Passage not found")

    passage = store.get_passage_by_id(
        "user_knowledge_base",
        passage_id=passage_id,
        file_id=file_id,
        owner_id=effective_user,
    )
    if not passage:
        raise HTTPException(status_code=404, detail="Passage not found")

    return {
        "passage_id": passage["passage_id"],
        "file_id": file_id,
        "doc_name": passage.get("doc_name") or file_record.get("filename", "Library Document"),
        "page": passage.get("page", 1),
        "section": passage.get("section", ""),
        "breadcrumb": passage.get("breadcrumb", ""),
        "text": passage["text"],
        "char_count": len(passage["text"]),
        "download_url": f"/api/v1/library/files/{file_id}/content",
        "source_type": "document",
        "verified_owner": True,
    }


@router.get("/files/{file_id}/content", summary="Download/retrieve file content")
async def get_file_content(
    file_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Retrieve raw file content bytes from storage with tenant authorization check."""
    file_record = workspace_store.get_library_file(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    file_owner = file_record.get("user_id")
    effective_user = owner_id
    if file_owner != effective_user:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to access this knowledge document.",
        )

    storage_path = file_record.get("storage_path")
    if not storage_path or not storage_service.exists(storage_path):
        raise HTTPException(status_code=404, detail="File content not found in storage")

    content_bytes = storage_service.get(storage_path)
    from app.services.document_download import document_download_response
    return document_download_response(content_bytes, file_record["filename"])


@router.delete("/files/{file_id}", summary="Delete knowledge file")
async def delete_file(
    file_id: str,
    owner_id: Optional[str] = Depends(get_session_owner_id),
):
    """Permanently delete file from disk, vector store, and database after verifying caller ownership."""
    file_record = workspace_store.get_library_file(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    file_owner = file_record.get("user_id")
    effective_user = owner_id
    if file_owner != effective_user:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You do not have permission to delete this knowledge document.",
        )
    if file_record.get("status") == "uploading":
        raise HTTPException(409, "Upload outcome is unknown. Confirm writer and remote upload settlement before recovery.")

    # Hide the document before external cleanup; retain retryable metadata on failure.
    if not workspace_store.update_library_file_status(file_id, "deleting", user_id=effective_user, expected_statuses=("indexed", "failed", "processing", "deleting")):
        raise HTTPException(503, "Document deletion could not be started. Please retry.")
    try:
        store = QdrantVectorStore()
        store.delete_document_chunks("user_knowledge_base", file_id=file_id, owner_id=effective_user)
    except Exception:
        logger.warning("Document vector cleanup failed")
        raise HTTPException(503, "Document deletion is incomplete. Please retry later.")

    # 2. Delete storage file and DB record
    try:
        success = workspace_store.delete_library_file(file_id, user_id=effective_user)
    except Exception:
        raise HTTPException(503, "Document deletion is incomplete. Please retry later.")
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True}
