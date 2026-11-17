from tests.private_accounts import private_account_headers
"""Integration tests for Genuine End-to-End Document Intelligence & RAG Ingestion Pipeline.

Validates the full lifecycle:
Upload -> Validate -> Store -> Extract -> Chunk -> Embed -> Index -> Verify -> Retrieve.
Enforces truthful indexing status transitions (pending/processing/indexed/failed),
cross-user tenant isolation, and vector index cleanup upon deletion.
"""
import base64
import io
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.services.chat_service import check_user_library_knowledge
from app.services.rag.qdrant_vector_store import QdrantVectorStore
from app.services.tools.base import ToolContext
from app.services.tools.builtin import DocumentLookupInput, DocumentLookupTool
from app.services.workspace_store import workspace_store

client = TestClient(app)


def create_synthetic_pdf(text: str) -> bytes:
    """Generate genuine in-memory PDF binary containing specific test text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    buf.seek(0)
    return buf.getvalue()


class TestDocumentIngestionAndRetrieval:
    """Comprehensive test suite verifying truthful indexing and reliable RAG retrieval."""

    def test_valid_text_document_ingestion_and_retrieval(self):
        """Verify upload, truthful indexing, and genuine retrieval of a text document."""
        user_id = "user_atlas_reviewer"
        doc_text = "The fictional JRUVA Project Atlas launches on 14 March 2032. The project coordinator is Mira Sen."
        b64_content = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")

        # 1. Upload via /upload-json
        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "project_atlas_spec.txt",
                "file_type": "text/plain",
                "data": f"data:text/plain;base64,{b64_content}",
            },
        )
        assert res.status_code == 200, res.text
        file_data = res.json()["file"]
        file_id = file_data["id"]

        # Truthful status: Must be marked 'indexed' and chunk_count >= 1
        assert file_data["status"] == "indexed"
        assert file_data["chunk_count"] >= 1

        try:
            # 2. Query /api/v1/library/search
            search_res = client.get(
                "/api/v1/library/search?q=When does Project Atlas launch?",
                headers=private_account_headers(user_id),
            )
            assert search_res.status_code == 200
            search_data = search_res.json()
            assert search_data["count"] >= 1
            top_passage = search_data["results"][0]
            assert "14 March 2032" in top_passage["text"]
            assert top_passage["doc_name"] == "project_atlas_spec.txt"

            # 3. Direct RAG grounding helper check
            effective_user = user_id
            chunks, citations = check_user_library_knowledge(
                "When does Project Atlas launch?",
                user_id=effective_user,
                top_k=2,
            )
            assert len(chunks) >= 1
            assert "14 March 2032" in chunks[0]["text"]
            assert len(citations) >= 1
            assert "project_atlas_spec.txt" in citations[0].title
        finally:
            # Cleanup
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    def test_valid_pdf_document_ingestion_and_retrieval(self):
        """Verify upload and indexing of a genuine PDF, extracting text and citations."""
        user_id = "user_pdf_engineer"
        pdf_text = "Project Atlas budget is 45 million credits. Headquarters are situated in Sector 7."
        pdf_bytes = create_synthetic_pdf(pdf_text)

        # Upload via multipart /upload
        files = {
            "file": ("atlas_budget.pdf", pdf_bytes, "application/pdf")
        }
        res = client.post(
            "/api/v1/library/upload",
            headers=private_account_headers(user_id),
            files=files,
        )
        assert res.status_code == 200, res.text
        file_data = res.json()["file"]
        file_id = file_data["id"]

        assert file_data["status"] == "indexed"
        assert file_data["chunk_count"] >= 1

        try:
            # Retrieve via search
            search_res = client.get(
                "/api/v1/library/search?q=What is the budget of Project Atlas in Sector 7?",
                headers=private_account_headers(user_id),
            )
            assert search_res.status_code == 200
            search_data = search_res.json()
            assert search_data["count"] >= 1
            matched = search_data["results"][0]
            assert "45 million credits" in matched["text"]
            assert matched["page"] == 1
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    def test_invalid_corrupted_pdf_rejection(self):
        """Verify that a corrupt or unreadable PDF rejects gracefully and does not fake indexing."""
        user_id = "user_qa_tester"
        corrupted_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nCORRUPT_BINARY_DATA_\xff\xfe\x00\x01\x02_TRUNCATED"

        files = {
            "file": ("corrupt_document.pdf", corrupted_bytes, "application/pdf")
        }
        res = client.post(
            "/api/v1/library/upload",
            headers=private_account_headers(user_id),
            files=files,
        )
        # Truthful rejection: HTTP 422 Unprocessable Entity
        assert res.status_code == 422
        err_msg = res.json()["detail"]
        assert "indexing failed" in err_msg.lower() or "unable to extract" in err_msg.lower()

        # Verify no fake indexed files exist for this user with this filename
        files_res = client.get("/api/v1/library/files", headers=private_account_headers(user_id))
        assert files_res.status_code == 200
        for f in files_res.json()["files"]:
            if f["filename"] == "corrupt_document.pdf":
                assert f["status"] == "failed"
                assert f["chunk_count"] == 0
                # Cleanup failed record
                client.delete(f"/api/v1/library/files/{f['id']}", headers=private_account_headers(user_id))

    def test_empty_document_rejection(self):
        """Verify that an empty (0-byte) document is rejected with HTTP 422."""
        user_id = "user_qa_tester"
        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "empty_document.txt",
                "file_type": "text/plain",
                "data": "data:text/plain;base64,",  # empty base64
            },
        )
        assert res.status_code == 422
        assert "empty" in res.json()["detail"].lower()

    def test_unsupported_extension_rejection(self):
        """Verify that forbidden/unsupported file formats are rejected with HTTP 400."""
        user_id = "user_qa_tester"
        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "executable_payload.exe",
                "file_type": "application/x-msdownload",
                "data": "data:application/octet-stream;base64,TVqQAAMAAAAEAAAA//8AALgAAAAAAAAA",
            },
        )
        assert res.status_code == 400
        assert "unsupported file extension" in res.json()["detail"].lower()

    def test_cross_user_isolation(self):
        """Verify that documents uploaded by User A are strictly inaccessible to User B."""
        user_a = "user_alpha_confidential"
        user_b = "user_beta_stranger"

        secret_text = "Project Classified X: The access passkey is OmegaTitan9988."
        b64_content = base64.b64encode(secret_text.encode("utf-8")).decode("ascii")

        # User A uploads secret document
        res_a = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_a),
            json={
                "filename": "classified_plan.txt",
                "data": f"data:text/plain;base64,{b64_content}",
            },
        )
        assert res_a.status_code == 200
        file_id = res_a.json()["file"]["id"]

        try:
            # 1. User A can retrieve it
            search_a = client.get(
                "/api/v1/library/search?q=What is the access passkey for Project Classified X?",
                headers=private_account_headers(user_a),
            )
            assert search_a.status_code == 200
            assert search_a.json()["count"] >= 1
            assert "OmegaTitan9988" in search_a.json()["results"][0]["text"]

            # 2. User B querying the exact same question gets 0 passages
            search_b = client.get(
                "/api/v1/library/search?q=What is the access passkey for Project Classified X?",
                headers=private_account_headers(user_b),
            )
            assert search_b.status_code == 200
            assert search_b.json()["count"] == 0

            # 3. Direct chat knowledge helper check for User B
            chunks_b, citations_b = check_user_library_knowledge(
                "What is the access passkey for Project Classified X?",
                user_id=user_b,
            )
            assert len(chunks_b) == 0
            assert len(citations_b) == 0

            # 4. User B cannot access file content or delete file
            content_b = client.get(
                f"/api/v1/library/files/{file_id}/content",
                headers=private_account_headers(user_b),
            )
            assert content_b.status_code == 403

            delete_b = client.delete(
                f"/api/v1/library/files/{file_id}",
                headers=private_account_headers(user_b),
            )
            assert delete_b.status_code == 403
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_a))

    def test_document_deletion_cleans_vector_index(self):
        """Verify that deleting a document purges all vectors from Qdrant, making it unsearchable."""
        user_id = "user_lifecycle_test"
        doc_text = "Temporary quantum telemetry beacon alpha operates at frequency 942.5 MHz."
        b64_content = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")

        # Upload
        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "quantum_beacon.txt",
                "data": f"data:text/plain;base64,{b64_content}",
            },
        )
        assert res.status_code == 200
        file_id = res.json()["file"]["id"]

        # Verify searchable
        search_before = client.get(
            "/api/v1/library/search?q=quantum telemetry beacon frequency",
            headers=private_account_headers(user_id),
        )
        assert search_before.json()["count"] >= 1

        # Delete document
        del_res = client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))
        assert del_res.status_code == 200

        # Verify no longer searchable
        search_after = client.get(
            "/api/v1/library/search?q=quantum telemetry beacon frequency",
            headers=private_account_headers(user_id),
        )
        assert search_after.json()["count"] == 0

    def test_reindex_idempotency_and_recovery(self):
        """Verify reindexing an existing document updates index chunks cleanly."""
        user_id = "user_reindex_test"
        doc_text = "Orbital station Kepler-44 completed structural retrofit on 12 June 2030."
        b64_content = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")

        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "kepler_station.txt",
                "data": f"data:text/plain;base64,{b64_content}",
            },
        )
        assert res.status_code == 200
        file_id = res.json()["file"]["id"]

        try:
            # Trigger reindex
            reindex_res = client.post(
                f"/api/v1/library/files/{file_id}/reindex",
                headers=private_account_headers(user_id),
            )
            assert reindex_res.status_code == 200
            file_data = reindex_res.json()["file"]
            assert file_data["status"] == "indexed"
            assert file_data["chunk_count"] >= 1

            # Verify still searchable
            search_res = client.get(
                "/api/v1/library/search?q=Kepler-44 retrofit date",
                headers=private_account_headers(user_id),
            )
            assert search_res.json()["count"] >= 1
            assert "12 June 2030" in search_res.json()["results"][0]["text"]
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    @pytest.mark.asyncio
    async def test_document_lookup_tool_execution(self):
        """Verify that DocumentLookupTool runs genuine hybrid search across user's indexed passages."""
        user_id = "user_agent_tool_tester"
        doc_text = "Autonomous drone fleet firmware version 4.19 patches waypoint drift anomaly."
        b64_content = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")

        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "drone_firmware.txt",
                "data": f"data:text/plain;base64,{b64_content}",
            },
        )
        assert res.status_code == 200
        file_id = res.json()["file"]["id"]

        try:
            tool = DocumentLookupTool()
            context = ToolContext(user_id=user_id, session_id="test_session")
            tool_res = await tool.run(DocumentLookupInput(query="waypoint drift anomaly firmware"), context)

            assert tool_res["total_passages"] >= 1
            top_p = tool_res["matched_passages"][0]
            assert "version 4.19" in top_p["text"]
            assert top_p["filename"] == "drone_firmware.txt"
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))
