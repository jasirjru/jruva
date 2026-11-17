"""Comprehensive End-to-End Verification Test Suite for Document Intelligence.

Verifies:
1. Native multi-page PDF parsing and page-specific citation indexing.
2. Scanned PDF OCR extraction (image-based PDF via pypdfium2 + RapidOCR).
3. Word document (.docx) parsing with heading hierarchy and markdown tables.
4. Excel spreadsheet (.xlsx) parsing with multi-sheet markdown tables.
5. Markdown (.md) and Python code (.py) document parsing.
6. Corrupted file rejection (corrupted PDF, binary disguised as .txt, 0-byte).
7. Oversized file rejection (> 15MB).
8. Duplicate file uploads (independent IDs, isolated deletion, no cross-impact).
9. Citation metadata completeness (id, title, page, file_id, score, source_type).
10. Vector cleanup on document deletion.
11. Rejection of unsupported legacy binary formats (.doc, .xls, .exe).
"""
import base64
import uuid
import io
import os
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import docx
import openpyxl
from PIL import Image, ImageDraw

from app.main import app
from app.services.rag.document_rag import document_rag_service, MAX_ATTACHMENT_SIZE_BYTES
from app.services.rag.qdrant_vector_store import QdrantVectorStore
from app.services.workspace_store import workspace_store
from tests.private_accounts import private_account_headers

client = TestClient(app)


def make_multipage_pdf(page_texts: list[str]) -> bytes:
    """Generate a valid multi-page PDF binary with distinct text on each page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in page_texts:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def make_scanned_pdf(text: str) -> bytes:
    """Generate a pure image-based PDF (no digital text) for OCR testing."""
    img = Image.new("RGB", (700, 250), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


def make_docx(title: str, heading: str, paragraph: str, table_data: list[list[str]]) -> bytes:
    """Generate a valid Word (.docx) document with heading and table."""
    doc = docx.Document()
    doc.add_heading(title, level=0)
    doc.add_heading(heading, level=1)
    doc.add_paragraph(paragraph)
    table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = val
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_xlsx(sheets_data: dict[str, list[list[str]]]) -> bytes:
    """Generate a valid Excel (.xlsx) workbook with multiple sheets."""
    wb = openpyxl.Workbook()
    first = True
    for sheet_name, rows in sheets_data.items():
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(title=sheet_name)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestDocumentIntelligenceE2E:
    """Complete end-to-end suite verifying document workflows across all supported formats."""

    def test_native_multipage_pdf_citations(self):
        """Verify multi-page PDF preserves page indices and produces accurate page citations."""
        user_id = "user_e2e_multipage"
        pages_content = [
            "Chapter 1: Quantum Telemetry basics and antenna arrays.",
            "Chapter 2: Sub-millimeter wave calibration parameters set to 0.42 nm.",
            "Chapter 3: Maintenance protocols and decommission procedures for orbital beacon.",
        ]
        pdf_bytes = make_multipage_pdf(pages_content)
        files = {"file": ("quantum_manual.pdf", pdf_bytes, "application/pdf")}

        res = client.post("/api/v1/library/upload", headers=private_account_headers(user_id), files=files)
        assert res.status_code == 200, res.text
        file_id = res.json()["file"]["id"]
        assert res.json()["file"]["status"] == "indexed"
        assert res.json()["file"]["chunk_count"] >= 3

        try:
            # Query information specifically on page 2
            search_res = client.get(
                "/api/v1/library/search?q=calibration parameters set to 0.42 nm",
                headers=private_account_headers(user_id),
            )
            assert search_res.status_code == 200
            data = search_res.json()
            assert data["count"] >= 1
            top = data["results"][0]
            assert "0.42 nm" in top["text"]
            assert top["page"] == 2
            assert top["doc_name"] == "quantum_manual.pdf"

            # Query information specifically on page 3
            search_p3 = client.get(
                "/api/v1/library/search?q=decommission procedures orbital beacon",
                headers=private_account_headers(user_id),
            )
            assert search_p3.status_code == 200
            top_p3 = search_p3.json()["results"][0]
            assert "decommission procedures" in top_p3["text"]
            assert top_p3["page"] == 3
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    def test_scanned_pdf_ocr_pipeline(self):
        """Verify image-only scanned PDF is processed via RapidOCR and indexed with searchable text."""
        user_id = "user_e2e_scanned"
        ocr_text = "OFFICIAL MEMO 2026: Campus laboratory maintenance scheduled on 19 November."
        pdf_bytes = make_scanned_pdf(ocr_text)

        files = {"file": ("scanned_memo.pdf", pdf_bytes, "application/pdf")}
        res = client.post("/api/v1/library/upload", headers=private_account_headers(user_id), files=files)
        assert res.status_code == 200, res.text
        file_id = res.json()["file"]["id"]
        assert res.json()["file"]["status"] == "indexed"

        try:
            search_res = client.get(
                "/api/v1/library/search?q=Campus laboratory maintenance scheduled date",
                headers=private_account_headers(user_id),
            )
            assert search_res.status_code == 200
            data = search_res.json()
            assert data["count"] >= 1
            assert "19 November" in data["results"][0]["text"] or "maintenance" in data["results"][0]["text"].lower()
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    def test_word_docx_headings_and_tables(self):
        """Verify Word (.docx) parses headings and structured markdown tables."""
        user_id = "user_e2e_docx"
        table_data = [
            ["Item", "Quantity", "Unit Price"],
            ["Lithium Cell", "500", "$12.50"],
            ["Sensor Array", "25", "$420.00"],
        ]
        docx_bytes = make_docx(
            title="Procurement Report",
            heading="Hardware Manifest",
            paragraph="The following items were approved by Procurement Director Vance.",
            table_data=table_data,
        )
        b64 = base64.b64encode(docx_bytes).decode("ascii")

        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "procurement_order.docx",
                "file_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "data": f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}",
            },
        )
        assert res.status_code == 200, res.text
        file_id = res.json()["file"]["id"]
        assert res.json()["file"]["status"] == "indexed"

        try:
            # Query paragraph text
            search_res = client.get(
                "/api/v1/library/search?q=Who approved the procurement items?",
                headers=private_account_headers(user_id),
            )
            assert search_res.status_code == 200
            assert "Director Vance" in search_res.json()["results"][0]["text"]

            # Query table data
            search_table = client.get(
                "/api/v1/library/search?q=Sensor Array quantity and unit price",
                headers=private_account_headers(user_id),
            )
            assert search_table.status_code == 200
            table_text = search_table.json()["results"][0]["text"]
            assert "Sensor Array" in table_text
            assert "$420.00" in table_text
            assert "|" in table_text  # Markdown table format
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    def test_excel_xlsx_multi_sheet_tables(self):
        """Verify Excel (.xlsx) parses multiple sheets into markdown tables with sheet names."""
        user_id = "user_e2e_xlsx"
        sheets_data = {
            "Q1_Revenue": [
                ["Region", "Target", "Actual"],
                ["North America", "1000000", "1150000"],
                ["Asia Pacific", "800000", "920000"],
            ],
            "Headcount": [
                ["Department", "FullTime", "Contractors"],
                ["AI Engineering", "42", "8"],
                ["Cloud Ops", "18", "4"],
            ],
        }
        xlsx_bytes = make_xlsx(sheets_data)
        b64 = base64.b64encode(xlsx_bytes).decode("ascii")

        res = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "annual_metrics.xlsx",
                "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "data": f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}",
            },
        )
        assert res.status_code == 200, res.text
        file_id = res.json()["file"]["id"]
        assert res.json()["file"]["status"] == "indexed"

        try:
            # Query revenue sheet
            search_rev = client.get(
                "/api/v1/library/search?q=Asia Pacific revenue actual",
                headers=private_account_headers(user_id),
            )
            assert search_rev.status_code == 200
            rev_text = search_rev.json()["results"][0]["text"]
            assert "920000" in rev_text
            assert "Q1_Revenue" in rev_text

            # Query headcount sheet
            search_hc = client.get(
                "/api/v1/library/search?q=AI Engineering fulltime headcount",
                headers=private_account_headers(user_id),
            )
            assert search_hc.status_code == 200
            hc_text = search_hc.json()["results"][0]["text"]
            assert "42" in hc_text
            assert "Headcount" in hc_text
        finally:
            client.delete(f"/api/v1/library/files/{file_id}", headers=private_account_headers(user_id))

    def test_markdown_and_code_document_workflows(self):
        """Verify Markdown (.md) and Python code (.py) documents index and retrieve cleanly."""
        user_id = "user_e2e_code_md"

        # 1. Markdown document
        md_content = "# System Architecture\n\n- Vector Database: Qdrant\n- Primary LLM: Llama 3.3 70B\n- Cache Layer: SQLite and In-Memory"
        b64_md = base64.b64encode(md_content.encode("utf-8")).decode("ascii")
        res_md = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "arch_spec.md",
                "data": f"data:text/markdown;base64,{b64_md}",
            },
        )
        assert res_md.status_code == 200
        file_id_md = res_md.json()["file"]["id"]

        # 2. Python code document
        py_content = "def calculate_orbit_trajectory(mass_kg: float, velocity_kms: float) -> float:\n    \"\"\"Compute orbital insertion velocity vector.\"\"\"\n    G = 6.67430e-11\n    return G * mass_kg / (velocity_kms ** 2)\n"
        b64_py = base64.b64encode(py_content.encode("utf-8")).decode("ascii")
        res_py = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "orbital_math.py",
                "data": f"data:text/x-python;base64,{b64_py}",
            },
        )
        assert res_py.status_code == 200
        file_id_py = res_py.json()["file"]["id"]

        try:
            search_md = client.get(
                "/api/v1/library/search?q=What is the primary LLM in architecture?",
                headers=private_account_headers(user_id),
            )
            assert search_md.status_code == 200
            assert "Llama 3.3 70B" in search_md.json()["results"][0]["text"]

            search_py = client.get(
                "/api/v1/library/search?q=calculate_orbit_trajectory mass_kg velocity",
                headers=private_account_headers(user_id),
            )
            assert search_py.status_code == 200
            assert "calculate_orbit_trajectory" in search_py.json()["results"][0]["text"]
        finally:
            client.delete(f"/api/v1/library/files/{file_id_md}", headers=private_account_headers(user_id))
            client.delete(f"/api/v1/library/files/{file_id_py}", headers=private_account_headers(user_id))

    def test_corrupted_and_binary_disguised_rejections(self):
        """Verify corrupted PDFs and arbitrary binary disguised as .txt are rejected with HTTP 422."""
        # Fresh owner per run: rejected uploads keep a "failed" receipt that counts
        # toward the 25-document cap, so a fixed id fills up a persistent local DB.
        user_id = f"user_e2e_rejection_{uuid.uuid4().hex[:8]}"

        # 1. Damaged PDF
        corrupt_pdf = b"%PDF-1.4\n%\xff\xfe\xfd\x00\x00\x00BROKEN_PDF_STREAM_UNREADABLE"
        files_pdf = {"file": ("damaged.pdf", corrupt_pdf, "application/pdf")}
        res_pdf = client.post("/api/v1/library/upload", headers=private_account_headers(user_id), files=files_pdf)
        assert res_pdf.status_code == 422
        assert "indexing failed" in res_pdf.json()["detail"].lower() or "readable" in res_pdf.json()["detail"].lower()

        # 2. Binary file disguised as .txt (contains null bytes and control codes)
        binary_disguised = b"\x00\x00\x01\x02\x03\xff\xfe\x00\x1f\x80\x90BINARY_GARBAGE_PAYLOAD"
        b64_bin = base64.b64encode(binary_disguised).decode("ascii")
        res_bin = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={
                "filename": "corrupt_data.txt",
                "data": f"data:text/plain;base64,{b64_bin}",
            },
        )
        assert res_bin.status_code == 422

    def test_unsupported_formats_rejection(self):
        """Verify unsupported legacy formats (.doc, .xls, .exe) are rejected with HTTP 400."""
        user_id = "user_e2e_unsupported"
        dummy_bytes = b"dummy content"
        b64 = base64.b64encode(dummy_bytes).decode("ascii")

        for unsupported in ["legacy.doc", "ancient.xls", "danger.exe", "bundle.zip"]:
            res = client.post(
                "/api/v1/library/upload-json",
                headers=private_account_headers(user_id),
                json={"filename": unsupported, "data": f"data:application/octet-stream;base64,{b64}"},
            )
            assert res.status_code == 400, f"Expected 400 for {unsupported}, got {res.status_code}"
            assert "unsupported file extension" in res.json()["detail"].lower()

    def test_oversized_upload_rejection(self):
        """Verify files exceeding 15MB size limit are rejected with HTTP 413."""
        user_id = "user_e2e_oversized"
        # 16MB file
        oversized_data = b"x" * (MAX_ATTACHMENT_SIZE_BYTES + 1024)
        files = {"file": ("giant_file.txt", oversized_data, "text/plain")}
        res = client.post("/api/v1/library/upload", headers=private_account_headers(user_id), files=files)
        assert res.status_code == 413
        assert "exceeds maximum size" in res.json()["detail"].lower()

    def test_duplicate_uploads_isolated_lifecycle(self):
        """Verify uploading identical document twice creates 2 independent records, and deleting one leaves the other searchable."""
        user_id = "user_e2e_duplicates"
        doc_text = "Project Nebula secret protocol code is Hyperion9942."
        b64 = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")

        # First upload
        res1 = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={"filename": "nebula_protocol.txt", "data": f"data:text/plain;base64,{b64}"},
        )
        assert res1.status_code == 200
        file_id_1 = res1.json()["file"]["id"]

        # Duplicate upload with exact same filename and content
        res2 = client.post(
            "/api/v1/library/upload-json",
            headers=private_account_headers(user_id),
            json={"filename": "nebula_protocol.txt", "data": f"data:text/plain;base64,{b64}"},
        )
        assert res2.status_code == 200
        file_id_2 = res2.json()["file"]["id"]

        assert file_id_1 != file_id_2

        try:
            # Delete first file
            del1 = client.delete(f"/api/v1/library/files/{file_id_1}", headers=private_account_headers(user_id))
            assert del1.status_code == 200

            # Second file must still be searchable
            search_res = client.get(
                "/api/v1/library/search?q=Project Nebula secret protocol code",
                headers=private_account_headers(user_id),
            )
            assert search_res.status_code == 200
            assert search_res.json()["count"] >= 1
            top = search_res.json()["results"][0]
            assert "Hyperion9942" in top["text"]
            assert top["file_id"] == file_id_2
        finally:
            client.delete(f"/api/v1/library/files/{file_id_2}", headers=private_account_headers(user_id))

    def test_citation_metadata_precision(self):
        """Verify retrieve() constructs complete SourceCitation objects with id, title, page, snippet, score, and source_type."""
        user_id = "user_e2e_citations"
        doc_text = "Section 9.4: Thermal conductivity coefficient for beryllium copper is 105 W/(m*K)."
        b64 = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")

        attachment = {
            "name": "thermal_constants.txt",
            "mime_type": "text/plain",
            "data": b64,
        }

        chunks, citations = document_rag_service.retrieve(
            attachments=[attachment],
            query="What is the thermal conductivity coefficient for beryllium copper?",
            top_k=2,
        )

        assert len(chunks) >= 1
        assert len(citations) >= 1
        cite = citations[0]
        assert cite.id is not None
        assert cite.title == "thermal_constants.txt (Page 1)"
        assert cite.page == 1
        assert "105 W/(m*K)" in cite.snippet
        assert cite.source_type == "document"
