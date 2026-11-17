"""Unit tests for Local Document RAG Service (parsing, BM25, and retrieval)."""
import base64
import pytest
from app.services.rag.document_rag import DocumentRAGService, BM25Index, tokenize

@pytest.fixture
def rag_service():
    return DocumentRAGService()

def test_is_document_attachment(rag_service):
    assert rag_service.is_document_attachment({"name": "report.pdf", "mime_type": "application/pdf"})
    assert rag_service.is_document_attachment({"name": "notes.md", "mime_type": "text/markdown"})
    assert rag_service.is_document_attachment({"name": "code.py", "mime_type": "text/plain"})
    assert not rag_service.is_document_attachment({"name": "photo.jpg", "mime_type": "image/jpeg"})

def test_extract_document_text(rag_service):
    text_sample = "Retrieval-Augmented Generation bridges LLMs with dynamic external knowledge bases."
    b64 = base64.b64encode(text_sample.encode("utf-8")).decode("ascii")
    attachment = {"name": "rag_overview.txt", "mime_type": "text/plain", "data": b64}
    
    pages = rag_service.extract_document_pages(attachment)
    assert len(pages) == 1
    assert "Retrieval-Augmented Generation" in pages[0]["text"]
    assert pages[0]["page"] == 1

def test_chunk_pages(rag_service):
    long_text = " ".join([f"token_{i}" for i in range(600)])
    pages = [{"doc_name": "large_doc.txt", "page": 1, "text": long_text}]
    chunks = rag_service.chunk_pages(pages, chunk_size_words=200, chunk_overlap_words=50)
    assert len(chunks) >= 3
    assert chunks[0]["doc_name"] == "large_doc.txt"

def test_bm25_index():
    docs = [
        tokenize("Faster-whisper provides ultra-fast local speech to text transcription."),
        tokenize("ChromaDB is a lightweight embedded vector database for RAG pipelines."),
        tokenize("Ollama runs quantized open-weight language models locally on your GPU or CPU."),
    ]
    index = BM25Index()
    index.fit(docs)

    scores = index.score(tokenize("vector database for RAG"))
    # Second document about ChromaDB and vector database should rank highest
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]

def test_retrieve_end_to_end(rag_service):
    doc1 = "Speech recognition using faster-whisper converts voice to text."
    doc2 = "RAG architecture uses BM25 and vector embeddings to ground LLM generations in facts."
    
    b64_1 = base64.b64encode(doc1.encode("utf-8")).decode("ascii")
    b64_2 = base64.b64encode(doc2.encode("utf-8")).decode("ascii")
    
    attachments = [
        {"name": "speech.txt", "mime_type": "text/plain", "data": b64_1},
        {"name": "rag.txt", "mime_type": "text/plain", "data": b64_2},
    ]
    
    chunks, citations = rag_service.retrieve(attachments, "Tell me about vector embeddings and RAG grounding", top_k=2)
    assert len(chunks) > 0
    assert len(citations) > 0
    assert "rag.txt" in citations[0].title
    assert "vector embeddings" in citations[0].snippet

    context = rag_service.build_grounding_context(chunks)
    assert "[LOCAL DOCUMENT RETRIEVAL GROUNDING (RAG)]" in context
    assert "rag.txt" in context


def test_image_attachment_ocr(rag_service):
    """Test that image attachments are processed via OCR and return recognized text."""
    import io
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "JHARKHAND RAI UNIVERSITY", fill=(0, 0, 0))
    draw.text((20, 100), "Passing rule: 40 percent in CIA and ESE", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

    attachment = {
        "name": "rules_notice.png",
        "mime_type": "image/png",
        "data": b64_img,
    }

    assert rag_service.is_image_attachment(attachment)
    assert rag_service.is_extractable_attachment(attachment)

    pages = rag_service.extract_document_pages(attachment)
    assert len(pages) == 1
    assert "JHARKHAND" in pages[0]["text"]
    assert pages[0]["doc_name"] == "rules_notice.png"


def test_format_markdown_table_pipe_escaping():
    from app.services.rag.document_rag import format_markdown_table

    rows = [
        ["Course | Code", "Subject Name", "Credits"],
        ["CS|101\nIntro", "Algorithms & Logic", "4"],
        ["MA|201", "Linear Algebra | Calculus", "3"],
    ]
    md = format_markdown_table(rows, title="Curriculum")
    assert "### Curriculum" in md
    assert "Course \\| Code" in md
    assert "CS\\|101 Intro" in md
    assert "Linear Algebra \\| Calculus" in md
    assert "| --- | --- | --- |" in md


def test_csv_extraction_structured_table(rag_service):
    csv_data = "Student ID,Course,Grade\nJRU101,Computer Networks,A+\nJRU102,Database Systems,A\n"
    b64 = base64.b64encode(csv_data.encode("utf-8")).decode("ascii")
    attachment = {
        "name": "grades.csv",
        "mime_type": "text/csv",
        "data": b64,
    }
    pages = rag_service.extract_document_pages(attachment)
    assert len(pages) == 1
    assert "### Table: grades.csv" in pages[0]["text"]
    assert "| Student ID | Course | Grade |" in pages[0]["text"]
    assert "| JRU101 | Computer Networks | A+ |" in pages[0]["text"]


def test_docx_table_extraction(rag_service):
    import io
    import docx

    doc = docx.Document()
    doc.add_heading("Syllabus Overview", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Subject"
    table.cell(0, 1).text = "Marks"
    table.cell(1, 0).text = "Deep Learning"
    table.cell(1, 1).text = "100"

    buf = io.BytesIO()
    doc.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    attachment = {
        "name": "syllabus.docx",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "data": b64,
    }
    pages = rag_service.extract_document_pages(attachment)
    assert len(pages) == 1
    assert "Syllabus Overview" in pages[0]["text"]
    assert "| Subject | Marks |" in pages[0]["text"]
    assert "| Deep Learning | 100 |" in pages[0]["text"]


def test_xlsx_table_extraction(rag_service):
    import io
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "FeeStructure"
    ws.append(["Category", "Amount (INR)"])
    ws.append(["Tuition Fee", "45000"])
    ws.append(["Exam Fee", "2500"])

    buf = io.BytesIO()
    wb.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    attachment = {
        "name": "fees.xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "data": b64,
    }
    pages = rag_service.extract_document_pages(attachment)
    assert len(pages) == 1
    assert "Sheet: FeeStructure" in pages[0]["text"]
    assert "| Category | Amount (INR) |" in pages[0]["text"]
    assert "| Tuition Fee | 45000 |" in pages[0]["text"]



