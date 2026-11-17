"""Comprehensive test suite for Phase 4: Cloud Document Intelligence & RAG Evaluation.

Evaluates DocumentRAGService against the 30-scenario benchmark fixture:
1. Exact factual retrieval
2. Multi-page reasoning
3. Table & CSV extraction
4. Contradictory passages
5. No-answer out-of-scope queries
6. Missing and corrupt documents
7. Cross-tenant library isolation
8. Prompt injection and homoglyph defense
9. Long-document pagination
"""

import json
import os
from typing import Any, Dict, List
import pytest

from app.services.rag.document_rag import (
    DocumentRAGService,
    detect_and_disarm_injections,
    document_rag_service,
)
from app.services.workspace_store import workspace_store


@pytest.fixture
def rag_dataset() -> Dict[str, Any]:
    """Load the 30-question evaluation benchmark dataset."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "rag_eval_dataset.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_eval_dataset_completeness(rag_dataset):
    """Verify evaluation benchmark fixture contains all 30 test scenarios across categories."""
    test_cases = rag_dataset["test_cases"]
    assert len(test_cases) == 30, f"Expected 30 test cases, found {len(test_cases)}"
    categories = {tc["category"] for tc in test_cases}
    expected_categories = {
        "exact_factual_retrieval",
        "multi_page_reasoning",
        "table_extraction",
        "contradictory_passages",
        "no_answer_refusal",
        "missing_or_corrupt_documents",
        "cross_tenant_isolation",
        "prompt_injection_resistance",
        "code_and_json_parsing",
        "markdown_hierarchy",
        "large_document_pagination",
    }
    assert expected_categories.issubset(categories)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exact Factual Retrieval Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_factual_retrieval(rag_dataset):
    """Test cases RAG-001 through RAG-004: Exact numbers and policy facts."""
    service = DocumentRAGService()
    factual_cases = [tc for tc in rag_dataset["test_cases"] if tc["category"] == "exact_factual_retrieval"]
    assert len(factual_cases) >= 4

    for case in factual_cases:
        doc = case["document"]
        attachment = {
            "name": doc["name"],
            "mime_type": doc["mime_type"],
            "data": doc["content"].encode("utf-8"),
        }
        chunks, citations = service.retrieve([attachment], case["query"], top_k=2)
        assert len(chunks) > 0, f"Failed to retrieve chunks for {case['id']}"
        assert len(citations) > 0, f"Failed to generate citations for {case['id']}"

        combined_text = " ".join(c["text"] for c in chunks)
        for kw in case["expected_answer_keywords"]:
            assert kw.lower() in combined_text.lower(), f"Expected keyword '{kw}' missing in {case['id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Page Reasoning & Pagination Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_page_reasoning(rag_dataset):
    """Test cases RAG-005 through RAG-007: Information distributed across multiple pages."""
    service = DocumentRAGService()
    mp_cases = [tc for tc in rag_dataset["test_cases"] if tc["category"] == "multi_page_reasoning"]

    for case in mp_cases:
        pages = [{"doc_name": "multi_page_doc.pdf", "page": p["page"], "text": p["text"]} for p in case["document_pages"]]
        chunks = service.chunk_pages(pages, chunk_size_words=100)
        assert len(chunks) >= len(pages)

        # Verify page number preservation
        page_numbers_in_chunks = {c["page"] for c in chunks}
        for expected_p in [p["page"] for p in case["document_pages"]]:
            assert expected_p in page_numbers_in_chunks


def test_large_document_pagination(rag_dataset):
    """Test cases RAG-029 and RAG-030: Sparse pages up to page 60 preserve page metadata."""
    service = DocumentRAGService()
    pagination_cases = [tc for tc in rag_dataset["test_cases"] if tc["category"] == "large_document_pagination"]

    for case in pagination_cases:
        pages = [{"doc_name": "large_report.pdf", "page": p["page"], "text": p["text"]} for p in case["document_pages"]]
        chunks = service.chunk_pages(pages, chunk_size_words=200)

        # Verify page numbers match original document page tags
        chunk_page_map = {c["page"]: c["text"] for c in chunks}
        for expected_page in case["expected_page_citation"]:
            assert expected_page in chunk_page_map, f"Page {expected_page} missing in chunk results"
            for kw in case["expected_answer_keywords"]:
                assert kw.lower() in chunk_page_map[expected_page].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Table and Structured Data Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_table_extraction(rag_dataset):
    """Test cases RAG-008 through RAG-011: Tabular CSV and markdown table retrieval."""
    service = DocumentRAGService()
    table_cases = [tc for tc in rag_dataset["test_cases"] if tc["category"] == "table_extraction"]

    for case in table_cases:
        doc = case["document"]
        attachment = {
            "name": doc["name"],
            "mime_type": doc["mime_type"],
            "data": doc["content"].encode("utf-8"),
        }
        chunks, citations = service.retrieve([attachment], case["query"], top_k=2)
        assert len(chunks) > 0, f"Failed to retrieve tabular data for {case['id']}"

        combined_text = " ".join(c["text"] for c in chunks)
        for kw in case["expected_answer_keywords"]:
            assert kw.lower() in combined_text.lower(), f"Keyword '{kw}' missing from table extraction in {case['id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Contradictory Passages & Policy Updates
# ─────────────────────────────────────────────────────────────────────────────

def test_contradictory_passages(rag_dataset):
    """Test cases RAG-012 through RAG-014: Amended/superseding policies."""
    service = DocumentRAGService()
    contra_cases = [tc for tc in rag_dataset["test_cases"] if tc["category"] == "contradictory_passages"]

    for case in contra_cases:
        pages = [{"doc_name": "policy_history.pdf", "page": p["page"], "text": p["text"]} for p in case["document_pages"]]
        chunks = service.chunk_pages(pages)

        # Build grounding context with all chunks
        grounding_context = service.build_grounding_context(chunks)
        for kw in case["expected_answer_keywords"]:
            assert kw.lower() in grounding_context.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing, Corrupt, and No-Answer Documents
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_and_corrupt_documents(rag_dataset):
    """Test cases RAG-018 through RAG-020: Graceful handling of empty or corrupted payloads."""
    service = DocumentRAGService()

    # Empty document
    empty_att = {"name": "empty.txt", "mime_type": "text/plain", "data": b""}
    chunks, citations = service.retrieve([empty_att], "anything")
    assert chunks == []
    assert citations == []

    # Corrupt fake PDF
    fake_pdf_att = {"name": "corrupt.pdf", "mime_type": "application/pdf", "data": b"NOT_A_REAL_PDF"}
    # Must not raise unhandled exception
    pages = service.extract_document_pages(fake_pdf_att)
    assert isinstance(pages, list)

    # Invalid base64 string
    bad_b64 = {"name": "bad.txt", "mime_type": "text/plain", "data": "%%%not-base64%%%"}
    pages_b64 = service.extract_document_pages(bad_b64)
    assert pages_b64 == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prompt Injection Resistance & Disarming
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt_injection_resistance(rag_dataset):
    """Test cases RAG-023 through RAG-025: Disarming system overrides, homoglyphs, and base64."""
    injection_cases = [tc for tc in rag_dataset["test_cases"] if tc["category"] == "prompt_injection_resistance"]

    for case in injection_cases:
        raw_text = case["document"]["content"]
        sanitized_text, is_injection = detect_and_disarm_injections(raw_text)

        # Injection must be neutralized
        assert is_injection is True, f"Injection was not detected in {case['id']}"
        assert "[disarmed_prompt_injection_attempt]" in sanitized_text.lower(), (
            f"Expected disarm marker in {case['id']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cross-Tenant Library Knowledge Isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_tenant_library_isolation(rag_dataset):
    """Test cases RAG-021 & RAG-022: User A's private library files are strictly inaccessible to User B."""
    service = DocumentRAGService()

    # Create distinct files in store for tenant A
    uid_a = "tenant-alpha-isolation-user"
    uid_b = "tenant-beta-isolation-user"

    file_a = workspace_store.record_library_file(
        filename="confidential_alpha.txt",
        file_type="text/plain",
        file_size=100,
        storage_path="confidential_alpha.txt",
        chunk_count=2,
        user_id=uid_a,
    )
    assert file_a is not None

    # Retrieve from workspace_store for tenant B
    user_b_files = workspace_store.list_library_files(user_id=uid_b)
    user_b_file_ids = {f["id"] for f in user_b_files}
    assert file_a["id"] not in user_b_file_ids, "Tenant B must not see Tenant A's file in library listing"

    # Direct retrieval verifies owner partition
    fetched = workspace_store.get_library_file(file_a["id"])
    assert fetched is not None
    assert fetched.get("user_id") == uid_a
    assert fetched.get("user_id") != uid_b

    # Clean up
    workspace_store.delete_library_file(file_a["id"], user_id=uid_a)
