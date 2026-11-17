"""Regression tests verifying that local dense embeddings are safely bypassed in low-memory mode."""

import base64
from unittest.mock import patch

from app.core.config import settings
from app.services.rag.document_rag import document_rag_service
from app.services.rag.qdrant_vector_store import get_embedding_model, is_embedding_enabled


class TestLocalEmbeddingsMemoryGuard:
    def test_embedding_disabled_in_low_memory_mode(self):
        """In low-memory mode with default config, is_embedding_enabled must return False."""
        with patch.object(settings, "ENABLE_LOCAL_EMBEDDINGS", False):
            enabled, reason = is_embedding_enabled()
            assert not enabled
            assert "disabled" in reason

    def test_get_embedding_model_returns_none_when_disabled(self):
        """get_embedding_model must return None without importing FastEmbed or downloading weights."""
        with patch.object(settings, "LOW_MEMORY_RAG_MODE", True), \
             patch.object(settings, "ENABLE_LOCAL_EMBEDDINGS", False):
            model = get_embedding_model()
            assert model is None

    def test_embedding_bypassed_when_rss_exceeds_budget(self):
        """Even if explicitly enabled, high RSS must disable local embeddings to guard 512MB limit."""
        with patch.object(settings, "ENVIRONMENT", "production"), \
             patch.object(settings, "ENABLE_LOCAL_EMBEDDINGS", True), \
             patch("app.services.rag.qdrant_vector_store.get_process_rss_mb", return_value=410.0), \
             patch.object(settings, "MAX_PROCESS_RSS_MB", 380.0):
            enabled, reason = is_embedding_enabled()
            assert not enabled
            assert "rss_exceeds_budget" in reason

    def test_document_rag_retrieves_via_bm25_jaccard_without_onnx(self):
        """document_rag.retrieve must produce grounded passages and citations in low-memory mode."""
        plain_text = "This is an institutional policy document. The research grant is $500,000. Protocol code is ALPHA-99."
        dummy_attachment = {
            "name": "sample_policy.txt",
            "content_type": "text/plain",
            "data": base64.b64encode(plain_text.encode("utf-8")).decode("utf-8"),
        }

        with patch.object(settings, "LOW_MEMORY_RAG_MODE", True), \
             patch.object(settings, "ENABLE_LOCAL_EMBEDDINGS", False):
            chunks, citations = document_rag_service.retrieve(
                attachments=[dummy_attachment],
                query="What is the protocol code and research grant?",
                top_k=2,
            )
            assert len(chunks) > 0
            assert len(citations) > 0
            assert any("ALPHA-99" in c["text"] or "$500,000" in c["text"] for c in chunks)
            assert citations[0].source_type == "document"
