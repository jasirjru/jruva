"""RAG (Retrieval-Augmented Generation) package for local document grounding."""
from app.services.rag.document_rag import document_rag_service, DocumentRAGService

__all__ = ["document_rag_service", "DocumentRAGService"]
