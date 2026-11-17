"""Download the retrieval models into MODEL_CACHE_DIR at build time.

Render free instances restart often; without this the first chat after each restart
downloads BGE-small and the FlashRank reranker from Hugging Face before answering.
Usage (Render build command):
    pip install -r backend/requirements.txt && MODEL_CACHE_DIR=backend/.model_cache python scripts/prefetch_models.py
The runtime needs the same MODEL_CACHE_DIR environment variable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))


def main() -> int:
    if not os.getenv("MODEL_CACHE_DIR", "").strip():
        print("MODEL_CACHE_DIR is not set; nothing to prefetch.")
        return 0
    from app.services.rag.qdrant_vector_store import get_embedding_model, get_reranker_model

    get_embedding_model()
    get_reranker_model()
    print(f"Models cached in {os.environ['MODEL_CACHE_DIR']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
