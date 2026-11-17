"""Domain Knowledge Ingestion Pipeline for JRU AI.

Reads domain documents from data/domain_corpus/, chunks text with section-awareness,
embeds via FastEmbed, and indexes vectors into Qdrant for zero-hallucination RAG grounding.
"""
import glob
import logging
import os
import re
from typing import Any, Dict, List

from app.services.rag.qdrant_vector_store import QdrantVectorStore, get_reranker_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "jru_knowledge_base"


def split_into_semantic_chunks(text: str, doc_name: str, max_chunk_chars: int = 1200, overlap_chars: int = 150) -> List[Dict[str, Any]]:
    """Split markdown text into hierarchical chunks with breadcrumb context."""
    clean_doc_title = doc_name.replace("_", " ").replace(".md", "").title()
    doc_header_match = re.search(r"^#\s+(.+?)(?:\n|$)", text)
    if doc_header_match:
        clean_doc_title = doc_header_match.group(1).strip()

    # Split by markdown headers (H1, H2, H3)
    sections = re.split(r"(?=(?:^|\n)#{1,3}\s+)", text)
    chunks = []
    chunk_counter = 0

    current_h2 = "General Overview"
    current_h3 = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        header_match = re.match(r"^(#{1,3})\s+(.+?)(?:\n|$)", section)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            if level == 1:
                clean_doc_title = title
            elif level == 2:
                current_h2 = title
                current_h3 = ""
            elif level == 3:
                current_h3 = title

        # Build structural breadcrumb
        if current_h3:
            breadcrumb = f"{clean_doc_title} > {current_h2} > {current_h3}"
        else:
            breadcrumb = f"{clean_doc_title} > {current_h2}"

        body = section
        is_table_heavy = "|" in body and body.count("\n|") >= 2

        # Keep small sections and tables together to preserve structured context
        if len(body) <= max_chunk_chars or is_table_heavy:
            chunk_counter += 1
            chunk_text = f"[{breadcrumb}]\n{body}"
            chunks.append({
                "id": f"{doc_name}-chunk-{chunk_counter}",
                "doc_name": doc_name,
                "section": current_h2,
                "subsection": current_h3,
                "breadcrumb": breadcrumb,
                "page": 1,
                "text": chunk_text,
            })
        else:
            paragraphs = body.split("\n\n")
            current_chunk = []
            current_len = 0

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                if current_len + len(p) > max_chunk_chars and current_chunk:
                    chunk_counter += 1
                    chunk_text = f"[{breadcrumb}]\n" + "\n\n".join(current_chunk)
                    chunks.append({
                        "id": f"{doc_name}-chunk-{chunk_counter}",
                        "doc_name": doc_name,
                        "section": current_h2,
                        "subsection": current_h3,
                        "breadcrumb": breadcrumb,
                        "page": 1,
                        "text": chunk_text,
                    })
                    if len(current_chunk[-1]) < overlap_chars:
                        current_chunk = [current_chunk[-1], p]
                        current_len = len(current_chunk[0]) + len(p)
                    else:
                        current_chunk = [p]
                        current_len = len(p)
                else:
                    current_chunk.append(p)
                    current_len += len(p)

            if current_chunk:
                chunk_counter += 1
                chunk_text = f"[{breadcrumb}]\n" + "\n\n".join(current_chunk)
                chunks.append({
                    "id": f"{doc_name}-chunk-{chunk_counter}",
                    "doc_name": doc_name,
                    "section": current_h2,
                    "subsection": current_h3,
                    "breadcrumb": breadcrumb,
                    "page": 1,
                    "text": chunk_text,
                })

    return chunks


def ingest_all_domain_documents(corpus_dir: str = "data/domain_corpus") -> int:
    """Ingest all markdown/text files in corpus directory into Qdrant."""
    if not os.path.exists(corpus_dir):
        # Also check relative to backend directory
        alt_dir = os.path.join(os.path.dirname(__file__), "../../../data/domain_corpus")
        if os.path.exists(alt_dir):
            corpus_dir = alt_dir
        else:
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files = glob.glob(os.path.join(corpus_dir, "*.md")) + glob.glob(os.path.join(corpus_dir, "*.txt"))
    logger.info(f"Found {len(files)} domain documents in {corpus_dir} for ingestion.")

    all_chunks = []
    for file_path in files:
        doc_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        doc_chunks = split_into_semantic_chunks(content, doc_name)
        logger.info(f"  - Parsed {doc_name}: {len(doc_chunks)} chunks generated.")
        all_chunks.extend(doc_chunks)

    logger.info(f"Total chunks across corpus: {len(all_chunks)}. Indexing into Qdrant '{COLLECTION_NAME}'...")
    store = QdrantVectorStore()
    count = store.index_chunks(COLLECTION_NAME, all_chunks)
    logger.info(f"Successfully indexed {count} vectors into '{COLLECTION_NAME}'.")
    return count


def verify_retrieval_query(query: str, top_k: int = 3):
    """Run a test query against Qdrant and FlashRank reranker."""
    from app.services.rag.qdrant_vector_store import NeuralReranker
    logger.info(f"Verifying retrieval for query: '{query}'")
    store = QdrantVectorStore()
    results = store.search_dense(COLLECTION_NAME, query, limit=top_k * 2)

    if not results:
        logger.warning("No candidate vectors retrieved!")
        return

    # Cross-encoder neural reranking
    reranker = NeuralReranker()
    ranked = reranker.rerank(query, results, top_n=top_k)

    logger.info(f"Top {len(ranked)} reranked results:")
    for idx, item in enumerate(ranked, 1):
        score = item.get("rerank_score", 0.0)
        logger.info(f"  [{idx}] Score: {score:.4f} | Doc: {item.get('doc_name')} (Page {item.get('page')})")
        logger.info(f"      Snippet: {item['text'][:140]}...")


def ingest_internal_ops_documents(corpus_dir: str = "data/internal_ops") -> int:
    """Ingest internal engineering runbooks and whitepapers into separate administrative collection."""
    if not os.path.exists(corpus_dir):
        alt_dir = os.path.join(os.path.dirname(__file__), "../../../data/internal_ops")
        if os.path.exists(alt_dir):
            corpus_dir = alt_dir
        else:
            logger.warning(f"Internal ops directory not found: {corpus_dir}; skipping.")
            return 0

    files = glob.glob(os.path.join(corpus_dir, "*.md")) + glob.glob(os.path.join(corpus_dir, "*.txt"))
    if not files:
        return 0

    logger.info(f"Found {len(files)} internal operations documents in {corpus_dir}.")
    all_chunks = []
    for file_path in files:
        doc_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        doc_chunks = split_into_semantic_chunks(content, doc_name)
        all_chunks.extend(doc_chunks)

    store = QdrantVectorStore()
    internal_col = "jru_internal_ops"
    count = store.index_chunks(internal_col, all_chunks)
    logger.info(f"Successfully indexed {count} vectors into isolated '{internal_col}'.")
    return count


if __name__ == "__main__":
    count_academic = ingest_all_domain_documents()
    count_ops = ingest_internal_ops_documents()
    print(f"\n--- Ingestion Completed: {count_academic} Academic Chunks, {count_ops} Internal Ops Chunks Indexed ---\n")
