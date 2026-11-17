"""Automated Evaluation Runner for RAG Retrieval & Reranking Benchmarks.

Measures Hit@1, Hit@3, and Mean Reciprocal Rank (MRR) on the 30-question ground truth dataset.
Evaluates both the baseline in-memory BM25 engine and the upgraded Qdrant + FlashRank pipeline.
"""
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag.document_rag import BM25Index, DocumentRAGService, tokenize
from tests.retrieval_eval_dataset import EVAL_DOCUMENTS, EVAL_QUESTIONS


def evaluate_baseline_bm25(chunks: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate baseline in-memory BM25 + N-gram Jaccard engine on evaluation questions."""
    tokenized_docs = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Index()
    bm25.fit(tokenized_docs)

    results_by_type: Dict[str, List[Dict[str, Any]]] = {"pdf": [], "code": [], "markdown": []}
    all_ranks: List[int] = []

    for q in questions:
        query = q["query"]
        target_id = q["target_chunk_id"]
        doc_type = q["doc_type"]

        query_tokens = tokenize(query)
        if not query_tokens:
            query_tokens = [w.lower() for w in query.split() if len(w) > 1]

        # BM25 Keyword score
        scores_bm25 = bm25.score(query_tokens)
        ranked_bm25 = sorted(enumerate(scores_bm25), key=lambda x: x[1], reverse=True)
        bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(ranked_bm25)}

        # Dense n-gram overlap
        query_clean = query.lower()
        query_ngrams = set(query_clean[i:i+3] for i in range(len(query_clean)-2)) if len(query_clean) >= 3 else set([query_clean])
        semantic_scores = []
        for c in chunks:
            chunk_clean = c["text"].lower()
            chunk_ngrams = set(chunk_clean[i:i+3] for i in range(len(chunk_clean)-2)) if len(chunk_clean) >= 3 else set()
            intersection = query_ngrams.intersection(chunk_ngrams)
            jaccard = len(intersection) / float(len(query_ngrams.union(chunk_ngrams))) if chunk_ngrams else 0.0
            title_boost = 0.2 if query_clean in c["doc_name"].lower() else 0.0
            semantic_scores.append(jaccard + title_boost)

        ranked_semantic = sorted(enumerate(semantic_scores), key=lambda x: x[1], reverse=True)
        semantic_rank_map = {idx: rank for rank, (idx, _) in enumerate(ranked_semantic)}

        # RRF k=60
        fused = []
        for idx in range(len(chunks)):
            r1 = bm25_rank_map.get(idx, len(chunks))
            r2 = semantic_rank_map.get(idx, len(chunks))
            rrf = (1.0 / (60.0 + r1 + 1)) + (1.0 / (60.0 + r2 + 1))
            fused.append((idx, rrf))

        fused_ranked = sorted(fused, key=lambda x: x[1], reverse=True)
        ranked_chunk_ids = [chunks[idx]["id"] for idx, _ in fused_ranked]

        rank = (ranked_chunk_ids.index(target_id) + 1) if target_id in ranked_chunk_ids else 999
        all_ranks.append(rank)

        results_by_type[doc_type].append({
            "query_id": q["id"],
            "query": query,
            "target": target_id,
            "top_retrieved": ranked_chunk_ids[:3],
            "rank": rank,
            "hit_at_1": rank == 1,
            "hit_at_3": rank <= 3,
            "reciprocal_rank": 1.0 / rank if rank <= 30 else 0.0,
        })

    def calc_metrics(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(res_list)
        if n == 0:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0}
        hit1 = sum(1 for r in res_list if r["hit_at_1"]) / n
        hit3 = sum(1 for r in res_list if r["hit_at_3"]) / n
        mrr = sum(r["reciprocal_rank"] for r in res_list) / n
        return {
            "hit_at_1_pct": round(hit1 * 100, 2),
            "hit_at_3_pct": round(hit3 * 100, 2),
            "mrr": round(mrr, 4),
        }

    all_res = [r for sublist in results_by_type.values() for r in sublist]
    return {
        "engine": "Baseline In-Memory BM25 + Jaccard RRF",
        "total_questions": len(questions),
        "overall": calc_metrics(all_res),
        "by_type": {
            "pdf": calc_metrics(results_by_type["pdf"]),
            "code": calc_metrics(results_by_type["code"]),
            "markdown": calc_metrics(results_by_type["markdown"]),
        },
    }


def evaluate_upgraded_pipeline(chunks: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate upgraded pipeline: Qdrant Vector Store + FastEmbed + FlashRank Cross-Encoder."""
    from app.services.rag.qdrant_vector_store import QdrantVectorStore, NeuralReranker

    vector_store = QdrantVectorStore(storage_path=":memory:")
    collection_name = "eval_benchmarks"
    vector_store.index_chunks(collection_name, chunks)
    reranker = NeuralReranker()

    # Fit BM25 for sparse retrieval stage
    tokenized_docs = [tokenize(f"{c.get('doc_name', '')} {c['text']}") for c in chunks]
    bm25 = BM25Index()
    bm25.fit(tokenized_docs)

    results_by_type: Dict[str, List[Dict[str, Any]]] = {"pdf": [], "code": [], "markdown": []}
    all_ranks: List[int] = []

    for q in questions:
        query = q["query"]
        target_id = q["target_chunk_id"]
        doc_type = q["doc_type"]

        # Stage 1A: Dense semantic search in Qdrant (top 15)
        dense_results = vector_store.search_dense(collection_name, query, limit=15)
        dense_rank_map = {res["id"]: rank for rank, res in enumerate(dense_results)}

        # Stage 1B: Sparse BM25 keyword search (top 15)
        query_tokens = tokenize(query) or [w.lower() for w in query.split() if len(w) > 1]
        scores_bm25 = bm25.score(query_tokens)
        ranked_bm25 = sorted(enumerate(scores_bm25), key=lambda x: x[1], reverse=True)
        bm25_rank_map = {chunks[idx]["id"]: rank for rank, (idx, _) in enumerate(ranked_bm25[:15])}

        # Stage 1C: Reciprocal Rank Fusion (k=60)
        candidate_ids = set(dense_rank_map.keys()).union(set(bm25_rank_map.keys()))
        chunk_map = {c["id"]: c for c in chunks}
        fused = []
        for cid in candidate_ids:
            r_dense = dense_rank_map.get(cid, 30)
            r_bm25 = bm25_rank_map.get(cid, 30)
            rrf = (1.0 / (60.0 + r_dense + 1)) + (1.0 / (60.0 + r_bm25 + 1))
            fused.append((chunk_map[cid], rrf))

        fused_candidates = [c for c, _ in sorted(fused, key=lambda x: x[1], reverse=True)[:15]]

        # Stage 2: Neural Cross-Encoder Reranking
        reranked_chunks = reranker.rerank(query=query, candidates=fused_candidates, top_n=10)
        ranked_chunk_ids = [c["id"] for c in reranked_chunks]

        rank = (ranked_chunk_ids.index(target_id) + 1) if target_id in ranked_chunk_ids else 999
        all_ranks.append(rank)

        results_by_type[doc_type].append({
            "query_id": q["id"],
            "query": query,
            "target": target_id,
            "top_retrieved": ranked_chunk_ids[:3],
            "rank": rank,
            "hit_at_1": rank == 1,
            "hit_at_3": rank <= 3,
            "reciprocal_rank": 1.0 / rank if rank <= 30 else 0.0,
        })

    def calc_metrics(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(res_list)
        if n == 0:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0}
        hit1 = sum(1 for r in res_list if r["hit_at_1"]) / n
        hit3 = sum(1 for r in res_list if r["hit_at_3"]) / n
        mrr = sum(r["reciprocal_rank"] for r in res_list) / n
        return {
            "hit_at_1_pct": round(hit1 * 100, 2),
            "hit_at_3_pct": round(hit3 * 100, 2),
            "mrr": round(mrr, 4),
        }

    all_res = [r for sublist in results_by_type.values() for r in sublist]
    return {
        "engine": "Upgraded Qdrant + FastEmbed + FlashRank Reranker",
        "total_questions": len(questions),
        "overall": calc_metrics(all_res),
        "by_type": {
            "pdf": calc_metrics(results_by_type["pdf"]),
            "code": calc_metrics(results_by_type["code"]),
            "markdown": calc_metrics(results_by_type["markdown"]),
        },
    }


if __name__ == "__main__":
    import json
    print("\n" + "=" * 65)
    print("1. RUNNING BASELINE RETRIEVAL EVALUATION")
    print("=" * 65)
    t0 = time.perf_counter()
    baseline_report = evaluate_baseline_bm25(EVAL_DOCUMENTS, EVAL_QUESTIONS)
    t_baseline = round(time.perf_counter() - t0, 3)
    print(f"Completed baseline evaluation in {t_baseline}s.")
    print(json.dumps(baseline_report, indent=2))

    print("\n" + "=" * 65)
    print("2. RUNNING UPGRADED QDRANT + FLASHRANK RETRIEVAL EVALUATION")
    print("=" * 65)
    t1 = time.perf_counter()
    upgraded_report = evaluate_upgraded_pipeline(EVAL_DOCUMENTS, EVAL_QUESTIONS)
    t_upgraded = round(time.perf_counter() - t1, 3)
    print(f"Completed upgraded evaluation in {t_upgraded}s.")
    print(json.dumps(upgraded_report, indent=2))
