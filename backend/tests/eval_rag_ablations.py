"""Reproducible RAG Retrieval Ablation Suite for Jruva AI Platform.

Ablation Pipelines Evaluated:
  Pipeline A: Dense Only (FastEmbed BAAI/bge-small-en-v1.5)
  Pipeline B: BM25 Only (Lexical token frequencies)
  Pipeline C: Dense + BM25 Hybrid with Reciprocal Rank Fusion (RRF k=60)
  Pipeline D: Dense + BM25 + FlashRank Neural Cross-Encoder (ms-marco-TinyBERT-L-2-v2)

Evaluates on the 100-Query Hardened Evaluation Dataset across:
  - Hit@1, Hit@3, Hit@5
  - Recall@5
  - Mean Reciprocal Rank (MRR)
  - Mean Retrieval Latency (ms)
  - Category-specific breakdown (direct factual, terminology mismatch, multi-section, code/docs, ambiguous, adversarial)
  - Retrieval proxies: exact context evidence coverage and top-1 target match
  - Generation, hallucination and refusal behavior are NOT evaluated

Outputs machine-readable benchmark results to docs/rag_ablation_results.json.
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag.document_rag import BM25Index, tokenize
from app.services.rag.qdrant_vector_store import NeuralReranker, QdrantVectorStore
from tests.eval_dataset_hardened_100 import PASSAGES, QUESTIONS

logging.basicConfig(level=logging.WARNING)


def run_pipeline_a_dense_only(vector_store: QdrantVectorStore, collection_name: str, query: str, top_k: int = 15) -> List[str]:
    """Pipeline A: Dense Semantic Vector Retrieval Only."""
    results = vector_store.search_dense(collection_name, query, limit=top_k)
    return [r["id"] for r in results]


def run_pipeline_b_bm25_only(bm25: BM25Index, chunks: List[Dict[str, Any]], query: str, top_k: int = 15) -> List[str]:
    """Pipeline B: BM25 Lexical Keyword Retrieval Only."""
    tokens = tokenize(query) or [w.lower() for w in query.split() if len(w) > 1]
    scores = bm25.score(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [chunks[idx]["id"] for idx, _ in ranked[:top_k]]


def run_pipeline_c_hybrid_rrf(
    vector_store: QdrantVectorStore,
    collection_name: str,
    bm25: BM25Index,
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 15,
    rrf_k: int = 60,
) -> List[str]:
    """Pipeline C: Dense + BM25 Hybrid with Reciprocal Rank Fusion (RRF)."""
    dense_results = vector_store.search_dense(collection_name, query, limit=top_k)
    dense_ranks = {r["id"]: idx for idx, r in enumerate(dense_results)}

    tokens = tokenize(query) or [w.lower() for w in query.split() if len(w) > 1]
    scores = bm25.score(tokens)
    ranked_bm25 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    bm25_ranks = {chunks[idx]["id"]: rank for rank, (idx, _) in enumerate(ranked_bm25[:top_k])}

    all_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
    fused: List[Tuple[str, float]] = []
    default_rank = top_k * 2

    for cid in all_ids:
        r_dense = dense_ranks.get(cid, default_rank)
        r_bm25 = bm25_ranks.get(cid, default_rank)
        score = (1.0 / (rrf_k + r_dense + 1)) + (1.0 / (rrf_k + r_bm25 + 1))
        fused.append((cid, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in fused[:top_k]]


def run_pipeline_d_hybrid_flashrank(
    vector_store: QdrantVectorStore,
    collection_name: str,
    bm25: BM25Index,
    reranker: NeuralReranker,
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 15,
    final_top_k: int = 5,
    rrf_k: int = 60,
) -> List[str]:
    """Pipeline D: Dense + BM25 Hybrid + FlashRank Neural Cross-Encoder Reranking."""
    candidate_ids = run_pipeline_c_hybrid_rrf(vector_store, collection_name, bm25, chunks, query, top_k=top_k, rrf_k=rrf_k)
    chunk_map = {c["id"]: c for c in chunks}
    candidates = [chunk_map[cid] for cid in candidate_ids if cid in chunk_map]

    reranked = reranker.rerank(query=query, candidates=candidates, top_n=final_top_k)
    return [c["id"] for c in reranked]


def evaluate_pipeline(
    pipeline_name: str,
    runner_fn,
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Hit@1, Hit@3, Hit@5, Recall@5, MRR, latency, and category breakdowns."""
    answerable_questions = [q for q in questions if q["target_chunk_id"] is not None]
    unanswerable_questions = [q for q in questions if q["target_chunk_id"] is None]

    category_stats: Dict[str, List[Dict[str, Any]]] = {}
    latencies: List[float] = []

    for q in answerable_questions:
        t0 = time.perf_counter()
        retrieved_ids = runner_fn(q["query"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        target = q["target_chunk_id"]
        cat = q["category"]
        if cat not in category_stats:
            category_stats[cat] = []

        rank = (retrieved_ids.index(target) + 1) if target in retrieved_ids else 999
        hit1 = 1 if rank == 1 else 0
        hit3 = 1 if rank <= 3 else 0
        hit5 = 1 if rank <= 5 else 0
        rr = (1.0 / rank) if rank <= 30 else 0.0
        recall5 = 1 if rank <= 5 else 0

        category_stats[cat].append({
            "query_id": q["id"],
            "rank": rank,
            "hit1": hit1,
            "hit3": hit3,
            "hit5": hit5,
            "recall5": recall5,
            "rr": rr,
        })

    def summarize(entries: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(entries)
        if n == 0:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "hit_at_5": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
        return {
            "hit_at_1_pct": round((sum(e["hit1"] for e in entries) / n) * 100.0, 2),
            "hit_at_3_pct": round((sum(e["hit3"] for e in entries) / n) * 100.0, 2),
            "hit_at_5_pct": round((sum(e["hit5"] for e in entries) / n) * 100.0, 2),
            "recall_at_5_pct": round((sum(e["recall5"] for e in entries) / n) * 100.0, 2),
            "mrr": round(sum(e["rr"] for e in entries) / n, 4),
        }

    all_entries = [e for cat in category_stats.values() for e in cat]
    overall = summarize(all_entries)
    overall["mean_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    breakdown = {cat: summarize(entries) for cat, entries in category_stats.items()}

    # Unanswerable queries are counted but not scored by this retrieval harness.
    unans_count = len(unanswerable_questions)

    return {
        "pipeline": pipeline_name,
        "total_answerable_queries": len(answerable_questions),
        "total_unanswerable_queries": unans_count,
        "overall": overall,
        "by_category": breakdown,
    }


def evaluate_context_coverage(
    retrieval_fn,
    questions: List[Dict[str, Any]],
    chunk_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Measure retrieved evidence coverage, not generated-answer quality.

    Exact evidence substring coverage uses only answerable queries with nonempty
    evidence labels. Top-1 target match uses all answerable queries. No answers
    are generated, so refusal, citation precision and hallucination are unmeasured.
    """
    total_covered = 0
    total_target_matches = 0
    evidence_labeled_count = 0
    answerable_count = 0
    unanswerable_count = 0

    for q in questions:
        query = q["query"]
        target = q["target_chunk_id"]
        evidence = q.get("expected_evidence", "")

        if target is not None:
            answerable_count += 1
            retrieved = retrieval_fn(query)
            top_retrieved_ids = retrieved[:3]

            # Check if expected evidence is present in top-3 context
            context_text = " ".join(chunk_map[cid]["text"] for cid in top_retrieved_ids if cid in chunk_map)
            if evidence.strip():
                evidence_labeled_count += 1
                if evidence.lower() in context_text.lower():
                    total_covered += 1

            # Retrieval target match, not validation of generated citations.
            if top_retrieved_ids and top_retrieved_ids[0] == target:
                total_target_matches += 1
        else:
            unanswerable_count += 1

    context_coverage_pct = round((total_covered / evidence_labeled_count) * 100.0, 2) if evidence_labeled_count else None
    target_match_pct = round((total_target_matches / answerable_count) * 100.0, 2) if answerable_count else None

    return {
        "evaluator_type": "Retrieval-only exact evidence coverage and top-1 target match",
        "context_coverage_pct": context_coverage_pct,
        "top1_target_match_pct": target_match_pct,
        "evidence_labeled_queries": evidence_labeled_count,
        "generation_evaluated": False,
        "answerable_evaluated": answerable_count,
        "unanswerable_not_evaluated": unanswerable_count,
    }


def main():
    print("=" * 75)
    print("JRUV AI PLATFORM: RAG RETRIEVAL ABLATION SUITE (100 QUERIES, 50 PASSAGES)")
    print("=" * 75)

    collection_name = "eval_ablations_100"
    vector_store = QdrantVectorStore(storage_path=":memory:")
    print(f"Indexing {len(PASSAGES)} passages into in-memory Qdrant with BGE-small embeddings...")
    t_idx = time.perf_counter()
    vector_store.index_chunks(collection_name, PASSAGES)
    print(f"Indexing complete in {round(time.perf_counter() - t_idx, 2)}s.")

    print("Fitting BM25 index on passage token corpus...")
    tokenized_docs = [tokenize(f"{c.get('doc_name', '')} {c['text']}") for c in PASSAGES]
    bm25 = BM25Index()
    bm25.fit(tokenized_docs)

    print("Initializing FlashRank Neural Cross-Encoder...")
    reranker = NeuralReranker()
    chunk_map = {c["id"]: c for c in PASSAGES}

    # Execute 4 Pipeline Ablations
    print("\n[1/4] Running Pipeline A: Dense Only (FastEmbed BGE-small)...")
    res_a = evaluate_pipeline(
        "Pipeline A: Dense Only (FastEmbed BGE-small)",
        lambda q: run_pipeline_a_dense_only(vector_store, collection_name, q, top_k=15),
        QUESTIONS,
    )

    print("[2/4] Running Pipeline B: BM25 Only (Lexical Token Matching)...")
    res_b = evaluate_pipeline(
        "Pipeline B: BM25 Only",
        lambda q: run_pipeline_b_bm25_only(bm25, PASSAGES, q, top_k=15),
        QUESTIONS,
    )

    print("[3/4] Running Pipeline C: Dense + BM25 Hybrid (RRF k=60)...")
    res_c = evaluate_pipeline(
        "Pipeline C: Dense + BM25 Hybrid (RRF k=60)",
        lambda q: run_pipeline_c_hybrid_rrf(vector_store, collection_name, bm25, PASSAGES, q, top_k=15),
        QUESTIONS,
    )

    print("[4/4] Running Pipeline D: Dense + BM25 + FlashRank Reranker...")
    res_d = evaluate_pipeline(
        "Pipeline D: Dense + BM25 + FlashRank Cross-Encoder",
        lambda q: run_pipeline_d_hybrid_flashrank(vector_store, collection_name, bm25, reranker, PASSAGES, q, top_k=15, final_top_k=5),
        QUESTIONS,
    )

    print("\n[5/5] Measuring Retrieved Context Coverage on Pipeline D...")
    context_eval = evaluate_context_coverage(
        lambda q: run_pipeline_d_hybrid_flashrank(vector_store, collection_name, bm25, reranker, PASSAGES, q, top_k=15, final_top_k=5),
        QUESTIONS,
        chunk_map,
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "passage_count": len(PASSAGES),
            "query_count": len(QUESTIONS),
            "answerable_queries": res_d["total_answerable_queries"],
            "unanswerable_queries": res_d["total_unanswerable_queries"],
            "categories": list(res_d["by_category"].keys()),
        },
        "ablations": [res_a, res_b, res_c, res_d],
        "context_quality": context_eval,
    }

    # Save to docs/rag_ablation_results.json
    output_path = Path(__file__).resolve().parent.parent.parent / "docs" / "rag_ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAblation results saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Pipeline':<45} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Latency':<8}")
    print("-" * 90)
    for res in [res_a, res_b, res_c, res_d]:
        p = res["pipeline"]
        o = res["overall"]
        print(f"{p:<45} | {o['hit_at_1_pct']:<7}% | {o['hit_at_3_pct']:<7}% | {o['hit_at_5_pct']:<7}% | {o['mrr']:<8} | {o['mean_latency_ms']}ms")
    print("=" * 90)

    print("\nRetrieval Context Proxies:")
    print(f"  Context Coverage: {context_eval['context_coverage_pct']}%")
    print(f"  Top-1 Target Match: {context_eval['top1_target_match_pct']}%")
    print("  Generation, hallucination, citation precision and refusal: not evaluated")


if __name__ == "__main__":
    main()
