# Jruva: Reproducible Retrieval Benchmarks

Run on 9 Oct 2026 (Windows, Python 3.12 venv, FastEmbed ONNX, Qdrant in-memory, Okapi BM25).
Script: `scripts/benchmark_retrieval_truth.py`. The question sets are small and hand-built development sets,
not held-out test sets, so treat the numbers as a sanity check of the pipeline, not a general accuracy claim.

---

## 1. Executive Summary & Root Cause Analysis

Prior documentation reported conflicting and potentially misleading performance claims for JRUVA's retrieval pipeline:
- **Historical Claim A:** "Dense-only Hit@5: 100%"
- **Historical Claim B:** "Dense + BM25 + FlashRank Hit@1: 88.89% on historical corpus"
- **Historical Claim C:** "Overnight retrieval benchmark Hit@1: 93.33%"
- **Historical Claim D:** "Overnight 100-query ablation Hit@1: 85.56%"
- **Historical Claim E:** "Reported 0.36ms retrieval benchmark latency"

This audit examined the source code (`eval_retrieval.py`, `eval_rag_ablations.py`, `qdrant_vector_store.py`), isolated timing boundaries, and executed independent reproductions across all corpora.

### Root Cause of the "0.36ms" Latency Artifact
The reported `0.36ms` latency was **not neural cross-encoder retrieval**, but an artifact of two architectural behaviors:
1. **Query Embedding Cache Pre-warming:** In `eval_rag_ablations.py`, Pipeline A (Dense Only) executed first across all 90 answerable queries. During that run, `get_or_compute_query_embedding()` stored 384-dimensional dense vectors in the global `QueryEmbeddingCache`. When Pipeline D subsequently executed, `query_embedding_cache.get(query)` took **0.006ms**, completely bypassing the FastEmbed ONNX neural inference (~3.91ms).
2. **FlashRank Graceful Degradation Passthrough:** `LOW_MEMORY_RAG_MODE=True` and `ENABLE_FLASHRANK=False` are default configurations in `Settings` to protect the 512MB RAM ceiling of cloud container tiers. In `NeuralReranker.rerank()`, lines 765–767 explicitly execute:
   ```python
   if not self.is_available or self.ranker is None:
       # Degraded / Low-memory mode: Return top candidates sorted by RRF directly
       return candidates[:limit]
   ```
   Because FlashRank was disabled, `rerank()` operated as an immediate Python slice (`candidates[:5]`), taking **0.001ms** instead of running cross-attention inference through TinyBERT (~25–45ms).
3. **Corpus Size:** The evaluation indexed only 50 short passages into in-memory Qdrant and BM25.

**Verdict:** The 0.36ms figure represents an in-memory dictionary lookup + 50-passage Qdrant HNSW filter + BM25 score + array slice. **Full cold neural retrieval with BGE-small ONNX takes ~4.2–5.5ms**. If real FlashRank cross-encoding were invoked, latency would be ~30–50ms.

---

## 2. Incomparability of Historical Hit@1 Claims

The audit discovered why historical Hit@1 varied from 85.56% to 93.33% and 88.89%:

| Metric Claim | Dataset / Origin | Passage Count | Query Count | Category Nature | Why Incomparable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **88.89% Hit@1** | Early prototype benchmark (`eval_retrieval_v1`) | 18 passages | 18 queries | Simple direct facts | 16/18 hits; small sample size, no distractors. |
| **93.33% Hit@1** | `retrieval_eval_dataset.py` (`eval_retrieval.py`) | 30 passages | 30 queries | Clean technical docs (PDF, Code, Markdown) | Small 30-item corpus; clean queries with distinct vocabulary; baseline had title-boosting logic. |
| **85.56% Hit@1** | `eval_dataset_hardened_100.py` (`eval_rag_ablations.py`) | 50 passages | 100 queries (90 answerable) | Hardened: includes ambiguity, lexical distractors, multi-section | Contains 10 ambiguous queries (Hit@1: 30.0%) and 15 terminology mismatches (Hit@1: 80.0%), pulling down the mean. |

**Conclusion:** These three numbers cannot be compared as improvements or regressions over one another; they reflect three completely different datasets with different difficulty distributions.

---

## 3. Independently Reproducible Benchmark Truth

Executed via `scripts/benchmark_retrieval_truth.py` on commit `0170f16`:

### Dataset A: 30-Query Benchmark (`retrieval_eval_dataset.py`)
- Passages: 30 (10 PDF, 10 Code, 10 Markdown)
- Queries: 30 answerable queries

| Pipeline | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@3 | nDCG@5 | Latency p50 | Latency p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A: Dense Only (BGE-small Cold)** | **96.67%** | 100.0% | 100.0% | 100.0% | 0.9833 | 0.9877 | 0.9877 | 5.04 ms | 7.21 ms |
| **Pipeline B: BM25 Only (Lexical)** | 90.00% | 100.0% | 100.0% | 100.0% | 0.9500 | 0.9631 | 0.9631 | 0.03 ms | 0.05 ms |
| **Pipeline C: Hybrid RRF (k=60)** | 93.33% | 100.0% | 100.0% | 100.0% | 0.9667 | 0.9754 | 0.9754 | 0.35 ms | 0.52 ms |

### Dataset B: 100-Query Hardened Benchmark (`eval_dataset_hardened_100.py`)
- Passages: 50 (Systems, JRUVA Specs, Academic Regulations, Distractors)
- Queries: 100 total (90 answerable, 10 unanswerable)

| Pipeline | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@3 | nDCG@5 | Latency p50 | Latency p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A: Dense Only (BGE-small Cold)** | 83.33% | **98.89%** | **100.0%** | **100.0%** | **0.9102** | **0.9312** | **0.9357** | 4.68 ms | 7.82 ms |
| **Pipeline B: BM25 Only (Lexical)** | 74.44% | 84.44% | 90.00% | 90.00% | 0.8100 | 0.8073 | 0.8301 | 0.03 ms | 0.06 ms |
| **Pipeline C: Hybrid RRF (k=60)** | **84.44%** | 94.44% | 96.67% | 96.67% | 0.8997 | 0.9073 | 0.9168 | 0.51 ms | 0.74 ms |
| **Pipeline D: Degraded Passthrough Rerank** | **84.44%** | 94.44% | 96.67% | 96.67% | 0.8957 | 0.9051 | 0.9146 | 0.49 ms | 0.69 ms |

---

## 4. Architectural Latency Component Breakdown

Measured with micro-benchmarking over isolated stages (100-passage corpus):

| Stage | What It Executes | Mean Latency | Architectural Role |
| :--- | :--- | :---: | :--- |
| **1. Cold Query Embedding** | ONNX Runtime CPU inference on BAAI/bge-small-en-v1.5 (uncached) | **3.909 ms** | Mandatory for new/unique user queries |
| **2. Warm Query Embedding** | LRU `QueryEmbeddingCache` retrieval (SHA-256 normalized string) | **0.006 ms** | Instant replay for repeated prompts |
| **3. Qdrant Vector Search** | HNSW index vector similarity search in in-memory Qdrant | **0.582 ms** | Dense semantic candidate retrieval (top 15) |
| **4. BM25 Lexical Scoring** | Okapi BM25 keyword matching across tokenized passage corpus | **0.075 ms** | Sparse lexical candidate retrieval (top 15) |
| **5. Reciprocal Rank Fusion** | RRF merging of dense and sparse candidate ranks (k=60) | **0.017 ms** | Hybrid rank aggregation |
| **6. Degraded Reranker** | Passthrough candidate slice (`[:5]`) | **0.001 ms** | Low-memory mode fallback |
| **7. FlashRank Cross-Encoder** | TinyBERT cross-attention reranking of 15 candidate passages | *~32.5 ms (est)* | Neural re-scoring (disabled in low-memory) |
| **Composite Cold End-to-End** | Cold Embedding + Qdrant + BM25 + RRF | **~4.58 ms** | Real cold retrieval time |
| **Composite Warm End-to-End** | Cached Embedding + Qdrant + BM25 + RRF + Degraded Passthrough | **~0.68 ms** | What was reported as "0.36ms" |

---

## 5. Key Verification Findings

1. **Dense Retrieval Is Dominant:** Dense-only retrieval with BGE-small achieves **100% Hit@5** on both the 30-query and 100-query datasets. No ground-truth document was ever dropped outside the top 5 by dense retrieval alone.
2. **BM25 Fails on Ambiguity & Paraphrasing:** BM25 alone drops to **10% Hit@1** on ambiguous queries and **53.33% Hit@1** on terminology mismatches, demonstrating why pure lexical search is insufficient for conversational AI.
3. **RRF Hybrid Synergies:** Hybrid RRF combines BM25's strength on exact acronyms and code identifiers with dense retrieval's semantic flexibility, lifting Hit@1 on 100 queries to **84.44%**.
4. **Reproducibility Guarantee:** All numbers documented in this report can be re-generated at any time using:
   ```powershell
   & "backend/.venv/Scripts/python.exe" scripts/benchmark_retrieval_truth.py
   ```
