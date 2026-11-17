# Jruva retrieval benchmarks

Run: 2026-11-17 15:20:49 UTC, Windows 11, AMD Ryzen 9 7940HX, Python 3.14, FastEmbed ONNX (BAAI/bge-small-en-v1.5), Qdrant in-memory, Okapi BM25.
Reproduce: `python scripts/benchmark_retrieval_truth.py` (writes `docs/retrieval_benchmark_truth.json`).

**Read this first.** Both question sets are small, hand-built development sets. There is no held-out test set.
The production fusion weights were tuned on the 100-question set, so its hybrid number is optimistic.
With 90 scored questions, differences of 1-2 points in MRR are within noise.

## 100-question development set

50 passages, 100 questions (90 answerable, 10 unanswerable; unanswerable questions are not scored).

| Pipeline | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | Latency p50 |
|---|---|---|---|---|---|---|
| Dense only (BGE-small) | 83.3% | 98.9% | 100.0% | 0.910 | 0.933 | 3.46 ms |
| BM25 only | 74.4% | 84.4% | 90.0% | 0.810 | 0.830 | 0.03 ms |
| Hybrid, equal-weight RRF (k=60) | 84.4% | 94.4% | 96.7% | 0.900 | 0.914 | 0.51 ms |
| Hybrid, production weighted RRF (k=20, dense 1.0, BM25 0.7) | 86.7% | 97.8% | 98.9% | 0.919 | 0.935 | 0.35 ms |
| Equal-weight RRF + reranker in passthrough mode | 84.4% | 94.4% | 96.7% | 0.896 | 0.914 | 0.35 ms |

## 30-question set

| Pipeline | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | Latency p50 |
|---|---|---|---|---|---|---|
| Dense only (BGE-small) | 96.7% | 100.0% | 100.0% | 0.983 | 0.988 | 3.76 ms |
| BM25 only | 90.0% | 100.0% | 100.0% | 0.950 | 0.963 | 0.02 ms |
| Hybrid, equal-weight RRF (k=60) | 93.3% | 100.0% | 100.0% | 0.967 | 0.975 | 0.35 ms |

## Latency by stage (100-passage corpus, mean)

| Stage | ms |
|---|---|
| cold query embedding onnx | 3.333 |
| warm query embedding cache | 0.005 |
| qdrant vector search | 0.473 |
| bm25 lexical scoring | 0.069 |
| rrf fusion | 0.015 |
| degraded passthrough rerank | 0.001 |
| real flashrank tinybert rerank | disabled_in_low_memory |
| composite cold neural pipeline | 3.89 |
| composite warm cached degraded pipeline | 0.563 |

## What the numbers say

- Dense retrieval does most of the work: it is the strongest single method on both sets.
- BM25 alone is weaker on paraphrased and ambiguous questions, but it is cheap and handles exact terms (codes, names, IDs).
- The production weighted fusion scores slightly above dense only on the 100-question set, but the weights were tuned on that set. Treat it as a tie until a separate test set exists.
- The TinyBERT reranker is optional and off in production (512 MB memory limit). In this benchmark it runs as a passthrough, so no reranker quality number is claimed here.
- The latency numbers are for a tiny in-memory corpus on a desktop CPU, and the hybrid rows reuse query embeddings cached by the dense run. They are not production latency.

The standalone benchmark repo (https://github.com/jasirjru/jruva-retrieval-eval) uses its own code, so its numbers can differ slightly from this file.
