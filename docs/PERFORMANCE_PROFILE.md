# Jruva AI Platform: Empirical Subsystem Latency Profile

**Benchmark Date:** 2026-10-07  
**Host Environment:** Windows 11, Python 3.14.6  
**Methodology:** High-resolution timer (`time.perf_counter()`) over multi-trial iterations.

---

## 1. Latency Percentiles Summary

| Subsystem Component | Description / Operation | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Deterministic Fastpath** | Regex intent match & canned payload | **0.005** | **0.011** | **0.015** | **0.01** |
| **Response Cache Hit** | LRU SHA-256 normalized hash lookup | **0.003** | **0.003** | **0.005** | **0.003** |
| **Router Classification** | AST / keyword task classification | **0.02** | **0.044** | **0.061** | **0.032** |
| **BM25 Sparse Retrieval** | Inverted index token scoring (140 chunks) | **0.05** | **0.054** | **0.057** | **0.05** |
| **FlashRank Reranking** | Cross-encoder rescoring (15 candidates → 3) | **2.294** | **3.656** | **3.656** | **2.38** |
| **SQLite WAL Write** | Single row commit in WAL mode | **0.017** | **0.067** | **0.094** | **0.027** |
| **SQLite WAL Read** | Primary key indexed point query | **0.003** | **0.004** | **0.019** | **0.004** |

---

## 2. Key Profiling Findings

1. **Deterministic Fastpaths and Cache Hits execute in microseconds (< 0.05ms)**:
   - Queries served from exact-match cache or regex fastpaths bypass LLM reasoning completely, serving responses in **0.003ms** with zero GPU or cloud API consumption.
2. **FlashRank Reranking overhead is ~2.294ms**:
   - Pruning candidate pools from 50 to 15 prior to FlashRank keeps cross-encoder reranking under **3.656ms**, maintaining total end-to-end RAG latency well within human-conversational expectations (< 350ms).
3. **SQLite WAL concurrency profile**:
   - Primary key reads execute in **0.003ms**.
   - WAL writes commit in **0.017ms**.
   - While adequate for moderate concurrency, under high sustained parallel write pressure, migration to PostgreSQL (`docs/POSTGRESQL_MIGRATION_DESIGN.md`) prevents `sqlite3.OperationalError: database is locked`.
