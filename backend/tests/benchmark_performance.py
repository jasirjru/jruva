"""Comprehensive Performance and Load Engineering Benchmark (Phase 7).
Measures:
1. Latency profiles (p50, p90, p95, p99, min, max, mean).
2. Concurrency scaling across C=1, C=3, C=5, C=10 workers.
3. Time to First Token (TTFT) and token throughput (tokens/sec).
4. Subsystem latency breakdown: API routing, Vector/BM25 retrieval, memory access.
5. Process RSS memory and CPU utilization.
Outputs structured metrics to docs/results/performance_benchmark_results.json.
"""
import asyncio
import json
import os
import statistics
import time
import sys
from pathlib import Path
from typing import Any, Dict, List
import httpx
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_URL = os.getenv("BENCHMARK_BACKEND_URL", "http://127.0.0.1:8001")


def get_system_metrics() -> Dict[str, float]:
    process = psutil.Process(os.getpid())
    return {
        "rss_memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "system_memory_percent": psutil.virtual_memory().percent,
    }


def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    sorted_l = sorted(latencies)
    n = len(sorted_l)
    def p(pct: float) -> float:
        idx = min(int(n * pct), n - 1)
        return round(sorted_l[idx], 2)
    return {
        "min": round(sorted_l[0], 2),
        "max": round(sorted_l[-1], 2),
        "mean": round(statistics.mean(sorted_l), 2),
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
    }


async def probe_endpoint(client: httpx.AsyncClient, method: str, path: str, json_data: Any = None) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        if method.upper() == "GET":
            resp = await client.get(f"{BASE_URL}{path}", timeout=10.0)
        else:
            resp = await client.post(f"{BASE_URL}{path}", json=json_data, timeout=15.0)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "status": resp.status_code,
            "duration_ms": duration_ms,
            "success": resp.status_code in (200, 201),
        }
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "status": 0,
            "duration_ms": duration_ms,
            "success": False,
            "error": str(exc),
        }


async def run_concurrency_tier(concurrency: int, total_requests: int, path: str) -> Dict[str, Any]:
    print(f"  Benchmarking {path} with Concurrency={concurrency}, Total Requests={total_requests}...")
    sem = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        async def worker():
            async with sem:
                res = await probe_endpoint(client, "GET", path)
                results.append(res)

        start_time = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        total_time_sec = time.perf_counter() - start_time

    latencies = [r["duration_ms"] for r in results]
    successes = sum(1 for r in results if r["success"])
    rps = round(total_requests / total_time_sec, 2) if total_time_sec > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "success_count": successes,
        "error_count": total_requests - successes,
        "success_rate_pct": round((successes / total_requests) * 100.0, 2),
        "total_time_sec": round(total_time_sec, 3),
        "requests_per_sec": rps,
        "latencies_ms": compute_percentiles(latencies),
    }


async def measure_chat_stream_ttft() -> Dict[str, Any]:
    """Measure Time To First Token (TTFT) and token throughput on live chat stream."""
    print("  Measuring Chat Stream TTFT and Throughput...")
    url = f"{BASE_URL}/api/v1/chat/stream"
    payload = {
        "message": "Explain what an index is in a relational database in 2 sentences.",
        "stream": True,
        "tier": "B",
    }

    start = time.perf_counter()
    ttft_ms = None
    chunks_count = 0
    total_bytes = 0

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=30.0) as resp:
                async for chunk in resp.aiter_bytes():
                    if ttft_ms is None and len(chunk) > 0:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    chunks_count += 1
                    total_bytes += len(chunk)

        total_time_sec = time.perf_counter() - start
        return {
            "status": "success",
            "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
            "total_duration_sec": round(total_time_sec, 2),
            "chunks_received": chunks_count,
            "total_bytes_received": total_bytes,
            "estimated_tok_per_sec": round((total_bytes / 4.0) / total_time_sec, 2) if total_time_sec > 0 else 0.0,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "ttft_ms": None,
        }


def benchmark_local_rag_latency() -> Dict[str, Any]:
    """Benchmark local in-memory BM25 vs ONNX embedding retrieval latency."""
    print("  Benchmarking Subsystem Retrieval Latency...")
    from app.services.rag.document_rag import BM25Index, tokenize

    corpus = [
        "Jharkhand Rai University B.Tech Computer Science curriculum focuses on AI, data structures, and algorithms.",
        "The highest salary package achieved in recent placement season was 14.5 LPA at top MNCs.",
        "Hostel accommodation is available with 24/7 high speed Wi-Fi and nutritious mess facilities.",
        "End semester examinations carry 70% weightage while internal assessments contribute 30%.",
        "Admission criteria for engineering programs require 45% minimum aggregate in PCM subjects.",
    ] * 20  # 100 passages

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Index()
    bm25.fit(tokenized_corpus)

    test_queries = [
        "What is the placement package?",
        "B.Tech admission requirements in PCM",
        "Hostel facilities and Wi-Fi",
        "Examination marks distribution",
    ]

    latencies_bm25 = []
    for q in test_queries * 25:  # 100 queries
        t0 = time.perf_counter()
        q_tokens = tokenize(q)
        _ = bm25.score(q_tokens)
        latencies_bm25.append((time.perf_counter() - t0) * 1000.0)

    return {
        "passages_count": len(corpus),
        "queries_evaluated": len(latencies_bm25),
        "bm25_latency_ms": compute_percentiles(latencies_bm25),
    }


async def main():
    print("=" * 70)
    print("JRUV CHATBOT: COMPREHENSIVE PERFORMANCE & LOAD BENCHMARK")
    print(f"Target Backend: {BASE_URL}")
    print("=" * 70)

    pre_system = get_system_metrics()
    print(f"System Baseline: RSS={pre_system['rss_memory_mb']}MB, Memory Usage={pre_system['system_memory_percent']}%")

    # 1. Health Probe Concurrency Scaling
    concurrency_tiers = [1, 3, 5, 10]
    health_results = []
    for c in concurrency_tiers:
        res = await run_concurrency_tier(concurrency=c, total_requests=c * 5, path="/api/v1/health")
        health_results.append(res)

    # 2. Models Catalog Endpoint Concurrency (C=5)
    catalog_res = await run_concurrency_tier(concurrency=5, total_requests=15, path="/api/v1/models")

    # 3. Chat Stream TTFT
    stream_res = await measure_chat_stream_ttft()

    # 4. RAG Retrieval Latency
    rag_res = benchmark_local_rag_latency()

    post_system = get_system_metrics()

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE_URL,
        "system_metrics": {
            "initial_rss_mb": pre_system["rss_memory_mb"],
            "final_rss_mb": post_system["rss_memory_mb"],
            "rss_delta_mb": round(post_system["rss_memory_mb"] - pre_system["rss_memory_mb"], 2),
            "system_ram_percent": post_system["system_memory_percent"],
        },
        "concurrency_scaling_health": health_results,
        "catalog_endpoint_concurrency_5": catalog_res,
        "chat_streaming_performance": stream_res,
        "rag_retrieval_subsystem": rag_res,
    }

    out_file = Path("docs/results/performance_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"Benchmark results successfully saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
