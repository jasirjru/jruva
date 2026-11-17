"""Isolated timing comparison: Synchronous SQLite critical-path vs Async dispatch critical-path."""
import sys
import os
import asyncio
import time
from uuid import uuid4
import statistics

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from app.services.session_store import session_store
from app.services.chat_service import ChatService

chat_service = ChatService()


def compute_stats(vals):
    s = sorted(vals)
    n = len(s)
    return {
        "min": min(s),
        "mean": statistics.mean(s),
        "p50": s[int(n * 0.50)],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": max(s),
    }


async def run_comparison(iterations: int = 500):
    # Warmup
    s_warm = uuid4()
    session_store.ensure_session(s_warm, "Warmup")
    session_store.record_turn(s_warm, "user", "Warmup msg")
    session_store.record_turn(s_warm, "assistant", "Warmup ans")

    sync_times = []
    async_dispatch_times = []

    print(f"Running N={iterations} iterations of Synchronous SQLite path...")
    for i in range(iterations):
        sess_id = uuid4()
        t0 = time.perf_counter()
        title = session_store.generate_title_from_message("What is the capital of France?")
        session_store.ensure_session(sess_id, title)
        session_store.record_turn(sess_id, "user", "What is the capital of France?")
        session_store.record_turn(sess_id, "assistant", "Paris is the capital of France.")
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        sync_times.append(t_elapsed)

    print(f"Running N={iterations} iterations of Async Task Dispatch path...")
    for i in range(iterations):
        sess_id = uuid4()
        t0 = time.perf_counter()
        chat_service._async_persist_cache_hit(sess_id, "What is the capital of France?", "Paris is the capital of France.")
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        async_dispatch_times.append(t_elapsed)

    # Let background tasks settle
    await asyncio.sleep(2.0)

    st_sync = compute_stats(sync_times)
    st_async = compute_stats(async_dispatch_times)

    print("\n=== ISOLATED SQLITE CRITICAL-PATH BENCHMARK: BEFORE vs AFTER ===")
    print(f"{'Metric':<10} | {'BEFORE (Sync SQLite on path)':<30} | {'AFTER (Async Dispatch on path)':<30}")
    print("-" * 75)
    print(f"{'min':<10} | {st_sync['min']:>26.3f} ms | {st_async['min']:>26.3f} ms")
    print(f"{'mean':<10} | {st_sync['mean']:>26.3f} ms | {st_async['mean']:>26.3f} ms")
    print(f"{'p50':<10} | {st_sync['p50']:>26.3f} ms | {st_async['p50']:>26.3f} ms")
    print(f"{'p95':<10} | {st_sync['p95']:>26.3f} ms | {st_async['p95']:>26.3f} ms")
    print(f"{'p99':<10} | {st_sync['p99']:>26.3f} ms | {st_async['p99']:>26.3f} ms")
    print(f"{'max':<10} | {st_sync['max']:>26.3f} ms | {st_async['max']:>26.3f} ms")


if __name__ == "__main__":
    asyncio.run(run_comparison(500))
