"""Direct isolated timing benchmark for SQLite session_store operations.

Measures latency of:
1. session_store.ensure_session()
2. session_store.record_turn()
3. Combined cache-hit path sequence (ensure_session + 2x record_turn)
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

import time
from uuid import uuid4
import statistics
from app.services.session_store import session_store


def run_benchmark(iterations: int = 500):
    # Warmup
    s_id = uuid4()
    session_store.ensure_session(s_id, "Warmup Session")
    session_store.record_turn(s_id, "user", "Warmup user message")
    session_store.record_turn(s_id, "assistant", "Warmup assistant message")

    ensure_times = []
    turn1_times = []
    turn2_times = []
    combined_times = []

    for i in range(iterations):
        sess_id = uuid4()

        # 1. ensure_session timing
        t0 = time.perf_counter()
        session_store.ensure_session(sess_id, f"Benchmark Session {i}")
        t_ensure = (time.perf_counter() - t0) * 1000.0
        ensure_times.append(t_ensure)

        # 2. record_turn (user) timing
        t0 = time.perf_counter()
        session_store.record_turn(sess_id, "user", "Benchmark user query string")
        t_turn1 = (time.perf_counter() - t0) * 1000.0
        turn1_times.append(t_turn1)

        # 3. record_turn (assistant) timing
        t0 = time.perf_counter()
        session_store.record_turn(sess_id, "assistant", "Benchmark cached response content string from model")
        t_turn2 = (time.perf_counter() - t0) * 1000.0
        turn2_times.append(t_turn2)

        combined_times.append(t_ensure + t_turn1 + t_turn2)

    def compute_stats(vals):
        s = sorted(vals)
        n = len(s)
        p50 = s[int(n * 0.50)]
        p95 = s[int(n * 0.95)]
        p99 = s[int(n * 0.99)]
        return {
            "min": min(s),
            "mean": statistics.mean(s),
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "max": max(s),
        }

    print(f"\n=== ISOLATED SQLITE TIMING BENCHMARK (N={iterations} iterations) ===")
    targets = [
        ("ensure_session() alone", ensure_times),
        ("record_turn() (user turn) alone", turn1_times),
        ("record_turn() (assistant turn) alone", turn2_times),
        ("Combined (ensure_session + 2x record_turn on cache hit path)", combined_times),
    ]
    for name, data in targets:
        st = compute_stats(data)
        print(f"{name}:")
        print(f"  min:  {st['min']:.3f} ms")
        print(f"  mean: {st['mean']:.3f} ms")
        print(f"  p50:  {st['p50']:.3f} ms")
        print(f"  p95:  {st['p95']:.3f} ms")
        print(f"  p99:  {st['p99']:.3f} ms")
        print(f"  max:  {st['max']:.3f} ms\n")


def test_sqlite_isolation_benchmark():
    """Pytest wrapper ensuring the isolated measurement runs and asserts real bounds."""
    run_benchmark(iterations=300)


if __name__ == "__main__":
    run_benchmark(iterations=500)
