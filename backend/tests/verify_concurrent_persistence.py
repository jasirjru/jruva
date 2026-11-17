"""Concurrent-load data-loss and correctness verification for JRU AI Chatbot.

Fires 15 concurrent requests (simulating 15 simultaneous users) on the cached chat path
using asyncio.gather, measures concurrent roundtrip latency (p50/p95/p99), and directly
queries SQLite (chatbot.db) after requests return to verify that 100% of admitted requests
have their session metadata and message turns persisted without data loss.
"""
import sys
import os
import asyncio
import time
import uuid
import sqlite3
import statistics
import httpx

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

from app.main import app
from app.services.session_store import session_store


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


async def run_concurrent_check(num_users: int = 15, requests_per_user: int = 4):
    transport = httpx.ASGITransport(app=app)
    admitted_sessions = []
    latencies = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=15.0) as client:
        # Warmup
        warmup_id = str(uuid.uuid4())
        w_res = await client.post(
            "/api/v1/chat",
            json={
                "session_id": warmup_id,
                "message": "benchmark-cached-probe-prompt",
                "model": "openai/gpt-oss-120b",
                "tier": "S+++",
                "stream": False,
            },
            headers={"X-User-ID": "warmup-user"},
        )
        assert w_res.status_code == 200, f"Warmup failed: {w_res.status_code}"

        async def user_worker(user_idx: int):
            user_id = f"concurrent-user-{user_idx}-{uuid.uuid4().hex[:6]}"
            for req_idx in range(requests_per_user):
                sess_id = str(uuid.uuid4())
                payload = {
                    "session_id": sess_id,
                    "message": "benchmark-cached-probe-prompt",
                    "model": "openai/gpt-oss-120b",
                    "tier": "S+++",
                    "stream": False,
                }
                t0 = time.perf_counter()
                res = await client.post(
                    "/api/v1/chat",
                    json=payload,
                    headers={"X-User-ID": user_id},
                )
                t_elapsed = (time.perf_counter() - t0) * 1000.0
                latencies.append(t_elapsed)

                if res.status_code == 200:
                    admitted_sessions.append(sess_id)
                # Short spacing between turns within the same user session
                await asyncio.sleep(0.05)

        # Launch all 15 users concurrently
        tasks = [user_worker(i) for i in range(num_users)]
        await asyncio.gather(*tasks)

        # Wait for background thread-pool persistence to complete
        await asyncio.sleep(1.0)

    # Direct SQLite verification of all admitted sessions
    conn = sqlite3.connect(session_store.db_path)
    cursor = conn.cursor()

    sessions_found = 0
    messages_verified = 0
    missing_sessions = []
    missing_messages = []

    for sid in admitted_sessions:
        cursor.execute("SELECT id, session_id FROM sessions WHERE session_id = ?", (sid,))
        s_row = cursor.fetchone()
        if s_row:
            sessions_found += 1
        else:
            missing_sessions.append(sid)

        cursor.execute("SELECT count(*) FROM messages WHERE session_id = ?", (sid,))
        m_count = cursor.fetchone()[0]
        if m_count == 2:
            messages_verified += 1
        else:
            missing_messages.append((sid, m_count))

    conn.close()

    st = compute_stats(latencies)

    print(f"\n=== CONCURRENT LOAD LATENCY & CORRECTNESS CHECK ({num_users} users, {len(latencies)} total requests) ===")
    print(f"Total Requests Dispatched:   {len(latencies)}")
    print(f"Total Admitted (HTTP 200):   {len(admitted_sessions)}")
    print(f"Total Rejected (HTTP 429):   {len(latencies) - len(admitted_sessions)}")
    print("-" * 55)
    print(f"Concurrent Latency min:      {st['min']:.3f} ms")
    print(f"Concurrent Latency mean:     {st['mean']:.3f} ms")
    print(f"Concurrent Latency p50:      {st['p50']:.3f} ms")
    print(f"Concurrent Latency p95:      {st['p95']:.3f} ms")
    print(f"Concurrent Latency p99:      {st['p99']:.3f} ms")
    print(f"Concurrent Latency max:      {st['max']:.3f} ms")
    print("-" * 55)
    print(f"Admitted Sessions to Verify: {len(admitted_sessions)}")
    print(f"Sessions Persisted in DB:    {sessions_found}/{len(admitted_sessions)}")
    print(f"Messages Persisted (2/sess): {messages_verified}/{len(admitted_sessions)}")
    print(f"Missing Sessions:            {len(missing_sessions)}")
    print(f"Incomplete Message Sets:     {len(missing_messages)}")

    assert len(missing_sessions) == 0, f"Data loss: {len(missing_sessions)} sessions missing in SQLite"
    assert len(missing_messages) == 0, f"Data loss: {len(missing_messages)} sessions have incomplete messages"
    print("\nResult: PASS - ZERO DATA LOSS under concurrent 15-user load.")


if __name__ == "__main__":
    asyncio.run(run_concurrent_check(15, 4))
