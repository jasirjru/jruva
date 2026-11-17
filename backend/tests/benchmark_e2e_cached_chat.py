"""End-to-end benchmark for cached-chat requests comparing Before (sync SQLite) vs After (async persistence).

Also validates zero data loss by directly querying SQLite database after the request returns.
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
from app.api.dependencies import get_chat_service
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


async def main():
    chat_svc = get_chat_service()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=10.0) as client:
        # Warmup
        warmup_payload = {
            "session_id": str(uuid.uuid4()),
            "message": "benchmark-cached-probe-prompt",
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        r = await client.post("/api/v1/chat", json=warmup_payload, headers={"X-User-ID": "warmup-user"})
        assert r.status_code == 200, f"Warmup failed: {r.status_code} {r.text}"

        iterations = 50

        # =========================================================================
        # 1. BEFORE FIX: Synchronous SQLite persistence directly on the request path
        # =========================================================================
        def sync_persist(session_id, message, content):
            title = chat_svc.store.generate_title_from_message(message)
            chat_svc.store.ensure_session(session_id, title)
            chat_svc.record_turn(session_id, "user", message)
            chat_svc.record_turn(session_id, "assistant", content)

        original_persist = chat_svc._persist_cache_hit
        chat_svc._persist_cache_hit = sync_persist

        before_latencies = []
        for i in range(iterations):
            uid = f"bench-sync-user-{i}"
            payload = {
                "session_id": str(uuid.uuid4()),
                "message": "benchmark-cached-probe-prompt",
                "model": "openai/gpt-oss-120b",
                "tier": "S+++",
                "stream": False,
            }
            t0 = time.perf_counter()
            res = await client.post("/api/v1/chat", json=payload, headers={"X-User-ID": uid})
            t_ms = (time.perf_counter() - t0) * 1000.0
            assert res.status_code == 200
            assert res.json().get("cached") is True
            before_latencies.append(t_ms)

        # =========================================================================
        # 2. AFTER FIX: Asynchronous task dispatch off the request path
        # =========================================================================
        chat_svc._persist_cache_hit = original_persist

        after_latencies = []
        for i in range(iterations):
            uid = f"bench-async-user-{i}"
            payload = {
                "session_id": str(uuid.uuid4()),
                "message": "benchmark-cached-probe-prompt",
                "model": "openai/gpt-oss-120b",
                "tier": "S+++",
                "stream": False,
            }
            t0 = time.perf_counter()
            res = await client.post("/api/v1/chat", json=payload, headers={"X-User-ID": uid})
            t_ms = (time.perf_counter() - t0) * 1000.0
            assert res.status_code == 200
            assert res.json().get("cached") is True
            after_latencies.append(t_ms)

        # =========================================================================
        # 3. VERIFY NO DATA LOSS: Query DB after request returns
        # =========================================================================
        verify_session_id = uuid.uuid4()
        verify_payload = {
            "session_id": str(verify_session_id),
            "message": "benchmark-cached-probe-prompt",
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        t0 = time.perf_counter()
        res_verify = await client.post(
            "/api/v1/chat",
            json=verify_payload,
            headers={"X-User-ID": "verify-user-1"},
        )
        req_latency_ms = (time.perf_counter() - t0) * 1000.0
        assert res_verify.status_code == 200
        verify_json = res_verify.json()
        assert verify_json.get("cached") is True

        # Allow background task to complete write
        await asyncio.sleep(0.2)

        # Query SQLite directly to confirm records exist in database
        db_path = session_store.db_path
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, session_id, title, created_at FROM sessions WHERE session_id = ?", (str(verify_session_id),))
        session_row = cursor.fetchone()

        cursor.execute(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (str(verify_session_id),),
        )
        message_rows = cursor.fetchall()
        conn.close()

    st_before = compute_stats(before_latencies)
    st_after = compute_stats(after_latencies)

    print("\n=== REAL END-TO-END CACHED-CHAT LATENCY (N=50 requests) ===")
    print(f"{'Metric':<10} | {'BEFORE (Sync SQLite on path)':<30} | {'AFTER (Async Persistence)':<30}")
    print("-" * 75)
    print(f"{'min':<10} | {st_before['min']:>26.3f} ms | {st_after['min']:>26.3f} ms")
    print(f"{'mean':<10} | {st_before['mean']:>26.3f} ms | {st_after['mean']:>26.3f} ms")
    print(f"{'p50':<10} | {st_before['p50']:>26.3f} ms | {st_after['p50']:>26.3f} ms")
    print(f"{'p95':<10} | {st_before['p95']:>26.3f} ms | {st_after['p95']:>26.3f} ms")
    print(f"{'p99':<10} | {st_before['p99']:>26.3f} ms | {st_after['p99']:>26.3f} ms")
    print(f"{'max':<10} | {st_before['max']:>26.3f} ms | {st_after['max']:>26.3f} ms")

    print("\n=== DATA LOSS VERIFICATION (Direct SQLite Query) ===")
    print(f"Request returned HTTP status: {res_verify.status_code} in {req_latency_ms:.3f} ms")
    print(f"Verified Session ID: {verify_session_id}")
    print(f"Session Row in 'sessions' table: {session_row}")
    print(f"Message Rows in 'messages' table (count={len(message_rows)}):")
    for r in message_rows:
        safe_content = r[3][:60].encode("ascii", "replace").decode("ascii")
        print(f"  - role: {r[2]}, content: {safe_content}...")

    assert session_row is not None, "Data loss detected: session row missing in SQLite"
    assert len(message_rows) == 2, f"Data loss detected: expected 2 messages, got {len(message_rows)}"
    print("\nData loss check result: PASS (all background writes persisted successfully)")


if __name__ == "__main__":
    asyncio.run(main())
