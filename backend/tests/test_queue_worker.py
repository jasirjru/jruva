"""Test latency of Synchronous vs asyncio.to_thread vs Dedicated Single Worker Queue."""
import sys
import os
import asyncio
import time
import uuid
import queue
import threading
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


# Dedicated Single Background Worker Queue
_persist_queue = queue.Queue()
_stop_event = threading.Event()

def _dedicated_worker():
    while not _stop_event.is_set():
        try:
            item = _persist_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        session_id, message, content = item
        try:
            title = session_store.generate_title_from_message(message)
            session_store.ensure_session(session_id, title)
            session_store.record_turn(session_id, "user", message)
            session_store.record_turn(session_id, "assistant", content)
        except Exception as e:
            pass
        finally:
            _persist_queue.task_done()

_worker_thread = threading.Thread(target=_dedicated_worker, daemon=True)
_worker_thread.start()


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
        await client.post("/api/v1/chat", json=warmup_payload, headers={"X-User-ID": "warmup-user"})

        N = 50

        # 1. Mode 1: Synchronous direct write (Original baseline)
        def sync_persist(session_id, message, content):
            title = chat_svc.store.generate_title_from_message(message)
            chat_svc.store.ensure_session(session_id, title)
            chat_svc.record_turn(session_id, "user", message)
            chat_svc.record_turn(session_id, "assistant", content)

        chat_svc._async_persist_cache_hit = sync_persist
        sync_latencies = []
        for i in range(N):
            uid = f"user-sync-{i}"
            payload = {
                "session_id": str(uuid.uuid4()),
                "message": "benchmark-cached-probe-prompt",
                "model": "openai/gpt-oss-120b",
                "tier": "S+++",
                "stream": False,
            }
            t0 = time.perf_counter()
            r = await client.post("/api/v1/chat", json=payload, headers={"X-User-ID": uid})
            sync_latencies.append((time.perf_counter() - t0) * 1000.0)
            assert r.status_code == 200

        # 2. Mode 2: asyncio.to_thread per request (The regressed fix)
        def to_thread_persist(session_id, message, content):
            def _sync():
                title = chat_svc.store.generate_title_from_message(message)
                chat_svc.store.ensure_session(session_id, title)
                chat_svc.record_turn(session_id, "user", message)
                chat_svc.record_turn(session_id, "assistant", content)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(asyncio.to_thread(_sync))
            except RuntimeError:
                _sync()

        chat_svc._async_persist_cache_hit = to_thread_persist
        to_thread_latencies = []
        for i in range(N):
            uid = f"user-thread-{i}"
            payload = {
                "session_id": str(uuid.uuid4()),
                "message": "benchmark-cached-probe-prompt",
                "model": "openai/gpt-oss-120b",
                "tier": "S+++",
                "stream": False,
            }
            t0 = time.perf_counter()
            r = await client.post("/api/v1/chat", json=payload, headers={"X-User-ID": uid})
            to_thread_latencies.append((time.perf_counter() - t0) * 1000.0)
            assert r.status_code == 200

        await asyncio.sleep(1.0)

        # 3. Mode 3: Dedicated Single Background Queue/Worker
        def queue_persist(session_id, message, content):
            _persist_queue.put_nowait((session_id, message, content))

        chat_svc._async_persist_cache_hit = queue_persist
        queue_latencies = []
        for i in range(N):
            uid = f"user-queue-{i}"
            payload = {
                "session_id": str(uuid.uuid4()),
                "message": "benchmark-cached-probe-prompt",
                "model": "openai/gpt-oss-120b",
                "tier": "S+++",
                "stream": False,
            }
            t0 = time.perf_counter()
            r = await client.post("/api/v1/chat", json=payload, headers={"X-User-ID": uid})
            queue_latencies.append((time.perf_counter() - t0) * 1000.0)
            assert r.status_code == 200

        # Wait for queue to drain
        _persist_queue.join()

    st_sync = compute_stats(sync_latencies)
    st_thread = compute_stats(to_thread_latencies)
    st_queue = compute_stats(queue_latencies)

    print("\n=== THREE-WAY COMPARISON (E2E N=50 requests) ===")
    print(f"{'Metric':<8} | {'1. Sync (Original)':<20} | {'2. asyncio.to_thread':<22} | {'3. Single Queue Worker':<22}")
    print("-" * 80)
    print(f"{'min':<8} | {st_sync['min']:>17.3f} ms | {st_thread['min']:>19.3f} ms | {st_queue['min']:>19.3f} ms")
    print(f"{'mean':<8} | {st_sync['mean']:>17.3f} ms | {st_thread['mean']:>19.3f} ms | {st_queue['mean']:>19.3f} ms")
    print(f"{'p50':<8} | {st_sync['p50']:>17.3f} ms | {st_thread['p50']:>19.3f} ms | {st_queue['p50']:>19.3f} ms")
    print(f"{'p95':<8} | {st_sync['p95']:>17.3f} ms | {st_thread['p95']:>19.3f} ms | {st_queue['p95']:>19.3f} ms")
    print(f"{'p99':<8} | {st_sync['p99']:>17.3f} ms | {st_thread['p99']:>19.3f} ms | {st_queue['p99']:>19.3f} ms")
    print(f"{'max':<8} | {st_sync['max']:>17.3f} ms | {st_thread['max']:>19.3f} ms | {st_queue['max']:>19.3f} ms")

    _stop_event.set()


if __name__ == "__main__":
    asyncio.run(main())
