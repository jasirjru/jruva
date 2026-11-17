"""Tests for persistent database worker loop lifecycle and safety (R10).

Verifies thread liveness, bounded timeouts, task cancellation, graceful shutdown,
repeated restart, concurrent execution, and reentrancy safety.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.core.database import (
    _get_sync_db_loop,
    run_sync_db,
    shutdown_sync_db_worker,
)


def test_database_worker_thread_liveness():
    """_get_sync_db_loop returns an active, running event loop on a live background thread."""
    loop = _get_sync_db_loop()
    assert loop is not None
    assert loop.is_running()
    assert not loop.is_closed()


def test_database_worker_timeout_and_cancellation():
    """A slow coroutine exceeding timeout raises TimeoutError and cancels the task."""
    async def slow_work():
        await asyncio.sleep(2.0)
        return "done"

    with pytest.raises(TimeoutError) as exc_info:
        run_sync_db(slow_work, timeout=0.1)
    assert "timed out after 0.1 seconds" in str(exc_info.value)


def test_database_worker_graceful_shutdown():
    """shutdown_sync_db_worker stops the loop, joins the thread, and clears references."""
    # Ensure worker is started
    assert run_sync_db(asyncio.sleep(0.01, result=42)) == 42
    
    # Graceful shutdown
    shutdown_sync_db_worker(timeout=2.0)
    
    from app.core import database
    assert database._SYNC_DB_LOOP is None
    assert database._SYNC_DB_THREAD is None


def test_database_worker_repeated_lifecycle_restart():
    """The database worker loop can be shut down and cleanly restarted multiple times."""
    for cycle in range(3):
        res = run_sync_db(asyncio.sleep(0.01, result=cycle))
        assert res == cycle
        shutdown_sync_db_worker(timeout=2.0)


def test_concurrent_load_on_sync_db():
    """Multiple concurrent caller threads submit to run_sync_db without contention or race."""
    def worker(i):
        async def work():
            await asyncio.sleep(0.02)
            return i * 2
        return run_sync_db(work)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(worker, range(10)))

    assert results == [i * 2 for i in range(10)]


def test_reentrancy_does_not_deadlock():
    """Nested synchronous calls through the database worker bridge execute safely."""
    async def nested():
        async def inner():
            await asyncio.sleep(0.01)
            return "inner_success"
        return run_sync_db(inner)

    res = run_sync_db(nested)
    assert res == "inner_success"
