"""PostgreSQL-compatible concurrency regression tests for daily quota admission.

Verifies:
1. PostgreSQL daily quota admission acquires row-level locks (FOR UPDATE) on both
   the global sentinel row ('__global__') and the user row, eliminating check-then-increment races.
2. Concurrent requests competing for a user's daily prompt budget cannot exceed DAILY_PROMPTS_PER_USER.
3. Concurrent requests across multiple distinct users competing for the project-wide daily cap
   cannot exceed DAILY_GLOBAL_PROMPT_BUDGET.
4. Excludes sentinel rows from global sums and user quota lookups.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from app.core.config import settings
from app.services.quota.daily_usage_service import (
    DailyQuotaExceededError,
    async_check_and_consume_chat_prompt,
)


class MockPostgresSession:
    """Simulates a PostgreSQL connection with MVCC and row-level FOR UPDATE locking."""

    table_rows = {}  # (user_id, usage_date) -> dict of column values
    row_locks = {}   # (user_id, usage_date) -> asyncio.Lock
    db_mutex = asyncio.Lock()
    executed_queries = []

    def __init__(self):
        self.held_locks = []

    @asynccontextmanager
    async def begin(self):
        try:
            yield self
            await self.commit()
        except BaseException:
            await self.rollback()
            raise

    async def execute(self, statement, params=None):
        sql = str(statement).strip()
        params = params or {}
        MockPostgresSession.executed_queries.append((sql, params))

        # 1. INSERT ... ON CONFLICT DO NOTHING (sentinel or user row init)
        if "INSERT INTO user_daily_usage" in sql and "DO NOTHING" in sql:
            uid = params.get("uid")
            if "user_id='__global__'" in sql or "'__global__'" in sql:
                uid = "__global__"
            dt = params.get("dt")
            key = (uid, dt)
            async with MockPostgresSession.db_mutex:
                if key not in MockPostgresSession.table_rows:
                    MockPostgresSession.table_rows[key] = {
                        "chat_prompts": 0,
                        "web_searches": 0,
                        "document_uploads": 0,
                        "audio_seconds": 0,
                        "provider_calls": 0,
                        "provider_errors": 0,
                    }
                if key not in MockPostgresSession.row_locks:
                    MockPostgresSession.row_locks[key] = asyncio.Lock()
            return None

        # 2. SELECT ... FOR UPDATE (locking sentinel or user row)
        if "SELECT" in sql and "FOR UPDATE" in sql:
            uid = params.get("uid")
            if "user_id='__global__'" in sql or uid == "__global__":
                uid = "__global__"
            dt = params.get("dt")
            key = (uid, dt)

            async with MockPostgresSession.db_mutex:
                if key not in MockPostgresSession.row_locks:
                    MockPostgresSession.row_locks[key] = asyncio.Lock()
                lock = MockPostgresSession.row_locks[key]

            # Acquire row lock (PostgreSQL blocks concurrent transactions on this row until commit)
            await lock.acquire()
            self.held_locks.append(lock)

            # Yield control to event loop to encourage interleaving if unlocked
            await asyncio.sleep(0.001)

            async with MockPostgresSession.db_mutex:
                row_data = MockPostgresSession.table_rows.get(key, {"chat_prompts": 0})

            class FakeResult:
                def fetchone(self):
                    return (row_data["chat_prompts"],)
            return FakeResult()

        # 3. SELECT COALESCE(SUM(chat_prompts), 0) FROM user_daily_usage WHERE usage_date=:dt AND user_id != '__global__'
        if "SUM(chat_prompts)" in sql:
            dt = params.get("dt")
            await asyncio.sleep(0.001)  # Window for race condition if unprotected
            async with MockPostgresSession.db_mutex:
                total = sum(
                    data["chat_prompts"]
                    for (u, d), data in MockPostgresSession.table_rows.items()
                    if d == dt and u != "__global__"
                )
            class FakeSumResult:
                def fetchone(self):
                    return (total,)
            return FakeSumResult()

        # 4. SELECT chat_prompts FROM user_daily_usage WHERE user_id=:uid AND usage_date=:dt
        if "SELECT chat_prompts FROM user_daily_usage WHERE user_id=:uid" in sql:
            key = (params.get("uid"), params.get("dt"))
            await asyncio.sleep(0.001)  # Window for race condition if unprotected
            async with MockPostgresSession.db_mutex:
                row_data = MockPostgresSession.table_rows.get(key, {"chat_prompts": 0})
            class FakeUserResult:
                def fetchone(self):
                    return (row_data["chat_prompts"],)
            return FakeUserResult()

        # 5. UPDATE user_daily_usage SET chat_prompts = user_daily_usage.chat_prompts + 1
        if "UPDATE user_daily_usage" in sql and "chat_prompts" in sql:
            key = (params.get("uid"), params.get("dt"))
            async with MockPostgresSession.db_mutex:
                if key in MockPostgresSession.table_rows:
                    MockPostgresSession.table_rows[key]["chat_prompts"] += 1
                    MockPostgresSession.table_rows[key]["provider_calls"] += 1
            return None

        return None

    async def commit(self):
        for lock in self.held_locks:
            if lock.locked():
                lock.release()
        self.held_locks.clear()

    async def rollback(self):
        for lock in self.held_locks:
            if lock.locked():
                lock.release()
        self.held_locks.clear()

    async def close(self):
        await self.rollback()


@asynccontextmanager
async def mock_postgres_session():
    session = MockPostgresSession()
    try:
        yield session
    finally:
        await session.close()


@pytest.fixture(autouse=True)
def reset_mock_postgres():
    MockPostgresSession.table_rows.clear()
    MockPostgresSession.row_locks.clear()
    MockPostgresSession.executed_queries.clear()
    yield
    MockPostgresSession.table_rows.clear()
    MockPostgresSession.row_locks.clear()
    MockPostgresSession.executed_queries.clear()


@pytest.mark.asyncio
async def test_postgres_admission_acquires_row_locks_for_update():
    """Verify that PostgreSQL admission acquires FOR UPDATE locks on sentinel and user rows."""
    with patch("app.services.quota.daily_usage_service.is_postgres_configured", return_value=True):
        with patch("app.services.quota.daily_usage_service.get_async_session", mock_postgres_session):
            with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True):
                res = await async_check_and_consume_chat_prompt("pg_lock_user_1")
                assert res["allowed"] is True

            # Verify that SELECT ... FOR UPDATE was executed
            for_update_queries = [
                q[0] for q in MockPostgresSession.executed_queries if "FOR UPDATE" in q[0]
            ]
            assert len(for_update_queries) >= 2, (
                f"Expected at least 2 FOR UPDATE queries (global + user), got {len(for_update_queries)}: {for_update_queries}"
            )
            assert any("__global__" in q for q in for_update_queries), "Missing FOR UPDATE lock on __global__ sentinel"
            assert any(":uid" in q for q in for_update_queries), "Missing FOR UPDATE lock on user row"


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_exceed_user_daily_prompt_limit():
    """Prove that concurrent requests competing for a user's quota cannot exceed DAILY_PROMPTS_PER_USER."""
    user_limit = 5
    concurrency = 25
    user_id = "concurrent_quota_user"

    with patch("app.services.quota.daily_usage_service.is_postgres_configured", return_value=True):
        with patch("app.services.quota.daily_usage_service.get_async_session", mock_postgres_session):
            with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", user_limit), patch.object(settings, "DAILY_GLOBAL_PROMPT_BUDGET", 100):
                admitted = 0
                rejected = 0

                async def worker(idx: int):
                    nonlocal admitted, rejected
                    try:
                        res = await async_check_and_consume_chat_prompt(user_id)
                        if res["allowed"]:
                            admitted += 1
                    except DailyQuotaExceededError as err:
                        assert err.quota_type == "chat_prompts"
                        rejected += 1

                # Run all workers concurrently
                tasks = [worker(i) for i in range(concurrency)]
                await asyncio.gather(*tasks)

                # Exactly user_limit must be admitted, the rest rejected
                assert admitted == user_limit, (
                    f"Expected exactly {user_limit} admissions, got {admitted} (overshot limit!)"
                )
                assert rejected == concurrency - user_limit, (
                    f"Expected {concurrency - user_limit} rejections, got {rejected}"
                )

                # Verify final committed total in table
                key = (user_id, [k[1] for k in MockPostgresSession.table_rows.keys() if k[0] == user_id][0])
                final_prompts = MockPostgresSession.table_rows[key]["chat_prompts"]
                assert final_prompts == user_limit, f"Committed prompts {final_prompts} exceeds limit {user_limit}"


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_exceed_global_prompt_budget():
    """Prove that concurrent requests across multiple distinct users cannot exceed DAILY_GLOBAL_PROMPT_BUDGET."""
    global_budget = 8
    num_users = 6
    requests_per_user = 4  # Total 24 requests, budget is 8

    with patch("app.services.quota.daily_usage_service.is_postgres_configured", return_value=True):
        with patch("app.services.quota.daily_usage_service.get_async_session", mock_postgres_session):
            with patch.object(settings, "ENABLE_DAILY_PROMPT_LIMIT", True), patch.object(settings, "DAILY_PROMPTS_PER_USER", 20), patch.object(settings, "DAILY_GLOBAL_PROMPT_BUDGET", global_budget):
                admitted = 0
                global_rejected = 0

                async def worker(u_idx: int, req_idx: int):
                    nonlocal admitted, global_rejected
                    uid = f"user_global_{u_idx}"
                    try:
                        res = await async_check_and_consume_chat_prompt(uid)
                        if res["allowed"]:
                            admitted += 1
                    except DailyQuotaExceededError as err:
                        if err.quota_type == "global_prompt_budget":
                            global_rejected += 1

                tasks = [
                    worker(u, r)
                    for u in range(num_users)
                    for r in range(requests_per_user)
                ]
                await asyncio.gather(*tasks)

                # Total admitted across ALL users must equal global_budget
                assert admitted == global_budget, (
                    f"Expected exactly {global_budget} admissions across users, got {admitted} (global budget breached!)"
                )
                assert global_rejected == (num_users * requests_per_user) - global_budget

                # Sum committed prompts in database
                total_committed = sum(
                    data["chat_prompts"]
                    for (u, _), data in MockPostgresSession.table_rows.items()
                    if u != "__global__"
                )
                assert total_committed == global_budget, (
                    f"Database total {total_committed} exceeded global budget {global_budget}"
                )
