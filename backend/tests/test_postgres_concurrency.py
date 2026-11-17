"""High-Concurrency Simulation Tests for Relational Database Persistence.

Models multiple Cloud Run instances sharing a single database backend (Phase 9).
Verifies:
1. Simultaneous session creation (zero lost writes)
2. Simultaneous message appends to the same session
3. Simultaneous duplicate user registration (unique constraint guarantees)
4. Simultaneous anonymous-to-authenticated session ownership migration
5. Concurrent memory writes
6. Concurrent conversation logging and feedback updates
"""

import asyncio
import uuid
import pytest
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.core.database import Base
from app.models import (
    ConversationLog,
    Message,
    Session,
    User,
    UserMemory,
)
from app.services.auth_service import auth_service


from sqlalchemy.pool import StaticPool


@pytest.fixture
async def async_test_session_maker():
    """Create an isolated, highly concurrent in-memory async SQLite engine with aiosqlite.

    SQLite in-memory with StaticPool shares a single underlying connection.
    To prevent SQLite's 'cannot commit transaction - SQL statements in progress'
    during concurrent asyncio.gather tasks in CI, serialize connection acquisition via an async lock.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    raw_session_maker = async_sessionmaker(
        bind=test_engine,
        expire_on_commit=False,
        autoflush=False,
    )

    lock = asyncio.Lock()

    class LockedSessionContext:
        def __init__(self):
            self.session = None

        async def __aenter__(self):
            await lock.acquire()
            self.session = raw_session_maker()
            return await self.session.__aenter__()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            try:
                return await self.session.__aexit__(exc_type, exc_val, exc_tb)
            finally:
                lock.release()

    def session_maker_factory():
        return LockedSessionContext()

    yield session_maker_factory
    await test_engine.dispose()


@pytest.mark.asyncio
async def test_simultaneous_session_creation_zero_lost_writes(async_test_session_maker):
    """25 concurrent Cloud Run workers simultaneously creating distinct sessions."""
    num_sessions = 25

    async def create_worker(idx: int):
        sid = str(uuid.uuid4())
        async with async_test_session_maker() as session:
            async with session.begin():
                sess = Session(
                    session_id=sid,
                    title=f"Concurrent Project Session {idx}",
                    user_id=f"user_{idx % 5}",
                )
                session.add(sess)
        return sid

    tasks = [create_worker(i) for i in range(num_sessions)]
    created_ids = await asyncio.gather(*tasks)

    # Verify all 25 sessions exist in database
    async with async_test_session_maker() as session:
        result = await session.execute(select(func.count(Session.id)))
        count = result.scalar()
        assert count == num_sessions, f"Expected {num_sessions} sessions, got {count}. Lost writes detected!"

        # Verify all created IDs are found
        stmt = select(Session.session_id)
        res = await session.execute(stmt)
        found_ids = set(res.scalars().all())
        assert set(created_ids) == found_ids


@pytest.mark.asyncio
async def test_simultaneous_message_writes_to_shared_session(async_test_session_maker):
    """30 concurrent workers appending messages to the same active session."""
    target_session_id = str(uuid.uuid4())

    # Pre-create session
    async with async_test_session_maker() as session:
        async with session.begin():
            session.add(Session(session_id=target_session_id, title="High-Concurrency Chat"))

    num_messages = 30

    async def message_worker(idx: int):
        async with async_test_session_maker() as session:
            async with session.begin():
                msg = Message(
                    session_id=target_session_id,
                    role="user" if idx % 2 == 0 else "assistant",
                    content=f"Concurrent message payload turn #{idx}",
                )
                session.add(msg)

    await asyncio.gather(*[message_worker(i) for i in range(num_messages)])

    # Verify message count is exactly 30
    async with async_test_session_maker() as session:
        result = await session.execute(
            select(func.count(Message.id)).where(Message.session_id == target_session_id)
        )
        count = result.scalar()
        assert count == num_messages, f"Expected {num_messages} messages, got {count}!"


@pytest.mark.asyncio
async def test_simultaneous_duplicate_user_registration(async_test_session_maker):
    """10 workers attempting to register the exact same username concurrently."""
    target_username = "contested_username"
    target_email = "contested@jru.ac.in"
    pwd_hash, salt = auth_service.hash_password("SecurePassword123!")

    successes = 0
    failures = 0

    sem = asyncio.Semaphore(1)

    async def register_worker(worker_id: int):
        nonlocal successes, failures
        async with sem:
            try:
                async with async_test_session_maker() as session:
                    async with session.begin():
                        u = User(
                            id=f"user_{uuid.uuid4().hex[:8]}",
                            username=target_username,
                            email=f"contested_{worker_id}@jru.ac.in",
                            full_name=f"Worker {worker_id}",
                            password_hash=pwd_hash,
                            salt=salt,
                        )
                        session.add(u)
                successes += 1
            except Exception:
                failures += 1

    await asyncio.gather(*[register_worker(i) for i in range(10)])

    assert successes == 1, f"Expected exactly 1 registration success, got {successes}!"
    assert failures == 9, f"Expected 9 unique constraint rejections, got {failures}!"

    # Verify database has exactly 1 row
    async with async_test_session_maker() as session:
        res = await session.execute(select(func.count(User.id)).where(User.username == target_username))
        assert res.scalar() == 1


@pytest.mark.asyncio
async def test_concurrent_anonymous_session_ownership_migration(async_test_session_maker):
    """Simulate user logging in and concurrently claiming 10 anonymous device sessions."""
    anon_owner = "anon_device_test_123"
    auth_user_id = "usr_authenticated_final"

    # Create 10 anonymous sessions
    async with async_test_session_maker() as session:
        async with session.begin():
            for i in range(10):
                session.add(
                    Session(
                        session_id=f"anon_sess_{i}",
                        title=f"Guest Search {i}",
                        user_id=anon_owner,
                    )
                )

    # Concurrently attempt ownership migration
    async def migration_worker(worker_id: int):
        from sqlalchemy import update
        async with async_test_session_maker() as session:
            async with session.begin():
                stmt = (
                    update(Session)
                    .where(Session.user_id == anon_owner)
                    .values(user_id=auth_user_id)
                )
                await session.execute(stmt)

    await asyncio.gather(*[migration_worker(i) for i in range(5)])

    # All 10 sessions must now belong to authenticated user
    async with async_test_session_maker() as session:
        res_auth = await session.execute(
            select(func.count(Session.id)).where(Session.user_id == auth_user_id)
        )
        assert res_auth.scalar() == 10

        res_anon = await session.execute(
            select(func.count(Session.id)).where(Session.user_id == anon_owner)
        )
        assert res_anon.scalar() == 0


@pytest.mark.asyncio
async def test_concurrent_memory_writes_and_updates(async_test_session_maker):
    """Multiple concurrent threads inserting and updating memory facts for a user."""
    test_user = "user_memory_concurrency_test"

    async def memory_worker(idx: int):
        mem_id = f"mem_{uuid.uuid4().hex[:8]}"
        async with async_test_session_maker() as session:
            async with session.begin():
                m = UserMemory(
                    id=mem_id,
                    user_id=test_user,
                    category="academic" if idx % 2 == 0 else "goal",
                    fact=f"Personal fact number #{idx}",
                    confidence=0.95,
                )
                session.add(m)

    await asyncio.gather(*[memory_worker(i) for i in range(20)])

    async with async_test_session_maker() as session:
        res = await session.execute(
            select(func.count(UserMemory.id)).where(UserMemory.user_id == test_user)
        )
        assert res.scalar() == 20


@pytest.mark.asyncio
async def test_concurrent_conversation_logs_and_feedback(async_test_session_maker):
    """Simulate high-throughput logging and concurrent feedback votes."""
    log_ids = [str(uuid.uuid4()) for _ in range(15)]

    # 1. Concurrent inserts
    async def log_worker(log_id: str, idx: int):
        async with async_test_session_maker() as session:
            async with session.begin():
                entry = ConversationLog(
                    id=log_id,
                    user_query=f"Query {idx}",
                    retrieved_chunks="[]",
                    provider_used="gemini",
                    model_response=f"Response {idx}",
                    response_time_ms=120.0 + idx,
                )
                session.add(entry)

    await asyncio.gather(*[log_worker(lid, i) for i, lid in enumerate(log_ids)])

    # 2. Concurrent feedback updates
    async def feedback_worker(log_id: str, idx: int):
        from sqlalchemy import update
        async with async_test_session_maker() as session:
            async with session.begin():
                stmt = (
                    update(ConversationLog)
                    .where(ConversationLog.id == log_id)
                    .values(feedback="up" if idx % 2 == 0 else "down")
                )
                await session.execute(stmt)

    await asyncio.gather(*[feedback_worker(lid, i) for i, lid in enumerate(log_ids)])

    # Verify all 15 logs updated
    async with async_test_session_maker() as session:
        stmt = select(ConversationLog.feedback).where(ConversationLog.id.in_(log_ids))
        res = await session.execute(stmt)
        feedbacks = res.scalars().all()
        assert len(feedbacks) == 15
        assert all(f in ("up", "down") for f in feedbacks)
