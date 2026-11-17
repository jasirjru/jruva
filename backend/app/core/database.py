"""Central Database Engine, Session Factory, and Pool Configuration for JRUVA.

Supports dual persistence backends:
1. PostgreSQL (production deployment target via Supabase, Neon, Cloud SQL, or generic PostgreSQL)
   - Uses SQLAlchemy 2.0 AsyncEngine + asyncpg
   - Configures serverless transaction pooler compatibility (statement_cache_size = 0)
   - NullPool avoids reusing asyncpg connections across main/worker loops
   - Process-wide session admission is bounded by pool_size + max_overflow
   - pool_pre_ping=True for resilient checkout
2. SQLite (embedded zero-dependency local development and testing fallback)
   - Preserves compatibility with existing test suites and offline workflows
"""
import asyncio
import inspect
import logging
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Canonical declarative base for all ORM models
Base = declarative_base()

# Global engine and sessionmaker singletons
_ASYNC_ENGINE: AsyncEngine | None = None
_ASYNC_SESSION_MAKER: async_sessionmaker[AsyncSession] | None = None
_DB_SESSION_SLOTS: threading.BoundedSemaphore | None = None
_DB_SESSION_SLOT_LOCK = threading.Lock()


STAGING_PROJECT_REF = "stagingprojectref000"
PRODUCTION_PROJECT_REF = "prodprojectref000000"


def mask_database_url(url: str) -> str:
    """Mask credentials and sensitive user details in database URLs for safe logging."""
    if not url:
        return "<EMPTY>"
    try:
        import urllib.parse
        parsed = urllib.parse.urlsplit(url)
        username = parsed.username or ""
        masked_user = (username[:4] + "***") if len(username) > 4 else ("***" if username else "")
        netloc = ""
        if masked_user:
            netloc += f"{masked_user}:******@"
        if parsed.hostname:
            netloc += parsed.hostname
        if parsed.port:
            netloc += f":{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return "<REDACTED_URL>"


def validate_database_target(url: str, environment: str = "development", allow_test_loopback: bool = False) -> None:
    """Enforce fail-closed database target isolation across environments."""
    env = (environment or "development").lower().strip()
    url_str = (url or "").strip()
    url_lower = url_str.lower()

    if env == "production":
        if not url_str:
            raise RuntimeError(
                "CRITICAL DATABASE CONFIGURATION FATAL: DATABASE_URL is missing or empty in production environment."
            )
        if url_lower.startswith("sqlite"):
            raise RuntimeError(
                "CRITICAL DATABASE CONFIGURATION FATAL: Production environment cannot use SQLite. "
                "Must be PostgreSQL to prevent split-brain data loss."
            )
        if not url_lower.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
            raise RuntimeError(
                f"CRITICAL DATABASE CONFIGURATION FATAL: Production database URL must use a PostgreSQL scheme; got '{url_lower.split('://')[0]}'."
            )
        if STAGING_PROJECT_REF in url_lower:
            raise RuntimeError(
                f"CRITICAL DATABASE ISOLATION FATAL: Production environment cannot target staging project ref '{STAGING_PROJECT_REF}'!"
            )
        if "jruva-staging" in url_lower or ("staging" in url_lower and "production" not in url_lower):
            raise RuntimeError(
                "CRITICAL DATABASE ISOLATION FATAL: Production environment cannot target a staging database host!"
            )
        if "127.0.0.1" in url_lower or "localhost" in url_lower:
            is_test_runner = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI_TESTING"))
            if not is_test_runner or (not allow_test_loopback and os.environ.get("ALLOW_TEST_LOOPBACK") != "1"):
                raise RuntimeError(
                    "CRITICAL DATABASE ISOLATION FATAL: Production environment cannot target a local loopback database! "
                    "Loopback override is strictly prohibited outside of automated test harnesses."
                )

    elif env in ("staging", "development", "test"):
        if PRODUCTION_PROJECT_REF in url_lower:
            raise RuntimeError(
                f"CRITICAL DATABASE ISOLATION FATAL: Non-production environment '{env}' cannot target production project ref '{PRODUCTION_PROJECT_REF}'! "
                "Aborting to protect production data."
            )
        if "jruva-production" in url_lower or ("prod" in url_lower and "staging" not in url_lower and "development" not in url_lower and "test" not in url_lower):
            raise RuntimeError(
                f"CRITICAL DATABASE ISOLATION FATAL: Non-production environment '{env}' cannot target a production database host!"
            )
        if env == "staging" and "supabase" in url_lower and STAGING_PROJECT_REF not in url_lower:
            raise RuntimeError(
                f"CRITICAL DATABASE ISOLATION FATAL: Staging environment targeting Supabase must match staging project '{STAGING_PROJECT_REF}'."
            )


def is_postgres_configured() -> bool:
    """Return True if backend is configured to use PostgreSQL and a URL is provided."""
    backend = settings.DATABASE_BACKEND.lower().strip()
    return backend == "postgres" and bool(settings.effective_database_url)


def validate_production_database(cfg=None) -> None:
    """Validate database configuration during application initialization."""
    current_settings = cfg or settings
    env = current_settings.ENVIRONMENT.lower().strip()
    backend = current_settings.DATABASE_BACKEND.lower().strip()

    if env == "production":
        if backend != "postgres":
            raise RuntimeError(
                f"Production environment requires 'postgres' database backend; got '{backend}'. "
                "Refusing ephemeral or unsupported database backend to prevent data loss."
            )
        if not current_settings.effective_database_url:
            raise RuntimeError(
                "CRITICAL DATABASE CONFIGURATION FATAL: DATABASE_BACKEND is set to 'postgres' "
                "in production environment, but DATABASE_URL is missing or empty. "
                "Refusing silent fallback to ephemeral SQLite to prevent split-brain data loss."
            )
        validate_database_target(current_settings.effective_database_url, env)
    else:
        if current_settings.effective_database_url:
            validate_database_target(current_settings.effective_database_url, env)


def get_sqlite_fallback_path() -> str:
    """Resolve local SQLite chatbot.db path."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "chatbot.db",
    )


def create_engine_instance(custom_url: str | None = None) -> AsyncEngine:
    """Create and configure a new AsyncEngine instance based on settings."""
    backend = settings.DATABASE_BACKEND.lower().strip()
    db_url = custom_url or settings.effective_database_url

    if backend == "postgres":
        if not db_url:
            if settings.ENVIRONMENT.lower() == "production":
                raise RuntimeError(
                    "CRITICAL DATABASE CONFIGURATION FATAL: DATABASE_BACKEND is set to 'postgres' "
                    "in production environment, but DATABASE_URL is missing or empty. "
                    "Refusing silent fallback to ephemeral SQLite to prevent split-brain data loss."
                )
            logger.warning(
                "[DatabaseWarning] DATABASE_BACKEND is 'postgres' but no URL provided in non-production mode. "
                "Falling back to local SQLite engine."
            )
            backend = "sqlite"
        else:
            validate_database_target(db_url, settings.ENVIRONMENT)

    if backend == "postgres":
        logger.info(
            f"Configuring PostgreSQL NullPool on {mask_database_url(db_url)} "
            f"with session limit={settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW}..."
        )
        # Ensure asyncpg driver prefix
        if db_url.startswith("postgres://"):
            db_url = "postgresql+asyncpg://" + db_url[len("postgres://"):]
        elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
            db_url = "postgresql+asyncpg://" + db_url[len("postgresql://"):]

        import uuid

        connect_args: dict[str, Any] = {
            # Critical for Supabase Supavisor / PgBouncer transaction-mode poolers
            "statement_cache_size": settings.DB_STATEMENT_CACHE_SIZE,
            "prepared_statement_cache_size": settings.DB_STATEMENT_CACHE_SIZE,
            # Generate unique prepared statement names to prevent DuplicatePreparedStatementError
            # when connections are pooled and multiplexed in transaction pooling mode.
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex}__",
        }

        # Handle SSL requirement if specified in URL query
        if "ssl=require" in db_url.lower():
            # asyncpg interprets ssl parameter natively
            connect_args["ssl"] = "require"

        engine = create_async_engine(
            db_url,
            poolclass=NullPool,
            pool_pre_ping=settings.DB_POOL_PRE_PING,
            connect_args=connect_args,
            echo=settings.DEBUG,
        )
        return engine

    # Fallback / Local SQLite async engine
    sqlite_path = get_sqlite_fallback_path()
    sqlite_url = f"sqlite+aiosqlite:///{sqlite_path}"
    logger.info(f"Configuring SQLite async engine with file {sqlite_path}...")
    try:
        import aiosqlite  # noqa: F401
        engine = create_async_engine(
            sqlite_url,
            connect_args={"check_same_thread": False, "timeout": 15.0},
            echo=settings.DEBUG,
        )
        return engine
    except ImportError:
        # If aiosqlite is not installed, use an in-memory/fallback standard sqlite
        logger.warning("aiosqlite not installed; SQLite async engine using sqlite+pysqlite standard.")
        engine = create_async_engine(
            f"sqlite+pysqlite:///{sqlite_path}",
            connect_args={"check_same_thread": False},
        )
        return engine


def get_async_engine() -> AsyncEngine:
    """Retrieve or lazily initialize the singleton AsyncEngine."""
    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is None:
        _ASYNC_ENGINE = create_engine_instance()
    return _ASYNC_ENGINE


def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    """Retrieve or lazily initialize the singleton session factory."""
    global _ASYNC_SESSION_MAKER
    if _ASYNC_SESSION_MAKER is None:
        engine = get_async_engine()
        _ASYNC_SESSION_MAKER = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _ASYNC_SESSION_MAKER


async def _settle_database_cleanup(awaitable):
    """Observe physical cleanup despite repeated cancellation before releasing capacity."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager providing a transactional AsyncSession."""
    global _DB_SESSION_SLOTS
    slots = None
    if is_postgres_configured():
        with _DB_SESSION_SLOT_LOCK:
            if _DB_SESSION_SLOTS is None:
                capacity = settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW
                if capacity < 1:
                    raise RuntimeError("PostgreSQL session capacity must be positive")
                _DB_SESSION_SLOTS = threading.BoundedSemaphore(capacity)
            slots = _DB_SESSION_SLOTS
        deadline = asyncio.get_running_loop().time() + settings.DB_POOL_TIMEOUT
        # Never block the loop or leave an acquiring background thread on cancellation.
        while not slots.acquire(blocking=False):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("PostgreSQL session capacity exhausted")
            await asyncio.sleep(0.01)
    session = None
    try:
        session = get_async_session_maker()()
        try:
            yield session
            await session.commit()
        except BaseException:
            await _settle_database_cleanup(session.rollback())
            raise
    finally:
        try:
            if session is not None:
                await _settle_database_cleanup(session.close())
        finally:
            if slots is not None:
                slots.release()


async def check_database_health() -> dict[str, Any]:
    """Verify database connectivity and report connection telemetry."""
    backend = settings.DATABASE_BACKEND.lower().strip()
    try:
        def _ping():
            async def _check():
                async with get_async_session() as s:
                    result = await s.execute(text("SELECT 1"))
                    return result.scalar() == 1
            return run_sync_db(_check)

        connected = await asyncio.to_thread(_ping)
        return {
            "status": "healthy" if connected else "unhealthy",
            "backend": backend,
            "connected": connected,
        }
    except Exception as exc:
        logger.error(f"[DatabaseHealthCheck] Connection check failed for backend '{backend}': {exc}")
        return {
            "status": "unhealthy",
            "backend": backend,
            "connected": False,
            "error": str(exc),
        }

_SYNC_DB_LOOP: asyncio.AbstractEventLoop | None = None
_SYNC_DB_THREAD: threading.Thread | None = None
_SYNC_DB_LOCK = threading.Lock()
_SHUTTING_DOWN = False


def _get_sync_db_loop() -> asyncio.AbstractEventLoop:
    """Maintain a dedicated, persistent worker event loop for all synchronous DB operations.

    Prevents AsyncEngine connection pools from being reused across different closed loops.
    """
    global _SYNC_DB_LOOP, _SYNC_DB_THREAD
    with _SYNC_DB_LOCK:
        if _SHUTTING_DOWN:
            raise RuntimeError("Database worker loop is shutting down and cannot accept new tasks.")
        if (
            _SYNC_DB_LOOP is None
            or _SYNC_DB_LOOP.is_closed()
            or _SYNC_DB_THREAD is None
            or not _SYNC_DB_THREAD.is_alive()
        ):
            if _SYNC_DB_LOOP and not _SYNC_DB_LOOP.is_closed():
                try:
                    _SYNC_DB_LOOP.close()
                except Exception:
                    pass
            loop = asyncio.new_event_loop()
            ready_event = threading.Event()

            def run_loop():
                asyncio.set_event_loop(loop)
                ready_event.set()
                try:
                    loop.run_forever()
                finally:
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    except Exception:
                        pass

            thread = threading.Thread(target=run_loop, name="sync-db-worker", daemon=True)
            thread.start()
            ready_event.wait(timeout=5.0)
            _SYNC_DB_LOOP = loop
            _SYNC_DB_THREAD = thread
        return _SYNC_DB_LOOP


def shutdown_sync_db_worker(timeout: float = 5.0) -> None:
    """Gracefully shuts down the persistent database worker loop, disposes async engine, and joins worker thread."""
    global _SYNC_DB_LOOP, _SYNC_DB_THREAD, _ASYNC_ENGINE, _ASYNC_SESSION_MAKER, _SHUTTING_DOWN
    with _SYNC_DB_LOCK:
        _SHUTTING_DOWN = True
        loop = _SYNC_DB_LOOP
        thread = _SYNC_DB_THREAD
        engine = _ASYNC_ENGINE

        if engine is not None and loop and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(engine.dispose(), loop)
                fut.result(timeout=min(timeout, 3.0))
            except Exception as e:
                logger.warning(f"Error disposing async engine during shutdown: {e}")
        _ASYNC_ENGINE = None
        _ASYNC_SESSION_MAKER = None

        if loop and loop.is_running():
            try:
                def _cancel_tasks():
                    for task in asyncio.all_tasks(loop):
                        if not task.done():
                            task.cancel()
                    loop.stop()
                loop.call_soon_threadsafe(_cancel_tasks)
            except Exception:
                pass

        if thread and thread.is_alive():
            thread.join(timeout=timeout)

        if loop and not loop.is_closed():
            try:
                loop.close()
            except Exception:
                pass

        _SYNC_DB_LOOP = None
        _SYNC_DB_THREAD = None
        _SHUTTING_DOWN = False


def reset_engine_for_testing() -> None:
    """Reset singletons so test suites can configure custom engines / URLs."""
    global _ASYNC_ENGINE, _ASYNC_SESSION_MAKER, _DB_SESSION_SLOTS
    if _ASYNC_ENGINE is not None:
        try:
            loop = _SYNC_DB_LOOP
            if loop and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_ASYNC_ENGINE.dispose(), loop)
                future.result(timeout=5.0)
        except Exception:
            pass
    _ASYNC_ENGINE = None
    _ASYNC_SESSION_MAKER = None
    _DB_SESSION_SLOTS = None


def run_sync_db(coro_or_fn, *args, timeout: float = 30.0, **kwargs):
    """Execute an async database coroutine synchronously on the persistent database event loop."""
    coro = coro_or_fn(*args, **kwargs) if inspect.iscoroutinefunction(coro_or_fn) or callable(coro_or_fn) else coro_or_fn
    if threading.current_thread() == _SYNC_DB_THREAD:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=min(timeout, 10.0))

    db_loop = _get_sync_db_loop()
    future = asyncio.run_coroutine_threadsafe(coro, db_loop)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise TimeoutError(f"Database operation timed out after {timeout} seconds.")
    except Exception:
        raise
