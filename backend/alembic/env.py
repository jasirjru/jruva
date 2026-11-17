import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set model metadata for autogenerate support
target_metadata = Base.metadata


def get_target_url() -> str:
    """Resolve database URL from config settings or environment."""
    backend_env = os.getenv("DATABASE_BACKEND", "").lower().strip()
    db_url_env = (os.getenv("DATABASE_URL") or os.getenv("STAGING_DATABASE_URL") or "").strip()

    # If explicitly postgres or URL has postgres scheme, target PostgreSQL
    if backend_env == "postgres" or (db_url_env and "postgres" in db_url_env) or settings.DATABASE_BACKEND.lower() == "postgres":
        url = db_url_env or settings.effective_database_url
        if url:
            if url.startswith("postgres://"):
                url = "postgresql+asyncpg://" + url[len("postgres://"):]
            elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
                url = "postgresql+asyncpg://" + url[len("postgresql://"):]
            return url
    if settings.effective_database_url and "postgres" in settings.effective_database_url:
        return settings.effective_database_url
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chatbot.db")
    return f"sqlite+aiosqlite:///{db_path}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_target_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create async engine and associate a connection with the context."""
    url = get_target_url()
    connect_args = {}
    if "postgres" in url:
        import uuid
        connect_args = {
            "statement_cache_size": settings.DB_STATEMENT_CACHE_SIZE,
            "prepared_statement_cache_size": settings.DB_STATEMENT_CACHE_SIZE,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex}__",
        }
    elif "sqlite" in url:
        connect_args = {"check_same_thread": False}

    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
