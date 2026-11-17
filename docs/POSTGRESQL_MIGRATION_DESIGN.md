# PostgreSQL Migration Architecture & Implementation Blueprint

**Document Version:** 2.0.0  
**Classification:** Core Database Architecture & Persistence  
**Date:** 2026-10-08  
**Status:** Implementation Completed on Branch `overnight-next-level-2026-10-08` (Pending Production Cutover)

---

## 1. Executive Summary & Migration Context

Prior to this implementation, JRUVA operated with single-instance SQLite databases (`backend/data/chatbot.db`, `backend/data/workspace.db`). On Google Cloud Run, this prevented horizontal autoscaling (`max_instances=1`) due to disk ephemerality, single-writer serialization locks, and risk of data loss upon container shutdown.

On development branch `overnight-next-level-2026-10-08`, Phase 1 of the PostgreSQL migration has been fully implemented:
1. **Async SQLAlchemy 2.0 Core (`backend/app/core/database.py`)** with `asyncpg` / `psycopg` drivers.
2. **Declarative ORM Layer (`backend/app/models/__init__.py`)** covering all 11 audited entities with cross-dialect primary keys.
3. **Async Alembic Migrations (`backend/alembic/versions/001_initial_schema.py`)** for reproducible, version-controlled schema provisioning.
4. **Dual-Persistence Service Adapters** (`auth_service`, `session_store`, `memory_service`, `conversation_log_service`, `workspace_store`) maintaining 100% backward compatibility with SQLite for development/testing while routing to PostgreSQL in production.
5. **Direct Persistence Path**: Bypasses the SQLite in-memory queue for critical message persistence when PostgreSQL is configured.
6. **Zero Silent Fallback**: If `DATABASE_BACKEND=postgres` and PostgreSQL fails, the system fails closed with controlled errors rather than silently creating an ephemeral split-brain SQLite database.
7. **Safe Standalone Migration Tool (`scripts/migrate_sqlite_to_postgres.py`)**: Dry-run by default, explicit `--execute` required, transaction-safe, row-reconciled, and masked credentials.
8. **Storage Abstraction (`backend/app/services/storage/`)**: Decouples document metadata from file blob storage.

---

## 2. Implemented Database Schema (11 Core Entities)

All tables use standard UTC timestamps (`DateTime(timezone=True)`) and structured JSON storage where appropriate.

```mermaid
erDiagram
    users ||--o{ user_memories : "has"
    users ||--o{ projects : "owns"
    users ||--o{ library_files : "owns"
    users ||--o{ image_records : "owns"
    users ||--o{ scheduled_tasks : "owns"
    users ||--o{ user_integrations : "configures"
    sessions ||--o{ messages : "contains"
    scheduled_tasks ||--o{ task_executions : "runs"

    users {
        string id PK
        string email UK
        string name
        string picture
        string provider
        timestamptz created_at
        timestamptz last_login
    }

    sessions {
        bigint id PK
        string user_id "FK-free (UUID, anon_*, NULL)"
        string title
        timestamptz created_at
        timestamptz updated_at
    }

    messages {
        bigint id PK
        bigint session_id FK
        string role
        text content
        string provider
        string model
        string tier
        jsonb citations
        timestamptz created_at
    }

    conversation_logs {
        bigint id PK
        string session_id
        string prompt
        string response
        string provider
        string model
        string task_type
        jsonb retrieved_chunks
        jsonb shadow_evaluations
        string latency_breakdown
        timestamptz created_at
    }

    user_memories {
        string id PK
        string user_id "Supports registered & anon"
        text content
        string source
        float confidence
        timestamptz created_at
    }

    projects {
        string id PK
        string user_id
        string name
        string description
        string instructions
        jsonb active_plugins
        timestamptz created_at
        timestamptz updated_at
    }

    library_files {
        string id PK
        string user_id
        string project_id
        string filename
        string file_type
        bigint file_size
        text storage_path
        int chunk_count
        string status
        timestamptz created_at
        timestamptz updated_at
    }

    image_records {
        string id PK
        string user_id
        string filename
        text storage_path
        string mime_type
        string resolution
        timestamptz created_at
    }

    scheduled_tasks {
        string id PK
        string user_id
        string name
        text prompt
        string cron_expression
        string timezone
        string status
        timestamptz last_run
        timestamptz created_at
        timestamptz updated_at
    }

    task_executions {
        string id PK
        string task_id "Nullable for legacy orphan safety"
        string status
        text output
        text error_message
        timestamptz executed_at
    }

    user_integrations {
        string id PK
        string user_id
        string provider
        text encrypted_credentials
        boolean is_active
        timestamptz updated_at
    }
```

### Critical Audit Constraints Addressed

1. **Flexible Session User Identity**:
   `sessions.user_id` supports registered UUID user IDs, unauthenticated guest IDs (`anon_*`), and legacy `NULL` users without foreign key constraint violations.
2. **Anonymous Memories**:
   `user_memories.user_id` supports both registered and anonymous user profiles.
3. **Legacy Execution Orphans**:
   `task_executions.task_id` is nullable and foreign key constraints allow legacy orphaned historical rows without breaking schema migrations.
4. **Cross-Dialect Autoincrement Primary Keys**:
   `Session.id` and `Message.id` utilize `BigInteger().with_variant(Integer, "sqlite")` to ensure `BIGSERIAL` on PostgreSQL and native autoincrement `INTEGER PRIMARY KEY` on SQLite.

---

## 3. Serverless Connection Pooling & Supavisor Compatibility

Cloud Run instances spin up and down dynamically. Using direct connection mode (port 5432) can exhaust PostgreSQL connection limits.

### Configuration Rules
- **Supabase / PgBouncer Transaction Mode (Port 6543)**:
  - Prepared statements across transactions cause `prepared statement "..." already exists` errors in transaction-mode poolers.
  - Solution implemented in `backend/app/core/database.py`:
    ```python
    connect_args["statement_cache_size"] = 0
    connect_args["prepared_statement_cache_size"] = 0
    ```
- **Connection Pool Parameters**:
  - `DB_POOL_SIZE = 5` (moderate per-container footprint for Cloud Run).
  - `DB_MAX_OVERFLOW = 5`.
  - `DB_POOL_TIMEOUT = 30.0` seconds.
  - `DB_POOL_RECYCLE = 300` seconds (avoids stale connections dropped by cloud proxies).
  - `DB_POOL_PRE_PING = True` (executes `SELECT 1` heartbeat check before checking out a connection).

---

## 4. Standalone Migration Utility (`scripts/migrate_sqlite_to_postgres.py`)

The migration script safely transfers data from SQLite to PostgreSQL without automated or accidental triggers.

### Key Safeguards
- **Manual Execution Only**: Script will not execute during application boot or CI runs.
- **Dry-Run by Default**: Unless `--execute` is explicitly provided, the script runs in read-only analysis mode.
- **Secret Masking**: Passwords in `DATABASE_URL` are redacted in all logs.
- **Batch Processing & Foreign Key Ordering**: Inserts data in dependency order (`users` -> `sessions` -> `messages`, etc.) using batches of 500 rows.
- **Verification Matrix**: Validates pre- and post-migration row counts, JSON structure in `retrieved_chunks`, ISO timestamp formats, and reports orphaned records.

```bash
# Dry run verification
python scripts/migrate_sqlite_to_postgres.py \
    --sqlite backend/data/chatbot.db \
    --postgres "postgresql+asyncpg://postgres:SECRET@db.supabase.co:6543/postgres"

# Live execution
python scripts/migrate_sqlite_to_postgres.py \
    --sqlite backend/data/chatbot.db \
    --postgres "postgresql+asyncpg://postgres:SECRET@db.supabase.co:6543/postgres" \
    --execute
```

---

## 5. Remaining Blocker for Horizontal Scaling (`max_instances > 1`)

While relational persistence is now ready for PostgreSQL, **uploaded document files currently reside on local container disk**:
- `backend/app/services/workspace_store.py` and `backend/app/api/v1/endpoints/library.py` store files locally via `storage_service`.
- In Cloud Run, container filesystems are ephemeral and not shared between instances.
- If instance A processes an upload, instance B cannot read the physical file for re-indexing or download.

**Resolution Required Before `max_instances > 1`**:
1. Implement `CloudObjectStorageBackend` using Google Cloud Storage (GCS) or AWS S3.
2. Update `STORAGE_BACKEND=gcs` and provide `GCS_BUCKET_NAME`.
3. Once blob storage is unified, increase Cloud Run `max_instances`.
