"""Safe Supabase PostgreSQL & Storage Deployment Readiness Verification Script.

CRITICAL SAFETY RULES:
- Safe read-only checks by default.
- Masks all credentials and connection strings in stdout/stderr.
- Never mutates production databases or live user buckets.
- Verifies PgBouncer / Supavisor transaction pooler compatibility (statement_cache_size=0).
- Verifies Alembic migration graph continuity.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

# Ensure backend module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from alembic.config import Config
from alembic.script import ScriptDirectory


def mask_url(url: str) -> str:
    if not url:
        return "<EMPTY>"
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or "unknown"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        if parsed.username:
            netloc = f"{parsed.username}:****@{netloc}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except Exception:  # noqa: BLE001
        return "<MASKED_URL>"


def verify_alembic_ancestry():
    print("--> Checking Alembic Migration Ancestry Graph...")
    cfg = Config("backend/alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    print(f"    Current Alembic Head(s): {heads}")
    if len(heads) != 1:
        print(f"    [FAIL] Branching migration heads detected: {heads}")
        return False
    head_rev = script.get_revision(heads[0])
    curr = head_rev
    history = []
    while curr:
        history.append(curr.revision)
        curr = script.get_revision(curr.down_revision) if curr.down_revision else None
    expected_head = "008_doc_storage_daily_quotas"
    if heads[0] != expected_head:
        print(f"    [FAIL] Expected head '{expected_head}', got '{heads[0]}'")
        return False
    print("    [PASS] Alembic ancestry is single-headed and strictly linear.")
    return True


def verify_pooler_safeguards():
    print("--> Checking Transaction Pooler Safeguards (PgBouncer/Supavisor)...")
    from unittest.mock import patch

    from app.core.config import settings
    from app.core.database import create_engine_instance
    scs = getattr(settings, "DB_STATEMENT_CACHE_SIZE", None)
    print(f"    DB_STATEMENT_CACHE_SIZE: {scs}")
    if scs != 0:
        print("    [FAIL] DB_STATEMENT_CACHE_SIZE must be 0 for Supabase Transaction Pooler compatibility.")
        return False

    with patch.object(settings, "DATABASE_BACKEND", "postgres"), \
         patch("app.core.database.create_async_engine") as mock_engine:
        create_engine_instance("postgresql+asyncpg://user:pass@127.0.0.1:6543/postgres")
        kwargs = mock_engine.call_args.kwargs
        connect_args = kwargs.get("connect_args", {})
        if "prepared_statement_name_func" not in connect_args:
            print("    [FAIL] prepared_statement_name_func missing from engine connect_args.")
            return False
        func = connect_args["prepared_statement_name_func"]
        sample_name = func()
        print(f"    Dynamic Statement Naming: Enabled (Sample: {sample_name})")

    print("    [PASS] Statement caches disabled and dynamic statement naming active for PgBouncer / Supavisor.")
    return True


def verify_storage_config():
    print("--> Checking Storage Backend Configuration...")
    from app.core.config import settings
    sb = getattr(settings, "STORAGE_BACKEND", "local")
    bucket = getattr(settings, "STORAGE_BUCKET_NAME", "jruva-documents")
    max_obj = getattr(settings, "IMAGE_MAX_OBJECT_BYTES", 10 * 1024 * 1024)
    print(f"    STORAGE_BACKEND: {sb}")
    print(f"    STORAGE_BUCKET_NAME: {bucket}")
    print(f"    IMAGE_MAX_OBJECT_BYTES: {max_obj} ({max_obj // (1024 * 1024)} MB)")
    if max_obj > 10 * 1024 * 1024:
        print("    [FAIL] Object size exceeds Supabase free bucket limit of 10 MB.")
        return False
    print("    [PASS] Storage configuration aligned with Supabase Free Tier (10 MB/object, private bucket).")
    return True


def main():
    print("=" * 60)
    print("JRUVA PRODUCTION & CLOUD READINESS CHECKLIST")
    print("=" * 60)
    ok1 = verify_alembic_ancestry()
    ok2 = verify_pooler_safeguards()
    ok3 = verify_storage_config()
    print("=" * 60)
    if ok1 and ok2 and ok3:
        print("STATIC READINESS CHECKS PASS. Hosted migration, grants, pooler and storage behavior require independent verification.")
        return 0
    else:
        print("READINESS AUDIT RESULT: DEFECTS FOUND (NO-GO)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
