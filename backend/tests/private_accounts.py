"""Explicit synthetic accounts for authenticated private-data test contracts."""
import hashlib

from sqlalchemy import text

from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.services.auth_service import auth_service


def private_account_headers(user_id: str):
    digest = hashlib.sha256(user_id.encode()).hexdigest()
    if is_postgres_configured():
        async def _insert_pg():
            async with get_async_session() as session, session.begin():
                await session.execute(
                    text(
                        "INSERT INTO users (id, username, email, full_name, role, password_hash, salt, google_sub, created_at, last_login) "
                        "VALUES (:id, :u, :e, :fn, 'student', 'unused-synthetic', 'unused-synthetic', :gs, NOW(), NOW()) "
                        "ON CONFLICT (id) DO NOTHING"
                    ),
                    {
                        "id": user_id,
                        "u": f"fixture-{digest[:16]}",
                        "e": f"{digest[:16]}@example.invalid",
                        "fn": "Synthetic account",
                        "gs": f"fixture-sub-{digest[:16]}",
                    },
                )
        run_sync_db(_insert_pg())
    else:
        with auth_service._get_connection() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
            fields = ["id", "username", "email", "full_name", "role", "google_sub"]
            vals = [user_id, "fixture-" + digest[:16], digest[:16] + "@example.invalid", "Synthetic account", "student", "fixture-sub-" + digest[:16]]
            if "hashed_password" in columns:
                fields.append("hashed_password")
                vals.append("unused-synthetic")
            if "password_hash" in columns:
                fields.append("password_hash")
                vals.append("unused-synthetic")
            if "salt" in columns:
                fields.append("salt")
                vals.append("unused-synthetic")
            placeholders = ", ".join(["?"] * len(fields))
            col_names = ", ".join(fields)
            conn.execute(
                f"INSERT OR IGNORE INTO users ({col_names}) VALUES ({placeholders})",
                tuple(vals),
            )
            conn.commit()
    return {"Authorization": "Bearer " + auth_service.create_jwt_token({"sub": user_id}), "X-Client-Id": user_id}

