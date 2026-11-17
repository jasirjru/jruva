"""Authentication Service for User Registration, Authentication, and JWT Management.

Supports dual persistence:
1. PostgreSQL via SQLAlchemy 2.0 AsyncSession + asyncpg (Production target)
2. SQLite embedded WAL via sqlite3 standard library (Local dev / testing fallback)
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import func, select

from app.core.config import settings
from app.core.database import is_postgres_configured, get_async_session, run_sync_db
from app.models import User, UserMemory, Session
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chatbot.db")


def _resolve_jwt_secret() -> str:
    """Resolve JWT secret key with strict production safety enforcement."""
    secret = (os.environ.get("JWT_SECRET_KEY") or "").strip()
    env = os.environ.get("ENVIRONMENT", "development").strip().lower()

    if secret and len(secret) >= 16:
        return secret

    if env == "production":
        raise RuntimeError(
            "CRITICAL SECURITY FATAL: JWT_SECRET_KEY environment variable is missing or insecure in production mode. "
            "Set a strong secret key (minimum 32 characters) in production environment."
        )

    logger.warning(
        "[SecurityWarning] JWT_SECRET_KEY not set. Using ephemeral per-process secret. "
        "Tokens will invalidate on server restart."
    )
    ephemeral = secrets.token_hex(32)
    os.environ["JWT_SECRET_KEY"] = ephemeral
    return ephemeral


JWT_SECRET = _resolve_jwt_secret()
JWT_ALGORITHM = "HS256"


class GoogleAccountConflict(ValueError):
    """A verified Google subject cannot claim an existing email's account."""


class AccountDeletionPending(ValueError):
    """An account being erased cannot receive a new session."""


def _b64url_encode(data: bytes) -> str:
    """Encode bytes to standard URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    """Decode standard URL-safe base64 with auto-padding."""
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


class AuthService:
    """Manages secure password hashing, user persistence (Postgres/SQLite), and JWT tokens."""

    def __init__(self, db_path: str = DB_PATH, secret_key: str = JWT_SECRET):
        self.db_path = db_path
        self.secret_key = secret_key.encode("utf-8")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_db(self) -> None:
        """Create users table with unique constraints on email and username."""
        if is_postgres_configured():
            # In PostgreSQL, migrations handle schema; no-op during service instantiation
            return

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        role TEXT DEFAULT 'student',
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
                if "username" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
                if "full_name" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
                if "google_sub" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
                if "deletion_pending" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN deletion_pending INTEGER NOT NULL DEFAULT 0")
                if "password_hash" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
                if "salt" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")
                if "last_login" not in columns:
                    conn.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub)")
                conn.execute("DELETE FROM users WHERE email = 'demo@jru.ac.in';")
                conn.commit()
        except Exception as e:
            logger.error(f"[SilentFailureGuard] Failed to initialize users table: {e}", exc_info=True)
            telemetry_service.record_silent_failure("auth_service", str(e))

    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash placeholder token using PBKDF2-HMAC-SHA256 with cryptographically random salt."""
        if not salt:
            salt = secrets.token_hex(32)
        pwd_bytes = password.encode("utf-8")
        salt_bytes = salt.encode("utf-8")
        derived = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000)
        return derived.hex(), salt

    def create_jwt_token(self, payload: Dict[str, Any], expires_in_seconds: int = 7 * 24 * 3600) -> str:
        """Generate signed HS256 JWT token with standard header and claims."""
        header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
        claims = dict(payload)
        now = int(time.time())
        claims["iat"] = now
        claims["exp"] = now + expires_in_seconds

        encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        encoded_claims = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))

        signing_input = f"{encoded_header}.{encoded_claims}".encode("ascii")
        signature = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()
        encoded_signature = _b64url_encode(signature)

        return f"{encoded_header}.{encoded_claims}.{encoded_signature}"

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate signature and expiration of JWT token."""
        try:
            parts = token.strip().split(".")
            if len(parts) != 3:
                return None
            encoded_header, encoded_claims, encoded_signature = parts

            signing_input = f"{encoded_header}.{encoded_claims}".encode("ascii")
            expected_sig = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()
            actual_sig = _b64url_decode(encoded_signature)

            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            claims = json.loads(_b64url_decode(encoded_claims).decode("utf-8"))
            if claims.get("exp", 0) < int(time.time()):
                logger.debug("Token expired")
                return None

            return claims
        except Exception as e:
            logger.warning(f"[SilentFailureGuard] JWT verification failed: {e}", exc_info=True)
            telemetry_service.record_silent_failure("auth_service", str(e))
            return None

    # ─── Async PostgreSQL Implementation ───────────────────────────────────────

    async def async_get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by ID asynchronously from PostgreSQL."""
        if not user_id:
            return None
        async with get_async_session() as session:
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                return None
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "deletion_pending": bool(user.deletion_pending),
                "created_at": str(user.created_at) if user.created_at else "",
            }

    async def async_authenticate_or_register_google_user(
        self,
        email: str,
        full_name: str,
        google_sub: str = "",
        avatar_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Authenticate or register Google OAuth user asynchronously in PostgreSQL with transaction."""
        email_clean = email.strip().lower()
        full_name_clean = full_name.strip()
        username_clean = email_clean.split("@")[0]

        async with get_async_session() as session:
            if not google_sub:
                raise ValueError("Missing verified Google subject")
            stmt = select(User).where(User.google_sub == google_sub)
            result = await session.execute(stmt)
            existing_user = result.scalar_one_or_none()
            if not existing_user:
                email_owner = await session.scalar(select(User.id).where(User.email == email_clean))
                if email_owner:
                    raise GoogleAccountConflict("Google subject does not match the existing account")

            if existing_user:
                if existing_user.deletion_pending:
                    raise AccountDeletionPending("Account deletion is pending")
                existing_user.last_login = func.now()
                user_record = {
                    "id": existing_user.id,
                    "username": existing_user.username,
                    "email": existing_user.email,
                    "full_name": existing_user.full_name or full_name_clean,
                    "role": existing_user.role or "student",
                    "created_at": str(existing_user.created_at) if existing_user.created_at else "",
                }
                user_id = existing_user.id
            else:
                user_id = str(uuid.uuid4())
                stmt_uname = select(User.id).where(User.username == username_clean)
                existing_uname = (await session.execute(stmt_uname)).scalar_one_or_none()
                if existing_uname:
                    username_clean = f"{username_clean}_{secrets.token_hex(2)}"

                pwd_hash, salt = self.hash_password(secrets.token_urlsafe(32))
                role = "student"
                new_user = User(
                    id=user_id,
                    google_sub=google_sub,
                    username=username_clean,
                    email=email_clean,
                    full_name=full_name_clean,
                    role=role,
                    password_hash=pwd_hash,
                    salt=salt,
                )
                session.add(new_user)
                await session.flush()
                user_record = {
                    "id": user_id,
                    "username": username_clean,
                    "email": email_clean,
                    "full_name": full_name_clean,
                    "role": role,
                    "created_at": str(new_user.created_at) if new_user.created_at else "",
                }

        token = self.create_jwt_token({
            "sub": user_id,
            "username": user_record["username"],
            "email": email_clean,
            "full_name": user_record.get("full_name", full_name_clean),
            "role": user_record.get("role", "student"),
        })
        return {"user": user_record, "token": token}

    async def async_delete_user(self, user_id: str) -> bool:
        import asyncio
        from app.services.account_erasure import erase_account
        return await asyncio.to_thread(erase_account, user_id, self)

    # ─── Dual-Backend Synchronous Public Contracts ─────────────────────────────

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by ID. Supports both PostgreSQL and SQLite seamlessly."""
        if is_postgres_configured():
            return run_sync_db(self.async_get_user_by_id, user_id)

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, username, email, full_name, role, created_at, deletion_pending
                FROM users WHERE id = ?
            """, (user_id,))
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "full_name": row["full_name"],
            "role": row["role"],
            "deletion_pending": bool(row["deletion_pending"]),
            "created_at": str(row["created_at"]),
        }

    def authenticate_or_register_google_user(
        self,
        email: str,
        full_name: str,
        google_sub: str = "",
        avatar_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Authenticate by verified Google subject or register an unclaimed email."""
        if is_postgres_configured():
            return run_sync_db(
                self.async_authenticate_or_register_google_user,
                email, full_name, google_sub, avatar_url
            )

        email_clean = email.strip().lower()
        full_name_clean = full_name.strip()
        username_clean = email_clean.split("@")[0]

        with self._get_connection() as conn:
            if not google_sub:
                raise ValueError("Missing verified Google subject")
            cursor = conn.execute(
                "SELECT id, username, email, full_name, role, created_at, deletion_pending FROM users WHERE google_sub = ?",
                (google_sub,),
            )
            row = cursor.fetchone()

            if row:
                if row["deletion_pending"]:
                    raise AccountDeletionPending("Account deletion is pending")
                user_id = row["id"]
                user_record = {
                    "id": user_id,
                    "username": row["username"],
                    "email": row["email"],
                    "full_name": row["full_name"] or full_name_clean,
                    "role": row["role"] or "student",
                    "created_at": str(row["created_at"]),
                }
                try:
                    conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
                    conn.commit()
                except Exception as login_err:
                    logger.warning(
                        f"[SilentFailureGuard] Failed to update last_login for user {user_id}: {login_err}",
                        exc_info=True,
                    )
                    telemetry_service.record_silent_failure("auth_service", str(login_err))
            else:
                if conn.execute("SELECT id FROM users WHERE email = ?", (email_clean,)).fetchone():
                    raise GoogleAccountConflict("Google subject does not match the existing account")
                user_id = str(uuid.uuid4())
                cur_user = conn.execute("SELECT id FROM users WHERE username = ?", (username_clean,)).fetchone()
                if cur_user:
                    username_clean = f"{username_clean}_{secrets.token_hex(2)}"

                pwd_hash, salt = self.hash_password(secrets.token_urlsafe(32))
                role = "student"
                user_cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
                insert_fields = ["id", "username", "email", "full_name", "role", "google_sub"]
                insert_vals = [user_id, username_clean, email_clean, full_name_clean, role, google_sub]
                if "password_hash" in user_cols:
                    insert_fields.append("password_hash")
                    insert_vals.append(pwd_hash)
                if "salt" in user_cols:
                    insert_fields.append("salt")
                    insert_vals.append(salt)
                if "hashed_password" in user_cols:
                    insert_fields.append("hashed_password")
                    insert_vals.append(pwd_hash)
                placeholders = ", ".join(["?"] * len(insert_fields))
                col_clause = ", ".join(insert_fields)
                conn.execute(f"INSERT INTO users ({col_clause}) VALUES ({placeholders})", tuple(insert_vals))
                conn.commit()
                user_record = self.get_user_by_id(user_id) or {
                    "id": user_id,
                    "username": username_clean,
                    "email": email_clean,
                    "full_name": full_name_clean,
                    "role": role,
                }

        token = self.create_jwt_token({
            "sub": user_id,
            "username": user_record["username"],
            "email": email_clean,
            "full_name": user_record.get("full_name", full_name_clean),
            "role": user_record.get("role", "student"),
        })
        return {"user": user_record, "token": token}

    def delete_user(self, user_id: str) -> bool:
        """Revoke access and erase owned stores, retaining retry state on failure."""
        if not user_id:
            return False
        from app.services.account_erasure import erase_account
        return erase_account(user_id, self)



auth_service = AuthService()
