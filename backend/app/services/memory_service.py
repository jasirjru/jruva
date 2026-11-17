"""Personal AI Memory Service for JRU AI (Jruva).

Manages user-specific long-term memory:
1. Academic profile (branch, semester, course, subjects)
2. Professional & study goals (GATE, campus placement, internships)
3. Learning & response preferences (code style, language, brevity)
4. Active project interests

Provides automatic extraction from conversations and full CRUD for the Memory Card.
Supports both PostgreSQL (production) and SQLite (development/fallback).
"""
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select, update

from app.core.database import get_async_session, is_postgres_configured, run_sync_db
from app.models import UserMemory

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chatbot.db")

PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
    r"(?i)\bsystem\s+prompt\s+(?:override|directive)\b",
    r"(?i)\byou\s+are\s+now\s+in\s+dan\s+mode\b",
    r"(?i)\b(?:print|leak|reveal)\s+(?:all\s+)?(?:passwords|api_keys|secrets)\b",
    r"(?i)\bapi[_-]?key\s*[:=]\s*\S+",
    r"(?i)\bsecret[_-]?key\s*[:=]\s*\S+",
    r"(?i)\bpassword\s*[:=]\s*\S+",
]

MAX_USER_MEMORIES = 40


def is_adversarial_memory_fact(fact: str) -> bool:
    """Detect prompt injection attempts or credential dumping in memory facts."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, fact):
            return True
    return False


class MemoryService:
    """Persistent repository for user memory cards and context personalization."""


    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        if not is_postgres_configured():
            self._init_sqlite_db()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_sqlite_db(self) -> None:
        """Initialize user_memories SQLite table with indexes for fast per-user recall."""
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        category TEXT NOT NULL DEFAULT 'general',
                        fact TEXT NOT NULL,
                        source TEXT DEFAULT 'auto',
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories (user_id, category)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_updated ON user_memories (updated_at DESC)
                """)
                conn.commit()
            logger.info(f"Initialized persistent user_memories table in {self.db_path}")
        except Exception as exc:
            logger.error(f"Failed to initialize user_memories database: {exc}")

    # ─── Memory Card CRUD (Async PostgreSQL implementations) ─────────────────

    async def async_get_user_memories(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieve all active memories for a given user from PostgreSQL."""
        if not user_id:
            return []
        try:
            async with get_async_session() as session:
                stmt = (
                    select(UserMemory)
                    .where(UserMemory.user_id == user_id)
                    .order_by(UserMemory.updated_at.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [
                    {
                        "id": row.id,
                        "userId": row.user_id,
                        "category": row.category,
                        "fact": row.fact,
                        "source": row.source,
                        "confidence": float(row.confidence or 1.0),
                        "createdAt": row.created_at.isoformat() if row.created_at else "",
                        "updatedAt": row.updated_at.isoformat() if row.updated_at else "",
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error(f"Postgres error fetching memories for user {user_id}: {exc}")
            raise

    async def async_add_memory(

        self,
        user_id: str,
        fact: str,
        category: str = "general",
        source: str = "manual",
        confidence: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Add or update a memory item for a user with duplicate prevention and injection defense."""
        clean_fact = fact.strip()
        if not user_id or not clean_fact:
            return None

        # Guard against prompt injection or secret dumping in memory
        if is_adversarial_memory_fact(clean_fact):
            logger.warning(f"[SecurityWarning] Blocked adversarial memory fact insertion for user {user_id}")
            return None

        # Bounded length
        clean_fact = clean_fact[:250].strip()
        clean_category = category.lower().strip() if category else "general"
        now = datetime.now(timezone.utc)

        try:
            async with get_async_session() as session:
                async with session.begin():
                    # Check total memory count bound
                    count_stmt = select(UserMemory).where(UserMemory.user_id == user_id)
                    count_res = await session.execute(count_stmt)
                    all_user_memories = count_res.scalars().all()

                    stmt = select(UserMemory).where(
                        UserMemory.user_id == user_id,
                        UserMemory.category == clean_category,
                    )
                    res = await session.execute(stmt)
                    existing_rows = res.scalars().all()


                    norm_new = re.sub(r"[^\w\s]", "", clean_fact.lower())
                    for row in existing_rows:
                        norm_existing = re.sub(r"[^\w\s]", "", (row.fact or "").lower())
                        if norm_new == norm_existing or norm_new in norm_existing or norm_existing in norm_new:
                            row.fact = clean_fact
                            row.confidence = confidence
                            row.updated_at = now
                            return {
                                "id": row.id,
                                "userId": user_id,
                                "category": clean_category,
                                "fact": clean_fact,
                                "source": source,
                                "confidence": confidence,
                                "createdAt": row.created_at.isoformat() if row.created_at else now.isoformat(),
                                "updatedAt": now.isoformat(),
                            }

                    mem_id = str(uuid.uuid4())
                    new_mem = UserMemory(
                        id=mem_id,
                        user_id=user_id,
                        category=clean_category,
                        fact=clean_fact,
                        source=source,
                        confidence=confidence,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(new_mem)

                return {
                    "id": mem_id,
                    "userId": user_id,
                    "category": clean_category,
                    "fact": clean_fact,
                    "source": source,
                    "confidence": confidence,
                    "createdAt": now.isoformat(),
                    "updatedAt": now.isoformat(),
                }
        except Exception as exc:
            logger.error(f"Postgres error adding memory for user {user_id}: {exc}")
            raise

    async def async_update_memory(
        self,
        memory_id: str,
        user_id: str,
        fact: str,
        category: Optional[str] = None,
    ) -> bool:
        """Update an existing memory item with ownership verification in PostgreSQL."""
        clean_fact = fact.strip()
        if not memory_id or not user_id or not clean_fact:
            return False

        now = datetime.now(timezone.utc)
        try:
            async with get_async_session() as session:
                async with session.begin():
                    values = {"fact": clean_fact, "updated_at": now}
                    if category:
                        values["category"] = category.lower().strip()

                    stmt = (
                        update(UserMemory)
                        .where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
                        .values(**values)
                    )
                    res = await session.execute(stmt)
                    return res.rowcount > 0
        except Exception as exc:
            logger.error(f"Postgres error updating memory {memory_id}: {exc}")
            raise

    async def async_delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific memory belonging to user in PostgreSQL."""
        if not memory_id or not user_id:
            return False
        try:
            async with get_async_session() as session:
                async with session.begin():
                    stmt = delete(UserMemory).where(
                        UserMemory.id == memory_id,
                        UserMemory.user_id == user_id,
                    )
                    res = await session.execute(stmt)
                    return res.rowcount > 0
        except Exception as exc:
            logger.error(f"Postgres error deleting memory {memory_id}: {exc}")
            raise

    async def async_clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a user in PostgreSQL."""
        if not user_id:
            return False
        try:
            async with get_async_session() as session:
                async with session.begin():
                    stmt = delete(UserMemory).where(UserMemory.user_id == user_id)
                    await session.execute(stmt)
                    return True
        except Exception as exc:
            logger.error(f"Postgres error clearing memories for user {user_id}: {exc}")
            raise

    # ─── Memory Card CRUD (Sync Facade with Backend Routing) ─────────────────

    def get_user_memories(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieve all active memories for a given user."""
        if not user_id:
            return []
        if is_postgres_configured():
            return run_sync_db(self.async_get_user_memories(user_id, limit))

        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, user_id, category, fact, source, confidence, created_at, updated_at
                    FROM user_memories
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": row["id"],
                        "userId": row["user_id"],
                        "category": row["category"],
                        "fact": row["fact"],
                        "source": row["source"],
                        "confidence": float(row["confidence"] or 1.0),
                        "createdAt": str(row["created_at"]),
                        "updatedAt": str(row["updated_at"]),
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error(f"SQLite error fetching memories for user {user_id}: {exc}")
            return []

    def add_memory(
        self,
        user_id: str,
        fact: str,
        category: str = "general",
        source: str = "manual",
        confidence: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Add or update a memory item for a user with duplicate prevention and injection defense."""
        clean_fact = fact.strip()
        if not user_id or not clean_fact:
            return None

        if is_adversarial_memory_fact(clean_fact):
            logger.warning(f"[SecurityWarning] Blocked adversarial memory fact insertion for user {user_id}")
            return None

        clean_fact = clean_fact[:250].strip()

        if is_postgres_configured():
            return run_sync_db(self.async_add_memory(user_id, clean_fact, category, source, confidence))

        clean_category = category.lower().strip() if category else "general"
        now = datetime.now(timezone.utc).isoformat()


        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, fact FROM user_memories
                    WHERE user_id = ? AND category = ?
                    """,
                    (user_id, clean_category),
                )
                existing_rows = cursor.fetchall()

                norm_new = re.sub(r"[^\w\s]", "", clean_fact.lower())
                for row in existing_rows:
                    norm_existing = re.sub(r"[^\w\s]", "", row["fact"].lower())
                    if norm_new == norm_existing or norm_new in norm_existing or norm_existing in norm_new:
                        conn.execute(
                            """
                            UPDATE user_memories
                            SET fact = ?, confidence = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (clean_fact, confidence, now, row["id"]),
                        )
                        conn.commit()
                        return {
                            "id": row["id"],
                            "userId": user_id,
                            "category": clean_category,
                            "fact": clean_fact,
                            "source": source,
                            "confidence": confidence,
                            "createdAt": now,
                            "updatedAt": now,
                        }

                mem_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO user_memories (id, user_id, category, fact, source, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mem_id, user_id, clean_category, clean_fact, source, confidence, now, now),
                )
                conn.commit()
                return {
                    "id": mem_id,
                    "userId": user_id,
                    "category": clean_category,
                    "fact": clean_fact,
                    "source": source,
                    "confidence": confidence,
                    "createdAt": now,
                    "updatedAt": now,
                }
        except Exception as exc:
            logger.error(f"SQLite error adding memory for user {user_id}: {exc}")
            return None

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        fact: str,
        category: Optional[str] = None,
    ) -> bool:
        """Update an existing memory item with ownership verification."""
        clean_fact = fact.strip()
        if not memory_id or not user_id or not clean_fact:
            return False

        if is_postgres_configured():
            return run_sync_db(self.async_update_memory(memory_id, user_id, clean_fact, category))

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_sqlite_connection() as conn:
                if category:
                    res = conn.execute(
                        """
                        UPDATE user_memories
                        SET fact = ?, category = ?, updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (clean_fact, category.lower().strip(), now, memory_id, user_id),
                    )
                else:
                    res = conn.execute(
                        """
                        UPDATE user_memories
                        SET fact = ?, updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (clean_fact, now, memory_id, user_id),
                    )
                conn.commit()
                return res.rowcount > 0
        except Exception as exc:
            logger.error(f"SQLite error updating memory {memory_id}: {exc}")
            return False

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific memory belonging to user."""
        if not memory_id or not user_id:
            return False

        if is_postgres_configured():
            return run_sync_db(self.async_delete_memory(memory_id, user_id))

        try:
            with self._get_sqlite_connection() as conn:
                res = conn.execute(
                    "DELETE FROM user_memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
                conn.commit()
                return res.rowcount > 0
        except Exception as exc:
            logger.error(f"SQLite error deleting memory {memory_id}: {exc}")
            return False

    def clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a user."""
        if not user_id:
            return False

        if is_postgres_configured():
            return run_sync_db(self.async_clear_user_memories(user_id))

        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
                conn.commit()
                return True
        except Exception as exc:
            logger.error(f"SQLite error clearing memories for user {user_id}: {exc}")
            return False

    # ─── Automatic Fact Extraction ──────────────────────────────────────────

    def extract_and_save_facts(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str = "",
    ) -> List[Dict[str, Any]]:
        """Intelligently detect personal facts from chat and store into Memory Card."""
        if not user_id or not user_message or len(user_message.strip()) < 5:
            return []

        text = user_message.strip()
        extracted: List[Dict[str, Any]] = []

        # 0. User Name & Identity (strictly validated candidates)
        from app.services.name_validator import extract_introduced_name
        introduced_name = extract_introduced_name(text)
        if introduced_name and introduced_name.lower() != "jruva":
            fact = f"Name: {introduced_name}"
            res = self.add_memory(user_id, fact, category="preference", source="auto", confidence=0.98)
            if res:
                extracted.append(res)

        # 1. Academic Patterns (Course, Department, Semester, Year)
        academic_patterns = [
            (
                r"\b(?:enrolled in|pursuing|studying in|studying|student of|branch is|course is)\s+([A-Za-z0-9.\s]*?\b(?:B\.?Tech|BCA|MCA|M\.?Tech|BBA|MBA|B\.?Pharm|D\.?Pharm|Diploma|Ph\.?D|Law|LLB|Engineering|Computer Science|Mining|Mechanical|Civil|Pharmacy)\b(?:\s+[A-Za-z0-9&]+)?)",
                "Course: {0}",
            ),
            (
                r"\b(?:in)\s+([A-Za-z0-9.\s]*?\b(?:B\.?Tech|BCA|MCA|M\.?Tech|BBA|MBA|B\.?Pharm|D\.?Pharm|Diploma|Ph\.?D|Law|LLB)\b(?:\s+[A-Za-z0-9&]+)?)",
                "Course: {0}",
            ),
            (
                r"\b(?:my branch is|my department is)\s+([A-Za-z0-9.\s&]+?)(?:[.,\n;]|\band\b|$)",
                "Department/Branch: {0}",
            ),
            (
                r"\b(?:I am in|I'm in)\s+(\d+(?:st|nd|rd|th)?\s+(?:year|semester|sem))\b",
                "Stage: {0}",
            ),
            (
                r"\b(?:my roll number is|my roll no is|my enrollment is)\s+([A-Za-z0-9-]+)\b",
                "Roll/Enrollment: {0}",
            ),
        ]

        for pat, tmpl in academic_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 2 and len(val) <= 60:
                    fact = tmpl.format(val.title() if len(val) > 4 else val)
                    res = self.add_memory(user_id, fact, category="academic", source="auto", confidence=0.95)
                    if res:
                        extracted.append(res)

        # 2. Career Goals & Exam Aspirations
        goal_patterns = [
            (
                r"\b(?:preparing for|aiming for|targeting)\s+([A-Za-z0-9\s]*?\b(?:GATE|CAT|placements?|campus placements?|interviews?|upsc|internships?|higher studies|masters?|off-campus)\b(?:\s+exam)?)",
                "Goal: Preparing for {0}",
            ),
            (
                r"\b(?:my goal is to|I want to become a|aspiring to be a|aim to be a)\s+([A-Za-z0-9\s]+?\b(?:software engineer|developer|data scientist|ai engineer|manager|researcher|pharmacist|lawyer|advocate)\b)",
                "Aspiration: Wants to become a {0}",
            ),
        ]

        for pat, tmpl in goal_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 3 and len(val) <= 60:
                    fact = tmpl.format(val.title())
                    res = self.add_memory(user_id, fact, category="goal", source="auto", confidence=0.9)
                    if res:
                        extracted.append(res)

        # 3. Learning & Response Preferences
        preference_patterns = [
            (
                r"\b(?:I prefer|prefer|always give me|explain in|please answer in)\s+([A-Za-z0-9\s]*?\b(?:python|c\+\+|java|javascript|hindi|english|hinglish|concise|bullet points|step by step)\b)",
                "Prefers: {0}",
            ),
            (
                r"\b(?:speak to me in|reply in)\s+(hindi|english|hinglish)\b",
                "Language preference: {0}",
            ),
        ]

        for pat, tmpl in preference_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 3 and len(val) <= 40:
                    fact = tmpl.format(val.title() if len(val) > 3 else val)
                    res = self.add_memory(user_id, fact, category="preference", source="auto", confidence=0.88)
                    if res:
                        extracted.append(res)

        # 4. Projects & Technical Interests
        interest_patterns = [
            (
                r"\b(?:building|working on|making a project on)\s+([A-Za-z0-9\s]+?\b(?:project|app|website|bot|system|model|tracker)\b)",
                "Building project: {0}",
            ),
            (
                r"\b(?:interested in|interest is in|passionate about)\s+([A-Za-z0-9\s]+?\b(?:machine learning|ai|robotics|web dev|cybersecurity|cloud|embedded)\b)",
                "Interested in: {0}",
            ),
        ]

        for pat, tmpl in interest_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 3 and len(val) <= 60:
                    fact = tmpl.format(val.title())
                    res = self.add_memory(user_id, fact, category="interest", source="auto", confidence=0.85)
                    if res:
                        extracted.append(res)

        return extracted

    def format_memory_prompt_block(self, user_id: str, max_items: int = 8) -> str:
        """Format active user memories into a compact, markdown system instruction block."""
        if not user_id:
            return ""

        memories = self.get_user_memories(user_id, limit=max_items)
        if not memories:
            return (
                "## User Memory Card Context:\n"
                "This user is signed in with their persistent account. No specific facts are saved in their Memory Card yet.\n"
                "- If the user asks whether their info is saved in memory or asks what you remember about them, confirm that they have a personal Memory Card (accessible via the 🧠 Memory Card button in their sidebar), and that any facts they tell you (or add to their card) are saved across sessions.\n"
                "- NEVER tell a signed-in user 'I don't retain personal data between conversations' or 'each session is independent'."
            )

        lines = [
            "## User Memory Card (Personal Context):",
            "The following verified facts are remembered about this user across sessions via their personal Memory Card.",
            "Use them naturally to contextualize and personalize your answers. If the user asks what you know about them or if their information is saved in your memory, confirm that these facts are saved in their Memory Card:",
        ]

        for mem in memories:
            cat = mem.get("category", "general").capitalize()
            fact = mem.get("fact", "").strip()
            if fact:
                lines.append(f"- [{cat}] {fact}")

        lines.append(
            "- Remind the user that they can view, add, or delete any saved facts anytime via the 🧠 Memory Card button in the sidebar."
        )
        return "\n".join(lines)


memory_service = MemoryService()
