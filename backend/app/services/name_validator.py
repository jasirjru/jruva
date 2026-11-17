"""Deterministic Name Validation, Candidate Extraction, and Intent Hygiene.

Provides robust, conservative parsing for user name introductions, preventing
catastrophic fastpath hijacking of location requests, state expressions,
and complex inquiries. Follows the principle: false negatives are far safer
than false-positive deterministic routing.
"""
import re
from typing import Optional, Set

# Prepositions and directional locatives (never part of a personal name)
PREPOSITIONS: Set[str] = {
    "at", "in", "from", "near", "to", "by", "on", "for", "with", "into", "towards",
    "around", "of", "under", "over", "between", "through", "across", "behind",
    "beside", "along", "about", "against", "before", "after", "during", "within",
}

# Question interrogatives (never part of a personal name)
QUESTION_WORDS: Set[str] = {
    "what", "where", "which", "who", "whom", "whose", "when", "why", "how",
}

# Action verbs, gerunds (-ing), and modal/state auxiliaries
VERBS_AND_ACTIONS: Set[str] = {
    "looking", "working", "trying", "searching", "studying", "getting", "going",
    "eating", "cooking", "coding", "building", "feeling", "thinking", "wondering",
    "planning", "doing", "making", "learning", "writing", "traveling", "travelling",
    "preparing", "having", "need", "want", "like", "have", "has", "is", "are", "was",
    "were", "visit", "find", "seeking", "reading", "watching", "listening", "asking",
    "seeking", "deploy", "deploying", "install", "installing", "debug", "debugging",
    "run", "running", "staying", "living", "moving", "taking", "coming", "leaving",
}

# Adjectives representing physical, emotional, or operational state
STATES_AND_ADJECTIVES: Set[str] = {
    "hungry", "tired", "sick", "happy", "sad", "ready", "fine", "good", "bored",
    "busy", "crazy", "interested", "afraid", "well", "alive", "excited", "nervous",
    "confused", "stuck", "curious", "lost", "sleepy", "glad", "sorry", "ill",
    "exhausted", "free", "available", "occupied", "upset", "angry", "satisfied",
}

# Common functional grammar tokens, determiners, pronouns, and time markers
GRAMMAR_AND_TIME_STOPWORDS: Set[str] = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
    "our", "their", "me", "you", "it", "and", "or", "but", "if", "because", "so",
    "tomorrow", "today", "yesterday", "tonight", "later", "now", "soon", "here",
    "there", "also", "just", "still", "already", "again", "very", "too", "quite",
    "some", "any", "all", "not", "no", "never", "always",
}

# Technical, institutional, and conversational nouns that are not personal names
NON_NAME_NOUNS: Set[str] = {
    "student", "engineer", "developer", "programmer", "coder", "project", "app",
    "website", "code", "python", "fastapi", "ai", "machine", "data", "bot",
    "assistant", "user", "admin", "guest", "person", "human", "guy", "girl",
    "man", "woman", "teacher", "professor", "doctor", "researcher", "intern",
    "home", "office", "airport", "station", "hotel", "restaurant", "food",
}

REJECT_NAME_TOKENS: Set[str] = (
    PREPOSITIONS
    | QUESTION_WORDS
    | VERBS_AND_ACTIONS
    | STATES_AND_ADJECTIVES
    | GRAMMAR_AND_TIME_STOPWORDS
    | NON_NAME_NOUNS
)

# Known external entities / universities to prevent institutional hijacking
EXTERNAL_INSTITUTION_TERMS: Set[str] = {
    "stanford", "harvard", "mit", "oxford", "cambridge", "berkeley", "cmu",
    "caltech", "princeton", "yale", "columbia", "germany", "canada", "usa",
    "uk", "australia", "apartment", "house", "lease", "rent", "flat", "landlord",
    "hostel", "pg", "room",
}

CODING_AND_QUERY_CONTEXT_TOKENS: Set[str] = {
    "python", "javascript", "js", "typescript", "ts", "sql", "script", "query",
    "code", "utc", "format", "function", "postgres", "postgresql", "mysql",
    "sqlite", "database", "syntax", "api", "variable", "datetime", "timestamp",
}


def validate_candidate_name(candidate: str) -> Optional[str]:
    """Validate whether an extracted string is a plausible human personal name.

    Returns the cleaned, title-cased name if valid, or None if rejected.
    Enforces strict structural and lexical constraints:
    - 1 to 3 words maximum
    - 2 to 30 characters
    - Must NOT contain prepositions, verbs, states, adjectives, or grammar stopwords
    """
    if not candidate:
        return None
    raw = candidate.strip()
    # Normalize punctuation and extra spaces
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", raw)
    tokens = [t.lower() for t in cleaned.split() if t]

    if not tokens or len(tokens) > 3:
        return None
    if len(raw) < 2 or len(raw) > 30:
        return None

    # Check against lexical rejection tables
    for t in tokens:
        if t in REJECT_NAME_TOKENS:
            return None
        # Reject single character words unless it's a known initial
        if len(t) == 1 and t not in {"j", "r", "u", "a", "k", "m", "s", "p", "v"}:
            return None

    return " ".join(t.capitalize() for t in tokens)


def extract_introduced_name(message: str) -> Optional[str]:
    """Extract and validate a user's name introduced in conversation.

    Supports:
    - Explicit: "My name is [Name]" or "Hello, my name is [Name]"
    - Anchored: "I am [Name]" or "I'm [Name]" or "Hi, I am [Name]"
    - Imperative: "Call me [Name]"

    Returns None for any ambiguous query or candidate containing locations, verbs,
    questions, or non-name tokens.
    """
    if not message:
        return None
    msg = message.strip()

    # Pattern A: Explicit "my name is..." / "my ame is..."
    m_exp = re.search(
        r"\b(?:my\s+(?:name|ame)\s+is)\s+([a-zA-Z\s.'-]+?)(?:[.,;!?]|\band\b|$)",
        msg,
        re.IGNORECASE,
    )
    if m_exp:
        res = validate_candidate_name(m_exp.group(1))
        if res:
            return res

    # Pattern B: Anchored "I am..." / "I'm..." / "im..." (strictly anchored to message start, optional greeting)
    m_iam = re.match(
        r"^(?:(?:hello|hi|hey|greetings)[,\s]+)?(?:i\s+am|i\'m|\bim\b)\s+([a-zA-Z\s.'-]+?)(?:[.,;!?]|\band\b|$)",
        msg,
        re.IGNORECASE,
    )
    if m_iam:
        res = validate_candidate_name(m_iam.group(1))
        if res:
            return res

    # Pattern C: Anchored "Call me..."
    m_call = re.match(
        r"^(?:please\s+)?call\s+me\s+([a-zA-Z\s.'-]+?)(?:[.,;!?]|\band\b|$)",
        msg,
        re.IGNORECASE,
    )
    if m_call:
        res = validate_candidate_name(m_call.group(1))
        if res:
            return res

    return None


def is_pure_date_query(raw_message: str, clean_message: str) -> bool:
    """Determine if a query is a standalone system clock/date inquiry.

    Returns False if the query contains programming, database, or script context
    so it cleanly falls through to the coding or reasoning model tier.
    """
    tokens = set(clean_message.split())
    if any(tok in CODING_AND_QUERY_CONTEXT_TOKENS for tok in tokens):
        return False

    pattern = (
        r"^(?:what\s+(?:is|s)\s+(?:the\s+)?(?:today\s*s?\s+date|current\s+date|date\s+today)"
        r"|today\s*s?\s+date|current\s+date|what\s+day\s+is\s+it\s+today)[?.\s]*$"
    )
    return bool(re.match(pattern, clean_message))


def is_jru_institutional_fee_query(raw: str, clean: str) -> bool:
    """Determine if a fee query specifically targets JRU and should trigger the fastpath.

    Rejects external universities, international admissions, or general rental deposits.
    """
    clean_tokens = set(clean.split())
    if any(ext in clean_tokens or ext in clean for ext in EXTERNAL_INSTITUTION_TERMS):
        return False

    is_jru_query = "jru" in clean or "jharkhand" in clean

    if is_jru_query:
        fee_terms = r"\b(fees?|tuition|caution\s*(?:money|deposit|fee)|fee\s+structure)\b"
        return bool(re.search(fee_terms, clean))

    # Bare standalone queries with no external university context
    if clean in {"fees", "fee", "fee structure", "tuition fee", "caution fee", "caution deposit"}:
        return True

    return False
