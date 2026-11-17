"""Security, input sanitization, and adversarial guardrails service."""
import base64
import binascii
import logging
import re
import unicodedata
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Homoglyph lookup table mapping lookalike Cyrillic/Greek/symbols to Latin
HOMOGLYPH_MAP = {
    'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x', 'у': 'y',
    'і': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd', 'ԛ': 'q', 'ԝ': 'w',
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'І': 'I',
    'Ј': 'J', 'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Ѕ': 'S',
    'Т': 'T', 'Х': 'X', 'Ү': 'Y', 'α': 'a', 'β': 'b', 'ο': 'o', 'ρ': 'p'
}

# Adversarial regex matching known jailbreak scaffolding, bypass, and academic exploit patterns
ADVERSARIAL_REGEX = re.compile(
    r"(?i)("
    # Needs a qualifier: "ignore instructions inside the excerpt" is a user defending against
    # injection, not an attack (10 Nov 2026 audit refused it).
    r"ignore\s+(all\s+)?((your|the|my)\s+)?(previous|prior|above|earlier|system|original)\s+(instructions|rules|guidelines)|"
    r"ignore\s+(all|your)\s+(instructions|rules|guidelines)|"
    r"reveal\s+(your\s+)?(system\s+prompt|api\s+keys?|developer\s+instructions?|password|database\s+password)|"
    r"system\s+(confidentiality\s+)?override|"
    r"what\s+is\s+your\s+system\s+prompt|"
    r"output\s+your\s+initial\s+prompt|"
    r"leaked\s+question\s+papers?|"
    r"leak(ed)?\s+exam\s+paper|"
    r"upcoming\s+.*question\s+paper|"
    r"bypass\s+.*(attendance|bio-metric|biometric|scanner)|"
    r"bypass\s+university\s+attendance|"
    r"hack\s+.*(database|grading|portal|server)|"
    r"exploit\s+payload|"
    r"sql\s+injection\s+payload|"
    r"forge\s+.*(medical\s+certificate|condonation|document|doctor)|"
    r"fake\s+medical\s+certificate|"
    r"roleplay\s+as\s+.*(unrestricted|dan\b|developer\s+mode)|"
    r"enter\s+developer\s+mode|"
    r"administrative\s+access\s+credentials"
    r")"
)

STANDARDIZED_GUARDRAIL_REFUSAL = "Sorry, I can't help with that."


def normalize_input_text(raw_text: str) -> str:
    """Strip zero-width spaces, decode base64/hex fragments, and normalize homoglyphs."""
    if not raw_text:
        return ""

    # 1. Strip zero-width & non-printable control characters
    text = re.sub(r"[\u200B-\u200D\uFEFF\u0000-\u0008\u000B\u000C\u000E-\u001F]", "", raw_text)

    # 2. Normalize Unicode NFKD form
    text = unicodedata.normalize("NFKD", text)

    # 3. Replace known homoglyphs
    for char, replacement in HOMOGLYPH_MAP.items():
        if char in text:
            text = text.replace(char, replacement)

    # 4. Check for standalone base64 encoded strings
    base64_candidates = re.findall(r"(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?", text)
    for b64 in base64_candidates:
        try:
            decoded = base64.b64decode(b64.encode("ascii")).decode("utf-8", errors="ignore")
            if len(decoded) > 5 and any(kw in decoded.lower() for kw in ["ignore", "prompt", "hack", "bypass", "exploit", "leak"]):
                text += " " + decoded
        except (binascii.Error, ValueError):
            # Not a valid base64 string, expected for arbitrary token matches
            pass
        except Exception as exc:
            logger.warning(f"[SilentFailureGuard] Unexpected error decoding base64 candidate: {exc}", exc_info=True)

    # 5. Check for hex encoded strings (e.g. \x69\x67\x6e\x6f\x72\x65)
    hex_candidates = re.findall(r"(?:\\x[0-9a-fA-F]{2}){4,}", text)
    for hx in hex_candidates:
        try:
            decoded_hex = bytes.fromhex(hx.replace("\\x", "")).decode("utf-8", errors="ignore")
            text += " " + decoded_hex
        except (binascii.Error, ValueError):
            # Not a valid hex string, expected for arbitrary token matches
            pass
        except Exception as exc:
            logger.warning(f"[SilentFailureGuard] Unexpected error decoding hex candidate: {exc}", exc_info=True)

    return text.strip()


def check_adversarial_guardrails(user_input: str) -> Tuple[str, Optional[str]]:
    """Inspect input for adversarial triggers and return refusal response if triggered."""
    normalized = normalize_input_text(user_input)

    # Check for adversarial match
    if ADVERSARIAL_REGEX.search(normalized):
        logger.warning(f"[Guardrail Triggered] Adversarial input detected (chars={len(user_input)})")
        return normalized, STANDARDIZED_GUARDRAIL_REFUSAL

    return normalized, None
