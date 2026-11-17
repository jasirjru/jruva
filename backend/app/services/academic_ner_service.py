"""Academic Named Entity Recognition (NER) & Slot-Filling Service for JRU AI.

Extracts structured academic entities from conversational student queries:
- program_name: Canonical academic degree/course (e.g., 'B.Tech CSE', 'MCA', 'B.Sc Agriculture')
- semester_num: Canonical semester integer (1-8) or lateral entry marker
- fee_category: Tuition, caution money, bus, hostel, exam fee
- statutory_topic: Attendance, condonation, grading, CBCS, back-paper, helpline
- registration_id: Official student roll/registration pattern
"""
import re
from typing import Any, Dict, List, Optional


PROGRAM_PATTERNS = [
    (r"\b(b\.?\s*tech\s*(?:in\s+)?cse|b\.?\s*tech\s*(?:in\s+)?computer\s*science|bachelor\s+of\s+technology\s+in\s+cse)\b", "B.Tech CSE"),
    (r"\b(b\.?\s*tech\s*(?:in\s+)?mining(?:\s*engineering)?|btech\s*mining)\b", "B.Tech Mining"),
    (r"\b(b\.?\s*tech\s*(?:in\s+)?mech(?:anical)?|btech\s*mech)\b", "B.Tech Mechanical"),
    (r"\b(b\.?\s*tech\s*(?:in\s+)?civil(?:\s*engineering)?|btech\s*civil)\b", "B.Tech Civil"),
    (r"\bb\.?\s*tech\s*(?:in\s+)?lateral\s*entry\b", "B.Tech Lateral Entry"),
    (r"\bb\.?\s*tech\b", "B.Tech CSE"),
    (r"\bmca\b|\bmaster\s+of\s+computer\s+applications\b", "MCA"),
    (r"\bbca\b|\bbachelor\s+of\s+computer\s+applications\b", "BCA"),
    (r"\bmba\b|\bmaster\s+of\s+business\s+administration\b", "MBA"),
    (r"\bbba\b|\bbachelor\s+of\s+business\s+administration\b", "BBA"),
    (r"\bb\.?\s*sc\s*(?:hons\.?)?\s*agri(?:culture)?\b|\bbsc\s*agri\b", "B.Sc Agriculture"),
    (r"\bbba\s*ll\.?\s*b\b", "BBA LL.B"),
    (r"\bll\.?\s*b\b", "LL.B"),
    (r"\bb\.?\s*pharm\b|\bbachelor\s+of\s+pharmacy\b", "B.Pharm"),
    (r"\bd\.?\s*pharm\b|\bdiploma\s+in\s+pharmacy\b", "D.Pharm"),
]

SEMESTER_PATTERNS = [
    (r"\b(?:1st|first)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*1\b", 1),
    (r"\b(?:2nd|second)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*2\b", 2),
    (r"\b(?:3rd|third)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*3\b|\blateral\s*entry\b", 3),
    (r"\b(?:4th|fourth)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*4\b", 4),
    (r"\b(?:5th|fifth)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*5\b", 5),
    (r"\b(?:6th|sixth)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*6\b", 6),
    (r"\b(?:7th|seventh)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*7\b", 7),
    (r"\b(?:8th|eighth)\s*(?:sem(?:ester)?)?\b|\bsem(?:ester)?\s*8\b", 8),
]

FEE_CATEGORY_PATTERNS = [
    (r"\b(caution\s*(?:money|fee|deposit))\b", "caution_deposit"),
    (r"\b(tuition\s*(?:fee|cost)|semester\s*fee)\b", "tuition_fee"),
    (r"\b(bus|transport(?:ation)?)\s*(?:facility|fee|charge)?\b", "transport_fee"),
    (r"\b(hostel|mess|accommodation)\s*(?:fee|charge)?\b", "hostel_fee"),
    (r"\b(exam(?:ination)?)\s*(?:fee|charge)?\b", "exam_fee"),
    (r"\b(scholarship|concession|waiver|discount)\b", "scholarship"),
    (r"\b(fees?|annual\s*fee|total\s*cost)\b", "tuition_fee"),
]

STATUTORY_TOPIC_PATTERNS = [
    (r"\b(attendance|75%|condonation|shortage|debarment)\b", "attendance"),
    (r"\b(grade|grading|cgpa|sgpa|cbcs|10[- ]point|percentage)\b", "grading_cbcs"),
    (r"\b(cia|ese|internal|end[- ]semester|passing\s*(?:marks|criteria)|threshold|dual\s*threshold)\b", "examination_passing"),
    (r"\b(backlog|back\s*paper|supplementary|re[- ]test|promotion)\b", "backlog_retest"),
    (r"\b(anti[- ]ragging|ragging|helpline|ugc)\b", "anti_ragging"),
    (r"\b(cdc|career\s*development|placement|recruiters?|salary|package)\b", "placements"),
    (r"\b(legal\s*aid|pro[- ]bono|rural\s*clinic)\b", "legal_aid"),
    (r"\b(malpractice|cheating|unfair\s*means|u\.?f\.?m\.?)\b", "disciplinary_malpractice"),
]

REGISTRATION_REGEX = re.compile(r"\bJRU/\d{4}/[A-Z0-9]+/\d{2,4}\b", re.IGNORECASE)


def extract_academic_entities(text: str) -> Dict[str, Any]:
    """Extract canonical academic entity slots from a user query."""
    if not text:
        return {
            "program_name": None,
            "semester_num": None,
            "fee_category": None,
            "statutory_topic": None,
            "registration_id": None,
            "has_entities": False,
        }

    raw = text.strip()
    lowered = raw.lower()

    # 1. Program Name
    program_name = None
    for pattern, canonical in PROGRAM_PATTERNS:
        if re.search(pattern, lowered):
            program_name = canonical
            break

    # 2. Semester Number
    semester_num = None
    for pattern, sem_val in SEMESTER_PATTERNS:
        if re.search(pattern, lowered):
            semester_num = sem_val
            break

    # 3. Fee Category
    fee_category = None
    for pattern, canonical_fee in FEE_CATEGORY_PATTERNS:
        if re.search(pattern, lowered):
            fee_category = canonical_fee
            break

    # 4. Statutory Topic
    statutory_topic = None
    for pattern, canonical_topic in STATUTORY_TOPIC_PATTERNS:
        if re.search(pattern, lowered):
            statutory_topic = canonical_topic
            break

    # 5. Registration ID
    registration_id = None
    reg_match = REGISTRATION_REGEX.search(raw)
    if reg_match:
        registration_id = reg_match.group(0).upper()

    has_entities = any([
        program_name is not None,
        semester_num is not None,
        fee_category is not None,
        statutory_topic is not None,
        registration_id is not None,
    ])

    return {
        "program_name": program_name,
        "semester_num": semester_num,
        "fee_category": fee_category,
        "statutory_topic": statutory_topic,
        "registration_id": registration_id,
        "has_entities": has_entities,
    }
