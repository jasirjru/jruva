"""Tri-Tier Adaptive Cognitive Complexity Classifier.

Analyzes input queries to allocate appropriate reasoning depth and latency budgets:
- REFLEX (< 100ms): Greetings, quick definitions, slash commands, unit conversions, simple arithmetic.
- MODERATE (1-3s): Standard multi-paragraph explanations, code snippets, summaries, how-to guides.
- DEEP_THINKING (4-15s): Mathematical proofs, algorithmic optimization, distributed architecture, logic puzzles.
"""
import re
from typing import Any, Dict, Literal, Optional

ComplexityTier = Literal["reflex", "moderate", "deep_thinking"]

# Reflex triggers: very short factual queries, single-token questions, basic greetings
REFLEX_PATTERNS = [
    r"^(hi+|he+y+|he+l+o+|hlo+|hola|good\s+morning|good\s+evening|namaste)(\s+.*)?$",
    r"^/(?:fees|courses|attendance|exams|grading|helpline|status|help|clear|scholarship)$",
    r"^(what\s+is|who\s+is|define|meaning\s+of)\s+([a-zA-Z\s]{2,20})\??$",
    r"^(convert\s+\d+\s*[a-zA-Z]+\s+to\s+[a-zA-Z]+|\d+\s*[\+\-\*\/\^]\s*\d+)\??$",
    r"^(capital\s+of\s+[a-zA-Z\s]+|currency\s+of\s+[a-zA-Z\s]+)\??$",
]

# Deep thinking triggers: math proofs, algorithms, multi-step system design, formal logic, STEM
DEEP_THINKING_PATTERNS = [
    r"\b(prove|proof|theorem|contrapositive|contradiction|induction)\b",
    r"\b(differential\s+equation|integral|calculus|eigenvalue|eigenvector|laplace|fourier)\b",
    r"\b(taylor\s+series|maclaurin|green(?:'s)?\s+theorem|stokes(?:')?\s+theorem|divergence\s+theorem|matrix\s+decomposition|svd|singular\s+value|abstract\s+algebra|group\s+theory|ring\s+theory|euler(?:'s)?\s+totient|chinese\s+remainder)\b",
    r"\b(lagrangian|hamiltonian|maxwell(?:'s)?\s+equations?|schrodinger|wavefunction|quantum\s+mechanics|special\s+relativity|lorentz\s+transformation|thermodynamics|entropy|carnot|electrodynamics)\b",
    r"\b(sn1|sn2|electrophilic\s+aromatic|nucleophile|electrophile|reaction\s+mechanism|stereochemistry|enantiomer|diastereomer|gibbs\s+free\s+energy|nernst\s+equation|michaelis[- ]menten|chemical\s+kinetics|crystal\s+field\s+theory|orbital\s+hybridization)\b",
    r"\b(dna\s+replication|central\s+dogma|okazaki|transcription|translation|ribosome|krebs\s+cycle|citric\s+acid|oxidative\s+phosphorylation|electron\s+transport\s+chain|atp\s+synthase|enzyme\s+kinetics|hardy[- ]weinberg|mendelian\s+genetics|meiosis|mitosis|crispr|action\s+potential)\b",
    r"\b(solve|calculate|derive|evaluate)\s+(the\s+)?(integral|derivative|limit|eigenvalues?|work|heat|entropy|wavelength|ph|potential|velocity|acceleration|probability)\b",
    r"\b(np[- ]hard|p\s*vs\s*np|dijkstra|dynamic\s+programming|bellman[- ]ford|a\s*\*\s*algorithm|red[- ]black\s+tree|b[- ]tree)\b",
    r"\b(concurrency|lock[- ]free|memory\s+order|mutex|deadlock|race\s+condition|paxos|raft|byzantine|distributed\s+consensus)\b",
    r"\b(step[- ]by[- ]step\s+reasoning|deep\s+thinking|think\s+deeply|reason\s+through|chain\s+of\s+thought)\b",
    r"\b(formal\s+verification|trade[- ]off\s+matrix|game\s+theory|nash\s+equilibrium)\b",
    r"\b(logic\s+puzzle|riddle|paradox|fermi\s+problem|cryptography|zero[- ]knowledge)\b",
]


def classify_cognitive_complexity(query: str) -> Dict[str, Any]:
    """Classify prompt into reflex, moderate, or deep_thinking tier."""
    if not query or not query.strip():
        return {
            "tier": "reflex",
            "reasoning_budget": 0,
            "target_model_preference": "groq/openai/gpt-oss-120b",
            "enable_cot": False,
            "description": "Instant reflex answer (< 100ms)",
        }

    raw = query.strip()
    lowered = raw.lower()
    word_count = len(lowered.split())

    # 1. Check Deep Thinking triggers first
    for pattern in DEEP_THINKING_PATTERNS:
        if re.search(pattern, lowered):
            return {
                "tier": "deep_thinking",
                "reasoning_budget": 4096,
                "target_model_preference": "groq/openai/gpt-oss-120b",
                "enable_cot": True,
                "description": "Deep multi-step cognitive reasoning (<think> enabled, 4-15s)",
            }

    # 2. Check Reflex triggers
    if word_count <= 8:
        for pattern in REFLEX_PATTERNS:
            if re.search(pattern, lowered):
                return {
                    "tier": "reflex",
                    "reasoning_budget": 0,
                    "target_model_preference": "system_fastpath",
                    "enable_cot": False,
                    "description": "Instant reflex answer (< 100ms)",
                }

    # 3. Long or multi-question queries qualify for deeper analysis
    if word_count >= 40 or lowered.count("?") >= 2 or "\n1." in lowered or "compare and contrast" in lowered:
        return {
            "tier": "deep_thinking",
            "reasoning_budget": 2048,
            "target_model_preference": "groq/openai/gpt-oss-120b",
            "enable_cot": True,
            "description": "Comprehensive comparative & architectural analysis",
        }

    # 4. Default: Moderate
    return {
        "tier": "moderate",
        "reasoning_budget": 256,
        "target_model_preference": "gemini/gemini-3.6-flash",
        "enable_cot": False,
        "description": "Balanced high-speed streaming response (1-3s)",
    }
