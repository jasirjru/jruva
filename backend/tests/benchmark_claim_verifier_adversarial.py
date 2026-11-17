"""Independent Adversarial Benchmark for the Existing Claim Verifier Heuristic.

Evaluates backend/tests/eval_generation.py (verify_claim_grounding) against
edge cases:
1. Negation flipping (retaining high token overlap)
2. Number swapping across unrelated entities
3. Entity substitution (novel names with high overlap)
4. Temporal / Date contradiction
5. Numerical unit mismatch (unsupported vs supported units)
6. Word-number vs digit-number paraphrase (e.g. 'forty percent' vs '40%')
7. Legitimate semantic paraphrases
8. Missing evidence / extrapolation
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.eval_generation import verify_claim_grounding, extract_numbers_and_quantities, extract_claims


ADVERSARIAL_TEST_CASES = [
    # Category 1: Negation Flipping (Expected: NOT GROUNDED / FALSE)
    {
        "id": "NEG-01",
        "category": "Negation Flipping",
        "expected_grounded": False,
        "context": "Tuition fees are strictly non-refundable under any circumstance after the 15th day of classes.",
        "claim": "Tuition fees are refundable under any circumstance after the 15th day of classes.",
        "description": "Dropped 'non-' prefix; inverts meaning with 95% token overlap and identical number."
    },
    {
        "id": "NEG-02",
        "category": "Negation Flipping",
        "expected_grounded": False,
        "context": "Students are not permitted to use calculators during the examination.",
        "claim": "Students are permitted to use calculators during the examination.",
        "description": "Removed 'not'; inverts restriction with 90% token overlap."
    },

    # Category 2: Entity / Number Swapping (Expected: NOT GROUNDED / FALSE)
    {
        "id": "SWAP-01",
        "category": "Number-Entity Misattribution",
        "expected_grounded": False,
        "context": "The campus computer lab has 45 workstations. The annual tuition fee for BTech is INR 85,000.",
        "claim": "The annual tuition fee for BTech is INR 45.",
        "description": "Attribute swap: number 45 exists in context but belongs to computer lab workstations."
    },
    {
        "id": "SWAP-02",
        "category": "Number-Entity Misattribution",
        "expected_grounded": False,
        "context": "Professor Sharma teaches CS101 with 60 students. Dr. Patel advises 4 graduate researchers.",
        "claim": "Professor Sharma advises 4 graduate researchers.",
        "description": "Entity swap: number 4 and researchers exist in context, but belong to Dr. Patel."
    },

    # Category 3: Entity Substitution (Expected: NOT GROUNDED / FALSE)
    {
        "id": "ENT-01",
        "category": "Entity Substitution",
        "expected_grounded": False,
        "context": "Dr. Mira Sen was appointed as the Project Coordinator for Project Atlas in Ranchi.",
        "claim": "Dr. Alan Turing was appointed as the Project Coordinator for Project Atlas in Ranchi.",
        "description": "Substituted subject name with high overall token overlap (8/10 tokens match)."
    },

    # Category 4: Numerical Units (Expected: NOT GROUNDED / FALSE)
    {
        "id": "UNIT-01",
        "category": "Numerical Unit Mismatch",
        "expected_grounded": False,
        "context": "The maximum acceptable execution latency ceiling for retrieval is 150 ms.",
        "claim": "The maximum acceptable execution latency ceiling for retrieval is 150 seconds.",
        "description": "Changed unit from 'ms' to 'seconds'. Same base digit."
    },
    {
        "id": "UNIT-02",
        "category": "Numerical Unit Mismatch (Uncovered Unit)",
        "expected_grounded": False,
        "context": "The container memory allocation ceiling is set to 380 MB.",
        "claim": "The container memory allocation ceiling is set to 380 GB.",
        "description": "Changed unit from 'MB' to 'GB' where GB/MB may not be captured by hardcoded regex unit list."
    },

    # Category 5: Word Numbers vs Digits (Expected: GROUNDED / TRUE - Test for False Negatives)
    {
        "id": "WORD-01",
        "category": "Word vs Digit Paraphrase",
        "expected_grounded": True,
        "context": "The passing grade requires an aggregate score of not less than forty percent across all modules.",
        "claim": "Students must achieve at least 40% aggregate score across all modules to pass.",
        "description": "Document uses English word 'forty percent'; claim uses digit '40%'. Valid ground-truth."
    },
    {
        "id": "WORD-02",
        "category": "Word vs Digit Paraphrase",
        "expected_grounded": True,
        "context": "The project will launch across three phases over a period of two years.",
        "claim": "The project rollout is scheduled in 3 phases over 2 years.",
        "description": "Document uses words 'three' and 'two'; claim uses digits '3' and '2'."
    },

    # Category 6: Legitimate Semantic Paraphrases (Expected: GROUNDED / TRUE)
    {
        "id": "PARA-01",
        "category": "Legitimate Semantic Paraphrase",
        "expected_grounded": True,
        "context": "Candidates must possess a Bachelor's degree in Engineering or an equivalent qualification with at least 60% marks.",
        "claim": "An engineering undergraduate degree with a minimum of 60% marks is required for candidates.",
        "description": "Legitimate syntactic restructuring of academic admission requirement."
    },
    {
        "id": "PARA-02",
        "category": "Legitimate Semantic Paraphrase",
        "expected_grounded": True,
        "context": "The university library remains accessible from 8:00 AM until 10:00 PM on all official instructional days.",
        "claim": "Library hours are 8:00 AM to 10:00 PM during instructional days.",
        "description": "Concise summary of library operating schedule."
    },

    # Category 7: Unsupported Extrapolation / Hallucination (Expected: NOT GROUNDED / FALSE)
    {
        "id": "EXTRA-01",
        "category": "Unsupported Extrapolation",
        "expected_grounded": False,
        "context": "JRUVA supports Google Gemini, Groq, and OpenRouter multi-provider routing.",
        "claim": "JRUVA provides full native integration with Anthropic Claude 3.5 Sonnet and AWS Bedrock.",
        "description": "Hallucinates nonexistent provider integrations."
    }
]


def run_benchmark():
    print("=" * 80)
    print("ADVERSARIAL BENCHMARK: EXISTING CLAIM VERIFIER HEURISTIC")
    print("=" * 80)

    total = len(ADVERSARIAL_TEST_CASES)
    correct = 0
    false_positives = []
    false_negatives = []

    for case in ADVERSARIAL_TEST_CASES:
        is_grounded, reason = verify_claim_grounding(case["claim"], case["context"])
        expected = case["expected_grounded"]
        matched = (is_grounded == expected)

        if matched:
            correct += 1
            verdict = "PASS (Correct)"
        else:
            if is_grounded and not expected:
                verdict = "FAIL (FALSE POSITIVE - Allowed hallucination!)"
                false_positives.append((case, reason))
            else:
                verdict = "FAIL (FALSE NEGATIVE - Flagged true paraphrase!)"
                false_negatives.append((case, reason))

        print(f"[{case['id']}] {case['category']}: {verdict}")
        print(f"  Claim: '{case['claim']}'")
        print(f"  Expected: {expected} | Actual: {is_grounded}")
        if reason:
            print(f"  Reason: {reason}")
        print("-" * 80)

    accuracy = (correct / total) * 100
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(f"Total Test Cases:       {total}")
    print(f"Correct Classifications: {correct} ({accuracy:.1f}%)")
    print(f"False Positives:        {len(false_positives)} (Failed to catch hallucination/negation/swap)")
    print(f"False Negatives:        {len(false_negatives)} (Wrongly flagged valid paraphrase as fake)")
    print("=" * 80)

    if false_positives:
        print("\nCRITICAL FALSE POSITIVES (Hallucinations incorrectly marked as Verified):")
        for fp, r in false_positives:
            print(f" - [{fp['id']}] {fp['category']}: {fp['description']}")
            print(f"   Claim: {fp['claim']}")

    if false_negatives:
        print("\nCRITICAL FALSE NEGATIVES (Legitimate answers incorrectly marked as Fake):")
        for fn, r in false_negatives:
            print(f" - [{fn['id']}] {fn['category']}: {fn['description']}")
            print(f"   Reason given: {r}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives)
    }

if __name__ == "__main__":
    run_benchmark()
