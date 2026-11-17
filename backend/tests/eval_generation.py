"""RAG Generation-Quality Evaluation Harness (Made-With-ML Methodology).

Evaluates the triad of RAG generation quality:
1. Context Relevance: Did retrieval return context relevant to the user query?
2. Faithfulness / Groundedness: Is the generated answer supported by the retrieved context?
   (Detects unsupported claims / hallucinations)
3. Answer Relevance: Does the final answer actually answer the user's question?

Methodology & Deterministic Approximation Notes:
------------------------------------------------
- Deterministic Evaluation: Avoids relying on volatile or external LLM-as-a-judge APIs,
  ensuring offline reproducibility, zero cost, and zero network flakiness.
- Claim Extraction: Splits answers into atomic propositional assertions (by sentence / clause).
- Grounding Verification: Cross-verifies numerical values, proper nouns, key predicates, and
  entity tokens against the retrieved context chunks. Contradicted or ungrounded facts are
  flagged as unsupported claims (hallucinations).
- Context Relevance: Measures the recall and precision of target factual units within
  the retrieved document context chunks.
- Answer Relevance: Evaluates query intent satisfaction, interrogative target resolution,
  and multi-part completeness.
- Slice Analysis: Computes metrics across simple_factual, ambiguous, multi_part, and noisy_typo slices.
"""
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag.document_rag import DocumentRAGService, tokenize
from tests.generation_eval_dataset import HELD_OUT_DOCUMENTS, HELD_OUT_EVAL_CASES


# ==============================================================================
# DETERMINISTIC CLAIM EXTRACTION & GROUNDING HEURISTICS
# ==============================================================================

def extract_claims(answer: str) -> List[str]:
    """Decompose answer into atomic propositional assertions (sentences and semicolon clauses)."""
    if not answer or not answer.strip():
        return []
    # Split on sentence terminals and semicolons
    raw_sentences = re.split(r"(?<=[.!?])\s+|;\s*", answer.strip())
    claims = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return claims


def extract_numbers_and_quantities(text: str) -> Set[str]:
    """Extract numerical amounts, percentages, currencies, time intervals, and credit values."""
    extracted: Set[str] = set()
    text_lower = text.lower()

    # 1. Currency + number (INR 85,000 / Rs. 5000)
    for m in re.finditer(r"\b(?:inr|rs\.?)\s*\d+(?:,\d+)*(?:\.\d+)?\b", text_lower):
        extracted.add(m.group(0).strip())

    # 2. Number with unit (e.g. 150ms, 4 credits, 8 semesters, 6-week, 10-point)
    for m in re.finditer(r"\b\d+(?:\.\d+)?\s*(?:ms|seconds|minutes|credits?|semesters?|weeks?|point|-point)\b", text_lower):
        extracted.add(m.group(0).strip())

    # 3. Percentages (e.g. 75%, 40%) - % has no trailing word boundary
    for m in re.finditer(r"\b\d+(?:\.\d+)?%", text_lower):
        extracted.add(m.group(0).strip())

    # 4. Formatted numbers with commas (e.g. 85,000, 1,50,000)
    for m in re.finditer(r"\b\d{1,3}(?:,\d{2,3})+\b", text_lower):
        extracted.add(m.group(0).strip())

    # 5. Standalone numbers not adjacent to commas
    for m in re.finditer(r"(?<![,\d])\b\d+(?:\.\d+)?\b(?![,\d])", text_lower):
        extracted.add(m.group(0).strip())

    return extracted


def verify_claim_grounding(claim: str, context: str) -> Tuple[bool, Optional[str]]:
    """Check if an individual claim is supported by the retrieved context.
    
    Returns:
        (is_grounded, failure_reason)
    """
    context_lower = context.lower()
    claim_lower = claim.lower()

    # 1. Numerical & Quantitative Consistency Check (Check longer/compound numbers first)
    claim_numbers = extract_numbers_and_quantities(claim)
    sorted_numbers = sorted(claim_numbers, key=lambda s: len(s), reverse=True)

    # Check for hallucinated numbers / amounts
    for num in sorted_numbers:
        # Normalize commas for comparison (e.g. 85000 vs 85,000)
        clean_num = num.replace(",", "")
        clean_ctx = context_lower.replace(",", "")
        if clean_num not in clean_ctx and num not in context_lower:
            return False, f"Unsupported quantity/number '{num}' not found in retrieved context"

    # 2. Key Entity & Predicate Overlap
    claim_tokens = tokenize(claim_lower)
    if not claim_tokens:
        return True, None

    context_tokens = set(tokenize(context_lower))
    overlap = [t for t in claim_tokens if t in context_tokens]
    overlap_ratio = len(overlap) / float(len(claim_tokens)) if claim_tokens else 0.0

    # If the claim introduces excessive ungrounded novel tokens (threshold < 0.50), flag as unsupported
    if overlap_ratio < 0.50:
        unsupported_tokens = [t for t in claim_tokens if t not in context_tokens]
        return False, f"Low semantic overlap ({round(overlap_ratio, 2)}): unsupported concepts {unsupported_tokens[:3]}"

    return True, None


# ==============================================================================
# CORE EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_context_relevance(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    ground_truth_facts: List[str],
    target_doc_id: str,
) -> Dict[str, Any]:
    """Metric 1: Context Relevance.
    
    Measures whether the retrieval pipeline fetched chunks containing the necessary
    ground-truth evidence and entities to answer the question.
    """
    if not retrieved_chunks:
        return {
            "context_relevance_score": 0.0,
            "target_doc_retrieved": False,
            "facts_present_count": 0,
            "total_facts_count": len(ground_truth_facts),
            "fact_coverage": 0.0,
        }

    combined_context = " ".join(c.get("text", "") for c in retrieved_chunks).lower()

    # Target document hit
    retrieved_doc_names = [c.get("doc_name", "") for c in retrieved_chunks]
    target_hit = any(target_doc_id in c.get("id", "") or target_doc_id in c.get("doc_name", "") for c in retrieved_chunks)

    # Ground-truth fact coverage in retrieved context
    facts_present = 0
    for fact in ground_truth_facts:
        fact_lower = fact.lower().replace(",", "")
        clean_ctx = combined_context.replace(",", "")
        if fact_lower in clean_ctx or all(token in clean_ctx for token in tokenize(fact_lower)):
            facts_present += 1

    fact_coverage = facts_present / float(len(ground_truth_facts)) if ground_truth_facts else 1.0

    # Query token presence in retrieved context
    q_tokens = tokenize(query.lower())
    q_overlap = sum(1 for t in q_tokens if t in combined_context)
    q_coverage = (q_overlap / float(len(q_tokens))) if q_tokens else 1.0

    # Composite context relevance: 70% fact coverage + 30% query concept overlap
    relevance_score = round((0.70 * fact_coverage) + (0.30 * q_coverage), 4)

    return {
        "context_relevance_score": relevance_score,
        "target_doc_retrieved": target_hit,
        "facts_present_count": facts_present,
        "total_facts_count": len(ground_truth_facts),
        "fact_coverage": round(fact_coverage, 4),
    }


def evaluate_faithfulness(answer: str, retrieved_context: str) -> Dict[str, Any]:
    """Metric 2: Faithfulness / Groundedness.
    
    Verifies that all assertions in the generated answer are grounded in the retrieved context.
    Detects unsupported claims and hallucinations.
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "faithfulness_score": 1.0,
            "total_claims": 0,
            "grounded_claims": 0,
            "unsupported_claims": 0,
            "hallucination_detected": False,
            "unsupported_details": [],
        }

    grounded_count = 0
    unsupported_count = 0
    unsupported_details = []

    for c in claims:
        is_grounded, failure_reason = verify_claim_grounding(c, retrieved_context)
        if is_grounded:
            grounded_count += 1
        else:
            unsupported_count += 1
            unsupported_details.append({"claim": c, "reason": failure_reason})

    faithfulness_score = grounded_count / float(len(claims)) if claims else 1.0

    return {
        "faithfulness_score": round(faithfulness_score, 4),
        "total_claims": len(claims),
        "grounded_claims": grounded_count,
        "unsupported_claims": unsupported_count,
        "hallucination_detected": unsupported_count > 0,
        "unsupported_details": unsupported_details,
    }


def evaluate_answer_relevance(
    query: str,
    answer: str,
    expected_entities: List[str],
    ground_truth_facts: List[str],
) -> Dict[str, Any]:
    """Metric 3: Answer Relevance.
    
    Measures whether the final answer directly and completely addresses the user's query.
    Penalizes non-answers, off-topic rambles, and missing target entities.
    """
    if not answer or not answer.strip():
        return {
            "answer_relevance_score": 0.0,
            "is_relevant": False,
            "entity_coverage": 0.0,
            "addressed_facts": 0,
        }

    ans_lower = answer.lower().replace(",", "")

    # 1. Expected entity resolution (e.g. amounts, codes, thresholds)
    entities_present = 0
    for ent in expected_entities:
        ent_lower = ent.lower().replace(",", "")
        if ent_lower in ans_lower or all(t in ans_lower for t in tokenize(ent_lower)):
            entities_present += 1

    entity_coverage = entities_present / float(len(expected_entities)) if expected_entities else 1.0

    # 2. Key fact presence
    facts_present = 0
    for fact in ground_truth_facts:
        fact_lower = fact.lower().replace(",", "")
        if fact_lower in ans_lower or all(t in ans_lower for t in tokenize(fact_lower)):
            facts_present += 1
    fact_coverage = facts_present / float(len(ground_truth_facts)) if ground_truth_facts else 1.0

    # 3. Non-committal / refusal check
    refusal_patterns = [
        r"i (do not|don't) know",
        r"i cannot answer",
        r"information is not available",
    ]
    has_refusal = any(re.search(p, ans_lower) for p in refusal_patterns)

    # Composite score: 60% entity coverage + 40% fact coverage
    raw_score = (0.60 * entity_coverage) + (0.40 * fact_coverage)
    if has_refusal and entity_coverage < 0.3:
        raw_score = 0.0

    score = round(raw_score, 4)
    return {
        "answer_relevance_score": score,
        "is_relevant": score >= 0.70,
        "entity_coverage": round(entity_coverage, 4),
        "facts_present": facts_present,
        "total_facts": len(ground_truth_facts),
    }


# ==============================================================================
# END-TO-END RAG GENERATION EVALUATION RUNNER
# ==============================================================================

class RAGGenerationEvaluator:
    """Orchestrates end-to-end evaluation of Context Relevance, Faithfulness, and Answer Relevance."""

    def __init__(self, documents: List[Dict[str, Any]] = HELD_OUT_DOCUMENTS) -> None:
        self.documents = documents
        self.rag_service = DocumentRAGService()

        # Build in-memory attachments for DocumentRAGService
        self.attachments = []
        for doc in self.documents:
            b64_data = base64.b64encode(doc["text"].encode("utf-8")).decode("ascii")
            self.attachments.append({
                "name": doc["doc_name"],
                "mime_type": "text/plain",
                "data": b64_data,
            })

    def run_evaluation(
        self,
        cases: List[Dict[str, Any]] = HELD_OUT_EVAL_CASES,
        mode: str = "grounded",
    ) -> Dict[str, Any]:
        """Run evaluation across all cases in the benchmark dataset.
        
        Args:
            cases: List of evaluation cases.
            mode: "grounded" to test faithful answers, "hallucinated" to test hallucination detection,
                  or "irrelevant" to test answer irrelevance detection.
        """
        results: List[Dict[str, Any]] = []
        slice_aggregates: Dict[str, List[Dict[str, Any]]] = {}

        for case in cases:
            query = case["query"]
            slice_name = case["slice"]
            target_doc = case["target_doc_id"]
            facts = case["ground_truth_facts"]
            entities = case["expected_entities"]

            # Select answer candidate based on evaluation mode
            if mode == "hallucinated":
                answer = case["hallucinated_answer"]
            elif mode == "irrelevant":
                answer = case["irrelevant_answer"]
            else:
                answer = case["grounded_answer"]

            # 1. Perform actual RAG retrieval
            chunks, citations = self.rag_service.retrieve(self.attachments, query, top_k=2)
            retrieved_context = self.rag_service.build_grounding_context(chunks)

            # 2. Evaluate Context Relevance
            ctx_eval = evaluate_context_relevance(query, chunks, facts, target_doc)

            # 3. Evaluate Faithfulness / Groundedness (checks answer against retrieved context)
            faith_eval = evaluate_faithfulness(answer, retrieved_context)

            # 4. Evaluate Answer Relevance (checks answer against query expectations)
            ans_eval = evaluate_answer_relevance(query, answer, entities, facts)

            record = {
                "id": case["id"],
                "slice": slice_name,
                "query": query,
                "mode": mode,
                "context_relevance": ctx_eval["context_relevance_score"],
                "fact_coverage": ctx_eval["fact_coverage"],
                "faithfulness": faith_eval["faithfulness_score"],
                "unsupported_claims": faith_eval["unsupported_claims"],
                "hallucination_detected": faith_eval["hallucination_detected"],
                "unsupported_details": faith_eval["unsupported_details"],
                "answer_relevance": ans_eval["answer_relevance_score"],
                "is_relevant": ans_eval["is_relevant"],
            }
            results.append(record)

            if slice_name not in slice_aggregates:
                slice_aggregates[slice_name] = []
            slice_aggregates[slice_name].append(record)

        # Aggregate calculations
        def aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
            n = len(records)
            if n == 0:
                return {}
            return {
                "mean_context_relevance": round(sum(r["context_relevance"] for r in records) / n, 4),
                "mean_faithfulness": round(sum(r["faithfulness"] for r in records) / n, 4),
                "mean_answer_relevance": round(sum(r["answer_relevance"] for r in records) / n, 4),
                "hallucination_rate": round(sum(1 for r in records if r["hallucination_detected"]) / n, 4),
                "answer_pass_rate": round(sum(1 for r in records if r["is_relevant"]) / n, 4),
                "total_unsupported_claims": sum(r["unsupported_claims"] for r in records),
            }

        slices_summary = {s: aggregate(recs) for s, recs in slice_aggregates.items()}
        overall_summary = aggregate(results)

        return {
            "evaluation_mode": mode,
            "total_queries": len(results),
            "overall_metrics": overall_summary,
            "slice_metrics": slices_summary,
            "individual_records": results,
        }


# ==============================================================================
# CLI EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    evaluator = RAGGenerationEvaluator()

    print("\n" + "=" * 75)
    print("  RAG GENERATION-QUALITY EVALUATION RUNNER (MADE-WITH-ML / CHECKLIST)")
    print("=" * 75)

    # 1. Grounded Run
    print("\n>>> 1. EVALUATING GROUNDED / FAITHFUL GENERATION CANDIDATES:")
    t0 = time.perf_counter()
    grounded_report = evaluator.run_evaluation(mode="grounded")
    t_grounded = round(time.perf_counter() - t0, 3)

    print(f"Completed in {t_grounded}s across {grounded_report['total_queries']} queries.")
    print("\n--- OVERALL AGGREGATE METRICS ---")
    for k, v in grounded_report["overall_metrics"].items():
        print(f"  {k:30}: {v}")

    print("\n--- SLICE BREAKDOWN METRICS ---")
    for s_name, s_met in grounded_report["slice_metrics"].items():
        print(f"\n  [Slice: {s_name}]")
        for k, v in s_met.items():
            print(f"    {k:28}: {v}")

    # 2. Hallucinated Run (Validating Hallucination Detection Sensitivity)
    print("\n" + "=" * 75)
    print(">>> 2. EVALUATING SYNTHETIC HALLUCINATED CANDIDATES (SENSITIVITY TEST):")
    t1 = time.perf_counter()
    hallucinated_report = evaluator.run_evaluation(mode="hallucinated")
    t_hallucinated = round(time.perf_counter() - t1, 3)

    print(f"Completed in {t_hallucinated}s.")
    print("\n--- HALLUCINATION DETECTION PERFORMANCE ---")
    for k, v in hallucinated_report["overall_metrics"].items():
        print(f"  {k:30}: {v}")

    # 3. Irrelevant Run (Validating Answer Relevance Sensitivity)
    print("\n" + "=" * 75)
    print(">>> 3. EVALUATING IRRELEVANT OFF-TOPIC CANDIDATES (RELEVANCE SENSITIVITY):")
    irrelevant_report = evaluator.run_evaluation(mode="irrelevant")
    for k, v in irrelevant_report["overall_metrics"].items():
        print(f"  {k:30}: {v}")
