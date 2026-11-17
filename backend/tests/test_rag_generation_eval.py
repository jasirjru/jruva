"""Automated Regression Test Suite for RAG Generation-Quality Evaluation.

Validates the evaluation framework across:
1. Context Relevance: Verifies high context relevance (> 0.80) across all 4 query slices.
2. Faithfulness & Groundedness: Ensures faithful answers achieve 1.0 faithfulness with zero hallucinations.
3. Hallucination Detection Sensitivity: Confirms 100% detection rate on ungrounded/hallucinated answers.
4. Answer Relevance: Ensures relevant answers achieve > 0.85 relevance while off-topic answers score < 0.20.
5. Slice Performance: Validates stability across simple factual, ambiguous, multi-part, and noisy inputs.
"""
import pytest
from typing import Dict, Any

from tests.generation_eval_dataset import HELD_OUT_DOCUMENTS, HELD_OUT_EVAL_CASES
from tests.eval_generation import (
    RAGGenerationEvaluator,
    evaluate_context_relevance,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    extract_claims,
    extract_numbers_and_quantities,
    verify_claim_grounding,
)


@pytest.fixture(scope="module")
def evaluator():
    return RAGGenerationEvaluator(documents=HELD_OUT_DOCUMENTS)


def test_claim_and_number_extraction_heuristics():
    """Verify deterministic extraction of claims and quantities."""
    text = (
        "The annual tuition fee is INR 85,000 per academic year. "
        "Students must maintain 75% attendance and complete 160 credits across 8 semesters."
    )
    claims = extract_claims(text)
    assert len(claims) == 2

    quantities = extract_numbers_and_quantities(text)
    assert "85,000" in quantities
    assert "75%" in quantities
    assert "160" in quantities or "160 credits" in quantities


def test_claim_grounding_verification():
    """Verify grounded claims pass while fabricated numbers fail."""
    context = (
        "JHARKHAND RAI UNIVERSITY ANNUAL FEE STRUCTURE\n"
        "B.Tech Computer Science: Annual tuition fee is INR 85,000 per academic year.\n"
        "Admission registration fee is INR 5,000."
    )

    # Valid grounded claim
    grounded_claim = "The annual tuition fee for B.Tech is INR 85,000."
    is_grounded, failure = verify_claim_grounding(grounded_claim, context)
    assert is_grounded is True
    assert failure is None

    # Hallucinated number
    hallucinated_claim = "The annual tuition fee for B.Tech is INR 1,50,000."
    is_grounded, failure = verify_claim_grounding(hallucinated_claim, context)
    assert is_grounded is False
    assert "1,50,000" in failure


def test_grounded_candidates_evaluation(evaluator):
    """Verify that all grounded candidates score >= 0.80 context relevance and 1.0 faithfulness."""
    report = evaluator.run_evaluation(HELD_OUT_EVAL_CASES, mode="grounded")
    overall = report["overall_metrics"]

    assert overall["mean_context_relevance"] >= 0.80, f"Low context relevance: {overall['mean_context_relevance']}"
    assert overall["mean_faithfulness"] == 1.0, f"Unfaithful claims in grounded set: {overall['mean_faithfulness']}"
    assert overall["hallucination_rate"] == 0.0, f"False positive hallucinations: {overall['hallucination_rate']}"
    assert overall["total_unsupported_claims"] == 0
    assert overall["mean_answer_relevance"] >= 0.90, f"Low answer relevance: {overall['mean_answer_relevance']}"
    assert overall["answer_pass_rate"] == 1.0


def test_hallucination_detection_sensitivity(evaluator):
    """Verify that synthetic hallucinated candidates are 100% detected."""
    report = evaluator.run_evaluation(HELD_OUT_EVAL_CASES, mode="hallucinated")
    overall = report["overall_metrics"]

    # Evaluator must catch all injected hallucinations
    assert overall["hallucination_rate"] == 1.0, f"Hallucination evading detection: {overall['hallucination_rate']}"
    assert overall["total_unsupported_claims"] == 16, f"Expected 16 unsupported claims, got {overall['total_unsupported_claims']}"
    assert overall["mean_faithfulness"] == 0.0, f"Expected 0.0 faithfulness on hallucinated answers, got {overall['mean_faithfulness']}"


def test_irrelevant_candidates_penalty(evaluator):
    """Verify that off-topic answers receive low Answer Relevance scores."""
    report = evaluator.run_evaluation(HELD_OUT_EVAL_CASES, mode="irrelevant")
    overall = report["overall_metrics"]

    assert overall["mean_answer_relevance"] < 0.20, f"Irrelevant answer scored too high: {overall['mean_answer_relevance']}"
    assert overall["answer_pass_rate"] == 0.0


def test_slice_performance_consistency(evaluator):
    """Verify performance consistency across all 4 slices."""
    report = evaluator.run_evaluation(HELD_OUT_EVAL_CASES, mode="grounded")
    slices = report["slice_metrics"]

    for slice_name in ["simple_factual", "ambiguous", "multi_part", "noisy_typo"]:
        assert slice_name in slices, f"Missing slice {slice_name}"
        s_data = slices[slice_name]
        assert s_data["mean_context_relevance"] >= 0.80, f"Slice {slice_name} failed context relevance: {s_data['mean_context_relevance']}"
        assert s_data["mean_faithfulness"] == 1.0, f"Slice {slice_name} failed faithfulness: {s_data['mean_faithfulness']}"
        assert s_data["hallucination_rate"] == 0.0, f"Slice {slice_name} had hallucinations: {s_data['hallucination_rate']}"
        assert s_data["mean_answer_relevance"] >= 0.90, f"Slice {slice_name} failed answer relevance: {s_data['mean_answer_relevance']}"
