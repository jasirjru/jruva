"""Retrieval evidence must never be reported as generated-answer quality."""
import json
from pathlib import Path

import pytest
from tests import eval_rag_ablations as ablations


def evaluate(runner, questions, chunks):
    return ablations.evaluate_context_coverage(runner, questions, chunks)


@pytest.mark.parametrize("questions", [[], [{"query": "Unsupported question", "target_chunk_id": None}]])
def test_retrieval_does_not_award_refusal_or_hallucination_scores(questions):
    report = evaluate(lambda query: ["irrelevant"], questions, {"irrelevant": {"text": "Unrelated passage"}})
    assert report.get("refusal_accuracy_pct") is None
    assert "unsupported_claim_rate_pct" not in report
    assert report["generation_evaluated"] is False
    assert report["unanswerable_not_evaluated"] == len(questions)


def test_context_coverage_and_target_match_are_retrieval_proxies():
    questions = [
        {"query": "Covered", "target_chunk_id": "target", "expected_evidence": "75% attendance"},
        {"query": "Missing", "target_chunk_id": "target", "expected_evidence": "160 credits"},
    ]
    report = evaluate(lambda query: ["target"], questions, {"target": {"text": "75% attendance is required"}})
    assert report["context_coverage_pct"] == 50.0
    assert report["top1_target_match_pct"] == 100.0
    assert "groundedness_pct" not in report
    assert "citation_precision_pct" not in report


def test_missing_evidence_is_not_counted_as_covered():
    report = evaluate(lambda query: ["target"], [{"query": "Missing label", "target_chunk_id": "target"}], {"target": {"text": "Any passage"}})
    assert report["context_coverage_pct"] is None
    assert report["evidence_labeled_queries"] == 0


def test_saved_ablation_report_labels_metrics_honestly():
    report_path = Path(__file__).resolve().parents[2] / "docs" / "rag_ablation_results.json"
    report = json.loads(report_path.read_text())
    assert "generation_quality" not in report
    assert report["context_quality"]["generation_evaluated"] is False
    assert "refusal_accuracy_pct" not in report["context_quality"]
