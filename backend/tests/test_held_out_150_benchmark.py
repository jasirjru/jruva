"""Pytest suite for the versioned 150-case held-out benchmark dataset.
Validates all 20 prompt slices against the deterministic evaluation harness.
"""
import pytest
from tests.held_out_150_dataset import HELD_OUT_150_CASES
from tests.eval_held_out_150 import HeldOut150Evaluator, default_faithful_generator


@pytest.fixture(scope="module")
def evaluator():
    return HeldOut150Evaluator()


def test_held_out_dataset_integrity():
    """Verify that the held-out benchmark contains exactly 150 diverse cases."""
    assert len(HELD_OUT_150_CASES) == 150
    categories = set(c["category"] for c in HELD_OUT_150_CASES)
    assert len(categories) == 20, f"Expected 20 categories, got {len(categories)}"
    
    ids = [c["id"] for c in HELD_OUT_150_CASES]
    assert len(ids) == len(set(ids)), "Case IDs must be unique"


@pytest.mark.parametrize("case", HELD_OUT_150_CASES, ids=[c["id"] for c in HELD_OUT_150_CASES])
def test_held_out_case_deterministic_eval(evaluator, case):
    """Verify each benchmark case using the reference faithful generator."""
    output = default_faithful_generator(case)
    res = evaluator.evaluate_case(
        case=case,
        answer=output["answer"],
        citations=output.get("citations", []),
    )
    assert res["passed"] is True, f"Case {case['id']} ({case['category']}) failed: {res['failures']}"
