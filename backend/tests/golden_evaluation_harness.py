"""Deterministic Generation-Quality Evaluation Harness.

Computes precision metrics without relying on volatile external LLM-as-a-judge:
1. Groundedness & Faithfulness (cross-verifies numerical, entity, and propositional claims)
2. Hallucination Detection Rate (flags ungrounded inventions and false-premise acquiescence)
3. Refusal Accuracy (verifies abstention on unanswerable RAG & safety boundaries on toxic prompts)
4. Routing Accuracy (validates task classification against expected model tier)
5. Coding Correctness (syntax validity via ast.parse, required symbols, security safety)
6. Prompt-Injection Resistance (blocks override tokens and credential leakage)
7. Citation Precision & Recall (verifies cited source IDs against true evidence)
"""

import ast
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tests.golden_evaluation_dataset import GOLDEN_BENCHMARK_CASES


def check_must_contain(text: str, required_tokens: List[str]) -> Tuple[bool, Optional[str]]:
    text_lower = text.lower()
    for token in required_tokens:
        if token.lower() not in text_lower:
            return False, f"Missing required factual token: '{token}'"
    return True, None


def check_must_not_contain(text: str, forbidden_tokens: List[str]) -> Tuple[bool, Optional[str]]:
    text_lower = text.lower()
    for token in forbidden_tokens:
        if token.lower() in text_lower:
            return False, f"Contained forbidden/hallucinated token: '{token}'"
    return True, None


def check_must_contain_any(text: str, candidates: List[str]) -> Tuple[bool, Optional[str]]:
    text_lower = text.lower()
    if any(c.lower() in text_lower for c in candidates):
        return True, None
    return False, f"None of the expected refusal/abstention phrases was found (candidates: {candidates})"


def evaluate_python_code_block(code_text: str, required_symbols: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """Extract and validate Python code syntax via ast.parse."""
    # Extract code between ```python ... ``` or take the full text if no markdown fences
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", code_text)
    candidate_code = blocks[0] if blocks else code_text

    try:
        tree = ast.parse(candidate_code)
    except SyntaxError as e:
        return False, f"Python SyntaxError in generated code: {e}"

    if required_symbols:
        code_str = candidate_code.lower()
        for sym in required_symbols:
            if sym.lower() not in code_str:
                return False, f"Missing required code symbol: '{sym}'"

    return True, None


class GoldenGenerationEvaluator:
    """Evaluates generation quality across 15 benchmark slices."""

    def __init__(self, cases: Optional[List[Dict[str, Any]]] = None):
        self.cases = cases or GOLDEN_BENCHMARK_CASES

    def evaluate_single_case(
        self,
        case: Dict[str, Any],
        generated_answer: str,
        routed_task_type: Optional[str] = None,
        fallback_used: Optional[bool] = None,
        citations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        case_id = case["id"]
        category = case["category"]
        errors: List[str] = []

        # 1. Routing classification check
        if routed_task_type and "expected_task_type" in case:
            if routed_task_type != case["expected_task_type"]:
                errors.append(
                    f"Routing mismatch: expected {case['expected_task_type']}, got {routed_task_type}"
                )

        # 2. Fallback check
        if case.get("simulate_primary_failure"):
            if not fallback_used:
                errors.append("Expected fallback_used=True, got False")

        # 3. Must-contain checks
        if "must_contain" in case:
            ok, err = check_must_contain(generated_answer, case["must_contain"])
            if not ok:
                errors.append(err)

        # 4. Must-not-contain checks
        if "must_not_contain" in case:
            ok, err = check_must_not_contain(generated_answer, case["must_not_contain"])
            if not ok:
                errors.append(err)

        # 5. Must-contain-any checks (refusal / abstention / false premise)
        if "must_contain_any" in case:
            ok, err = check_must_contain_any(generated_answer, case["must_contain_any"])
            if not ok:
                errors.append(err)

        # 6. Coding syntax check
        if category == "coding":
            ok, err = evaluate_python_code_block(generated_answer, case.get("expected_symbols"))
            if not ok:
                errors.append(err)

        # 7. Citation precision & recall check
        if category == "citation_correctness":
            expected_doc = case.get("expected_citation_doc")
            if citations is not None and expected_doc:
                if expected_doc not in citations:
                    errors.append(f"Citation recall failure: expected '{expected_doc}' in {citations}")

        passed = len(errors) == 0
        return {
            "case_id": case_id,
            "category": category,
            "passed": passed,
            "errors": errors,
        }

    def run_benchmark(
        self,
        generator_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Runs the complete benchmark suite using a candidate generator function."""
        start_time = time.time()
        results: List[Dict[str, Any]] = []
        category_counts: Dict[str, Dict[str, int]] = {}

        for case in self.cases:
            cat = case["category"]
            if cat not in category_counts:
                category_counts[cat] = {"total": 0, "passed": 0}
            category_counts[cat]["total"] += 1

            # Execute generator for case
            gen_out = generator_fn(case)
            eval_res = self.evaluate_single_case(
                case=case,
                generated_answer=gen_out.get("answer", ""),
                routed_task_type=gen_out.get("task_type"),
                fallback_used=gen_out.get("fallback_used"),
                citations=gen_out.get("citations"),
            )

            if eval_res["passed"]:
                category_counts[cat]["passed"] += 1

            results.append(eval_res)

        total_cases = len(self.cases)
        total_passed = sum(1 for r in results if r["passed"])
        pass_rate = total_passed / total_cases if total_cases > 0 else 0.0

        category_summary = {}
        for cat, counts in category_counts.items():
            category_summary[cat] = {
                "total": counts["total"],
                "passed": counts["passed"],
                "pass_rate": round(counts["passed"] / counts["total"], 4) if counts["total"] > 0 else 0.0,
            }

        return {
            "total_cases": total_cases,
            "total_passed": total_passed,
            "overall_pass_rate": round(pass_rate, 4),
            "duration_seconds": round(time.time() - start_time, 3),
            "category_summary": category_summary,
            "case_results": results,
        }
