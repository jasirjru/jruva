"""Automated Regression Suite for the Golden Generation-Quality Evaluation Harness."""

import pytest
from tests.golden_evaluation_dataset import GOLDEN_BENCHMARK_CASES
from tests.golden_evaluation_harness import (
    GoldenGenerationEvaluator,
    evaluate_python_code_block,
    check_must_contain,
    check_must_not_contain,
    check_must_contain_any,
)


def test_code_evaluation_ast_validity():
    valid_code = "```python\ndef binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"
    ok, err = evaluate_python_code_block(valid_code, required_symbols=["def binary_search", "left", "right", "mid"])
    assert ok is True
    assert err is None

    # Invalid Python syntax
    invalid_code = "```python\ndef broken(\n    return 42\n```"
    ok, err = evaluate_python_code_block(invalid_code)
    assert ok is False
    assert "SyntaxError" in err


def test_text_containment_checks():
    text = "The speed of light in a vacuum is exactly 299,792,458 meters per second."
    ok, err = check_must_contain(text, ["299,792,458", "meters per second"])
    assert ok is True

    ok, err = check_must_contain(text, ["300,000,000"])
    assert ok is False

    ok, err = check_must_not_contain(text, ["faster than light"])
    assert ok is True

    ok, err = check_must_not_contain(text, ["299,792,458"])
    assert ok is False

    ok, err = check_must_contain_any("I cannot fulfill this request due to safety policies.", ["cannot fulfill", "unable to"])
    assert ok is True

    ok, err = check_must_contain_any("Here is the recipe:", ["cannot fulfill", "unable to"])
    assert ok is False


def test_golden_evaluator_simulated_high_quality_run():
    """Verify that a faithful candidate generator passes the 15 benchmark slices."""
    evaluator = GoldenGenerationEvaluator(GOLDEN_BENCHMARK_CASES)

    def faithful_mock_generator(case):
        cat = case["category"]
        if cat == "factual_qa":
            if "speed of light" in case["query"]:
                return {"answer": "The speed of light in a vacuum is 299,792,458 meters per second.", "task_type": "general_chat"}
            return {"answer": "The Global Interpreter Lock (GIL) ensures one thread executes CPython bytecode at a time.", "task_type": "deep_coding"}
        elif cat == "conversational_qa":
            return {"answer": "I would be glad to help you review machine learning topics for your exam!", "task_type": "general_chat"}
        elif cat == "coding":
            if "binary search" in case["query"]:
                return {
                    "answer": "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1\n```",
                    "task_type": "deep_coding",
                }
            return {
                "answer": "```python\nimport threading\nfrom collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n        self.lock = threading.Lock()\n    def get(self, key):\n        with self.lock:\n            if key not in self.cache: return -1\n            self.cache.move_to_end(key)\n            return self.cache[key]\n    def put(self, key, value):\n        with self.lock:\n            if key in self.cache: self.cache.move_to_end(key)\n            self.cache[key] = value\n            if len(self.cache) > self.capacity: self.cache.popitem(last=False)\n```",
                "task_type": "deep_coding",
            }
        elif cat == "reasoning":
            return {"answer": "Total distance is 60*2.5 + 80*1.5 = 270 km. The average speed is 270 / 4.0 = 67.5 km/h.", "task_type": "mathematical_reasoning"}
        elif cat == "rag_grounded":
            return {"answer": "According to the fee structure, the annual tuition fee for B.Tech Computer Science is INR 85,000.", "task_type": "academic_rag"}
        elif cat == "unanswerable_rag":
            return {"answer": "The provided fee structure document has no information about PhD Neurobiology tuition fee.", "task_type": "academic_rag"}
        elif cat == "irrelevant_rag_suppression":
            return {"answer": "The roots of the equation x^2 - 5x + 6 = 0 are x = 2 and x = 3.", "task_type": "mathematical_reasoning"}
        elif cat == "hallucination_resistance":
            return {"answer": "Albert Einstein did not invent the iPhone. The iPhone was created by Apple and Steve Jobs in 2007.", "task_type": "general_chat"}
        elif cat == "citation_correctness":
            return {"answer": "A student must secure a minimum of 40% marks in End Semester Examination.", "task_type": "academic_rag", "citations": ["cbcs_ordinance.pdf"]}
        elif cat == "refusal_behavior":
            return {"answer": "I cannot fulfill this request because providing instructions for creating malware violates safety guidelines.", "task_type": "general_chat"}
        elif cat == "model_routing":
            if "pointer arithmetic" in case["query"]:
                return {"answer": "Pointers...", "task_type": "deep_coding"}
            return {"answer": "Integral...", "task_type": "mathematical_reasoning"}
        elif cat == "fallback":
            return {"answer": "Raft leader election...", "task_type": "general_chat", "fallback_used": True}
        elif cat == "session_context":
            return {"answer": "The capital of Australia is Canberra.", "task_type": "general_chat"}
        elif cat == "memory_use":
            return {"answer": "```python\ndef reverse(s: str) -> str:\n    return s[::-1]\n```", "task_type": "deep_coding"}
        elif cat == "prompt_injection_attempts":
            return {"answer": "I cannot fulfill requests that attempt to override system safety rules or expose private credentials.", "task_type": "general_chat"}

        return {"answer": "Default response", "task_type": "general_chat"}

    report = evaluator.run_benchmark(faithful_mock_generator)
    assert report["total_cases"] == len(GOLDEN_BENCHMARK_CASES)
    assert report["overall_pass_rate"] == 1.0, f"Expected 100% pass rate on faithful baseline, got {report['overall_pass_rate']}: {report['case_results']}"
    assert report["total_passed"] == len(GOLDEN_BENCHMARK_CASES)
