"""Unit tests for Deep Coding Intelligence & Execution Sandbox."""
import pytest
from app.services.coding.code_intelligence_service import code_intelligence_service


def test_ast_linear_and_quadratic_complexity():
    """Verify Big-O time complexity estimation for linear and nested loops."""
    # 1. Linear O(N)
    code_linear = """
def linear_search(arr, target):
    for x in arr:
        if x == target:
            return True
    return False
"""
    res1 = code_intelligence_service.analyze_python_code(code_linear)
    assert res1["valid"] is True
    assert "O(N)" in res1["estimated_time_complexity"]
    assert res1["max_nesting_depth"] == 1

    # 2. Quadratic O(N^2)
    code_quadratic = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
    res2 = code_intelligence_service.analyze_python_code(code_quadratic)
    assert res2["valid"] is True
    assert "O(N^2)" in res2["estimated_time_complexity"]
    assert res2["max_nesting_depth"] == 2


def test_ast_syntax_error_handling():
    """Verify graceful handling of invalid Python syntax."""
    bad_code = "def broken_fn(x return x +"
    res = code_intelligence_service.analyze_python_code(bad_code)
    assert res["valid"] is False
    assert "syntax_error" in res


def test_ast_recursion_detection():
    """Verify recursive function identification."""
    rec_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    res = code_intelligence_service.analyze_python_code(rec_code)
    assert res["valid"] is True
    assert res["is_recursive"] is True


def test_sandbox_execution_disabled_by_default(monkeypatch):
    """Verify arbitrary Python execution is disabled by default in production and safe environments."""
    monkeypatch.delenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    exec_code = "print('hello world')"
    res = code_intelligence_service.execute_safe_sandbox(exec_code)
    assert res["success"] is False
    assert res["status"] == "execution_unavailable"
    assert "disabled in public production" in res["error"]


def test_sandbox_execution_explicit_dev_opt_in(monkeypatch):
    """Verify explicit local development opt-in allows local code execution."""
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    exec_code = """
nums = [4, 1, 8, 3]
sorted_nums = sorted(nums)
print(f"Sorted: {sorted_nums}")
"""
    res = code_intelligence_service.execute_safe_sandbox(exec_code)
    assert res["success"] is True
    assert "Sorted: [1, 3, 4, 8]" in res["stdout"]
    assert res["execution_time_ms"] > 0


def test_sandbox_security_rejection_in_dev_mode(monkeypatch):
    """Verify prohibited modules are blocked even when local dev execution is enabled."""
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    hack_code = "import os; os.system('echo hacked')"
    res = code_intelligence_service.execute_safe_sandbox(hack_code)
    assert res["success"] is False
    assert "Security Violation" in res["error"]


def test_sandbox_timeout_protection_in_dev_mode(monkeypatch):
    """Verify infinite loop is terminated by sandbox timeout in local dev mode."""
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    infinite_code = """
x = 0
while True:
    x += 1
"""
    res = code_intelligence_service.execute_safe_sandbox(infinite_code, timeout_seconds=0.2)
    assert res["success"] is False
    assert "TimeoutError" in res["error"]


def test_boundary_test_cases_generation():
    """Verify automated synthesis of edge test cases."""
    tests = code_intelligence_service.synthesize_boundary_test_cases("binary_search")
    assert len(tests) >= 5
    names = [t["name"] for t in tests]
    assert "Empty / Null Input" in names
    assert "Duplicate Elements" in names


def test_solve_coding_query():
    """Verify end-to-end coding query resolution from markdown."""
    query = """
Analyze the time complexity of this code:
```python
def find_max(arr):
    m = arr[0]
    for x in arr:
        if x > m:
            m = x
    return m
```
"""
    res = code_intelligence_service.solve_coding_query(query)
    assert res is not None
    assert res["success"] is True
    assert "O(N)" in res["markdown"]
    assert "Recommended Boundary Test Cases" in res["markdown"]


def test_code_smells_detection():
    """Verify AST detection of mutable default arguments and quadratic string concatenation."""
    bad_code = """
def append_item(val, container=[]):
    s = ""
    for i in range(10):
        s += str(i)
    return s
"""
    res = code_intelligence_service.analyze_python_code(bad_code)
    assert res["valid"] is True
    smell_types = [s["type"] for s in res["code_smells"]]
    assert "MutableDefaultArgument" in smell_types
    assert "QuadraticAccumulationInLoop" in smell_types


def test_algorithmic_pattern_classification():
    """Verify classification of canonical algorithmic design patterns."""
    p1 = code_intelligence_service.classify_algorithmic_pattern("Find longest substring without repeating characters using sliding window")
    assert p1["pattern"] == "Sliding Window"
    assert "O(N)" in p1["canonical_time"]

    p2 = code_intelligence_service.classify_algorithmic_pattern("Solve shortest path with dijkstra min-heap")
    assert p2["pattern"] == "Dijkstra's Algorithm"

    p3 = code_intelligence_service.classify_algorithmic_pattern("Check cycle in graph with union find disjoint set")
    assert p3["pattern"] == "Disjoint Set Union (DSU)"


def test_algorithmic_templates():
    """Verify code template synthesis across patterns and languages."""
    t1 = code_intelligence_service.get_algorithmic_template("binary_search", "python")
    assert "def binary_search" in t1["code"]
    assert "O(log N)" in t1["code"]

    t2 = code_intelligence_service.get_algorithmic_template("sliding_window", "python")
    assert "def max_subarray_sum" in t2["code"]

    t3 = code_intelligence_service.get_algorithmic_template("binary_search", "typescript")
    assert "function binarySearch" in t3["code"]


def test_run_test_suite_evaluation(monkeypatch):
    """Verify automated execution of assertions in sandbox harness when explicitly enabled in dev mode."""
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    solution_code = """
def add(a, b):
    return a + b
"""
    tests = [
        {"name": "test_positives", "call": "add(2, 3)", "expected": 5},
        {"name": "test_negatives", "call": "add(-1, -4)", "expected": -5},
        {"name": "test_failing", "call": "add(2, 2)", "expected": 5},
    ]
    res = code_intelligence_service.run_test_suite(solution_code, tests)
    assert res["success"] is True
    assert res["total_tests"] == 3
    assert res["passed_tests"] == 2
    assert res["failed_tests"] == 1
    assert res["results"][0]["passed"] is True
    assert res["results"][2]["passed"] is False

