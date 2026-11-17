"""Comprehensive API tests for the /api/v1/coding endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_api_coding_analyze_python():
    """Verify /api/v1/coding/analyze correctly computes AST Big-O and detects code smells."""
    code = """
def process(items, default_cache=[]):
    total = 0
    for x in items:
        total += x
    return total
"""
    response = client.post("/api/v1/coding/analyze", json={"code": code, "language": "python"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert "O(N)" in data["estimated_time_complexity"]
    assert data["max_nesting_depth"] == 1
    assert len(data["code_smells"]) >= 1
    assert any(s["type"] == "MutableDefaultArgument" for s in data["code_smells"])


def test_api_coding_analyze_polyglot():
    """Verify /api/v1/coding/analyze handles non-Python language classification."""
    code = "function search(arr, target) { let l = 0, r = arr.length - 1; /* binary search */ }"
    response = client.post("/api/v1/coding/analyze", json={"code": code, "language": "javascript"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["language"] == "javascript"
    assert "Binary Search" in data["algorithmic_pattern"]["pattern"]


def test_api_coding_execute_disabled_by_default():
    """Verify /api/v1/coding/execute returns safe error and disables execution by default/production."""
    code = "res = [x * 3 for x in range(4)]\nprint('Output:', res)"
    response = client.post("/api/v1/coding/execute", json={"code": code, "timeout_seconds": 0.5})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data.get("status") == "execution_unavailable"
    assert "Arbitrary Python code execution is disabled" in data["error"]


def test_api_coding_execute_dev_opt_in(monkeypatch):
    """Verify /api/v1/coding/execute executes only when explicitly enabled in local dev."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    code = "res = [x * 3 for x in range(4)]\nprint('Output:', res)"
    response = client.post("/api/v1/coding/execute", json={"code": code, "timeout_seconds": 0.5})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Output: [0, 3, 6, 9]" in data["stdout"]
    assert data["execution_time_ms"] > 0


def test_api_coding_execute_security_block(monkeypatch):
    """Verify /api/v1/coding/execute rejects dangerous module imports even when opted in."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    code = "import subprocess\nsubprocess.run(['dir'])"
    response = client.post("/api/v1/coding/execute", json={"code": code, "timeout_seconds": 0.5})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "Security Violation" in data["error"]


def test_api_coding_test_cases_synthesis():
    """Verify /api/v1/coding/test-cases returns canonical boundary cases."""
    response = client.post("/api/v1/coding/test-cases", json={"func_name": "merge_sort"})
    assert response.status_code == 200
    data = response.json()
    assert data["function_name"] == "merge_sort"
    assert data["total_cases"] >= 5
    case_names = [c["name"] for c in data["cases"]]
    assert "Empty / Null Input" in case_names
    assert "Already Sorted" in case_names


def test_api_coding_templates():
    """Verify /api/v1/coding/templates returns algorithmic templates."""
    response = client.post("/api/v1/coding/templates", json={"pattern": "binary_search", "language": "python"})
    assert response.status_code == 200
    data = response.json()
    assert data["pattern"] == "binary_search"
    assert "def binary_search" in data["code"]
    assert "O(log N)" in data["code"]


def test_api_coding_run_tests_disabled_by_default():
    """Verify /api/v1/coding/run-tests is safely disabled by default."""
    code = "def multiply(a, b): return a * b"
    tests = [{"name": "test_1", "call": "multiply(3, 4)", "expected": 12}]
    response = client.post("/api/v1/coding/run-tests", json={"code": code, "tests": tests})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "Arbitrary Python code execution is disabled" in data["error"]


def test_api_coding_run_tests_dev_opt_in(monkeypatch):
    """Verify /api/v1/coding/run-tests executes assertions and scores results when opted in."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ENABLE_UNSAFE_LOCAL_CODE_EXECUTION", "true")
    code = "def multiply(a, b): return a * b"
    tests = [
        {"name": "test_1", "call": "multiply(3, 4)", "expected": 12},
        {"name": "test_2", "call": "multiply(-2, 5)", "expected": -10},
        {"name": "test_3", "call": "multiply(0, 9)", "expected": 0},
    ]
    response = client.post("/api/v1/coding/run-tests", json={"code": code, "tests": tests})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_tests"] == 3
    assert data["passed_tests"] == 3
    assert data["failed_tests"] == 0
