"""Regression tests: chat math input must never execute Python code.

Before the fix, "derivative of __import__('os').system(...)" reached SymPy's
parse_expr / sympify (which use eval) and ran on the server.
"""
from pathlib import Path

import pytest
import sympy as sp
from app.services.math.calculus_engine import (
    CalculusEngine,
    UnsafeMathInputError,
    validate_math_input,
)

ENGINE = CalculusEngine()

ATTACK_TEMPLATES = [
    "derivative of __import__('os').makedirs('{probe}')",
    "integrate __import__('os').makedirs('{probe}')",
    "limit of __import__('os').makedirs('{probe}') as x approaches 0",
    "taylor series of __import__('os').makedirs('{probe}')",
    "solve ode __import__('os').makedirs('{probe}') = 0",
    "derivative of x.__class__.__mro__",
    "derivative of open('{probe}', 'w')",
    "derivative of exp.func",
    "differentiate (lambda: 1)()",
]


@pytest.mark.parametrize("template", ATTACK_TEMPLATES)
def test_math_query_never_executes_code(tmp_path: Path, template: str) -> None:
    probe = (tmp_path / "pwned").as_posix().lower()
    query = template.format(probe=probe)

    try:
        result = ENGINE.solve_math_query(query)
    except Exception:
        result = None

    assert not Path(probe).exists(), f"code executed for: {query}"
    assert not (result and result.get("success")), f"unsafe input was solved: {query}"


@pytest.mark.parametrize(
    "text",
    ["__import__('os')", "x.__class__", "exp.func", "'abc'", "a" * 300, ""],
)
def test_validate_math_input_rejects_unsafe_text(text: str) -> None:
    with pytest.raises(UnsafeMathInputError):
        validate_math_input(text)


def test_safe_parse_has_no_sympify_fallback() -> None:
    with pytest.raises(ValueError):
        ENGINE._safe_parse("x +* )")


def test_normal_derivative_still_works() -> None:
    result = ENGINE.solve_math_query("derivative of x^2 * sin(x)")
    assert result and result["success"]
    x = sp.Symbol("x")
    assert sp.simplify(sp.sympify(result["result_str"]) - sp.diff(x**2 * sp.sin(x), x)) == 0


def test_normal_integral_and_ode_still_work() -> None:
    integral = ENGINE.solve_math_query("integrate x^2 from 0 to 3")
    assert integral and integral["success"]

    ode = ENGINE.solve_math_query("solve ode y'' + 4y = 0")
    assert ode and ode["success"]


def test_matrix_string_cells_are_parsed_safely(tmp_path: Path) -> None:
    probe = (tmp_path / "matrix_pwned").as_posix().lower()
    with pytest.raises(ValueError):
        ENGINE.matrix_operations([["1", f"__import__('os').makedirs('{probe}')"], [0, 1]])
    assert not Path(probe).exists()

    ok = ENGINE.matrix_operations([["1", "2"], [3, 4]])
    assert ok["success"]
