"""Unit tests for Exact Symbolic Mathematics & Calculus Computation Engine."""
import pytest
from app.services.math.calculus_engine import calculus_engine


def test_differentiation_basic_and_higher_order():
    """Verify exact first and second derivatives."""
    # d/dx(x^3 + 2*x^2 - 5*x + 7) = 3*x^2 + 4*x - 5
    res1 = calculus_engine.differentiate("x^3 + 2*x^2 - 5*x + 7", var="x", order=1)
    assert res1["success"] is True
    assert "3*x**2 + 4*x - 5" in res1["result_str"]
    assert "3 x^{2} + 4 x - 5" in res1["result_latex"]

    # d^2/dx^2(sin(x)) = -sin(x)
    res2 = calculus_engine.differentiate("sin(x)", var="x", order=2)
    assert res2["success"] is True
    assert "-sin(x)" in res2["result_str"]


def test_integration_indefinite_and_definite():
    """Verify exact closed-form indefinite and definite integrals."""
    # integral of 2*x + 3 = x^2 + 3*x
    res1 = calculus_engine.integrate("2*x + 3", var="x")
    assert res1["success"] is True
    assert res1["is_definite"] is False
    assert "x**2 + 3*x" in res1["result_str"]

    # definite integral of x^2 from 0 to 3 = 3^3 / 3 = 9
    res2 = calculus_engine.integrate("x^2", var="x", lower=0, upper=3)
    assert res2["success"] is True
    assert res2["is_definite"] is True
    assert res2["result_str"] == "9"


def test_limits_lhopital_and_trig():
    """Verify limit calculations including 0/0 indeterminate forms."""
    # lim_{x->0} sin(x)/x = 1
    res = calculus_engine.calculate_limit("sin(x)/x", var="x", point=0)
    assert res["success"] is True
    assert res["result_str"] == "1"


def test_ode_first_and_second_order():
    """Verify analytical ordinary differential equation solutions."""
    # y'' + 4*y = 0 -> y(x) = C1*sin(2*x) + C2*cos(2*x)
    res = calculus_engine.solve_ode("y'' + 4*y = 0", func="y", var="x")
    assert res["success"] is True
    assert "sin(2*x)" in res["result_str"] or "cos(2*x)" in res["result_str"]


def test_matrix_determinant_and_eigenvalues():
    """Verify matrix operations, determinant, and eigenvalues."""
    # Matrix [[4, 1], [2, 3]]
    # det = 4*3 - 2*1 = 10
    # eigenvalues: (4-L)(3-L) - 2 = L^2 - 7L + 10 = (L-5)(L-2) = 0 -> L=2, L=5
    mat = [[4, 1], [2, 3]]
    res = calculus_engine.matrix_operations(mat)
    assert res["success"] is True
    assert res["determinant"] == "10"
    assert "2" in res["eigenvalues"]
    assert "5" in res["eigenvalues"]


def test_taylor_series_expansion():
    """Verify Maclaurin series expansion of exp(x)."""
    # exp(x) = 1 + x + x^2/2 + x^3/6 + ...
    res = calculus_engine.taylor_series("exp(x)", var="x", point=0, order=4)
    assert res["success"] is True
    assert "1" in res["result_str"]
    assert "x" in res["result_str"]
    assert "x**2/2" in res["result_str"]


def test_solve_math_query_natural_language():
    """Verify natural language query parsing and execution."""
    # 1. Integrate
    q1 = "Integrate 3*x^2 + 2*x with respect to x"
    res1 = calculus_engine.solve_math_query(q1)
    assert res1 is not None
    assert "x**3 + x**2" in res1["result_str"]

    # 2. Differentiate
    q2 = "Find the derivative of cos(x) with respect to x"
    res2 = calculus_engine.solve_math_query(q2)
    assert res2 is not None
    assert "-sin(x)" in res2["result_str"]

    # 3. Limit
    q3 = "Compute the limit of sin(x)/x as x approaches 0"
    res3 = calculus_engine.solve_math_query(q3)
    assert res3 is not None
    assert res3["result_str"] == "1"
