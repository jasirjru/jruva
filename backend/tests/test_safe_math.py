"""Tests for the safe math expression evaluator used by function plots."""

import numpy as np
import pytest
from app.services.image.safe_math import UnsafeExpressionError, evaluate_expression

X = np.linspace(-2.0, 2.0, 9)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("x", X),
        ("x^2 + 3*x - 1", X**2 + 3 * X - 1),
        ("sin(x) + cos(x)", np.sin(X) + np.cos(X)),
        ("-x / 2", -X / 2),
        ("exp(abs(x)) % 3", np.exp(np.abs(X)) % 3),
        ("pi * e", np.full_like(X, np.pi * np.e)),
        ("2", np.full_like(X, 2.0)),
    ],
)
def test_evaluates_allowed_math(expression, expected):
    result = evaluate_expression(expression, X)
    assert result.shape == X.shape
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "().__class__.__base__.__subclasses__()",
        "x.__class__",
        "np.sin(x)",
        "open('secrets.txt')",
        "[c for c in range(3)]",
        "(lambda: 1)()",
        "'text'",
        "x[0]",
        "sin(x, 1)",
        "sin(x=1)",
        "True",
        "y + 1",
        "x +",
        "x" + " + x" * 120,
    ],
)
def test_rejects_unsafe_or_invalid_expressions(expression):
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression(expression, X)


def test_huge_power_overflows_instead_of_hanging():
    result = evaluate_expression("9^9^9", X)
    assert np.all(np.isinf(result))
