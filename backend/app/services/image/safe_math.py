"""Safe evaluation of user-supplied math expressions such as ``sin(x) + x^2``.

Expressions come from chat text, so they must never reach Python's ``eval``.
This module parses the expression with ``ast`` and evaluates only a small,
fixed set of node types: numbers, the variable ``x``, a few constants,
arithmetic operators and whitelisted numpy functions. Anything else
(attribute access, subscripts, comprehensions, keyword arguments, strings,
lambdas) raises ``UnsafeExpressionError``.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

import numpy as np

MAX_EXPRESSION_LENGTH = 200
MAX_NODES = 100

ALLOWED_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
}

ALLOWED_CONSTANTS: dict[str, float] ={"pi": float(np.pi), "e": float(np.e)}

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses syntax outside the allowed math subset."""


def evaluate_expression(expression: str, x: np.ndarray) -> np.ndarray:
    """Evaluate ``expression`` for every value in ``x`` and return an array.

    ``^`` is treated as power. Numbers are evaluated as float64, so huge
    powers overflow to ``inf`` instead of building giant Python integers.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError("Expression is too long.")

    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression: {exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise UnsafeExpressionError("Expression is too complex.")

    with np.errstate(all="ignore"):
        result = _eval_node(tree.body, x)
    return np.broadcast_to(np.asarray(result, dtype=np.float64), x.shape).copy()


def _eval_node(node: ast.AST, x: np.ndarray):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError("Only numbers are allowed as constants.")
        return np.float64(node.value)

    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id in ALLOWED_CONSTANTS:
            return np.float64(ALLOWED_CONSTANTS[node.id])
        raise UnsafeExpressionError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _eval_node(node.left, x)
        right = _eval_node(node.right, x)
        return BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_eval_node(node.operand, x))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise UnsafeExpressionError("Only basic math functions are allowed.")
        if node.keywords or len(node.args) != 1:
            raise UnsafeExpressionError("Math functions take exactly one argument.")
        return ALLOWED_FUNCTIONS[node.func.id](_eval_node(node.args[0], x))

    raise UnsafeExpressionError(f"Unsupported syntax: {type(node).__name__}")
