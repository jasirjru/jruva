"""Exact Symbolic Mathematics & Calculus Computation Engine.

Provides deterministic, non-hallucinated symbolic calculus, differential equations,
matrix linear algebra, limits, and Taylor expansions powered by SymPy.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from app.services.telemetry.telemetry_service import telemetry_service

logger = logging.getLogger(__name__)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# SymPy's parse_expr evaluates the parsed text as Python code, so user text must be
# validated before it reaches the parser. Only plain math characters are accepted:
# no underscores (blocks dunder access such as __import__), no quotes (no string
# literals), and no attribute access after a name or bracket.
MAX_MATH_INPUT_LENGTH = 200
_ALLOWED_MATH_CHARS = re.compile(r"^[0-9a-zA-Z+\-*/^().,\s=]*$")
_ATTRIBUTE_ACCESS = re.compile(r"[A-Za-z)\]]\s*\.")
_ODE_PRIMES = re.compile(r"(?<=[A-Za-z])'+")


def _build_safe_globals() -> dict[str, Any]:
    """SymPy namespace for parse_expr with Python builtins removed."""
    namespace: dict[str, Any] = {}
    exec("from sympy import *", namespace)  # noqa: S102 - fixed import, no user input
    namespace["__builtins__"] = {}
    return namespace


_SAFE_GLOBALS = _build_safe_globals()


class UnsafeMathInputError(ValueError):
    """Raised when text sent to the math engine is not a plain math expression."""


def validate_math_input(text: str, allow_primes: bool = False) -> str:
    """Return the text if it is a plain math expression, else raise UnsafeMathInputError."""
    candidate = text.strip()
    if not candidate or len(candidate) > MAX_MATH_INPUT_LENGTH:
        raise UnsafeMathInputError("Math expression is empty or too long.")
    checked = _ODE_PRIMES.sub("", candidate) if allow_primes else candidate
    if not _ALLOWED_MATH_CHARS.match(checked):
        raise UnsafeMathInputError("Math expression contains unsupported characters.")
    if _ATTRIBUTE_ACCESS.search(checked):
        raise UnsafeMathInputError("Attribute access is not allowed in math expressions.")
    return candidate


class CalculusEngine:
    """Deterministic Symbolic Math & Calculus Computation Service."""

    def __init__(self):
        self.transformations = TRANSFORMATIONS

    def _parse(self, text: str, local_dict: dict[str, Any] | None = None) -> sp.Expr:
        """Run parse_expr in a namespace without Python builtins."""
        return parse_expr(
            text,
            local_dict=local_dict or {},
            global_dict=dict(_SAFE_GLOBALS),
            transformations=self.transformations,
        )

    def _safe_parse(self, expr_str: str) -> sp.Expr:
        """Validate and parse a user math expression. Raises on unsafe or invalid input."""
        cleaned = validate_math_input(str(expr_str))
        cleaned = re.sub(r"\^", "**", cleaned)
        # Normalize common function notations
        cleaned = re.sub(r"\bln\b", "log", cleaned)
        cleaned = re.sub(r"\be\b(?!\w)", "E", cleaned)
        cleaned = re.sub(r"\bpi\b", "pi", cleaned, flags=re.IGNORECASE)
        try:
            return self._parse(cleaned)
        except Exception as exc:
            logger.warning("Failed to parse math expression (length %d): %s", len(cleaned), exc)
            telemetry_service.record_silent_failure("math_sandbox", type(exc).__name__)
            raise ValueError("Could not parse math expression.") from exc

    def differentiate(
        self, expr_str: str, var: str = "x", order: int = 1
    ) -> Dict[str, Any]:
        """Compute the nth derivative of an expression with respect to var."""
        x = sp.Symbol(var)
        expr = self._safe_parse(expr_str)
        deriv = sp.diff(expr, x, order)
        latex_expr = sp.latex(expr)
        latex_deriv = sp.latex(deriv)

        order_str = f"^{{({order})}}" if order > 1 else "'"
        deriv_notation = f"\\frac{{d^{{{order}}}}}{{d{var}^{{{order}}}}}" if order > 1 else f"\\frac{{d}}{{d{var}}}"

        steps = [
            f"**Given Function:** $f({var}) = {latex_expr}$",
            f"**Derivative Operator:** ${deriv_notation} \\left( {latex_expr} \\right)$",
            f"**Exact Derivative:** $f{order_str}({var}) = {latex_deriv}$",
        ]

        return {
            "success": True,
            "operation": "differentiation",
            "expression": expr_str,
            "variable": var,
            "order": order,
            "result_str": str(deriv),
            "result_latex": latex_deriv,
            "markdown": "\n\n".join(steps),
        }

    def integrate(
        self,
        expr_str: str,
        var: str = "x",
        lower: Optional[Union[int, float, str]] = None,
        upper: Optional[Union[int, float, str]] = None,
    ) -> Dict[str, Any]:
        """Compute definite or indefinite integral of an expression."""
        x = sp.Symbol(var)
        expr = self._safe_parse(expr_str)
        latex_expr = sp.latex(expr)

        if lower is not None and upper is not None:
            # Definite integral
            a = self._safe_parse(str(lower))
            b = self._safe_parse(str(upper))
            result = sp.integrate(expr, (x, a, b))
            result_latex = sp.latex(result)
            integral_sym = f"\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} {latex_expr} \\, d{var}"
            steps = [
                f"**Definite Integral:**\n\n$$\n{integral_sym} = {result_latex}\n$$",
                f"**Exact Value:** `{result}` (evaluates to $\\approx {round(float(result.evalf()), 6)}$)"
                if result.is_number and not result.has(sp.I) else f"**Exact Value:** ${result_latex}$",
            ]
            is_definite = True
        else:
            # Indefinite integral
            result = sp.integrate(expr, x)
            result_latex = sp.latex(result)
            integral_sym = f"\\int {latex_expr} \\, d{var}"
            steps = [
                f"**Indefinite Integral:**\n\n$$\n{integral_sym} = {result_latex} + C\n$$",
                f"**Anti-derivative:** ${result_latex} + C$ *(where $C$ is the arbitrary constant of integration)*",
            ]
            is_definite = False

        return {
            "success": True,
            "operation": "integration",
            "is_definite": is_definite,
            "expression": expr_str,
            "variable": var,
            "result_str": str(result),
            "result_latex": result_latex,
            "markdown": "\n\n".join(steps),
        }

    def calculate_limit(
        self,
        expr_str: str,
        var: str = "x",
        point: Union[int, float, str] = 0,
        direction: str = "+-",
    ) -> Dict[str, Any]:
        """Compute the limit of an expression as var approaches point."""
        x = sp.Symbol(var)
        expr = self._safe_parse(expr_str)
        p = self._safe_parse(str(point))

        lim_val = sp.limit(expr, x, p, dir=direction)
        latex_expr = sp.latex(expr)
        latex_lim = sp.latex(lim_val)
        latex_p = sp.latex(p)

        dir_sym = "^+" if direction == "+" else ("^-" if direction == "-" else "")
        limit_notation = f"\\lim_{{{var} \\to {latex_p}{dir_sym}}} \\left( {latex_expr} \\right)"

        steps = [
            f"**Limit Expression:**\n\n$$\n{limit_notation} = {latex_lim}\n$$",
            f"**Evaluated Result:** ${latex_lim}$",
        ]

        return {
            "success": True,
            "operation": "limit",
            "expression": expr_str,
            "variable": var,
            "point": str(point),
            "result_str": str(lim_val),
            "result_latex": latex_lim,
            "markdown": "\n\n".join(steps),
        }

    def solve_ode(
        self,
        ode_str: str,
        func: str = "y",
        var: str = "x",
        ics: Optional[Dict[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve an ordinary differential equation (ODE)."""
        x = sp.Symbol(var)
        y_fn = sp.Function(func)
        y = y_fn(x)

        # Validate user text before rewriting; the rewrite below adds trusted .diff() calls.
        validate_math_input(ode_str, allow_primes=True)
        if not re.fullmatch(r"[A-Za-z]", func) or not re.fullmatch(r"[A-Za-z]", var):
            raise UnsafeMathInputError("ODE function and variable must be single letters.")

        # Pre-process ODE string: e.g. y'' + 4*y = 0 -> y(x).diff(x, 2) + 4*y(x)
        normalized = ode_str.strip()
        normalized = re.sub(rf"(?<![a-zA-Z]){func}\'\'\'", f"{func}({var}).diff({var}, 3)", normalized)
        normalized = re.sub(rf"(?<![a-zA-Z]){func}\'\'", f"{func}({var}).diff({var}, 2)", normalized)
        normalized = re.sub(rf"(?<![a-zA-Z]){func}\'", f"{func}({var}).diff({var}, 1)", normalized)
        normalized = re.sub(rf"(?<![a-zA-Z]){func}(?!\s*[\(\']|[a-zA-Z])", f"{func}({var})", normalized)

        if "'" in normalized:
            raise UnsafeMathInputError("Unsupported derivative notation in ODE.")

        local_dict = {func: y_fn, var: x}
        if "=" in normalized:
            lhs_str, rhs_str = normalized.split("=", 1)
            lhs = self._parse(lhs_str.strip(), local_dict=local_dict)
            rhs = self._parse(rhs_str.strip(), local_dict=local_dict)
            eq = sp.Eq(lhs, rhs)
        else:
            eq = sp.Eq(self._parse(normalized, local_dict=local_dict), 0)

        sol = sp.dsolve(eq, y, ics=ics)
        latex_sol = sp.latex(sol)
        latex_eq = sp.latex(eq)

        steps = [
            f"**Differential Equation:**\n\n$$\n{latex_eq}\n$$",
            f"**General / Particular Solution:**\n\n$$\n{latex_sol}\n$$",
        ]

        return {
            "success": True,
            "operation": "ode",
            "equation": ode_str,
            "result_str": str(sol),
            "result_latex": latex_sol,
            "markdown": "\n\n".join(steps),
        }

    def matrix_operations(
        self, matrix_list: List[List[Union[int, float, str]]]
    ) -> Dict[str, Any]:
        """Compute determinant, inverse, eigenvalues, and eigenvectors of a matrix."""
        # sp.Matrix would sympify (eval) string cells, so parse them through the safe path.
        safe_rows = [
            [self._safe_parse(cell) if isinstance(cell, str) else cell for cell in row]
            for row in matrix_list
        ]
        M = sp.Matrix(safe_rows)
        rows, cols = M.shape
        det_val = None
        inv_mat = None
        eigenvals = {}

        is_square = rows == cols
        if is_square:
            det_val = M.det()
            if det_val != 0:
                try:
                    inv_mat = M.inv()
                except Exception:
                    inv_mat = None
            try:
                eigenvals = {str(k): v for k, v in M.eigenvals().items()}
            except Exception:
                eigenvals = {}

        latex_M = sp.latex(M)
        steps = [f"**Matrix $A$ ({rows}x{cols}):**\n\n$$\nA = {latex_M}\n$$"]

        if is_square:
            steps.append(f"**Determinant:** $\\det(A) = {sp.latex(det_val)}$")
            if inv_mat is not None:
                steps.append(f"**Inverse Matrix $A^{{-1}}$:**\n\n$$\nA^{{-1}} = {sp.latex(inv_mat)}\n$$")
            if eigenvals:
                eval_str = ", ".join([f"\\lambda = {sp.latex(k)} \\text{{ (multiplicity {v})}}" for k, v in eigenvals.items()])
                steps.append(f"**Eigenvalues:** ${eval_str}$")

        return {
            "success": True,
            "operation": "matrix",
            "shape": (rows, cols),
            "determinant": str(det_val) if det_val is not None else None,
            "eigenvalues": eigenvals,
            "markdown": "\n\n".join(steps),
        }

    def taylor_series(
        self,
        expr_str: str,
        var: str = "x",
        point: Union[int, float] = 0,
        order: int = 5,
    ) -> Dict[str, Any]:
        """Compute Taylor/Maclaurin series expansion of expr around point."""
        x = sp.Symbol(var)
        expr = self._safe_parse(expr_str)
        series_res = sp.series(expr, x, point, order)
        latex_series = sp.latex(series_res)
        series_type = "Maclaurin" if point == 0 else f"Taylor (around $x = {point}$)"

        steps = [
            f"**{series_type} Series Expansion (Order {order}):**",
            f"$$\n{sp.latex(expr)} = {latex_series}\n$$",
        ]

        return {
            "success": True,
            "operation": "taylor_series",
            "expression": expr_str,
            "order": order,
            "point": point,
            "result_str": str(series_res),
            "result_latex": latex_series,
            "markdown": "\n\n".join(steps),
        }

    def solve_math_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Automatically detect calculus/math intent in natural query and compute exact solution."""
        lower = query.lower().strip()

        # 1. Integration Intent: e.g. "integrate x^2 * sin(x)" or "integral of 1/(1+x^2) from 0 to 1"
        int_match = re.search(
            r"(?:integrate|integral\s+of|compute\s+the\s+integral\s+of)\s+(?P<expr>.+?)(?:\s+from\s+(?P<lower>[-\w\.\/]+)\s+to\s+(?P<upper>[-\w\.\/]+))?(?:\s+with\s+respect\s+to\s+(?P<var>[a-zA-Z]))?$",
            lower,
        )
        if int_match:
            expr_str = int_match.group("expr").strip()
            var = int_match.group("var") or "x"
            lower_val = int_match.group("lower")
            upper_val = int_match.group("upper")
            try:
                return self.integrate(expr_str, var=var, lower=lower_val, upper=upper_val)
            except Exception as e:
                logger.warning(
                    f"[SilentFailureGuard] Calculus integration failed for expression '{expr_str}': {e}",
                    exc_info=True,
                )
                telemetry_service.record_silent_failure("math_sandbox", str(e))

        # 2. Derivative Intent: e.g. "derivative of x^3 * exp(2x)" or "differentiate sin(x)/x"
        diff_match = re.search(
            r"(?:derivative\s+of|differentiate|find\s+the\s+derivative\s+of)\s+(?P<expr>.+?)(?:\s+with\s+respect\s+to\s+(?P<var>[a-zA-Z]))?$",
            lower,
        )
        if diff_match:
            expr_str = diff_match.group("expr").strip()
            var = diff_match.group("var") or "x"
            # check order
            order = 1
            if "second derivative" in lower or "order 2" in lower:
                order = 2
            elif "third derivative" in lower:
                order = 3
            try:
                return self.differentiate(expr_str, var=var, order=order)
            except Exception as e:
                logger.warning(
                    f"[SilentFailureGuard] Calculus differentiation failed for expression '{expr_str}': {e}",
                    exc_info=True,
                )
                telemetry_service.record_silent_failure("math_sandbox", str(e))

        # 3. Limit Intent: e.g. "limit of (sin(x))/x as x -> 0"
        lim_match = re.search(
            r"(?:limit\s+of|compute\s+the\s+limit\s+of)\s+(?P<expr>.+?)\s+(?:as|when)\s+(?P<var>[a-zA-Z])\s*(?:->|approaches|to)\s*(?P<point>[-\w\.\/]+)",
            lower,
        )
        if lim_match:
            expr_str = lim_match.group("expr").strip()
            var = lim_match.group("var") or "x"
            point_val = lim_match.group("point").strip()
            try:
                return self.calculate_limit(expr_str, var=var, point=point_val)
            except Exception as e:
                logger.warning(
                    f"[SilentFailureGuard] Calculus limit computation failed for expression '{expr_str}': {e}",
                    exc_info=True,
                )
                telemetry_service.record_silent_failure("math_sandbox", str(e))

        # 4. Taylor Series Intent: e.g. "taylor series of cos(x)" or "maclaurin series of exp(x)"
        taylor_match = re.search(
            r"(?:taylor\s+series|maclaurin\s+series)\s+(?:of\s+)?(?P<expr>.+?)(?:\s+around\s+(?P<pt>[-\w\.\/]+))?$",
            lower,
        )
        if taylor_match:
            expr_str = taylor_match.group("expr").strip()
            pt = taylor_match.group("pt") or 0
            try:
                return self.taylor_series(expr_str, point=float(pt) if pt != 0 else 0)
            except Exception as e:
                logger.warning(
                    f"[SilentFailureGuard] Calculus Taylor series expansion failed for expression '{expr_str}': {e}",
                    exc_info=True,
                )
                telemetry_service.record_silent_failure("math_sandbox", str(e))

        # 5. Differential Equation Intent: e.g. "solve ode y'' + 4y = 0"
        ode_match = re.search(
            r"(?:solve\s+(?:the\s+)?ode|differential\s+equation)\s+(?P<eq>.+)$",
            lower,
        )
        if ode_match:
            eq_str = ode_match.group("eq").strip()
            try:
                return self.solve_ode(eq_str)
            except Exception as e:
                logger.warning(
                    f"[SilentFailureGuard] Calculus ODE solution failed for equation '{eq_str}': {e}",
                    exc_info=True,
                )
                telemetry_service.record_silent_failure("math_sandbox", str(e))

        return None


calculus_engine = CalculusEngine()
