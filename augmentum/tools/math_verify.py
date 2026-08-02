"""Math verification tool — validates mathematical expressions and equations."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

# Blocklist for SymPy/Python code injection via parse_expr.
# These patterns must never appear in expressions sent to the executor.
_SYMPY_BLOCKED_PATTERNS = re.compile(
    r"__|import|exec|eval|open|compile|getattr|setattr|delattr"
    r"|globals|locals|vars|dir|type|classmethod|staticmethod"
    r"|__builtins__|os\.|sys\.|subprocess|shutil|pathlib"
    r"|breakpoint|input|print\s*\(",
    re.IGNORECASE,
)

# Allowlist: only these characters are permitted in symbolic expressions.
# Letters (variable names, function names), digits, whitespace, and math symbols.
_SYMPY_ALLOWED_CHARS = re.compile(r"^[a-zA-Z0-9\s\+\-\*/\^(),.\[\]=<>!|&_%]+$")

# Operators allowed in the safe numeric evaluator.
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Named constants / functions allowed in numeric evaluation.
_SAFE_NAMES: dict[str, object] = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "inf": math.inf,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)

    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        val = _SAFE_NAMES[node.id]
        if callable(val):
            raise ValueError(f"'{node.id}' is a function — use it with parentheses")
        return float(val)  # type: ignore[arg-type]

    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPS[type(node.op)](operand)

    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _SAFE_NAMES.get(node.func.id)
        if func is not None and callable(func):
            args = [_safe_eval_node(a) for a in node.args]
            return float(func(*args))
        raise ValueError(f"Function '{node.func.id}' is not allowed")

    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def _normalize_expression(expr: str) -> str:
    """Normalize common model output quirks before AST parsing.

    Shared logic with calculator.py — handles carets, currency symbols,
    thousands separators, and trailing equals.
    """
    e = expr.strip()
    # Strip currency symbols
    e = re.sub(r"[$£€]", "", e)
    # Strip trailing =
    e = e.rstrip("= ")
    # ^ → ** (caret = exponent in math notation, XOR in Python)
    e = re.sub(r"(?<=[\d\)\s])(\^)(?=[\d\(\s-])", "**", e)
    # Thousands separators: 1,000 → 1000 (not function arg commas)
    e = re.sub(r"(\d),(\d{3})(?=[,.\d\s\)\+\-\*/^%]|$)", r"\1\2", e)
    e = re.sub(r"(\d),(\d{3})(?=[,.\d\s\)\+\-\*/^%]|$)", r"\1\2", e)
    return e


def _safe_numeric_eval(expr: str) -> float:
    """Evaluate a numeric expression using a restricted AST walker.

    Only allows basic arithmetic, math constants (pi, e), and a small set
    of math functions (sqrt, abs, sin, cos, etc.).
    """
    normalized = _normalize_expression(expr)
    tree = ast.parse(normalized, mode="eval")
    return _safe_eval_node(tree)


class MathVerifyTool(Tool):
    """Verify mathematical expressions and equations."""

    @property
    def name(self) -> str:
        return "math_verify"

    @property
    def description(self) -> str:
        return "Verify mathematical expressions and equations"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                },
                "expected": {
                    "type": "string",
                    "description": "Expected result to compare against (optional)",
                },
                "verify_type": {
                    "type": "string",
                    "enum": ["numeric", "symbolic", "equation"],
                    "description": "Verification mode",
                    "default": "numeric",
                },
            },
            "required": ["expression"],
        }

    @property
    def timeout(self) -> float:
        return 10.0

    @property
    def cache_ttl(self) -> float:
        return 0.0

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        executor_base_url: str = "http://executor:5000",
    ) -> None:
        self._http_client = http_client
        self._executor_base_url = executor_base_url.rstrip("/")

    def validate_input(self, **kwargs) -> bool:
        expression = kwargs.get("expression", "")
        return isinstance(expression, str) and len(expression.strip()) > 0

    # ------------------------------------------------------------------
    # Executor helpers
    # ------------------------------------------------------------------

    async def _run_on_executor(self, code: str) -> dict | None:
        """Send Python code to the executor container. Returns parsed JSON or None."""
        if self._http_client is None:
            return None
        try:
            response = await self._http_client.post(
                f"{self._executor_base_url}/execute",
                json={"code": code, "timeout": 15},
                timeout=25.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.debug("math_verify_executor_unavailable", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Verification modes
    # ------------------------------------------------------------------

    async def _verify_numeric(
        self, expression: str, expected: str | None
    ) -> ToolResult:
        """Evaluate an expression numerically and optionally compare to expected."""
        try:
            result = _safe_numeric_eval(expression)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Could not evaluate expression: {exc}",
            )

        if expected is not None:
            try:
                expected_val = _safe_numeric_eval(expected)
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Could not evaluate expected value: {exc}",
                )

            match = math.isclose(result, expected_val, rel_tol=1e-9, abs_tol=1e-12)
            return ToolResult(
                success=True,
                output=(
                    f"Expression: {expression}\n"
                    f"Result:     {result}\n"
                    f"Expected:   {expected_val}\n"
                    f"Match:      {'YES' if match else 'NO'}"
                ),
                metadata={"result": result, "expected": expected_val, "match": match},
            )

        return ToolResult(
            success=True,
            output=f"Expression: {expression}\nResult:     {result}",
            metadata={"result": result},
        )

    def _sanitize_sympy_input(self, expr: str) -> str | None:
        """Sanitize an expression for safe use with SymPy's parse_expr.

        Returns the escaped string, or None if the input is rejected.
        """
        if _SYMPY_BLOCKED_PATTERNS.search(expr):
            return None
        if not _SYMPY_ALLOWED_CHARS.match(expr):
            return None
        # Escape backslashes first, then single quotes (correct order).
        return expr.replace("\\", "\\\\").replace("'", "\\'")

    async def _verify_symbolic(
        self, expression: str, expected: str | None
    ) -> ToolResult:
        """Use SymPy (via executor) to simplify and compare symbolically."""
        safe_expr = self._sanitize_sympy_input(expression)
        if safe_expr is None:
            return ToolResult(
                success=False,
                error="Expression contains disallowed characters or patterns",
            )

        code_lines = [
            "from sympy import sympify, simplify, S",
            "from sympy.parsing.sympy_parser import parse_expr",
            f"expr = parse_expr({safe_expr!r})",
            "simplified = simplify(expr)",
            "print(f'simplified: {{simplified}}')",
        ]
        if expected is not None:
            safe_expected = self._sanitize_sympy_input(expected)
            if safe_expected is None:
                return ToolResult(
                    success=False,
                    error="Expected value contains disallowed characters or patterns",
                )
            code_lines.extend([
                f"expected = parse_expr('{safe_expected}')",
                "diff = simplify(expr - expected)",
                "match = diff == 0",
                "print(f'expected: {{expected}}')",
                "print(f'match: {{match}}')",
            ])

        code = "\n".join(code_lines)
        result = await self._run_on_executor(code)

        if result is None:
            # Executor unavailable — fall back to numeric.
            return await self._verify_numeric(expression, expected)

        if not result.get("success", False):
            return ToolResult(
                success=False,
                error=f"SymPy verification failed: {result.get('error', 'unknown error')}",
            )

        stdout = result.get("stdout", "")
        # Parse the printed output.
        simplified_match = re.search(r"simplified:\s*(.+)", stdout)
        simplified_str = simplified_match.group(1).strip() if simplified_match else str(expression)

        match_match = re.search(r"match:\s*(True|False)", stdout)
        is_match = match_match.group(1) == "True" if match_match else None

        output_parts = [f"Expression:  {expression}", f"Simplified:  {simplified_str}"]
        if expected is not None:
            output_parts.append(f"Expected:    {expected}")
        if is_match is not None:
            output_parts.append(f"Match:       {'YES' if is_match else 'NO'}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={"simplified": simplified_str, "match": is_match},
        )

    async def _verify_equation(
        self, expression: str, expected: str | None
    ) -> ToolResult:
        """Verify that a solution satisfies an equation using SymPy via executor."""
        if expected is None:
            return ToolResult(
                success=False,
                error="Equation verification requires an 'expected' solution value",
            )

        safe_expr = self._sanitize_sympy_input(expression)
        if safe_expr is None:
            return ToolResult(
                success=False,
                error="Expression contains disallowed characters or patterns",
            )
        safe_expected = self._sanitize_sympy_input(expected)
        if safe_expected is None:
            return ToolResult(
                success=False,
                error="Expected value contains disallowed characters or patterns",
            )

        code = (
            "from sympy import sympify, simplify\n"
            "from sympy.parsing.sympy_parser import parse_expr\n"
            "from sympy.abc import x\n"
            f"eq = parse_expr({safe_expr!r})\n"
            f"sol = parse_expr({safe_expected!r})\n"
            "result = simplify(eq.subs(x, sol))\n"
            "satisfies = result == 0\n"
            "print(f'result: {{result}}')\n"
            "print(f'satisfies: {{satisfies}}')\n"
        )
        result = await self._run_on_executor(code)

        if result is None:
            return ToolResult(
                success=False,
                error="Equation verification requires the executor container (unavailable)",
            )

        if not result.get("success", False):
            return ToolResult(
                success=False,
                error=f"Equation verification failed: {result.get('error', 'unknown error')}",
            )

        stdout = result.get("stdout", "")
        satisfies_match = re.search(r"satisfies:\s*(True|False)", stdout)
        satisfies = satisfies_match.group(1) == "True" if satisfies_match else False

        return ToolResult(
            success=True,
            output=(
                f"Equation:   {expression}\n"
                f"Solution:   x = {expected}\n"
                f"Satisfies:  {'YES' if satisfies else 'NO'}"
            ),
            metadata={"satisfies": satisfies},
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        *,
        expression: str,
        expected: str | None = None,
        verify_type: str = "numeric",
    ) -> ToolResult:
        """Verify a mathematical expression.

        Modes:
        - numeric:  Parse and evaluate, compare with expected (if given).
        - symbolic: Use SymPy to simplify and compare symbolically.
        - equation: Verify that a solution satisfies an equation.
        """
        if not expression.strip():
            return ToolResult(success=False, error="Empty expression")

        if verify_type == "numeric":
            return await self._verify_numeric(expression, expected)
        if verify_type == "symbolic":
            return await self._verify_symbolic(expression, expected)
        if verify_type == "equation":
            return await self._verify_equation(expression, expected)

        return ToolResult(
            success=False,
            error=f"Unknown verify_type: {verify_type} (expected numeric, symbolic, or equation)",
        )
