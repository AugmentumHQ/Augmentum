"""Calculator tool — safe mathematical expression evaluation."""

from __future__ import annotations

import ast
import math
import operator

from augmentum.tools.base import Tool, ToolCategory, ToolResult

# Whitelisted math functions and constants
_SAFE_FUNCTIONS: dict[str, object] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "pow": pow,
    # math module
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
}

_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Supported binary operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unknown function: {func_name}")
        args = [_safe_eval(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_calculate(expression: str) -> float | int:
    """Safely evaluate a mathematical expression.

    Uses AST parsing instead of eval() for security.
    Normalizes common model output quirks before parsing:
    - ``^`` → ``**`` (models use caret for exponentiation)
    - Strips leading ``$`` and ``£``/``€`` (models include currency symbols)
    - Strips trailing ``=`` (models sometimes append it)
    - Strips commas in numbers (``1,000`` → ``1000``)
    """
    import re as _re

    # Basic input sanitization — use word boundaries to avoid false positives
    # (e.g. "cos" should not match "os")
    forbidden = r"\b(import|exec|eval|open|compile|getattr|setattr|delattr|globals|locals)\b|__"
    if _re.search(forbidden, expression):
        raise ValueError("Expression contains forbidden keywords")

    # --- Normalize common model output quirks ---

    # Strip currency symbols (models include $, £, €)
    expr = _re.sub(r"[$£€]", "", expression)

    # Strip trailing = (models sometimes write "2+3=")
    expr = expr.rstrip("= ")

    # Replace ^ with ** (caret = exponent in math notation, XOR in Python)
    # Only replace when surrounded by digits/parens/spaces — not inside function names
    expr = _re.sub(r"(?<=[\d\)\s])(\^)(?=[\d\(\s-])", "**", expr)

    # Strip thousands separators: 1,000,000 → 1000000
    # Match digit,digit{3} patterns (not function arg separators like max(3,7))
    expr = _re.sub(r"(\d),(\d{3})(?=[,.\d\s\)\+\-\*/^%]|$)", r"\1\2", expr)
    # Repeat for multi-group: 1,000,000
    expr = _re.sub(r"(\d),(\d{3})(?=[,.\d\s\)\+\-\*/^%]|$)", r"\1\2", expr)

    tree = ast.parse(expr, mode="eval")
    return _safe_eval(tree)


class CalculatorTool(Tool):
    """Safe mathematical expression evaluator."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions safely. Supports arithmetic, "
            "trig functions, logarithms, and constants (pi, e). "
            "Examples: '2 + 3 * 4', 'sqrt(144)', 'sin(pi/4)', 'log(100, 10)'"
        )

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
                    "description": "Mathematical expression to evaluate",
                },
            },
            "required": ["expression"],
        }

    @property
    def timeout(self) -> float:
        return 2.0

    @property
    def cache_ttl(self) -> float:
        return 0.0

    async def execute(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")
        if not expression:
            return ToolResult(success=False, error="No expression provided")

        try:
            result = safe_calculate(expression)
            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": expression, "result": result},
            )
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return ToolResult(success=False, error=f"Calculation error: {e}")
        except SyntaxError:
            return ToolResult(success=False, error=f"Invalid expression syntax: {expression}")
