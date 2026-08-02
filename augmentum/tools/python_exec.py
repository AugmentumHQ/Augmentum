"""Python executor client — sends code to the sandboxed executor container."""

from __future__ import annotations

import ast
import re
import time
from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

# Maximum output size to prevent blowing up the LLM context
_MAX_OUTPUT_CHARS = 50_000

# Shorthand calculation preamble — makes a bare expression behave like a
# calculator: math functions/constants (sqrt, sin, log, factorial, gcd, pi,
# e, tau, inf) and a few numeric helpers are in scope without an import line.
_CALC_PREAMBLE = (
    "from math import *\n"
    "from fractions import Fraction\n"
    "from decimal import Decimal\n"
    "from statistics import mean, median, mode, pstdev, pvariance, stdev, variance\n"
)


def maybe_wrap_calc(code: str) -> str:
    """Shorthand calculation harness: if ``code`` is a single BARE EXPRESSION
    (no ``print``, no statements, one line), evaluate and print it with math
    functions preloaded — so the model can pass ``"17*23"`` or ``"sqrt(2)*100"``
    and get the answer back, instead of remembering ``print(...)`` + ``import
    math`` (the #1 reason a math call returned "(no output)"). Anything that
    isn't a lone expression — multi-line code, assignments, statements, or code
    that already prints — is returned untouched and runs exactly as before."""
    stripped = code.strip()
    if not stripped or "\n" in stripped or "print(" in stripped:
        return code
    try:
        # mode="eval" parses ONLY a single expression; a statement raises.
        ast.parse(stripped, mode="eval")
    except SyntaxError:
        return code  # a statement (assignment, import, …) — run as written
    return f"{_CALC_PREAMBLE}print({stripped})"

# Patterns that indicate potentially dangerous code even inside a sandbox.
# The executor container has its own isolation, but defense-in-depth matters.
_BLOCKED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bos\.system\s*\("), "os.system() is not allowed"),
    (re.compile(r"\bsubprocess\b"), "subprocess module is not allowed"),
    (re.compile(r"\b__import__\s*\("), "__import__() is not allowed"),
    (re.compile(r"\bexec\s*\("), "exec() is not allowed"),
    (re.compile(r"\beval\s*\("), "eval() is not allowed — use the expression directly"),
    (re.compile(r"\bcompile\s*\("), "compile() is not allowed"),
    (re.compile(r"\bopen\s*\("), "open() is not allowed — use print() for output"),
]


class PythonExecTool(Tool):
    """Execute Python code in a sandboxed environment."""

    @property
    def name(self) -> str:
        return "python_exec"

    @property
    def description(self) -> str:
        return (
            "Execute Python in a sandbox for exact calculation, math "
            "verification, data processing, or text manipulation. SHORTHAND: "
            "pass a bare expression (e.g. `17*23`, `sqrt(2)*100`, "
            "`factorial(12)`) and it's evaluated and printed for you, with math "
            "functions preloaded — use it to check arithmetic instead of doing "
            "it in your head. For multi-line code, print() what you want back. "
            "Standard library only."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "Code rejected": "The code contains blocked patterns (os.system, subprocess, exec, eval). Rewrite using only safe standard library functions.",
            "circuit breaker open": "The code executor is temporarily unavailable after repeated failures. Wait a moment and try again.",
            "timed out": "The code took too long. Simplify the computation or reduce data size.",
            "Executor request failed": "The code execution service is not responding. Answer from your own knowledge instead.",
            "ModuleNotFoundError": "That Python package is not installed in the sandbox. Use only standard library modules.",
            "SyntaxError": "The code has a syntax error. Check for missing colons, parentheses, or indentation.",
        }

    @property
    def requires_services(self) -> list[str]:
        return ["executor"]

    @property
    def produces(self) -> list[str]:
        return ["text", "structured_data"]

    @property
    def model_hint(self) -> str:
        return (
            "Prefer this over mental arithmetic for any non-trivial calculation "
            "or to verify a number before you state it. Shorthand: a bare "
            "expression like `sqrt(2)*100` is auto-evaluated and printed (math "
            "functions preloaded); otherwise write code and print() the result. "
            "Standard library only (no numpy/pandas/requests)."
        )

    def health_check(self) -> bool:
        """Check if the executor container is reachable."""
        import time as _time
        if _time.monotonic() < self._circuit_open_until:
            return False
        return True

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["code"],
        }

    @property
    def cacheable(self) -> bool:
        return False

    _FAILURE_THRESHOLD: int = 3
    _COOLDOWN_SECONDS: float = 60.0

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str = "http://executor:5000",
    ) -> None:
        self._client = http_client
        self._base_url = base_url.rstrip("/")
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    def validate_input(self, **kwargs) -> bool:
        code = kwargs.get("code", "")
        return isinstance(code, str) and len(code.strip()) > 0

    async def execute(self, *, code: str, timeout: int = 30) -> ToolResult:
        """Send code to the executor container and return the result."""
        if not code.strip():
            return ToolResult(success=False, error="No code provided")

        # Shorthand calc: a bare expression is auto-evaluated + printed with
        # math preloaded. No-op for real multi-line code.
        code = maybe_wrap_calc(code)

        timeout = max(1, min(int(timeout), 120))

        # Static analysis — reject obviously dangerous patterns
        for pattern, reason in _BLOCKED_PATTERNS:
            if pattern.search(code):
                return ToolResult(
                    success=False,
                    error=f"Code rejected: {reason}",
                    metadata={"code": code, "language": "python"},
                )

        # Circuit breaker — reject early if executor is known to be down
        if time.monotonic() < self._circuit_open_until:
            return ToolResult(
                success=False,
                error="Code execution temporarily unavailable (executor circuit breaker open). Try again in a moment.",
                metadata={"code": code, "language": "python"},
            )

        was_open = self._circuit_open_until > 0.0 and self._consecutive_failures >= self._FAILURE_THRESHOLD

        try:
            response = await self._client.post(
                f"{self._base_url}/execute",
                json={"code": code, "timeout": timeout},
                timeout=float(timeout) + 10.0,  # HTTP timeout > execution timeout
            )
            response.raise_for_status()
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._FAILURE_THRESHOLD:
                self._circuit_open_until = time.monotonic() + self._COOLDOWN_SECONDS
                log.warning(
                    "circuit_breaker_opened",
                    failures=self._consecutive_failures,
                    cooldown_seconds=self._COOLDOWN_SECONDS,
                )
            log.warning("python_exec_request_failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"Executor request failed: {sanitize_error_detail(str(exc))}",
                metadata={"code": code, "language": "python"},
            )

        # Circuit breaker — reset on successful HTTP round-trip
        if was_open:
            log.info("circuit_breaker_closed")
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

        try:
            data = response.json()
        except Exception:
            log.debug("executor_response_parse_failed", exc_info=True)
            return ToolResult(
                success=False,
                error="Failed to parse executor response as JSON",
                metadata={"code": code, "language": "python"},
            )

        success = data.get("success", False)
        stdout = data.get("stdout", "")
        stderr = sanitize_error_detail(data.get("stderr", ""))
        return_value = data.get("return_value")
        error = sanitize_error_detail(data.get("error") or "")  or None
        metrics = data.get("metrics", {})

        # Build a human-readable output string.
        parts: list[str] = []
        if stdout:
            parts.append(f"Output:\n{stdout}")
        if return_value is not None:
            parts.append(f"Return value: {return_value}")
        if stderr and success:
            parts.append(f"Warnings:\n{stderr}")
        if error and not success:
            parts.append(f"Error:\n{error}")

        output = "\n\n".join(parts) if parts else "(no output)"

        # Truncate oversized output to prevent context blowup
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + f"\n\n... (truncated, {len(output)} total chars)"

        return ToolResult(
            success=success,
            output=output,
            error=error or "",
            metadata={
                # The exact source that ran on the user's machine — post
                # calc-wrapping, so what's shown is what executed. Carried
                # untruncated so the UI can render a reviewable code block;
                # the model never sees metadata, so this costs no context.
                "code": code,
                "language": "python",
                "stdout": stdout,
                "stderr": stderr,
                "return_value": return_value,
                "elapsed_seconds": metrics.get("elapsed_seconds"),
            },
        )
