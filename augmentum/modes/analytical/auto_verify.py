"""Automated verification — tool-based checks on APPLY phase output.

Detects verifiable content (math expressions, code blocks, factual claims
backed by search results) in the APPLY output and runs automated checks
using real tools (math_verify, python_exec) rather than asking the LLM
to self-verify.

Each check produces a VerificationCheck with pass/fail status, details,
and a human-readable summary suitable for injection into the VERIFY phase
prompt so the LLM reviewer has concrete evidence to work with.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VerificationCheck:
    """Result of a single automated verification check."""

    check_type: str          # "math", "code", "fact"
    input_text: str          # the expression/code/claim that was checked
    passed: bool
    details: str             # human-readable explanation
    error: str = ""          # error message if check couldn't run
    skipped: bool = False    # True if check was skipped (e.g. unavailable deps)


@dataclass
class AutoVerifyResult:
    """Aggregated results from all automated verification checks."""

    checks: list[VerificationCheck] = field(default_factory=list)
    all_passed: bool = True
    summary: str = ""

    @property
    def has_checks(self) -> bool:
        return len(self.checks) > 0

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed and not c.skipped)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.skipped)

    @property
    def skip_count(self) -> int:
        return sum(1 for c in self.checks if c.skipped)


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Math expressions: "= 42", "result is 3.14", inline equations
# Captures lines like "25 * 1.08 = 27" or "Total: $1,350"
_MATH_LINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:STEP \d+:?\s*)?.*?"
    r"(\d[\d,]*\.?\d*\s*[\+\-\*/\^]\s*\d[\d,]*\.?\d*"
    r"(?:\s*[\+\-\*/\^]\s*\d[\d,]*\.?\d*)*"
    r"\s*=\s*\d[\d,]*\.?\d*)",
    re.MULTILINE,
)

# Simpler: explicit "X = Y" where both are numbers
_EQUATION_PATTERN = re.compile(
    r"([\d,]+\.?\d*\s*[\+\-\*/\^%]\s*[\d,]+\.?\d*"
    r"(?:\s*[\+\-\*/\^%]\s*[\d,]+\.?\d*)*)"
    r"\s*=\s*([\d,]+\.?\d*)",
)

# Code blocks: ```python ... ``` or ```py ... ```
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Fact claims referencing search results
_FACT_CLAIM_PATTERN = re.compile(
    r"(?:according to|based on|from the search|the (?:data|results?) (?:show|indicate|suggest))",
    re.IGNORECASE,
)


def _clean_number(s: str) -> str:
    """Remove commas and whitespace from a number string."""
    return s.replace(",", "").replace(" ", "").strip()


def extract_math_expressions(text: str) -> list[tuple[str, str]]:
    """Extract (expression, expected_result) pairs from text.

    Returns tuples of (expression_str, expected_result_str) suitable
    for passing to math_verify.
    """
    results = []
    for m in _EQUATION_PATTERN.finditer(text):
        expr = _clean_number(m.group(1))
        expected = _clean_number(m.group(2))
        # Skip trivially simple expressions (just a number)
        if not any(op in expr for op in "+-*/^%"):
            continue
        # Skip very large numbers that are likely IDs or dates
        try:
            float(expected)
        except ValueError:
            continue
        results.append((expr, expected))
    return results


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from text."""
    blocks = []
    for m in _CODE_BLOCK_PATTERN.finditer(text):
        code = m.group(1).strip()
        if code and len(code) > 10:  # skip trivially small snippets
            blocks.append(code)
    return blocks


def extract_fact_claims(
    text: str, search_context: str,
) -> list[tuple[str, str]]:
    """Extract factual claims that reference search results.

    Returns (claim_sentence, relevant_search_snippet) pairs.
    """
    if not search_context:
        return []

    # Split on sentence boundaries (period followed by space/newline or end),
    # avoiding splits on decimal numbers like "3.5"
    _SENTENCE_SPLIT = re.compile(r"\.(?:\s|$)")

    claims = []
    for m in _FACT_CLAIM_PATTERN.finditer(text):
        # Find sentence boundaries using proper sentence-end detection
        # (period followed by whitespace or end of string, not mid-number)
        start = 0
        for sm in _SENTENCE_SPLIT.finditer(text[:m.start()]):
            start = sm.end()
        end_match = _SENTENCE_SPLIT.search(text, m.end())
        end = end_match.start() if end_match else len(text)
        sentence = text[start:end].strip(". \n")
        if len(sentence) > 20:
            # Find relevant snippet in search context
            # Use content words (3+ chars) and numbers from the claim
            words = set(
                w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence)
            )
            # Also include significant numbers (not tiny ones like "1" or "2")
            claim_nums = set(re.findall(r"\b\d+\.?\d*\b", sentence))
            best_snippet = ""
            best_overlap = 0
            for block in search_context.split("\n\n"):
                block_words = set(
                    w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", block)
                )
                block_nums = set(re.findall(r"\b\d+\.?\d*\b", block))
                # Count word overlap + number overlap
                overlap = len(words & block_words) + len(claim_nums & block_nums)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_snippet = block[:300]

            if best_snippet and best_overlap >= 2:
                claims.append((sentence, best_snippet))

    return claims


# ---------------------------------------------------------------------------
# Dependency checking for code blocks
# ---------------------------------------------------------------------------

# Packages installed in the executor container (Dockerfile) + stdlib.
# We check top-level module names only.
_EXECUTOR_PACKAGES: frozenset[str] = frozenset({
    # Installed in executor Dockerfile
    "numpy", "np",
    "pandas", "pd",
    "scipy",
    "sympy",
    "matplotlib", "mpl",
    "dateutil",
    # Common stdlib modules
    "os", "sys", "re", "json", "math", "random", "collections",
    "itertools", "functools", "operator", "string", "textwrap",
    "datetime", "time", "calendar", "copy", "typing", "dataclasses",
    "enum", "abc", "io", "pathlib", "tempfile", "shutil", "glob",
    "csv", "configparser", "argparse", "logging", "warnings",
    "unittest", "pprint", "statistics", "decimal", "fractions",
    "heapq", "bisect", "array", "struct", "hashlib", "hmac",
    "secrets", "base64", "binascii", "html", "xml", "urllib",
    "http", "email", "mailbox", "mimetypes", "socket", "ssl",
    "select", "signal", "threading", "multiprocessing", "subprocess",
    "queue", "contextvars", "ast", "dis", "inspect", "traceback",
    "types", "weakref", "contextlib", "atexit", "gc",
    "difflib", "pdb", "profile", "timeit", "cProfile",
    "zipfile", "gzip", "bz2", "lzma", "tarfile",
    "sqlite3", "dbm", "pickle", "shelve", "marshal",
    "platform", "sysconfig", "site", "code", "codeop",
    "compileall", "py_compile", "token", "tokenize", "keyword",
    "linecache", "fnmatch", "fileinput", "stat", "posixpath",
    "ntpath", "genericpath", "uuid", "ipaddress",
    # Common aliases the model might use for installed packages
    "plt",
})


def _extract_imports(code: str) -> list[str]:
    """Extract top-level module names from import statements using AST.

    Returns a list of module names (e.g. ["numpy", "requests", "sklearn"]).
    Falls back to regex if AST parsing fails (syntax errors in model code).
    """
    modules: list[str] = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # "import numpy.linalg" → top-level is "numpy"
                    modules.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module.split(".")[0])
    except SyntaxError:
        # Fallback: regex-based extraction for code with syntax errors
        for m in re.finditer(
            r"^\s*(?:import|from)\s+([\w.]+)", code, re.MULTILINE,
        ):
            modules.append(m.group(1).split(".")[0])

    return modules


def check_code_dependencies(code: str) -> list[str]:
    """Check if a code block uses packages not available in the executor.

    Returns a list of unavailable module names, or empty list if all
    dependencies are satisfied.
    """
    imports = _extract_imports(code)
    unavailable = []
    for mod in imports:
        mod_lower = mod.lower()
        if mod_lower not in _EXECUTOR_PACKAGES and mod not in sys.stdlib_module_names:
            unavailable.append(mod)
    return unavailable


# ---------------------------------------------------------------------------
# Automated verification runner
# ---------------------------------------------------------------------------


async def run_auto_verification(
    apply_output: str,
    tool_registry: ToolRegistry | None,
    search_context: str = "",
) -> AutoVerifyResult:
    """Run automated verification checks on APPLY phase output.

    Checks:
    1. Math: Extract equations, verify with math_verify tool
    2. Code: Extract Python blocks, execute with python_exec tool
    3. Facts: Cross-reference claims against search results

    Returns an AutoVerifyResult with all check results.
    """
    result = AutoVerifyResult()

    if not tool_registry:
        return result

    # --- Math verification ---
    math_tool = tool_registry.get("math_verify")
    math_expressions = extract_math_expressions(apply_output)

    if math_tool and math_expressions:
        for expr, expected in math_expressions[:5]:  # cap at 5 checks
            try:
                tool_result = await math_tool.execute(
                    expression=expr, expected=expected,
                )
                match = tool_result.metadata.get("match", None)
                check = VerificationCheck(
                    check_type="math",
                    input_text=f"{expr} = {expected}",
                    passed=bool(match) if match is not None else tool_result.success,
                    details=tool_result.output or tool_result.error,
                )
            except Exception as exc:
                check = VerificationCheck(
                    check_type="math",
                    input_text=f"{expr} = {expected}",
                    passed=False,
                    details="",
                    error=f"Math verification failed: {exc}",
                )
            result.checks.append(check)
            log.info(
                "auto_verify_math",
                expression=expr[:80],
                expected=expected,
                passed=check.passed,
            )

    # --- Code execution ---
    exec_tool = tool_registry.get("python_exec")
    code_blocks = extract_code_blocks(apply_output)

    if exec_tool and code_blocks:
        for code in code_blocks[:3]:  # cap at 3 blocks
            code_preview = code[:200] + ("..." if len(code) > 200 else "")

            # Check for unavailable dependencies before executing
            unavailable = check_code_dependencies(code)
            if unavailable:
                check = VerificationCheck(
                    check_type="code",
                    input_text=code_preview,
                    passed=True,  # not a failure — just can't verify
                    details=(
                        f"Skipped: code requires packages not available in "
                        f"sandbox: {', '.join(unavailable)}. "
                        f"Code structure appears valid but execution was not "
                        f"attempted."
                    ),
                    skipped=True,
                )
                result.checks.append(check)
                log.info(
                    "auto_verify_code_skipped",
                    code_preview=code[:80],
                    unavailable=unavailable,
                )
                continue

            try:
                tool_result = await exec_tool.execute(code=code, timeout=15)
                check = VerificationCheck(
                    check_type="code",
                    input_text=code_preview,
                    passed=tool_result.success,
                    details=tool_result.output or tool_result.error,
                )
            except Exception as exc:
                check = VerificationCheck(
                    check_type="code",
                    input_text=code_preview,
                    passed=False,
                    details="",
                    error=f"Code execution failed: {exc}",
                )
            result.checks.append(check)
            log.info(
                "auto_verify_code",
                code_preview=code[:80],
                passed=check.passed,
            )

    # --- Fact cross-reference ---
    fact_claims = extract_fact_claims(apply_output, search_context)
    for claim, snippet in fact_claims[:3]:  # cap at 3 checks
        # Simple keyword overlap check — not LLM-based
        claim_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim)
        )
        snippet_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", snippet)
        )
        overlap = claim_words & snippet_words
        # Extract numbers from both claim and snippet for numeric verification
        claim_numbers = set(re.findall(r"\b\d+\.?\d*\b", claim))
        snippet_numbers = set(re.findall(r"\b\d+\.?\d*\b", snippet))
        numbers_match = bool(
            claim_numbers and claim_numbers & snippet_numbers
        )

        supported = len(overlap) >= 3 or numbers_match
        check = VerificationCheck(
            check_type="fact",
            input_text=claim[:200],
            passed=supported,
            details=(
                f"Claim supported by search results "
                f"(keyword overlap: {len(overlap)}, "
                f"numbers match: {numbers_match})"
                if supported
                else f"Claim may not be fully supported by search results "
                f"(keyword overlap: {len(overlap)}, "
                f"numbers match: {numbers_match}). "
                f"Source snippet: {snippet[:150]}"
            ),
        )
        result.checks.append(check)
        log.info(
            "auto_verify_fact",
            claim_preview=claim[:80],
            passed=check.passed,
            overlap=len(overlap),
        )

    # --- Build summary ---
    # Skipped checks don't count as failures
    result.all_passed = (
        all(c.passed for c in result.checks if not c.skipped)
        if result.checks else True
    )
    result.summary = _build_summary(result)

    return result


def _build_summary(result: AutoVerifyResult) -> str:
    """Build a human-readable summary of all verification checks."""
    if not result.checks:
        return ""

    lines = ["## Automated Verification Results"]
    skip_note = f", {result.skip_count} skipped" if result.skip_count else ""
    lines.append(
        f"Ran {len(result.checks)} automated check(s): "
        f"{result.pass_count} passed, {result.fail_count} failed{skip_note}."
    )
    lines.append("")

    for i, check in enumerate(result.checks, 1):
        if check.skipped:
            status = "SKIPPED"
        elif check.passed:
            status = "PASS"
        else:
            status = "FAIL"
        lines.append(f"### Check {i} [{check.check_type.upper()}] — {status}")
        lines.append(f"Input: {check.input_text}")
        if check.details:
            lines.append(f"Result: {check.details}")
        if check.error:
            lines.append(f"Error: {check.error}")
        lines.append("")

    if not result.all_passed:
        lines.append(
            "**Some automated checks failed.** The reviewer should "
            "pay special attention to the flagged items above."
        )

    return "\n".join(lines)
