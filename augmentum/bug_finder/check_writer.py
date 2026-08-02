"""Check-writer subagent — LLM that generates codebase-specific
AST checks.

The "model edits its own tests" piece. Given a pillar from the
comprehender (e.g. "every user-scoped table accepts user_id"), the
check writer generates a Python module implementing an AST check
that surfaces violations.

The generated module has one entry point::

    def run(root: Path) -> list[dict]:
        '''Return findings.'''

…and is saved to the workspace's ``.augmentum/bug_finder/custom_checks/``
directory. The ``custom_check_runner`` discovers and executes every
such module on every subsequent audit.

Two outputs the writer can produce:

1. **AST-based check** — pure Python that walks the workspace's
   syntax tree looking for the pattern. Fast, deterministic, no
   third-party deps. Best for invariants over code shape.
2. **Grep-based check** — string-pattern scan when AST is overkill.
   Slower than AST but simpler to write reliably.

This module ships the prompts + parser + writer; the orchestrator
decides WHEN to invoke (typically: after the comprehender produces
pillars, if the pillar isn't already covered by a custom check).
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from pathlib import Path

from augmentum.agents.loop import SubagentResult, SubagentSpec, run_subagent
from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.role_models import Role
from augmentum.bug_finder.workspace_substrate import (
    custom_checks_dir,
    ensure_substrate,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Generous-but-bounded budget. Writing one check is ~3-5 LLM
# round-trips of reading sample code + drafting the check.
DEFAULT_CHECK_WRITER_BUDGET = SubagentBudget(
    max_iterations=10,
    max_wallclock_seconds=600,
    max_tokens=60_000,
)


CHECK_WRITER_SYSTEM_PROMPT = """\
You are the bug-finder CHECK_WRITER. Given a single pillar
(architectural invariant) the comprehender identified, write a
Python module that checks the invariant against the codebase. The
module becomes a permanent part of the bug_finder's audit suite for
this workspace — every future run executes it.

## Pillar input

You receive ONE pillar with:
  * ``name`` — short identifier (e.g. ``user_id_scoping``)
  * ``statement`` — the invariant as a sentence
  * ``evidence`` — file:line anchors where the invariant is upheld

Read those anchors first to understand the pattern; then write the
check.

## Module shape — what to produce

The module's text MUST satisfy this contract:

```python
\"\"\"Check for <pillar name>: <statement>.\"\"\"
from __future__ import annotations
import ast
from pathlib import Path

# Optional: skip directories that aren't your concern
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def run(root: Path) -> list[dict]:
    \"\"\"Return findings as a list of dicts with keys:
        severity (str: critical|high|medium|low|info)
        category (str: stable rule id e.g. "user_id_scoping_missing")
        file     (str: repo-relative path)
        line     (int)
        message  (str)
        fix      (str: optional suggested remediation)
    \"\"\"
    findings: list[dict] = []
    for py_file in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in py_file.parts):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        rel = str(py_file.relative_to(root)).replace("\\\\", "/")
        # walk the tree, identify violations, append to findings
        # ...
    return findings
```

## Discipline

- Use ONLY the Python stdlib (``ast``, ``re``, ``pathlib``).
  Custom checks must run without third-party deps.
- Make the check **idempotent** — running it twice on the same code
  must return the same findings.
- **Skip test files** unless the pillar is specifically about test
  hygiene.
- **Skip non-Python files** when the pillar is Python-specific. For
  other languages, write a grep-based check (still in Python).
- Quality over quantity: a 10-line module that catches the
  invariant cleanly beats a 200-line one that over-fires. False
  positives erode trust.

## Output

End your response with a single fenced Python block containing the
complete module source. The orchestrator extracts the source and
saves it as ``custom_checks/<pillar_name>.py``. No JSON wrapper —
just the Python code in a fenced block.

Don't include test cases or docstrings beyond what's natural. Don't
attempt to ``import`` the check at module load time. Don't print to
stdout.

If you cannot write a useful check for this pillar (the invariant
is too vague, too codebase-specific to a runtime state, etc.), emit
a fenced block containing only::

    # skipped: <one-line reason>

The orchestrator will skip persistence and the lead will know.
"""


CHECK_WRITER_USER_TEMPLATE = """\
Write a custom check for the pillar below.

**Pillar name:** `{pillar_name}`

**Statement:** {pillar_statement}

**Evidence anchors:** the comprehender flagged these locations as
upholding the invariant — read 1-2 of them to understand the
shape:

{evidence_block}

Output the module as a single fenced Python code block. Use only
stdlib. Make the check idempotent. Avoid false positives.
"""


# ---------------------------------------------------------------------------
# Parser — extract the Python source from the LLM output
# ---------------------------------------------------------------------------


_PY_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL,
)


def parse_check_source(output: str) -> str:
    """Return the Python source the LLM emitted, or ``""`` if none.

    Prefers a fenced ``python`` block. Falls back to a generic fenced
    block. Stripped of leading/trailing whitespace.
    """
    if not output:
        return ""
    blocks = [m.group(1).strip() for m in _PY_BLOCK_RE.finditer(output)]
    if not blocks:
        return ""
    # Last fenced block wins — same convention as the rest of the
    # subagent parsers in this codebase.
    return blocks[-1]


def _looks_like_skip(source: str) -> bool:
    """``True`` when the LLM emitted a "skipped: ..." sentinel."""
    head = source.strip().splitlines()
    if not head:
        return False
    first = head[0].strip()
    return first.startswith("# skipped:") or first == "# skipped"


def slug_for_pillar(pillar_name: str) -> str:
    """Filesystem-safe slug for a pillar name → ``custom_checks/<slug>.py``.

    Deterministic so the orchestrator can tell which pillars already
    have a check (compare against existing filenames) without storing
    a separate mapping.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", pillar_name).strip("_")[:50]
    return safe or "pillar_check"


# ---------------------------------------------------------------------------
# Safety guardrails — the generated module is EXECUTED by
# ``custom_check_runner`` (importlib + exec_module). Gating at
# write-time keeps unsafe code from ever reaching disk. The coder
# container is the runtime boundary; this is defense-in-depth.
# ---------------------------------------------------------------------------


# Stdlib modules a pure AST/grep check legitimately needs. Anything
# outside this set (subprocess, socket, os, sys, importlib, pickle,
# shutil, ctypes, requests, …) is rejected — those have no business in
# a read-only structural check and are the usual escape hatches.
_ALLOWED_IMPORT_ROOTS = frozenset({
    "ast", "re", "pathlib", "typing", "collections", "itertools",
    "functools", "dataclasses", "json", "string", "math", "enum",
    "textwrap", "__future__",
})

# Builtins that enable arbitrary code execution / sandbox escape when
# referenced by bare name. ``re.compile`` is an *attribute* (attr=
# "compile"), so banning the bare ``compile`` Name doesn't touch it.
_DANGEROUS_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "breakpoint",
})

# Attribute escapes — the classic ``().__class__.__bases__[0].
# __subclasses__()`` chain and friends.
_DANGEROUS_ATTRS = frozenset({
    "__globals__", "__builtins__", "__subclasses__", "__bases__",
    "__mro__", "__code__", "__class__", "__dict__",
})


def _import_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    if node.level:  # relative import — not allowed in a standalone check
        return ["<relative>"]
    return [(node.module or "").split(".")[0]]


def is_valid_check_source(source: str) -> tuple[bool, str]:
    """Validate the emitted source structurally AND for safety.

    Returns ``(ok, reason)``. ``ok=False`` means the module is unfit
    for persistence — the orchestrator must NOT save it. Rejects:
      * empty / skip-sentinel output
      * syntax errors
      * a missing top-level ``run`` function
      * imports outside the stdlib allowlist (blocks subprocess/socket/…)
      * import-time side effects (top-level loops, ``with``, bare calls)
      * code-execution builtins (eval/exec/__import__) and dunder
        attribute escapes anywhere in the tree
    """
    if not source.strip():
        return False, "empty source"
    if _looks_like_skip(source):
        return False, "skipped by check_writer"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"

    # 1. Top-level statements: imports, defs, classes, module constants,
    #    and a docstring only. A bare top-level call / loop / with-block
    #    would execute at import time — reject it.
    has_run = False
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            for root in _import_roots(node):
                if root not in _ALLOWED_IMPORT_ROOTS:
                    return False, f"disallowed import: {root or 'relative'}"
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name == "run":
                has_run = True
            continue
        if isinstance(node, ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign):
            continue  # module constant; import allowlist bounds the RHS
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring / bare literal
        if isinstance(node, ast.Pass):
            continue
        return False, f"disallowed top-level statement: {type(node).__name__}"

    if not has_run:
        return False, "module missing top-level `run` function"

    # 2. Dangerous-symbol scan over the entire tree.
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in _DANGEROUS_NAMES:
            return False, f"forbidden builtin: {n.id}"
        if isinstance(n, ast.Attribute) and n.attr in _DANGEROUS_ATTRS:
            return False, f"forbidden attribute: {n.attr}"

    return True, ""


# ---------------------------------------------------------------------------
# Subagent runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckWriterResult:
    """One generation pass for one pillar."""

    pillar_name: str
    source: str                       # the python source emitted
    written_to: Path | None           # where it was saved (None when skipped)
    subagent_result: SubagentResult
    runtime_seconds: float
    valid: bool
    skip_reason: str = ""


async def run_check_writer(
    *,
    workspace_root: Path,
    pillar_name: str,
    pillar_statement: str,
    evidence: tuple[str, ...],
    model: str,
    backend,
    tools: tuple = (),
    budget: SubagentBudget = DEFAULT_CHECK_WRITER_BUDGET,
    progress_callback=None,
) -> CheckWriterResult:
    """Generate one custom check, persist when valid.

    Returns a structured result regardless of outcome — invalid /
    skipped responses set ``valid=False`` and explain why.
    """
    evidence_block = "\n".join(
        f"- {ev}" for ev in evidence[:6]
    ) or "(no evidence anchors provided)"

    spec = SubagentSpec(
        role=Role.PLANNER.value,   # reuse planner role (same tool set / budget)
        model=model,
        system_prompt=CHECK_WRITER_SYSTEM_PROMPT,
        initial_user_message=CHECK_WRITER_USER_TEMPLATE.format(
            pillar_name=pillar_name,
            pillar_statement=pillar_statement,
            evidence_block=evidence_block,
        ),
        tools=tools,
        budget=budget,
        instance_id=f"check_writer_{pillar_name}",
        progress_callback=progress_callback,
        temperature=0.0,
    )
    start = time.monotonic()
    result = await run_subagent(spec, backend=backend)
    elapsed = time.monotonic() - start

    source = parse_check_source(result.output)
    valid, reason = is_valid_check_source(source)

    written_to: Path | None = None
    if valid:
        ensure_substrate(workspace_root)
        target = custom_checks_dir(workspace_root) / f"{slug_for_pillar(pillar_name)}.py"
        try:
            target.write_text(source, encoding="utf-8")
            written_to = target
            log.info(
                "bug_finder_custom_check_written",
                workspace=str(workspace_root),
                pillar=pillar_name, path=str(target),
                source_lines=source.count("\n") + 1,
            )
        except OSError as exc:
            log.warning(
                "bug_finder_custom_check_write_failed",
                pillar=pillar_name, error=str(exc),
            )
            valid = False
            reason = f"write failed: {exc}"

    return CheckWriterResult(
        pillar_name=pillar_name,
        source=source,
        written_to=written_to,
        subagent_result=result,
        runtime_seconds=elapsed,
        valid=valid,
        skip_reason=reason if not valid else "",
    )


def is_implemented() -> bool:
    return True
