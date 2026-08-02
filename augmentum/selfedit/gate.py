"""The fitness gate for self-editing — "is this candidate good enough to ship?"

Self-improvement is only safe where outcomes are verifiable (the lesson from
DGM / SICA / AlphaEvolve), so the gate is the real foundation: nothing gets
promoted to the live tree unless it passes. This module is the GENERIC engine
(an ordered list of checks → a structured verdict) plus the STANDARD Augmentum
check set (compile, lint, targeted tests, smoke-import) so the whole app is
validated consistently no matter which surface a self-edit touched.

Design notes:
* Every check runs as a SUBPROCESS against the candidate DIRECTORY, so it tests
  the candidate's code on disk — not the modules already imported into the
  running server. That's what makes the verdict trustworthy.
* Checks are tool-availability-aware: a missing tool (e.g. ruff not installed)
  → ``skip`` with a note, never a false ``fail``. A required check that runs and
  fails → the gate fails. Skips and advisory failures don't sink the gate.
* Pure + injectable: ``run_gate`` takes Check objects, so tests can pass
  synthetic checks with zero subprocesses.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# A check's body: returns (status, detail) where status is "pass"|"fail"|"skip".
CheckRunner = Callable[[], Awaitable[tuple[str, str]]]

_PASS, _FAIL, _SKIP = "pass", "fail", "skip"


@dataclass
class Check:
    name: str
    run: CheckRunner
    required: bool = True  # a required check that FAILS sinks the gate


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    required: bool = True
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "detail": self.detail,
            "required": self.required, "duration_ms": self.duration_ms,
        }


@dataclass
class GateVerdict:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


async def run_gate(checks: list[Check]) -> GateVerdict:
    """Run checks in order, collect results, decide. The gate PASSES iff no
    *required* check failed (skips and advisory failures don't sink it)."""
    results: list[CheckResult] = []
    for c in checks:
        t0 = time.monotonic()
        try:
            status, detail = await c.run()
        except Exception as exc:  # noqa: BLE001 — a crashing check is a failed check
            status, detail = _FAIL, f"check crashed: {exc!r}"
        if status not in (_PASS, _FAIL, _SKIP):
            status, detail = _FAIL, f"check returned bad status {status!r}: {detail}"
        results.append(CheckResult(
            name=c.name, status=status, detail=(detail or "")[:4000],
            required=c.required, duration_ms=int((time.monotonic() - t0) * 1000),
        ))
    failed_required = [r for r in results if r.required and r.status == _FAIL]
    passed = not failed_required
    return GateVerdict(passed=passed, checks=results, summary=_summarize(results, passed))


def _summarize(results: list[CheckResult], passed: bool) -> str:
    n_pass = sum(1 for r in results if r.status == _PASS)
    n_fail = sum(1 for r in results if r.status == _FAIL)
    n_skip = sum(1 for r in results if r.status == _SKIP)
    head = "PASS" if passed else "FAIL"
    bad = ", ".join(r.name for r in results if r.required and r.status == _FAIL)
    tail = f" — failed: {bad}" if bad else ""
    return f"{head} ({n_pass} passed, {n_fail} failed, {n_skip} skipped){tail}"


# ---------------------------------------------------------------------------
# Subprocess runner + the standard Augmentum check set
# ---------------------------------------------------------------------------

async def _run(argv: list[str], *, cwd: str, timeout: float = 300.0) -> tuple[int, str]:
    """Run a subprocess; return (exit_code, combined_output). Never raises.
    Secret-scrubbed env — the gate runs the candidate's tests/compile, so we don't
    hand the app's credentials to code we're verifying (W11)."""
    from augmentum.selfedit.sandbox import scrubbed_env
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=scrubbed_env(),
        )
    except Exception as exc:  # noqa: BLE001 — tool missing / not executable
        return 127, f"could not launch {argv[0]}: {exc!r}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        return 124, f"timed out after {timeout:.0f}s"
    return (proc.returncode or 0), (out or b"").decode("utf-8", errors="replace")


def _resolve_tool(name: str) -> str | None:
    """Find an executable on PATH or alongside the running interpreter (handles
    venvs where Scripts/bin isn't on PATH — e.g. ruff in .venv/Scripts/)."""
    found = shutil.which(name)
    if found:
        return found
    d = os.path.dirname(sys.executable)
    for cand in (os.path.join(d, name), os.path.join(d, f"{name}.exe")):
        if os.path.exists(cand):
            return cand
    return None


def _module_available(module: str) -> bool:
    """Is an importable module present in the verify interpreter? The gate runs
    checks as ``sys.executable -m <module>`` subprocesses, and the subprocess
    shares this process's site-packages, so an in-process ``find_spec`` correctly
    predicts whether the subprocess will find it — with no extra process spawn."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Output signatures that mean the oracle couldn't RUN its own framework (an
# infrastructure problem) rather than the code genuinely failing — these become
# SKIP (inconclusive), never FAIL. A false FAIL here would reject good code and,
# worse, teach the archive that a working change "broke the app."
_INFRA_FAIL_SIGNATURES = ("No module named pytest", "No module named 'pytest'",
                          "could not launch")


def _is_infra_failure(code: int, out: str) -> bool:
    return code == 127 or any(sig in (out or "") for sig in _INFRA_FAIL_SIGNATURES)


def compile_check(target_dir: str) -> Check:
    """Byte-compile the tree — catches syntax errors anywhere. Required."""
    async def run():
        code, out = await _run([sys.executable, "-m", "compileall", "-q", target_dir], cwd=target_dir)
        return (_PASS, "") if code == 0 else (_FAIL, out[-2000:])
    return Check("compile", run, required=True)


def ruff_check(target_dir: str, *, required: bool = True) -> Check:
    """Lint with ruff. Skips (not fails) if ruff isn't installed."""
    ruff = _resolve_tool("ruff")
    async def run():
        if not ruff:
            return (_SKIP, "ruff not found")
        code, out = await _run([ruff, "check", target_dir], cwd=target_dir)
        return (_PASS, "") if code == 0 else (_FAIL, out[-2000:])
    return Check("ruff", run, required=required)


def pytest_check(test_paths: list[str], *, cwd: str, required: bool = True) -> Check:
    """Run a targeted pytest subset (-x, stop on first failure).

    Tool-availability-aware (like ``ruff_check``): if pytest isn't installed in
    the verify interpreter, SKIP — never a false FAIL. This is the confirm oracle
    for feature/bugfix intents; an unrunnable oracle must degrade to an honest
    coverage gap (→ human_required), not reject correct code as "broke the app.\""""
    async def run():
        if not test_paths:
            return (_SKIP, "no tests selected")
        if not _module_available("pytest"):
            return (_SKIP, "pytest not available in the verify environment — "
                           "feature confirmation unavailable (install pytest to enable)")
        code, out = await _run([sys.executable, "-m", "pytest", "-q", "-x", *test_paths], cwd=cwd, timeout=600.0)
        if code == 0:
            return (_PASS, out[-1500:])
        # An infrastructure error (interpreter/tooling couldn't run) is inconclusive,
        # not a real test failure — SKIP so it becomes a coverage gap, not a reject.
        if _is_infra_failure(code, out):
            return (_SKIP, f"pytest could not run (infrastructure): {out[-1500:]}")
        return (_FAIL, out[-3000:])
    return Check("pytest", run, required=required)


def smoke_import_check(module: str, *, cwd: str, required: bool = True) -> Check:
    """Prove the candidate can import a module (cheap boot sanity). A genuine
    ImportError is a REAL signal (the module is broken) → FAIL; only a failure to
    launch the interpreter itself (infrastructure) → SKIP."""
    async def run():
        code, out = await _run([sys.executable, "-c", f"import {module}"], cwd=cwd)
        if code == 0:
            return (_PASS, "")
        if code == 127 or "could not launch" in (out or ""):
            return (_SKIP, f"interpreter could not launch (infrastructure): {out[-1000:]}")
        return (_FAIL, out[-2000:])
    return Check(f"import:{module}", run, required=required)


def default_app_gate(
    target_dir: str, *, test_paths: list[str] | None = None,
    smoke_modules: list[str] | None = None,
) -> list[Check]:
    """The standard Augmentum validation set — one consistent gate for the whole
    app. ``target_dir`` is the candidate root (e.g. a worktree path); paths are
    relative to it. Compile + lint always; tests + smoke-imports when supplied."""
    checks: list[Check] = [
        compile_check(os.path.join(target_dir, "augmentum")),
        ruff_check(os.path.join(target_dir, "augmentum")),
    ]
    for mod in smoke_modules or []:
        checks.append(smoke_import_check(mod, cwd=target_dir))
    if test_paths:
        checks.append(pytest_check(test_paths, cwd=target_dir))
    return checks
