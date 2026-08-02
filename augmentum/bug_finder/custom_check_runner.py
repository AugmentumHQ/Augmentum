"""Discover + execute LLM-generated custom checks.

The check_writer subagent saves AST checks to
``<workspace>/.augmentum/bug_finder/custom_checks/<name>.py``. This
module discovers them at audit-start time, loads each via
``importlib``, and calls each module's ``run(root: Path) -> list[dict]``.

Each check is sandboxed (its own module namespace, no inheritance
from prior loads). Errors in one check don't cascade — the loader
catches them and logs, then proceeds.

The output uses the same ``ScannerFinding`` shape as ``dev_tools``
and ``generic_scanners`` so the lead consumes custom + generic +
augmentum-dev findings uniformly.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

from augmentum.bug_finder.dev_tools import ScannerFinding, _make_rule_id
from augmentum.bug_finder.workspace_substrate import custom_checks_dir
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_VALID_SEVS = frozenset({"critical", "high", "medium", "low", "info"})


def _normalize_severity(value: object) -> str:
    sev = str(value or "medium").strip().lower()
    return sev if sev in _VALID_SEVS else "medium"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_custom_checks(workspace_root: Path) -> list[Path]:
    """Return every ``.py`` file in the workspace's custom_checks dir.

    Files starting with ``_`` are excluded — convention for
    suppressed-by-user checks ("rename to _foo.py to disable").
    """
    dir_ = custom_checks_dir(workspace_root)
    if not dir_.is_dir():
        return []
    return sorted(
        p for p in dir_.glob("*.py")
        if not p.name.startswith("_")
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomCheckRunResult:
    """One check's execution outcome."""

    check_name: str           # filename without .py
    path: Path
    succeeded: bool
    findings: tuple[ScannerFinding, ...] = ()
    error: str = ""
    duration_seconds: float = 0.0


def _run_one(check_path: Path, root: Path) -> CustomCheckRunResult:
    """Load + execute one check module."""
    check_name = check_path.stem
    start = time.monotonic()

    # SECURITY: re-validate the source at LOAD time, not just write time.
    # The check_writer gates what *it* persists, but this runner execs every
    # ``.py`` in the custom_checks dir regardless of provenance — a file
    # planted in a cloned/untrusted repo (or one that bypassed the writer)
    # would otherwise reach exec_module on the HOST process. Re-applying the
    # same AST allowlist here makes execution safe by construction: imports
    # are bounded to a stdlib read-only set, eval/exec/__import__ and dunder
    # escapes are rejected, and import-time side effects are disallowed.
    try:
        source = check_path.read_text(encoding="utf-8")
    except OSError as exc:
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error=f"read error: {type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - start,
        )
    # Lazy import: check_writer pulls in the agents.loop subagent stack, which
    # this lightweight runner has no other need for.
    from augmentum.bug_finder.check_writer import is_valid_check_source

    ok, reason = is_valid_check_source(source)
    if not ok:
        log.warning(
            "bug_finder_custom_check_rejected",
            check=check_name, path=str(check_path), reason=reason,
        )
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error=f"rejected by safety gate: {reason}",
            duration_seconds=time.monotonic() - start,
        )

    spec = importlib.util.spec_from_file_location(
        f"_bf_custom_check_{check_name}", str(check_path),
    )
    if spec is None or spec.loader is None:
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error="spec_from_file_location returned None",
            duration_seconds=time.monotonic() - start,
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — check author error
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error=f"import error: {type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - start,
        )

    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error="module has no top-level `run` function",
            duration_seconds=time.monotonic() - start,
        )
    try:
        raw = run_fn(root)
    except Exception as exc:  # noqa: BLE001
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error=f"run error: {type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - start,
        )
    if not isinstance(raw, list):
        return CustomCheckRunResult(
            check_name=check_name, path=check_path,
            succeeded=False,
            error="run() did not return a list",
            duration_seconds=time.monotonic() - start,
        )

    findings: list[ScannerFinding] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        file = str(r.get("file") or "").strip()
        try:
            line = int(r.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        category = str(
            r.get("category") or check_name,
        ).strip() or check_name
        findings.append(ScannerFinding(
            scanner=f"custom:{check_name}",
            severity=_normalize_severity(r.get("severity")),
            category=category,
            file=file,
            line=line,
            message=str(r.get("message") or "").strip(),
            fix=str(r.get("fix") or "").strip(),
            rule_id=_make_rule_id(
                f"custom:{check_name}", category, file, line,
            ),
        ))

    return CustomCheckRunResult(
        check_name=check_name, path=check_path,
        succeeded=True,
        findings=tuple(findings),
        duration_seconds=time.monotonic() - start,
    )


def run_all_custom_checks(
    workspace_root: Path,
    *,
    timeout_per_check: float = 60.0,
) -> list[CustomCheckRunResult]:
    """Discover + execute every custom check in the workspace.

    Returns one ``CustomCheckRunResult`` per check, including
    failures. Caller can flatten via ``.findings`` for the agent
    tool layer's consumption.

    ``timeout_per_check`` is advisory — the current synchronous
    implementation doesn't enforce it via process-level kill;
    individual checks should respect best-practice runtime bounds
    on their own (the check_writer prompt instructs this).
    """
    out: list[CustomCheckRunResult] = []
    for path in list_custom_checks(workspace_root):
        log.debug("bug_finder_custom_check_dispatch", path=str(path))
        result = _run_one(path, workspace_root)
        if result.succeeded:
            log.info(
                "bug_finder_custom_check_complete",
                check=result.check_name,
                findings=len(result.findings),
                seconds=round(result.duration_seconds, 2),
            )
        else:
            log.warning(
                "bug_finder_custom_check_failed",
                check=result.check_name,
                error=result.error,
                seconds=round(result.duration_seconds, 2),
            )
        out.append(result)
    return out


def collect_all_findings(
    workspace_root: Path,
) -> list[ScannerFinding]:
    """Flatten every custom check's findings into one list."""
    all_findings: list[ScannerFinding] = []
    for result in run_all_custom_checks(workspace_root):
        if result.succeeded:
            all_findings.extend(result.findings)
    return all_findings
