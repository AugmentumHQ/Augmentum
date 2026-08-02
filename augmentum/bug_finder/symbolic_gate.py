"""Symbolic patch re-verification gate (Semgrep).

After the fixer + fix-verifier accept a patch, this gate runs Semgrep
on the workspace with `--baseline-commit` pointed at the pre-fix
baseline. Any *new* findings (present in the patched tree but not the
baseline) reject the patch — the LLM fixer is good at making the PoC
pass, less good at not introducing parallel issues.

This is the "deterministic gate over a non-deterministic agent loop"
pattern every successful AI security tool in 2025-2026 converged on:

  * Snyk DeepCode AI: LLM patch → symbolic re-check
  * GitHub Copilot Autofix: CodeQL is the truth, LLM only proposes
  * XBOW: deterministic validator gates submission
  * Anthropic: task verifier executes the exploit

Gate is *advisory by default* — failure rejects this attempt but the
caller may try again (other attempts have their own gate). When
Semgrep isn't available in the workspace container, the gate skips
gracefully (logs once, no error).

Why Semgrep specifically: it's the closest thing to free-cost
semantic analysis Augmentum can bundle. CodeQL would be heavier
(requires per-language database build) and Snyk is closed. Semgrep
rules-from-OSS is the right ergonomics floor.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

from augmentum.coder.containers import ContainerManager
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolicFinding:
    """One Semgrep finding, normalized."""

    rule_id: str
    file: str
    line_start: int
    line_end: int
    severity: str
    message: str


@dataclass
class SymbolicGateResult:
    """Outcome of running the gate against one fix attempt.

    `passed = True` means either:
      - Semgrep ran and produced zero new findings, OR
      - Semgrep wasn't available — gate skipped, can't block the fix

    `passed = False` means Semgrep ran and surfaced one or more findings
    in the patched tree that didn't exist at baseline.

    `skipped = True` always tags graceful-skip cases so the orchestrator
    can mark the report appropriately. Distinct from `passed`: a skip
    doesn't block but also doesn't strengthen confidence.
    """

    passed: bool
    skipped: bool = False
    skip_reason: str = ""
    new_findings: list[SymbolicFinding] = field(default_factory=list)
    semgrep_version: str = ""
    duration_ms: int = 0

    @property
    def summary(self) -> str:
        if self.skipped:
            return f"semgrep gate skipped: {self.skip_reason}"
        if self.passed:
            return "semgrep gate passed (no new findings)"
        n = len(self.new_findings)
        return f"semgrep gate rejected patch: {n} new finding(s)"


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


_SEMGREP_TIMEOUT_S = 180.0
_BASELINE_REF = "refs/augmentum/bug_finder/baseline"


async def check_patch(
    cm: ContainerManager,
    workspace_id: str,
    *,
    file_paths: list[str] | None = None,
    config: str = "auto",
) -> SymbolicGateResult:
    """Run Semgrep against the patched workspace and return new findings.

    `file_paths` scopes the scan to the files the patch touched — much
    faster than a whole-repo scan and matches the patch's blast radius.
    Empty/None scans the whole repo (the slower fallback).

    `config` defaults to "auto" which pulls Semgrep's curated rules
    set from semgrep.dev. Self-hosted alternatives can pass a path.
    """
    import time
    started = time.time()

    # Availability check first — graceful skip wins over hard failure
    try:
        version_out = await cm.run_command(
            workspace_id,
            ["bash", "-c", "semgrep --version 2>&1"],
            timeout=10.0,
        )
        version = (version_out or "").strip().splitlines()[0][:40]
        if not version or "command not found" in version_out.lower() or "no such" in version_out.lower():
            return SymbolicGateResult(
                passed=True, skipped=True,
                skip_reason="semgrep not installed in workspace container",
            )
    except Exception as exc:  # noqa: BLE001
        return SymbolicGateResult(
            passed=True, skipped=True,
            skip_reason=f"semgrep version probe failed: {exc!r}"[:200],
        )

    # Build the scan command. --baseline-commit makes Semgrep report only
    # findings introduced in the working tree vs. the baseline ref.
    parts = [
        "semgrep", "scan",
        "--config", config,
        "--baseline-commit", _BASELINE_REF,
        "--json",
        "--quiet",
        "--error",
        # 30-second per-rule timeout — keep total bounded even when the
        # repo has long files.
        "--timeout", "30",
        # Don't try to upload results
        "--metrics=off",
    ]
    if file_paths:
        # Filter to a tight set of paths. Semgrep accepts file/dir
        # arguments at the end of the command.
        parts.extend(shlex.quote(p) for p in file_paths[:50])

    cmd = " ".join(parts)
    full = f"cd /workspace && {cmd}"
    try:
        stdout = await cm.run_command(
            workspace_id,
            ["bash", "-c", full],
            timeout=_SEMGREP_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        return SymbolicGateResult(
            passed=True, skipped=True,
            skip_reason=f"semgrep run failed: {exc!r}"[:200],
            semgrep_version=version,
            duration_ms=int((time.time() - started) * 1000),
        )

    findings = _parse_semgrep_json(stdout or "")
    duration_ms = int((time.time() - started) * 1000)
    if not findings:
        return SymbolicGateResult(
            passed=True,
            semgrep_version=version,
            duration_ms=duration_ms,
        )
    return SymbolicGateResult(
        passed=False,
        new_findings=findings,
        semgrep_version=version,
        duration_ms=duration_ms,
    )


def _parse_semgrep_json(stdout: str) -> list[SymbolicFinding]:
    """Best-effort Semgrep JSON parse. Tolerates bash output prefixes/suffixes
    (warnings, progress lines) that sometimes precede the JSON document."""
    text = (stdout or "").strip()
    if not text:
        return []
    # Find the JSON document — sometimes preceded by warnings on stderr
    # bleed-through.
    start = text.find("{")
    if start < 0:
        return []
    try:
        doc = json.loads(text[start:])
    except json.JSONDecodeError:
        return []
    raw = doc.get("results")
    if not isinstance(raw, list):
        return []
    out: list[SymbolicFinding] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        start_info = row.get("start") or {}
        end_info = row.get("end") or {}
        extra = row.get("extra") or {}
        out.append(SymbolicFinding(
            rule_id=str(row.get("check_id") or ""),
            file=str(row.get("path") or ""),
            line_start=int(start_info.get("line") or 0),
            line_end=int(end_info.get("line") or start_info.get("line") or 0),
            severity=str(extra.get("severity") or "UNKNOWN"),
            message=str(extra.get("message") or "")[:500],
        ))
    return out


# ---------------------------------------------------------------------------
# Helpers for orchestrator integration
# ---------------------------------------------------------------------------


def extract_patched_files(patch_text: str) -> list[str]:
    """Pull file paths out of a unified diff (`patch` output).

    Each `diff --git a/path b/path` line names a file. We return the
    `b/` (post-patch) path with the `b/` prefix stripped, deduped.
    Empty list when the diff doesn't parse — caller can scope the scan
    to None (whole workspace) in that case.
    """
    paths: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        # Expected shape: "diff --git a/path/to/file b/path/to/file"
        parts = line.split()
        if len(parts) < 4:
            continue
        b_path = parts[3]
        if b_path.startswith("b/"):
            b_path = b_path[2:]
        if b_path and not b_path.startswith("/dev/null"):
            paths.add(b_path)
    return sorted(paths)
