"""Codebase-agnostic deterministic scanners.

Unlike ``dev_tools.py`` (which wraps Augmentum's bespoke
augmentum-dev scripts), this module integrates **published, generic**
analysis tools that work on any Python codebase:

* **Bandit** — security-focused linter. Detects SQL injection
  patterns, hardcoded credentials, weak crypto, ``assert`` in
  production, etc. Mature project with ~50 plugin checks. Output:
  structured JSON.
* **Ruff** — fast Python linter with hundreds of rule packs
  (pyflakes / pycodestyle / pyupgrade / flake8-bugbear / pylint).
  Output: structured JSON.

Both run as ``python -m <tool>`` so they're available wherever the
augmentum process is — no separate binary install, no PATH lookup,
no platform-specific shims. Each wrapper returns the same
``ScannerFinding`` shape ``dev_tools.py`` uses, so the agent layer
can mix Augmentum-specific and generic scanners transparently.

This is the "de-augmentumification" layer — the bug_finder substrate
works on any Python codebase the moment we run these scanners
against it. No augmentum-dev convention required.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape — reuses the same record type dev_tools uses
# ---------------------------------------------------------------------------


from augmentum.bug_finder.dev_tools import (  # re-export for caller convenience
    ScannerFinding,
    _make_rule_id,
)

_VALID_SEVS = frozenset({"critical", "high", "medium", "low", "info"})


def _normalize_severity(value: object) -> str:
    sev = str(value or "medium").strip().lower()
    return sev if sev in _VALID_SEVS else "medium"


# ---------------------------------------------------------------------------
# Bandit — security scanner
# ---------------------------------------------------------------------------


# Bandit's severity → our scale
_BANDIT_SEV_MAP = {
    "HIGH":   "high",
    "MEDIUM": "medium",
    "LOW":    "low",
}


def run_bandit(
    root: Path,
    *,
    skip_dirs: tuple[str, ...] = (
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        "tests", "test", "dist", "build",
    ),
    timeout: float = 300.0,
) -> list[ScannerFinding]:
    """Run bandit against ``root`` and return structured findings.

    Bandit's defaults are conservative; we don't pass ``--severity-level
    LOW`` because bandit's "LOW" tier is often informational noise.
    Tests are skipped by default — bug_finder's job is auditing
    production code, not test fixtures.

    Skip paths are converted to bandit's expected absolute-prefix form
    (``<root>/<dir>``) so substrings in the absolute path above ``root``
    don't accidentally exclude target files. This matters under
    pytest's ``tmp_path`` (which contains ``test_`` in its parent).
    """
    if not root.is_dir():
        return []
    # Bandit's --exclude takes comma-separated paths; the matcher uses
    # substring containment against the full file path. Anchoring each
    # skip to ``<root>/<dir>`` avoids accidental matches from path
    # ancestors that contain "test", ".venv", etc.
    skip_anchored = [
        str(root.joinpath(d)).replace("\\", "/")
        for d in skip_dirs
    ]
    skip_arg = ",".join(skip_anchored)
    cmd = [
        sys.executable, "-m", "bandit",
        "-r", str(root),
        "-f", "json",
        "--quiet",
        "-x", skip_arg,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        log.warning("bug_finder_bandit_timeout", root=str(root))
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "bug_finder_bandit_failed", root=str(root), error=str(exc),
        )
        return []

    # Bandit returns non-zero exit code when findings present — that's
    # NOT an error. JSON is on stdout regardless.
    raw_text = result.stdout or ""
    try:
        payload = json.loads(raw_text)
    except (ValueError, TypeError):
        log.warning(
            "bug_finder_bandit_parse_failed",
            root=str(root),
            output_preview=raw_text[:200],
        )
        return []

    findings: list[ScannerFinding] = []
    for raw in payload.get("results") or []:
        if not isinstance(raw, dict):
            continue
        bandit_sev = str(raw.get("issue_severity") or "MEDIUM").upper()
        severity = _BANDIT_SEV_MAP.get(bandit_sev, "medium")
        file = str(raw.get("filename") or "").strip()
        try:
            line = int(raw.get("line_number") or 0)
        except (TypeError, ValueError):
            line = 0
        category = str(raw.get("test_id") or "uncategorized").strip()
        # Make ``file`` relative to root for stable suppression matching
        if file and root in Path(file).parents:
            try:
                file = str(Path(file).relative_to(root)).replace("\\", "/")
            except ValueError:
                pass
        message = str(raw.get("issue_text") or "").strip()
        # Bandit's "test_name" gives a human-readable rule name
        rule_name = str(raw.get("test_name") or "").strip()
        if rule_name:
            message = f"[{rule_name}] {message}"
        findings.append(ScannerFinding(
            scanner="bandit",
            severity=severity,
            category=category,
            file=file,
            line=line,
            message=message,
            fix="",   # Bandit doesn't emit fix hints
            rule_id=_make_rule_id("bandit", category, file, line),
        ))
    return findings


# ---------------------------------------------------------------------------
# Ruff — fast lint with hundreds of rules
# ---------------------------------------------------------------------------


# Ruff rules don't have severity tiers; we map by rule-prefix family.
_RUFF_SEVERITY_BY_PREFIX = (
    # Security / correctness packs land as medium-to-high
    ("S",    "high"),     # flake8-bandit (security)
    ("B",    "medium"),   # flake8-bugbear (likely bugs)
    ("BLE",  "medium"),   # flake8-blind-except
    ("E",    "low"),      # pycodestyle errors
    ("F",    "medium"),   # pyflakes
    ("RUF",  "low"),      # ruff-specific
    ("PERF", "info"),     # performance hints
    ("UP",   "info"),     # pyupgrade
    ("PL",   "low"),      # pylint
    ("C9",   "low"),      # mccabe complexity
)


def _ruff_severity_for_rule(rule_code: str) -> str:
    """Bucket the rule code into a severity tier."""
    if not rule_code:
        return "low"
    # Sort by prefix length desc so the most specific prefix wins
    for prefix, sev in sorted(
        _RUFF_SEVERITY_BY_PREFIX, key=lambda kv: -len(kv[0]),
    ):
        if rule_code.startswith(prefix):
            return sev
    return "low"


_RUFF_RULE_SELECT = (
    # Reasonable defaults for "find genuinely-problematic patterns"
    # across any Python codebase, without flooding the agent with
    # style nits. Pulled from ruff's own recommended pack list.
    "F",      # pyflakes (real errors)
    "E",      # pycodestyle errors (syntax + indentation)
    "B",      # flake8-bugbear (likely bugs)
    "BLE",    # blind except
    "S",      # security (flake8-bandit)
    "RUF",    # ruff-specific extra
)


_RUFF_DEFAULT_EXCLUDES = (
    "tests", "test", "testing",
    "node_modules", "dist", "build",
    ".venv", "venv", "__pycache__",
    "site-packages",
    "docs", "examples",
)


def run_ruff(
    root: Path,
    *,
    rule_packs: tuple[str, ...] = _RUFF_RULE_SELECT,
    exclude_dirs: tuple[str, ...] = _RUFF_DEFAULT_EXCLUDES,
    timeout: float = 120.0,
) -> list[ScannerFinding]:
    """Run ruff with a curated rule-pack selection against ``root``.

    ``rule_packs`` selects which Ruff rule prefixes to enable. Default
    is the "find real bugs, skip style" subset: F (pyflakes), E
    (errors), B (bugbear), BLE (blind except), S (security), RUF
    (ruff extras). Override to broaden (add PL for pylint) or narrow.

    ``exclude_dirs`` mirrors bandit's skip list so production-vs-test
    distinctions are consistent across both scanners. Without these
    exclusions, S101 (``assert`` in code) floods the output with
    test-file noise that masks real findings.
    """
    if not root.is_dir():
        return []
    selector = ",".join(rule_packs)
    cmd = [
        sys.executable, "-m", "ruff", "check",
        "--select", selector,
        "--output-format", "json",
        "--exit-zero",                     # don't error-out on findings
        "--no-fix",                        # don't auto-apply suggestions
    ]
    for d in exclude_dirs:
        cmd += ["--exclude", d]
    cmd.append(str(root))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        log.warning("bug_finder_ruff_timeout", root=str(root))
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "bug_finder_ruff_failed", root=str(root), error=str(exc),
        )
        return []

    raw_text = result.stdout or ""
    if not raw_text.strip():
        return []
    try:
        payload = json.loads(raw_text)
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, list):
        return []

    findings: list[ScannerFinding] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        rule_code = str(raw.get("code") or "").strip()
        # Bandit category mapping — keep the original rule code as the
        # category so suppression matching can pin on rule, not message.
        category = rule_code or "uncategorized"
        file = str(raw.get("filename") or "").strip()
        if file and root in Path(file).parents:
            try:
                file = str(Path(file).relative_to(root)).replace("\\", "/")
            except ValueError:
                pass
        loc = raw.get("location") or {}
        try:
            line = int((loc or {}).get("row") or 0)
        except (TypeError, ValueError):
            line = 0
        message_parts = [
            rule_code,
            str(raw.get("message") or "").strip(),
        ]
        message = " — ".join(p for p in message_parts if p)
        fix = ""
        fix_info = raw.get("fix") or {}
        if isinstance(fix_info, dict):
            fix = str(fix_info.get("message") or "").strip()
        findings.append(ScannerFinding(
            scanner="ruff",
            severity=_ruff_severity_for_rule(rule_code),
            category=category,
            file=file,
            line=line,
            message=message,
            fix=fix,
            rule_id=_make_rule_id("ruff", category, file, line),
        ))
    return findings


# ---------------------------------------------------------------------------
# Convenience: run every generic scanner and merge
# ---------------------------------------------------------------------------


def run_generic_suite(root: Path) -> dict[str, list[ScannerFinding]]:
    """Run every generic scanner. Returns ``{slug: findings}``."""
    return {
        "bandit": run_bandit(root),
        "ruff":   run_ruff(root),
    }


@dataclass(frozen=True)
class GenericScannerSuiteResult:
    """Aggregate result from one full generic-scanner sweep."""

    findings_by_scanner: dict[str, list[ScannerFinding]]
    wallclock_seconds: float

    @property
    def total_findings(self) -> int:
        return sum(
            len(rows) for rows in self.findings_by_scanner.values()
        )


def run_generic_suite_timed(root: Path) -> GenericScannerSuiteResult:
    """Like ``run_generic_suite`` but also returns timing info."""
    start = time.monotonic()
    results = run_generic_suite(root)
    elapsed = time.monotonic() - start
    return GenericScannerSuiteResult(
        findings_by_scanner=results,
        wallclock_seconds=elapsed,
    )
