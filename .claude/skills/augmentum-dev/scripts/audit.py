#!/usr/bin/env python3
"""Augmentum project health audit.

A full-scope project-health system. Bundles every individual scanner,
plus dependency-CVE scanning, stale-exception detection, doc-fact
verification, optional runtime smoke + test-execution, computes a
0-100 health score, surfaces cross-tool hotspots and a per-subsystem
breakdown, and appends to a trend history so regression / improvement
is visible across sessions.

Usage
-----
    python scripts/audit.py                       # default: full health pass + delta
    python scripts/audit.py --quiet               # summary only
    python scripts/audit.py --verbose             # also show subsystem breakdown
    python scripts/audit.py --smoke               # also verify imports + migrations apply
    python scripts/audit.py --with-contracts      # also probe every GET route in-process
    python scripts/audit.py --with-tests          # also actually run pytest (slow)
    python scripts/audit.py --skip-deps           # skip pip-audit (faster, less complete)
    python scripts/audit.py --trend N             # show last N runs from history
    python scripts/audit.py --format=json         # machine-readable output
    python scripts/audit.py --format=markdown     # PR-comment friendly output
    python scripts/audit.py --update-baseline     # commit current state as baseline
    python scripts/audit.py --no-history          # skip history append (useful for dry runs)

Exit codes
----------
    0   no regressions vs baseline (and smoke + tests passed if requested)
    1   regression detected, OR smoke / test failure when those modes
        are enabled
    2   a sub-tool failed to run (script error, not a project-health
        finding)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Make our OWN stdout/stderr UTF-8 so any scanner's output (a "✓", a "→"
# in a finding) can't crash the whole audit on a cp1252 Windows console.
# (Child processes already inherit PYTHONIOENCODING=utf-8; this covers
# the parent's own print() calls.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — older Python / non-reconfigurable stream
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
REFS_DIR = SCRIPTS_DIR.parent / "references"
BASELINE_FILE = REFS_DIR / "audit_baseline.json"
HISTORY_FILE = REFS_DIR / "audit_history.jsonl"
EXCEPTIONS_FILE = REFS_DIR / "security_exceptions.json"


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


ROOT = _find_root()


def _prepend_project_venv_site_packages() -> None:
    """Make the project's ``.venv`` site-packages importable.

    The registry audit imports ``augmentum.registry`` -> ``pydantic``. When
    audit.py is invoked with a bare ``python`` from PATH, that's often the
    system Python which doesn't have the project deps installed — the
    registry check then fails with ``No module named 'pydantic'`` and
    counts as a regression. Surfacing the venv on sys.path lets the check
    use the deps the project actually pins, without forcing the user to
    pip-install pydantic system-wide or rewrite their invocation to use
    ``.venv/bin/python``.

    No-op if no ``.venv`` exists or its site-packages can't be located —
    the registry block will still fail loudly, which is the intended
    design (see ``_check_registry`` docstring).
    """
    venv = ROOT / ".venv"
    if not venv.is_dir():
        return
    # Windows layout: .venv/Lib/site-packages
    # Unix layout:    .venv/lib/python3.X/site-packages
    candidates = [venv / "Lib" / "site-packages"]
    lib = venv / "lib"
    if lib.is_dir():
        for child in lib.iterdir():
            if child.name.startswith("python") and (child / "site-packages").is_dir():
                candidates.append(child / "site-packages")
    for sp in candidates:
        if sp.is_dir() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
            return


_prepend_project_venv_site_packages()


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    return sys.stdout.isatty() and shutil.get_terminal_size().columns > 0


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _red(t: str) -> str: return _c("91", t)
def _green(t: str) -> str: return _c("92", t)
def _yellow(t: str) -> str: return _c("93", t)
def _cyan(t: str) -> str: return _c("96", t)
def _bold(t: str) -> str: return _c("1", t)
def _dim(t: str) -> str: return _c("2", t)


# ---------------------------------------------------------------------------
# Per-tool runner + output parsers
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    script: str
    parser: callable  # text -> dict[str, int]
    file_extractor: callable | None = None  # text -> list[str] of file paths


def _extract_files(text: str) -> list[str]:
    """Pull augmentum/ or ui/ file paths out of a tool's stdout."""
    paths: set[str] = set()
    pattern = re.compile(r"((?:augmentum|ui|tests|scripts|docs)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+)")
    for m in pattern.finditer(text):
        paths.add(m.group(1).replace("\\", "/"))
    return sorted(paths)


def _parse_validate_wiring(text: str) -> dict[str, int]:
    # Tolerate both ``42 errors, 1 warnings`` (legacy) and ``42 error(s),
    # 1 warning(s)`` (current). validate_wiring.py drifted to the
    # parenthesised form to match runtime_checks; the parser must
    # accept either or it silently emits zero metrics.
    m = re.search(r"(\d+)\s+errors?(?:\(s\))?,\s+(\d+)\s+warnings?(?:\(s\))?", text)
    if not m:
        return {}
    return {"errors": int(m.group(1)), "warnings": int(m.group(2))}


def _parse_dead_code(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    m = re.search(r"Orphaned Endpoints \((\d+)\)", text)
    if m:
        out["orphaned_endpoints"] = int(m.group(1))
    m = re.search(r"Ghost Calls \((\d+)\)", text)
    out["ghost_calls"] = int(m.group(1)) if m else 0
    m = re.search(r"Dependency Drift \((\d+)\)", text)
    if m:
        out["dependency_drift"] = int(m.group(1))
    return out


def _parse_code_quality(text: str) -> dict[str, int]:
    # Scope to the Summary block — the detail lines above echo phrases
    # like 'Mixed error patterns: 75x JSONResponse(error)' that the
    # broad regex misread as the count.
    sm = re.search(r"Summary[\s\S]*", text)
    text = sm.group(0) if sm else text
    out: dict[str, int] = {}
    for label, key in [
        ("Missing CSS classes", "missing_css"),
        ("Dead CSS classes", "dead_css"),
        ("Silent JS catches", "silent_catches"),
        ("WS contract gaps", "ws_gaps"),
        ("Mixed error patterns", "mixed_errors"),
        ("Console.log", "console_log"),
        ("Tech debt markers", "tech_debt"),
        ("_model_map misuse", "model_map_misuse"),
    ]:
        m = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
        if m:
            out[key] = int(m.group(1))
    return out


def _parse_runtime_checks(text: str) -> dict[str, int]:
    # runtime_checks.py prints either "0 errors, 209 warning(s)" or
    # "5 error(s), 87 warning(s)" depending on count. Accept both forms,
    # mirroring _parse_wiring above — otherwise the audit silently logs
    # a parser failure when the codebase reaches zero errors.
    m = re.search(r"(\d+)\s+errors?(?:\(s\))?,\s+(\d+)\s+warnings?(?:\(s\))?", text)
    if not m:
        return {}
    return {"errors": int(m.group(1)), "warnings": int(m.group(2))}


def _parse_db_safety(text: str) -> dict[str, int]:
    # db_safety.py closes with the same "N error(s), M warning(s)" line.
    m = re.search(r"(\d+)\s+errors?(?:\(s\))?,\s+(\d+)\s+warnings?(?:\(s\))?", text)
    if not m:
        return {}
    return {"errors": int(m.group(1)), "warnings": int(m.group(2))}


def _parse_async_blocking(text: str) -> dict[str, int]:
    # async_blocking.py closes with the same "N error(s), M warning(s)" line.
    m = re.search(r"(\d+)\s+errors?(?:\(s\))?,\s+(\d+)\s+warnings?(?:\(s\))?", text)
    if not m:
        return {}
    return {"errors": int(m.group(1)), "warnings": int(m.group(2))}


def _parse_db_contention(text: str) -> dict[str, int]:
    """Parse db_contention.py's live-log scanner output.

    Reports zero-counted metrics even on skip so the audit history
    keeps a consistent shape (otherwise the trend chart has gaps every
    time the scanner ran without a live container). The skip case is
    distinguishable later via ``skipped=1``.
    """
    out: dict[str, int] = {"locked": 0, "slow_begin": 0, "skipped": 0}
    if "Skipped —" in text:
        out["skipped"] = 1
        return out
    m = re.search(r"database is locked errors:\s+(\d+)", text)
    if m:
        out["locked"] = int(m.group(1))
    m = re.search(r"slow BEGIN events:\s+(\d+)", text)
    if m:
        out["slow_begin"] = int(m.group(1))
    return out


def _parse_security(text: str) -> dict[str, int]:
    """Parse the security_check.py headline.

    The summary line has the shape ``N new finding(s): X critical,
    Y high, Z medium, W low`` but tiers with zero findings are omitted
    by the printer (so ``no critical`` simply doesn't appear). Parse
    each tier independently; default missing ones to 0 so the audit
    sees a complete metrics dict.
    """
    out: dict[str, int] = {}
    # Clean-state shortcut: the scanner prints "No new security findings."
    # with no total line. Treat that as zeros across the board so the audit
    # doesn't false-fail when security has nothing to report (mirrors the
    # red_team parser's clean-state path).
    if "No new security findings" in text:
        return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    m = re.search(r"(\d+)\s+new finding\(s\)", text)
    if not m:
        return out
    out["total"] = int(m.group(1))
    for tier in ("critical", "high", "medium", "low"):
        tm = re.search(rf"(\d+)\s+{tier}\b", text)
        out[tier] = int(tm.group(1)) if tm else 0
    return out


def _parse_test_coverage(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    m = re.search(r"Modules:\s+(\d+)/(\d+)", text)
    if m:
        out["modules_covered"] = int(m.group(1))
        out["modules_total"] = int(m.group(2))
    m = re.search(r"Routes:\s+(\d+)/(\d+)", text)
    if m:
        out["routes_covered"] = int(m.group(1))
        out["routes_total"] = int(m.group(2))
    m = re.search(r"(\d+)\s+coverage gaps", text)
    if m:
        out["coverage_gaps"] = int(m.group(1))
    return out


def _parse_red_team(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    # Clean-state shortcut: the scanner prints "No findings. Attack surface
    # looks clean." with no total line. Treat that as zeros across the board
    # so the audit doesn't false-fail when red_team has nothing to report.
    if "No findings" in text or "Attack surface looks clean" in text:
        return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    m = re.search(r"(\d+)\s+total findings", text)
    if m:
        out["total"] = int(m.group(1))
    for label, key in [
        ("CRITICAL", "critical"),
        ("HIGH", "high"),
        ("MEDIUM", "medium"),
        ("LOW", "low"),
    ]:
        m = re.search(rf"{label}\s+\((\d+)\)", text)
        if m:
            out[key] = int(m.group(1))
    return out


TOOLS: list[Tool] = [
    Tool("wiring",       "validate_wiring.py",  _parse_validate_wiring,   _extract_files),
    Tool("dead_code",    "dead_code.py",        _parse_dead_code,         _extract_files),
    Tool("code_quality", "code_quality.py",     _parse_code_quality,      _extract_files),
    Tool("runtime",      "runtime_checks.py",   _parse_runtime_checks,    _extract_files),
    Tool("security",     "security_check.py",   _parse_security,          _extract_files),
    Tool("coverage",     "test_coverage.py",    _parse_test_coverage,     None),
    Tool("red_team",     "red_team_scan.py",    _parse_red_team,          _extract_files),
    Tool("db_safety",    "db_safety.py",        _parse_db_safety,         _extract_files),
    Tool("db_contention","db_contention.py",    _parse_db_contention,     _extract_files),
    Tool("async_blocking","async_blocking.py",  _parse_async_blocking,    _extract_files),
]


# Higher = better (so "regression" means the value DROPPED).
_HIGHER_IS_BETTER = {
    ("coverage", "modules_covered"),
    ("coverage", "routes_covered"),
}


# ---------------------------------------------------------------------------
# Health score weights
# ---------------------------------------------------------------------------
#
# Each metric contributes (value * weight) points of deduction from the
# 100-point ceiling. Weights are calibrated so the current baseline scores
# in the 60-80 range — high enough to leave room for improvement, low
# enough that catastrophic regressions visibly tank the score.
#
# Severity intuition:
#   5.0 → one finding ≈ -5 points (CRITICAL security, blocking wiring error)
#   1.0 → one finding ≈ -1 point  (HIGH security, ghost call, ws gap)
#   0.5 → one finding ≈ -0.5      (MEDIUM-ish, runtime errors)
#   0.1 → bulk noise, lots of them allowed
#   0.05 → very noisy bulk metrics (silent catches, console.log)

_WEIGHTS: dict[tuple[str, str], float] = {
    # Wiring contract — errors are blocking, warnings are bulk
    ("wiring", "errors"):                5.0,
    ("wiring", "warnings"):              0.02,
    # Dead code / drift
    ("dead_code", "orphaned_endpoints"): 0.02,
    ("dead_code", "ghost_calls"):        1.0,    # silent UI bug — real
    ("dead_code", "dependency_drift"):   0.3,
    # Code quality (mostly bulk metrics — small per-finding weight)
    ("code_quality", "missing_css"):     0.01,
    ("code_quality", "dead_css"):        0.0005,
    ("code_quality", "silent_catches"):  0.003,
    ("code_quality", "ws_gaps"):         0.3,
    ("code_quality", "mixed_errors"):    0.2,
    ("code_quality", "console_log"):     0.01,
    ("code_quality", "tech_debt"):       0.1,
    # Each ``_model_map misuse`` is a real silent regression — the
    # 2026-05-26 NIM bug pattern. Weighted similar to ws_gaps because
    # the failure mode (model "not served" despite being resolved) is
    # equally user-visible. Easy fix per site (one OR-clause).
    ("code_quality", "model_map_misuse"): 0.3,
    # Runtime patterns
    ("runtime", "errors"):               0.05,
    ("runtime", "warnings"):             0.01,
    # Security (weights calibrated assuming exceptions catch known false
    # positives — counts here are "scanner flags that have not been
    # exception-listed yet", which include real and false positives)
    ("security", "critical"):            0.5,
    ("security", "high"):                0.3,
    ("security", "medium"):              0.1,
    ("security", "low"):                 0.02,
    # Coverage gap = an untested module
    ("coverage", "coverage_gaps"):       0.05,
    # Red team (similar calibration to security)
    ("red_team", "critical"):            0.3,
    ("red_team", "high"):                0.3,
    ("red_team", "medium"):              0.1,
    ("red_team", "low"):                 0.02,
    # DB safety — a non-WAL state-layer connection is an error (real corruption
    # risk); the rest are warnings (legacy / needs-review patterns)
    ("db_safety", "errors"):             1.0,
    ("db_safety", "warnings"):           0.05,
    # Async event-loop blocking — an unambiguous loop blocker (time.sleep /
    # requests.* / subprocess.run on the loop) is an error; spawn-and-return
    # and heuristic sync-embedding findings are warnings (review-and-offload).
    ("async_blocking", "errors"):        1.0,
    ("async_blocking", "warnings"):      0.1,
    # Smoke — one failure tanks the score
    ("smoke", "smoke_failures"):         25.0,
    # Contract probe (--with-contracts) — an in-code crash on a GET route is a
    # real regression; an authz flip (protected route answered without creds)
    # is a security break, weighted like a critical. Both ride the baseline/
    # delta machine, so mock-induced first-run noise becomes the baseline and
    # only NEW breaks move the score.
    ("contracts", "in_code_crash"):      1.0,
    ("contracts", "hard_block"):         5.0,
    ("contracts", "new_hang"):           0.3,
    # Dependency CVEs (each one is a real vuln)
    ("deps", "vulnerabilities"):         1.0,
    # Audit-infrastructure rot
    ("exceptions", "stale_entries"):     0.3,
    ("exceptions", "parse_failed"):      5.0,    # the file itself is broken
    # Documentation accuracy (each verifiable claim that's wrong)
    ("doc_facts", "doc_inaccuracies"):   0.5,
    # Documentation coverage — a code⟷doc SET drift (subsystem with no
    # Map row, mode with no Handler row, provider with no card, …). Light
    # nudge (informational): the baseline records the accepted floor, so
    # only NEW gaps move the number. Any doc_coverage.*_undocumented
    # metric not listed here is auto-weighted 0.2 in _compute_score, so a
    # new CoverageSpec is scored without touching this table.
    ("doc_coverage", "subsystems_undocumented"):    0.2,
    ("doc_coverage", "modes_undocumented"):         0.2,
    ("doc_coverage", "provider_cards_undocumented"): 0.2,
    # Test execution (when --with-tests is on)
    ("tests", "failed"):                 1.0,
    ("tests", "errors"):                 1.0,
    ("tests", "timed_out"):              10.0,
}


def _compute_score(metrics: dict[str, dict[str, int]]) -> float:
    """Derive a 0-100 health score from weighted findings."""
    deduction = 0.0
    for (tool, metric), weight in _WEIGHTS.items():
        value = metrics.get(tool, {}).get(metric, 0)
        deduction += value * weight
    # Auto-weight any doc_coverage spec metric not explicitly listed above
    # (so a new CoverageSpec is scored without editing _WEIGHTS).
    for metric, value in metrics.get("doc_coverage", {}).items():
        if ("doc_coverage", metric) not in _WEIGHTS and metric.endswith("_undocumented"):
            deduction += value * 0.2
    return max(0.0, round(100.0 - deduction, 1))


def _score_label(score: float) -> str:
    if score >= 90:
        return "EXCELLENT"
    if score >= 75:
        return "GOOD"
    if score >= 60:
        return "FAIR"
    if score >= 40:
        return "DEGRADED"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# Children inherit a Windows console codepage (cp1252) for stdout by default,
# so a scanner that prints ✓ / → / … raises UnicodeEncodeError *before* it
# reaches its summary line — the audit then reports "parser found no metrics"
# and silently drops that scanner's findings (inflating the score). Force the
# subprocess to emit UTF-8 and decode it leniently here.
_CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _run_tool(tool: Tool) -> tuple[dict[str, int], list[str], str | None]:
    script = SCRIPTS_DIR / tool.script
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=_CHILD_ENV,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {}, [], "timed out after 120s"
    except Exception as exc:
        return {}, [], f"failed to launch: {exc}"

    text = _strip_ansi((proc.stdout or "") + "\n" + (proc.stderr or ""))
    metrics = tool.parser(text)
    files = tool.file_extractor(text) if tool.file_extractor else []
    if not metrics:
        return {}, files, "parser found no metrics in output"
    return metrics, files, None


# ---------------------------------------------------------------------------
# Smoke check (--smoke)
# ---------------------------------------------------------------------------


def _check_registry() -> tuple[dict[str, int], list[str]]:
    """Run the declarative-action-substrate audit + drift checks.

    Spec: docs/superpowers/specs/2026-06-04-declarative-action-substrate-design.md

    Returns:
        registered:  number of Settings registered in the canonical registry.
        findings:    metadata-completeness failures (label/desc/section).
        drift:       drift between registry declarations and the historical
                     4 declaration sites (config.py / _TOOL_SETTINGS / etc).
    """
    issues: list[str] = []
    metrics: dict[str, int] = {}
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from augmentum.registry import audit as registry_audit  # noqa: PLC0415
        from augmentum.registry import verify as registry_verify  # noqa: PLC0415
        from augmentum.registry.builtin import (  # noqa: PLC0415
            load_into_default_registry,
        )
        from augmentum.registry.registry import (  # noqa: PLC0415
            _reset_for_tests,
            get_registry,
        )

        _reset_for_tests()
        load_into_default_registry()
        r = get_registry()
        metrics["registered"] = len(r.list_all())
        audit_findings = registry_audit.check_all()
        metrics["findings"] = len(audit_findings)
        drift_findings = registry_verify.check_all()
        metrics["drift"] = len(drift_findings)
        metrics["check_failed"] = 0  # Sentinel: 0 on success, 1 on import-fail.
        for f in audit_findings[:10]:
            issues.append(
                f"audit: {f['key']}: {f['rule']} — {f['message']}"
            )
        for f in drift_findings[:10]:
            issues.append(
                f"drift: {f['key']}: {f['rule']} — {f['message']}"
            )
    except Exception as exc:  # noqa: BLE001
        # Surface as a regression rather than silently skipping. Otherwise
        # an env that can't import pydantic (host dev shell without it
        # installed, missing venv, etc.) silently loses the drift check —
        # exactly the catch that would have caught settings-registry-
        # without-Pydantic-field bugs. The metric stays at 1 until the
        # env is fixed or the run moves to a container where pydantic is
        # available.
        return {"check_failed": 1}, [f"registry audit failed: {exc}"]
    return metrics, issues


def _run_smoke() -> tuple[dict[str, int], list[str]]:
    """Verify the app still imports and migrations apply on a fresh DB.

    Returns (metrics, error_messages). metrics contains
    ``smoke_failures`` (count of failed sub-checks).
    """
    failures: list[str] = []

    # 1. Import server module — covers the whole route + provider import graph.
    proc = subprocess.run(
        [sys.executable, "-c",
         "from augmentum.proxy.server import create_app; print('OK')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0 or "OK" not in proc.stdout:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        last = msg[-1] if msg else "unknown error"
        failures.append(f"import server: {last[:200]}")

    # 2. Apply every migration on a fresh in-memory DB.
    code = (
        "import asyncio\n"
        "async def go():\n"
        "    from augmentum.state.backends.sqlite import SQLiteBackend\n"
        "    b = SQLiteBackend(':memory:')\n"
        "    await b.connect()\n"
        "    await b.close()\n"
        "    print('OK')\n"
        "asyncio.run(go())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0 or "OK" not in proc.stdout:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        last = msg[-1] if msg else "unknown error"
        failures.append(f"migrations apply: {last[:200]}")

    return {"smoke_failures": len(failures)}, failures


def _run_contracts() -> tuple[dict[str, int], list[str]]:
    """Probe every GET route in-process and report in-code crashes + authz
    flips. A strict superset of --smoke: --smoke proves the app imports +
    migrations apply; this actually exercises each GET handler's request-parse
    + wiring path (catching e.g. a dangling lazy import a static smoke misses).

    Runs contract_test.py as a subprocess (the in-process app build is heavy)
    and reads its JSON from a temp file — the script's stdout carries app log
    noise, so the file is the clean channel.
    """
    import tempfile

    script = SCRIPTS_DIR / "contract_test.py"
    if not script.exists():
        return {}, ["contract_test.py not found"]
    out = Path(tempfile.gettempdir()) / "augmentum_contracts.json"
    try:
        out.unlink()
    except OSError:
        pass
    try:
        subprocess.run(
            [sys.executable, str(script), "--quiet", f"--out={out}"],
            cwd=ROOT, capture_output=True, text=True, timeout=300, env=_CHILD_ENV,
        )
    except subprocess.TimeoutExpired:
        return {}, ["contract_test timed out after 300s"]
    except Exception as exc:  # noqa: BLE001
        return {}, [f"contract_test failed to launch: {exc}"]
    if not out.exists():
        return {}, ["contract_test produced no result file"]
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {}, [f"contract_test output not valid JSON: {exc}"]

    # Score only NEW breaks (not in contracts_baseline.json) so recorded
    # harness noise doesn't tank the score — run contract_test.py
    # --update-baseline once to record the known-broken floor.
    metrics = {
        "in_code_crash": int(data.get("new_regression", data.get("regression", 0))),
        "hard_block": int(data.get("new_hard_block", data.get("hard_block", 0))),
        "new_hang": int(data.get("new_hang", 0)),
    }
    issues = [
        f"{'[NEW] ' if f.get('new') else ''}{f.get('route')}: "
        f"{f.get('exception') or f.get('status')} @ {f.get('locus') or f.get('handler')}"
        for f in data.get("findings", [])
        if f.get("severity") in ("regression", "hard_block") and f.get("new")
    ]
    return metrics, issues


# ---------------------------------------------------------------------------
# Stale-exception detector
# ---------------------------------------------------------------------------


def _check_stale_exceptions() -> tuple[dict[str, int], list[str]]:
    """Find security_exceptions.json entries that reference files no longer
    in the repo. Catches audit-infrastructure rot.
    """
    if not EXCEPTIONS_FILE.exists():
        return {"stale_entries": 0, "parse_failed": 0}, []
    try:
        data = json.loads(EXCEPTIONS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return ({"stale_entries": 0, "parse_failed": 1},
                [f"security_exceptions.json failed to parse: {exc}"])

    stale = 0
    issues: list[str] = []
    for exc in data.get("exceptions", []):
        files = exc.get("files", [])
        for f in files:
            if f == "*":
                continue
            target = ROOT / f
            if not target.exists():
                stale += 1
                issues.append(f"{exc.get('id', '?')}: references missing file {f}")
    return {"stale_entries": stale, "parse_failed": 0}, issues


# ---------------------------------------------------------------------------
# Doc-fact verifier
# ---------------------------------------------------------------------------


def _highest_migration() -> int:
    migrations_dir = ROOT / "augmentum" / "state" / "migrations"
    if not migrations_dir.exists():
        return 0
    nums: list[int] = []
    for f in migrations_dir.glob("*.sql"):
        try:
            nums.append(int(f.stem.split("_")[0]))
        except (ValueError, IndexError):
            pass
    return max(nums) if nums else 0


def _count_user_scoped_tables() -> int:
    """Tables that have a user_id column anywhere in their migration history."""
    migrations_dir = ROOT / "augmentum" / "state" / "migrations"
    if not migrations_dir.exists():
        return 0
    tables: set[str] = set()
    create_pat = re.compile(
        r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)\s*\([^;]*\buser_id\b",
        re.IGNORECASE | re.DOTALL,
    )
    alter_pat = re.compile(
        r"ALTER TABLE\s+(\w+)\s+ADD COLUMN\s+user_id",
        re.IGNORECASE,
    )
    for f in migrations_dir.glob("*.sql"):
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in create_pat.finditer(text):
            tables.add(m.group(1).lower())
        for m in alter_pat.finditer(text):
            tables.add(m.group(1).lower())
    return len(tables)


def _check_doc_facts() -> tuple[dict[str, int], list[str]]:
    """Verify specific countable claims in CLAUDE.md / SKILL.md against reality.

    Currently checks:
      - SKILL.md "highest as of this doc: **N**"  vs actual highest
        migration filename number
      - CLAUDE.md "User-scoped tables (N): "      vs actual count of
        tables with user_id columns (CREATE or ALTER)

    See also ``_check_doc_facts_via_model`` for the Phase 0 model-backed
    implementation (enabled with ``--use-model``). This regex path stays
    as the default for backward compatibility while the model surface
    expands.
    """
    issues: list[str] = []

    skill_md = SCRIPTS_DIR.parent / "SKILL.md"
    if skill_md.exists():
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"highest as of this doc:\s*\*\*(\d+)\*\*", text)
        if m:
            claimed = int(m.group(1))
            actual = _highest_migration()
            if claimed != actual:
                issues.append(
                    f"SKILL.md migration number: claims {claimed}, actual is {actual}"
                )

    claude_md = ROOT / "CLAUDE.md"
    if claude_md.exists():
        text = claude_md.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"[Uu]ser-scoped tables?\s*\((\d+)\)", text)
        if m:
            claimed = int(m.group(1))
            actual = _count_user_scoped_tables()
            # Tolerate small drift (3) — exact-match is fragile because
            # tables can be added without altering the doc immediately.
            if abs(claimed - actual) > 3:
                issues.append(
                    f"CLAUDE.md user-scoped table count: claims {claimed}, "
                    f"migrations show {actual}"
                )

    return {"doc_inaccuracies": len(issues)}, issues


def _check_doc_facts_via_model() -> tuple[dict[str, int], list[str]]:
    """Phase 0 model-backed doc-fact verifier.

    Walks every <!--fact:NAME-->...<!--/--> block in the tracked docs
    (CLAUDE.md, SKILL.md) and verifies the embedded value matches the
    current rendered fact from ``facts.registry.FACTS``.

    Returns the same shape as the legacy regex check so audit.py
    bookkeeping is identical. Falls back to legacy if the model
    package can't be imported (e.g. cache dir wiped, sqlite missing).
    """
    issues: list[str] = []
    try:
        # Add skill dir to sys.path so model/ + facts/ are importable.
        skill_dir = SCRIPTS_DIR.parent
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        from facts import FACTS, render_fact  # noqa: PLC0415
        from model import open_model, refresh  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"doc_inaccuracies": -1}, [
            f"model unavailable, --use-model falling back to no-op: {exc}"
        ]

    db = open_model(ROOT)
    refresh(db, ROOT)

    fact_block_re = re.compile(r"<!--fact:([\w.]+)-->(.*?)<!--/-->", re.DOTALL)
    for rel in ("CLAUDE.md", ".claude/skills/augmentum-dev/SKILL.md"):
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in fact_block_re.finditer(text):
            name = m.group(1)
            claimed = m.group(2)
            if name not in FACTS:
                issues.append(f"{rel}: unknown fact name {name!r} (no FACTS entry)")
                continue
            current = render_fact(db, name)
            if current.strip() != claimed.strip():
                claim_short = claimed if len(claimed) < 60 else claimed[:57] + "..."
                cur_short = current if len(current) < 60 else current[:57] + "..."
                issues.append(
                    f"{rel}: fact {name!r} claims {claim_short!r}, current is {cur_short!r}"
                )
    return {"doc_inaccuracies": len(issues)}, issues


def _check_doc_coverage() -> tuple[dict[str, int], list[str]]:
    """Diff every registered code⟷doc SET (coverage/specs.py) and flag
    keys the docs don't declare.

    The autonomy hook that keeps the docs honest as the codebase grows:
    when a new subsystem package / dispatch mode / provider lands, it
    surfaces here on the next audit instead of the doc silently rotting.
    Emits one ``<spec>_undocumented`` metric per spec so each list is
    tracked (and baselined) independently. Adding a new tracked list is
    a single ``CoverageSpec`` in coverage/specs.py — no audit edit.
    """
    issues: list[str] = []
    try:
        skill_dir = SCRIPTS_DIR.parent
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        from doc_coverage import SPECS, evaluate  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {}, [f"coverage engine unavailable: {exc}"]

    metrics: dict[str, int] = {}
    for spec in SPECS:
        res = evaluate(spec, ROOT)
        metrics[f"{spec.name}_undocumented"] = len(res.missing)
        for item in res.missing:
            issues.append(f"[{spec.name}] {item} - fix: {spec.fix_location}")
    return metrics, issues


def _run_model_queries() -> tuple[dict[str, int], list[str]]:
    """Run every registered codebase-model query and emit diagnosis.

    Returns ({query_name: count}, [human-readable diagnosis lines]).
    Lines are prefixed with the query name + a trailing ``  diag:``
    marker so they sort with their parent count when displayed.

    Phase 1 ships one query (``orphaned_endpoints``); Phases 2+ append
    incomplete_settings, dead_css, etc.
    """
    metrics: dict[str, int] = {}
    diagnoses: list[str] = []
    try:
        skill_dir = SCRIPTS_DIR.parent
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        from model import open_model, refresh  # noqa: PLC0415
        from queries import ALL_QUERIES  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {}, [f"model unavailable for query layer: {exc}"]

    db = open_model(ROOT)
    refresh(db, ROOT)

    for mod in ALL_QUERIES:
        rows = list(db.execute(mod.QUERY))
        count = len(rows)
        metrics[mod.NAME] = count
        if count == 0:
            continue
        if not hasattr(mod, "DIAGNOSE"):
            continue
        diag_rows = list(db.execute(mod.DIAGNOSE))
        if not diag_rows:
            continue
        for r in diag_rows:
            cells = []
            sample = ""
            # sqlite3.Row iteration yields values, not keys.
            for k in r.keys():  # noqa: SIM118
                v = r[k]
                if k.startswith("sample"):
                    txt = str(v) if v else ""
                    sample = (txt[:97] + "...") if len(txt) > 100 else txt
                    continue
                cells.append(f"{k}={v}")
            line = f"{mod.NAME}: " + "  ".join(cells)
            if sample:
                line += f"  e.g. {sample}"
            diagnoses.append(line)

    return metrics, diagnoses


# ---------------------------------------------------------------------------
# Dependency CVE scan (pip-audit)
# ---------------------------------------------------------------------------


def _check_deps() -> tuple[dict[str, int], list[str]]:
    """Run pip-audit if available, count vulnerabilities.

    Gracefully handles pip-audit not being installed (returns no metrics
    but a friendly hint). Skips dev-only deps via --no-deps default.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format", "json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        return {}, ["pip-audit not installed (pip install pip-audit)"]
    except subprocess.TimeoutExpired:
        return {}, ["pip-audit timed out after 120s"]
    except Exception as exc:
        return {}, [f"pip-audit failed: {exc}"]

    # pip-audit exits 1 when vulns are found — that's expected, not an error.
    if proc.returncode not in (0, 1):
        first_err = (proc.stderr or "").strip().splitlines()
        hint = first_err[-1] if first_err else "unknown"
        # "No module named pip_audit" → not installed
        if "No module named" in hint:
            return {}, ["pip-audit not installed (pip install pip-audit)"]
        return {}, [f"pip-audit error: {hint[:200]}"]

    text = proc.stdout.strip()
    if not text:
        return {"vulnerabilities": 0}, []
    try:
        data = json.loads(text)
    except Exception:
        return {}, ["pip-audit output not valid JSON"]

    vulns = 0
    affected: list[str] = []
    # Normalise the two output shapes pip-audit has shipped over time.
    iter_source: list = []
    if isinstance(data, list):
        iter_source = data
    elif isinstance(data, dict):
        iter_source = data.get("dependencies", [])
    for entry in iter_source:
        if not isinstance(entry, dict):
            continue
        vs = entry.get("vulns", []) or []
        if vs:
            name = entry.get("name", "?")
            ver = entry.get("version", "?")
            ids = ", ".join(v.get("id", "?") for v in vs[:3])
            affected.append(f"{name} {ver}: {ids}")
            vulns += len(vs)
    return {"vulnerabilities": vulns}, affected


# ---------------------------------------------------------------------------
# Test execution (--with-tests)
# ---------------------------------------------------------------------------


def _run_tests() -> tuple[dict[str, int], list[str]]:
    """Run pytest on the offline unit-test set; count pass / fail / error.

    Excludes ``tests/live/`` (requires running services) and eval matrices.
    Caps at 10 minutes; a timeout marks the run as a regression on its own.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/",
             "-q", "--no-header", "-p", "no:cacheprovider",
             "--ignore=tests/live", "--ignore=tests/eval_results",
             "-m", "not live and not slow",
             "--maxfail=50"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        return {}, ["pytest not installed"]
    except subprocess.TimeoutExpired:
        return ({"timed_out": 1, "passed": 0, "failed": 0, "errors": 0},
                ["pytest exceeded 600s — likely a hang or runaway test"])
    except Exception as exc:
        return {}, [f"pytest failed to launch: {exc}"]

    text = _strip_ansi(proc.stdout + "\n" + proc.stderr)
    passed = failed = errors = 0
    m = re.search(r"(\d+)\s+passed", text)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", text)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+)\s+error", text)
    if m:
        errors = int(m.group(1))

    metrics = {"passed": passed, "failed": failed, "errors": errors, "timed_out": 0}
    issues: list[str] = []
    if failed or errors:
        issues.append(
            f"pytest: {failed} failed, {errors} errors, {passed} passed"
        )
    return metrics, issues


# ---------------------------------------------------------------------------
# Per-subsystem breakdown
# ---------------------------------------------------------------------------


def _subsystem_for(path: str) -> str:
    """Map an in-repo path to a subsystem bucket."""
    parts = path.replace("\\", "/").split("/")
    if not parts:
        return "(unknown)"
    if parts[0] == "augmentum" and len(parts) >= 3:
        return f"augmentum/{parts[1]}/{parts[2].split('.')[0]}" if parts[1] in {"proxy", "modes"} else f"augmentum/{parts[1]}"
    if parts[0] == "augmentum" and len(parts) >= 2:
        return f"augmentum/{parts[1]}"
    if parts[0] in {"ui", "tests", "scripts", "docs"} and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def _subsystem_breakdown(file_lists: dict[str, list[str]]) -> list[tuple[str, int, list[str]]]:
    """Aggregate flag counts per subsystem.

    Returns a list of (subsystem, finding_count, contributing_tools)
    sorted worst-first.
    """
    counts: dict[str, int] = defaultdict(int)
    tools: dict[str, set[str]] = defaultdict(set)
    for tool, files in file_lists.items():
        for f in files:
            sub = _subsystem_for(f)
            counts[sub] += 1
            tools[sub].add(tool)
    rows = [(sub, n, sorted(tools[sub])) for sub, n in counts.items()]
    rows.sort(key=lambda r: -r[1])
    return rows


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def _append_history(entry: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_history(n: int) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Baseline + delta
# ---------------------------------------------------------------------------


def _load_baseline() -> dict[str, dict[str, int]]:
    if not BASELINE_FILE.exists():
        return {}
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _delta_summary(
    baseline: dict[str, dict[str, int]],
    current: dict[str, dict[str, int]],
) -> tuple[list[str], list[str], list[str], int]:
    """Return (regressions, improvements, unchanged, regression_count).

    Regressions whose source scanner files were modified since the
    baseline timestamp are tagged ``[SELF-CHANGE]`` and excluded from
    ``regression_count`` — they're metric-definition changes, not
    codebase regressions. Closes the broken-zero-baseline class of
    bug we hit when fixing the wiring parser.
    """
    regressions: list[str] = []
    improvements: list[str] = []
    unchanged: list[str] = []
    regression_count = 0

    # Causality lookup is best-effort — if the import fails (e.g.
    # bundled into a stripped release without the full skill tree),
    # fall back to flat regression counting so the audit still works.
    explain_regression = None
    baseline_mtime = 0.0
    try:
        skill_dir = SCRIPTS_DIR.parent
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        from causality import explain_regression as _explain  # noqa: PLC0415
        explain_regression = _explain
        if BASELINE_FILE.exists():
            baseline_mtime = BASELINE_FILE.stat().st_mtime
    except Exception:  # noqa: BLE001
        pass

    for tool_name, metrics in current.items():
        for metric, value in metrics.items():
            base_value = baseline.get(tool_name, {}).get(metric)
            if base_value is None:
                unchanged.append(f"{tool_name}.{metric}={value} (new)")
                continue
            delta = value - base_value
            if delta == 0:
                unchanged.append(f"{tool_name}.{metric}={value}")
                continue
            higher_is_better = (tool_name, metric) in _HIGHER_IS_BETTER
            improved = (delta > 0) if higher_is_better else (delta < 0)
            arrow = ("^" if delta > 0 else "v")
            line = f"{tool_name}.{metric}: {base_value} -> {value}  ({arrow}{abs(delta)})"
            if improved:
                improvements.append(line)
                continue

            # Regression — check whether the scanner itself changed
            # since the baseline.
            self_change = False
            if explain_regression is not None and baseline_mtime > 0:
                try:
                    prov = explain_regression(
                        ROOT, tool_name, metric, since_ts=baseline_mtime,
                    )
                    self_change = prov.self_changed
                    if self_change:
                        files_csv = ", ".join(prov.files_changed[:3])
                        if len(prov.files_changed) > 3:
                            files_csv += f" (+{len(prov.files_changed) - 3} more)"
                        line += f"  [SELF-CHANGE: {files_csv}]"
                except Exception:  # noqa: BLE001
                    pass

            regressions.append(line)
            if not self_change:
                regression_count += 1

    return regressions, improvements, unchanged, regression_count


# ---------------------------------------------------------------------------
# Cross-tool correlation (files appearing in multiple tools' outputs)
# ---------------------------------------------------------------------------


def _correlate(file_lists: dict[str, list[str]], threshold: int = 2) -> list[tuple[str, list[str]]]:
    """Return files that appear in >= ``threshold`` tools' outputs."""
    counts: dict[str, list[str]] = defaultdict(list)
    for tool_name, files in file_lists.items():
        for f in files:
            counts[f].append(tool_name)
    hot = [(path, sorted(set(tools)))
           for path, tools in counts.items()
           if len(set(tools)) >= threshold]
    hot.sort(key=lambda x: (-len(x[1]), x[0]))
    return hot


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


def _format_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _format_markdown(payload: dict) -> str:
    score = payload["score"]
    label = payload["score_label"]
    lines = [
        f"## Augmentum project audit",
        "",
        f"**Health score: {score}/100 ({label})**",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for tool, metrics in payload["metrics"].items():
        for k, v in metrics.items():
            lines.append(f"| `{tool}.{k}` | {v} |")
    if payload.get("regressions"):
        lines += ["", "### Regressions vs baseline", ""]
        for r in payload["regressions"]:
            lines.append(f"- {r}")
    if payload.get("improvements"):
        lines += ["", "### Improvements vs baseline", ""]
        for i in payload["improvements"]:
            lines.append(f"- {i}")
    if payload.get("hot_files"):
        lines += ["", "### Cross-tool hotspots", ""]
        for path, tools in payload["hot_files"][:10]:
            lines.append(f"- `{path}` (flagged by: {', '.join(tools)})")
    if payload.get("smoke_errors"):
        lines += ["", "### Smoke failures", ""]
        for err in payload["smoke_errors"]:
            lines.append(f"- {err}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trend mode
# ---------------------------------------------------------------------------


def _show_trend(n: int) -> int:
    history = _read_history(n)
    if not history:
        print(_yellow(f"No history yet. Run audit a few times to populate {HISTORY_FILE.relative_to(ROOT)}."))
        return 0
    print(_bold(f"Trend: last {len(history)} runs"))
    print()
    # Header
    print(f"  {'when':<20}  {'score':>6}  {'sec_crit':>8}  {'sec_high':>8}  {'ghost':>5}  {'wire_err':>8}  {'cov_gap':>7}")
    for entry in history:
        ts = entry.get("timestamp", "?")
        score = entry.get("score", 0)
        m = entry.get("metrics", {})
        sec_crit = m.get("security", {}).get("critical", 0)
        sec_high = m.get("security", {}).get("high", 0)
        ghost = m.get("dead_code", {}).get("ghost_calls", 0)
        wire_err = m.get("wiring", {}).get("errors", 0)
        cov_gap = m.get("coverage", {}).get("coverage_gaps", 0)
        print(f"  {ts:<20}  {score:>6.1f}  {sec_crit:>8}  {sec_high:>8}  {ghost:>5}  {wire_err:>8}  {cov_gap:>7}")
    if len(history) >= 2:
        print()
        first, last = history[0], history[-1]
        first_score = first.get("score", 0)
        last_score = last.get("score", 0)
        delta = last_score - first_score
        sign = "+" if delta >= 0 else ""
        col = _green if delta >= 0 else _red
        print(col(f"  net change: {sign}{round(delta, 1)} points over {len(history)} runs"))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_format(args: list[str]) -> str:
    for a in args:
        if a.startswith("--format="):
            return a.split("=", 1)[1]
    return "text"


def _parse_int_arg(args: list[str], flag: str, default: int) -> int:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                pass
    return default


def main() -> int:
    args = sys.argv[1:]

    if "--trend" in args:
        n = _parse_int_arg(args, "--trend", 10)
        return _show_trend(n)

    update_baseline = "--update-baseline" in args
    quiet = "--quiet" in args
    verbose = "--verbose" in args
    smoke = "--smoke" in args
    with_contracts = "--with-contracts" in args
    with_tests = "--with-tests" in args
    skip_deps = "--skip-deps" in args
    no_history = "--no-history" in args
    use_model = "--use-model" in args
    record_fix_event = "--record-fix-event" in args
    fmt = _parse_format(args)

    is_text = fmt == "text"

    if is_text:
        print(_bold("Augmentum project audit"))
        print(_dim(f"  root: {ROOT}"))
        print(_dim(f"  baseline: {BASELINE_FILE.relative_to(ROOT) if BASELINE_FILE.exists() else '(none yet)'}"))
        if smoke:
            print(_dim("  smoke: enabled"))
        print()

    current: dict[str, dict[str, int]] = {}
    file_lists: dict[str, list[str]] = {}
    failures: list[str] = []

    for tool in TOOLS:
        if is_text and not quiet:
            print(_cyan(f"  -> running {tool.name} ({tool.script})"))
        metrics, files, err = _run_tool(tool)
        if err:
            failures.append(f"{tool.name}: {err}")
            continue
        current[tool.name] = metrics
        file_lists[tool.name] = files
        if is_text and not quiet and metrics:
            for k, v in metrics.items():
                print(_dim(f"      {k}={v}"))

    smoke_errors: list[str] = []
    if smoke:
        if is_text and not quiet:
            print(_cyan("  -> running smoke (import + migrations)"))
        smoke_metrics, smoke_errors = _run_smoke()
        current["smoke"] = smoke_metrics
        if is_text and not quiet:
            for err in smoke_errors:
                print(_red(f"      smoke error: {err}"))
            print(_dim(f"      smoke_failures={smoke_metrics['smoke_failures']}"))

    if with_contracts:
        if is_text and not quiet:
            print(_cyan("  -> running contracts (in-process GET probe of every route)"))
        contract_metrics, contract_issues = _run_contracts()
        current["contracts"] = contract_metrics
        if is_text and not quiet:
            for line in contract_issues[:15]:
                print(_yellow(f"      contract: {line}"))
            for k, v in contract_metrics.items():
                print(_dim(f"      {k}={v}"))

    # Audit-infrastructure rot
    if is_text and not quiet:
        print(_cyan("  -> running exceptions (security_exceptions.json validation)"))
    exc_metrics, exc_issues = _check_stale_exceptions()
    current["exceptions"] = exc_metrics
    if is_text and not quiet:
        for line in exc_issues:
            print(_yellow(f"      stale: {line}"))
        for k, v in exc_metrics.items():
            print(_dim(f"      {k}={v}"))

    # Doc-fact verifier
    if is_text and not quiet:
        suffix = " [model]" if use_model else ""
        print(_cyan(f"  -> running doc_facts (CLAUDE.md / SKILL.md verifiable claims){suffix}"))
    if use_model:
        doc_metrics, doc_issues = _check_doc_facts_via_model()
    else:
        doc_metrics, doc_issues = _check_doc_facts()
    current["doc_facts"] = doc_metrics

    # Doc-coverage: subsystem packages missing a Subsystem Map row. The
    # autonomy hook — new packages surface here instead of rotting.
    if is_text and not quiet:
        print(_cyan("  -> running doc_coverage (code<->doc set drift: subsystems / modes / providers)"))
    cov_metrics, cov_issues = _check_doc_coverage()
    current["doc_coverage"] = cov_metrics
    if is_text and not quiet:
        for line in cov_issues:
            print(_yellow(f"      undocumented: {line}"))
        for k, v in cov_metrics.items():
            print(_dim(f"      {k}={v}"))

    # Codebase-model queries — Phase 1+ scanners as SQL.
    # Reports counts as ``model.<query_name>`` metrics; diagnosis lines
    # surface inline so dashboards see WHERE the debt is, not just how
    # much. Skipped entirely in legacy mode.
    if use_model:
        if is_text and not quiet:
            print(_cyan("  -> running model queries (Phase 1+ diagnostics)"))
        model_metrics, model_diagnoses = _run_model_queries()
        current["model"] = model_metrics
        if is_text and not quiet:
            for k, v in model_metrics.items():
                print(_dim(f"      {k}={v}"))
            for line in model_diagnoses:
                print(_dim(f"      diag: {line}"))
    if is_text and not quiet:
        for line in doc_issues:
            print(_yellow(f"      doc: {line}"))
        for k, v in doc_metrics.items():
            print(_dim(f"      {k}={v}"))

    # Declarative action substrate — registry metadata + drift.
    if is_text and not quiet:
        print(_cyan("  -> running registry (declarative-action-substrate)"))
    reg_metrics, reg_issues = _check_registry()
    if reg_metrics:
        current["registry"] = reg_metrics
    if is_text and not quiet:
        for line in reg_issues[:10]:
            print(_yellow(f"      {line}"))
        for k, v in reg_metrics.items():
            print(_dim(f"      {k}={v}"))

    # Dependency CVE scan (default-on; opt-out for speed)
    deps_issues: list[str] = []
    if not skip_deps:
        if is_text and not quiet:
            print(_cyan("  -> running deps (pip-audit)"))
        deps_metrics, deps_issues = _check_deps()
        if deps_metrics:
            current["deps"] = deps_metrics
        if is_text and not quiet:
            for line in deps_issues[:5]:
                print(_yellow(f"      cve: {line}"))
            if deps_metrics:
                for k, v in deps_metrics.items():
                    print(_dim(f"      {k}={v}"))
            else:
                # No metrics — pip-audit unavailable. Friendly hint, not a failure.
                print(_dim(f"      (deps check skipped: {(deps_issues[0] if deps_issues else 'unavailable')})"))

    # Test execution (opt-in; slow)
    test_issues: list[str] = []
    if with_tests:
        if is_text and not quiet:
            print(_cyan("  -> running tests (pytest, offline subset, up to 600s)"))
        test_metrics, test_issues = _run_tests()
        if test_metrics:
            current["tests"] = test_metrics
        if is_text and not quiet:
            for line in test_issues:
                print(_yellow(f"      tests: {line}"))
            if test_metrics:
                for k, v in test_metrics.items():
                    print(_dim(f"      {k}={v}"))

    if is_text:
        print()

    if failures and is_text:
        print(_red(_bold("Tool failures:")))
        for f in failures:
            print(_red(f"  ! {f}"))
        print()

    score = _compute_score(current)
    label = _score_label(score)

    baseline = _load_baseline()
    regressions: list[str] = []
    improvements: list[str] = []
    reg_count = 0
    if baseline:
        regressions, improvements, _u, reg_count = _delta_summary(baseline, current)

    hot_files = _correlate(file_lists, threshold=2)

    # Always append to history except when explicitly disabled or doing
    # a baseline update (those runs are calibration, not measurements).
    if not no_history and not update_baseline:
        _append_history({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": score,
            "score_label": label,
            "metrics": current,
            "regression_count": reg_count,
        })

    # Phase 3: opt-in fix-event recording into the model. Off by default
    # so CI runs don't pollute pattern-mining history; on for interactive
    # use ("did my fix actually move the needle?").
    if record_fix_event and baseline:
        try:
            skill_dir = SCRIPTS_DIR.parent
            if str(skill_dir) not in sys.path:
                sys.path.insert(0, str(skill_dir))
            from fix_log import record_fix_event as _record  # noqa: PLC0415
            from model import open_model  # noqa: PLC0415

            deltas: dict[str, tuple[int, int, int]] = {}
            for tool_name, metrics in current.items():
                for metric, value in metrics.items():
                    base = baseline.get(tool_name, {}).get(metric)
                    if base is None or value == base:
                        continue
                    deltas[f"{tool_name}.{metric}"] = (base, value, value - base)
            db = open_model(ROOT)
            _record(db, ROOT, deltas=deltas)
            if is_text and not quiet:
                print(_dim(f"  fix_event recorded ({len(deltas)} metric delta(s))"))
        except Exception as exc:  # noqa: BLE001
            if is_text and not quiet:
                print(_yellow(f"  fix_event recording failed: {exc}"))

    subsystem_rows = _subsystem_breakdown(file_lists)
    payload = {
        "score": score,
        "score_label": label,
        "metrics": current,
        "regressions": regressions,
        "improvements": improvements,
        "hot_files": [(p, t) for p, t in hot_files[:10]],
        "subsystems": [{"name": s, "flags": n, "tools": t} for s, n, t in subsystem_rows[:15]],
        "smoke_errors": smoke_errors,
        "deps_issues": deps_issues,
        "test_issues": test_issues,
        "doc_issues": doc_issues,
        "exception_issues": exc_issues,
        "tool_failures": failures,
    }

    test_failed = with_tests and current.get("tests", {}).get("failed", 0) + current.get("tests", {}).get("errors", 0) + current.get("tests", {}).get("timed_out", 0) > 0

    if fmt == "json":
        print(_format_json(payload))
        return _exit_code(reg_count, smoke, smoke_errors, failures, test_failed)
    if fmt == "markdown":
        print(_format_markdown(payload))
        return _exit_code(reg_count, smoke, smoke_errors, failures, test_failed)

    # ---- text format ----

    # Score
    score_color = _green if score >= 75 else (_yellow if score >= 60 else _red)
    print(_bold(f"Health score: {score_color(f'{score}/100 ({label})')}"))
    print()

    if not baseline and not update_baseline:
        print(_yellow("No baseline yet -- run with --update-baseline to record one."))
        print(_dim("  Above is the current snapshot."))
        return _exit_code(reg_count, smoke, smoke_errors, failures, test_failed)

    if update_baseline:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(_green(f"  OK baseline written: {BASELINE_FILE.relative_to(ROOT)}"))
        return _exit_code(reg_count, smoke, smoke_errors, failures, test_failed)

    # Delta
    print(_bold("Delta vs baseline"))
    if regressions:
        print(_red(_bold(f"  [REGRESSIONS] ({len(regressions)}):")))
        for line in regressions:
            print(_red(f"      {line}"))
    if improvements:
        print(_green(_bold(f"  [IMPROVEMENTS] ({len(improvements)}):")))
        for line in improvements:
            print(_green(f"      {line}"))
    if not regressions and not improvements:
        print(_green("  = no change -- baseline matched"))
    print()

    # Hotspots
    if hot_files:
        print(_bold("Cross-tool hotspots (files flagged by >= 2 tools):"))
        for path, tools in hot_files[:10]:
            print(_yellow(f"  {path}  ({len(tools)} tools: {', '.join(tools)})"))
        if len(hot_files) > 10:
            print(_dim(f"  ... and {len(hot_files) - 10} more"))
        print()

    # Per-subsystem breakdown (verbose mode only)
    if verbose:
        breakdown = _subsystem_breakdown(file_lists)
        if breakdown:
            print(_bold("Subsystem flag count (where the debt lives):"))
            for sub, count, t in breakdown[:10]:
                print(f"  {sub:<30} {count:>4}  ({', '.join(t)})")
            if len(breakdown) > 10:
                print(_dim(f"  ... and {len(breakdown) - 10} more"))
            print()

    # Final verdict
    code = _exit_code(reg_count, smoke, smoke_errors, failures, test_failed)
    if code == 0:
        print(_green(_bold("OK: no regressions.")))
    elif code == 1:
        if smoke and smoke_errors:
            print(_red(_bold(f"FAIL: smoke check found {len(smoke_errors)} failure(s).")))
        if test_failed:
            t = current.get("tests", {})
            if t.get("timed_out"):
                print(_red(_bold(f"FAIL: pytest exceeded the 10-minute cap.")))
            else:
                print(_red(_bold(f"FAIL: pytest reports {t.get('failed', 0)} failed, {t.get('errors', 0)} errors.")))
        if reg_count > 0:
            print(_red(_bold(f"FAIL: {reg_count} regression(s) detected.")))
        print(_dim("  If the increase is intentional (e.g. a new subsystem legitimately"))
        print(_dim("  adds known-defer findings), bump the baseline:"))
        print(_dim("      python scripts/audit.py --update-baseline"))
    else:
        print(_red(_bold("FAIL: tool failure(s) - see above.")))
    return code


def _exit_code(
    reg_count: int,
    smoke: bool,
    smoke_errors: list[str],
    failures: list[str],
    test_failed: bool,
) -> int:
    if failures:
        return 2
    if smoke and smoke_errors:
        return 1
    if test_failed:
        return 1
    if reg_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
