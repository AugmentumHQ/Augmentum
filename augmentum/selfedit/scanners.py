"""Bridge: the `augmentum-dev` audit/scanners → a self-edit Verifier.

`augmentum-dev` is the working dev-time prototype of this system's verification
layer. Rather than reinvent its 10 scanners, we COMPOSE them: run
`audit.py --format=json` against a candidate and turn its score + per-scanner
metrics into a **mechanical no-regression** Verifier — i.e. "did this change make
the codebase's health/safety/wiring/security worse?" (`confirms_intent=False` —
it proves the change didn't *break* things, not that it did what was asked).

This is the single biggest jump in mechanical-oracle coverage available: wiring,
dead-code, security, db-safety, async-blocking, coverage, red-team — exactly the
accidental harms a self-editing agent would introduce. The same tool that finds
the debt to fix also verifies a fix introduced no new debt.

The audit runner is injectable, so the comparison/parse logic is pure and
testable without spawning the (slow) full audit. `default_audit_runner` is the
real subprocess; it runs the *candidate's own* audit (cwd = candidate worktree).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.selfedit.verifier import (
    FAIL,
    ORACLE_MECHANICAL,
    PASS,
    SKIP,
    Verifier,
    VerifierResult,
    register_verifier,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# metric keys that are informational (higher is fine), not debt to minimize.
_INFORMATIONAL = frozenset({
    "modules_covered", "modules_total", "routes_covered", "routes_total",
    "registered", "score",
})


@dataclass
class AuditReport:
    """Parsed `audit.py --format=json` output."""
    score: float = 0.0
    metrics: dict[str, dict] = field(default_factory=dict)   # scanner -> {metric: count}
    regressions: list = field(default_factory=list)
    smoke_errors: list = field(default_factory=list)
    tool_failures: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def parse_audit_json(text: str) -> AuditReport:
    """Parse audit JSON. Raises ValueError on unparseable input (caller decides)."""
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"audit output not JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise ValueError("audit output not a JSON object")
    return AuditReport(
        score=float(d.get("score", 0.0) or 0.0),
        metrics=d.get("metrics") or {},
        regressions=d.get("regressions") or [],
        smoke_errors=d.get("smoke_errors") or [],
        tool_failures=d.get("tool_failures") or [],
        raw=d,
    )


@dataclass
class AuditDelta:
    score_delta: float
    broke_boot: bool                       # smoke or a scanner itself failed
    worsened: list[str] = field(default_factory=list)   # "scanner.metric: a->b"
    regressed: bool = False


def audit_delta(candidate: AuditReport, baseline: AuditReport, *, score_eps: float = 0.05) -> AuditDelta:
    """Compare a candidate's audit to the baseline (HEAD / known-good). A
    regression is: the candidate broke the boot/a scanner, OR its overall score
    dropped beyond the noise floor. Per-metric increases are collected as
    evidence even when the score holds."""
    broke = bool(candidate.smoke_errors or candidate.tool_failures)
    worsened: list[str] = []
    for scanner, m in candidate.metrics.items():
        if not isinstance(m, dict):
            continue
        base_m = baseline.metrics.get(scanner) or {}
        for key, cval in m.items():
            if key in _INFORMATIONAL or not isinstance(cval, int | float):
                continue
            bval = base_m.get(key, 0)
            if isinstance(bval, int | float) and cval > bval:
                worsened.append(f"{scanner}.{key}: {bval}->{cval}")
    score_delta = candidate.score - baseline.score
    regressed = broke or score_delta < -score_eps
    return AuditDelta(score_delta=score_delta, broke_boot=broke, worsened=worsened, regressed=regressed)


# An audit runner: given a target dir, return the audit JSON text.
AuditRunner = Callable[[str], Awaitable[str]]


def audit_verifier(*, run_audit: AuditRunner, baseline: AuditReport,
                   required: bool = True, cost: int = 8) -> Verifier:
    """A mechanical no-regression Verifier from the audit. Runs the candidate's
    audit, compares to ``baseline``, fails on regression. confirms_intent=False
    (it proves no new debt/breakage, not that the change did what was asked)."""
    async def _run(ctx: dict) -> VerifierResult:
        target = ctx.get("candidate_dir") or "."
        try:
            text = await run_audit(target)
            cand = parse_audit_json(text)
        except Exception as exc:  # noqa: BLE001 — audit unavailable → skip, don't false-fail
            return VerifierResult("audit", ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                                  required=required, detail=f"audit unavailable: {exc!r}")
        delta = audit_delta(cand, baseline)
        status = FAIL if delta.regressed else PASS
        detail = f"score {baseline.score:.1f}->{cand.score:.1f}"
        if delta.broke_boot:
            detail += " | BOOT/scan broke"
        if delta.worsened:
            detail += " | worsened: " + "; ".join(delta.worsened[:8])
        return VerifierResult("audit", ORACLE_MECHANICAL, status, confirms_intent=False,
                              score=1.0 if status == PASS else 0.0, required=required, detail=detail)
    return Verifier("audit", ORACLE_MECHANICAL, _run, ("*",), confirms_intent=False,
                    cost=cost, required=required)


def register_audit_verifier(*, run_audit: AuditRunner, baseline: AuditReport,
                            required: bool = True) -> None:
    register_verifier(audit_verifier(run_audit=run_audit, baseline=baseline, required=required))


# ---------------------------------------------------------------------------
# Incremental selection — which scanners care about which changed files.
# (Seed for running only the relevant scanners per change; v1 runs the full
# audit, but this documents/enables the fine-grained path.)
# ---------------------------------------------------------------------------

def select_scanners(changed_paths: list[str]) -> set[str]:
    """The scanners relevant to a set of changed paths. Over-selection is safe
    (it just runs an extra scanner); under-selection would miss a regression, so
    bias toward including."""
    sel: set[str] = set()
    for raw in changed_paths:
        p = raw.replace("\\", "/")
        py, js = p.endswith(".py"), p.endswith(".js")
        css, html, sql = p.endswith(".css"), p.endswith(".html"), p.endswith(".sql")
        if p.startswith("augmentum/") and py:
            sel |= {"code_quality", "security", "red_team", "async_blocking", "runtime", "coverage"}
        if "_routes.py" in p:
            sel |= {"wiring", "dead_code"}
        if p in ("augmentum/config.py", "augmentum/proxy/config_routes.py",
                 "augmentum/proxy/server.py") or p == "ui/scripts/settings.js":
            sel.add("wiring")
        if p.startswith("ui/") and (js or css or html):
            sel |= {"code_quality", "dead_code"}
        if sql or "/state/" in p:
            sel.add("db_safety")
        if p.startswith("tests/") and py:
            sel.add("coverage")
    return sel


_AUDIT_SCRIPT = ".claude/skills/augmentum-dev/scripts/audit.py"


def _resolve_audit_target(target_dir: str) -> tuple[str, str]:
    """(script, cwd) for the audit. The script path is resolved ABSOLUTELY so the
    run doesn't depend on cwd (the old relative path silently no-op'd when cwd was
    a tree without the script). If ``target_dir`` has no checkout of the script —
    e.g. the ``--no-checkout`` self-edit clone — fall back to the live source mount
    (full checkout) so the BASELINE audit still produces JSON instead of nothing.
    A candidate worktree (full checkout) keeps auditing itself."""
    here = os.path.join(target_dir, _AUDIT_SCRIPT)
    if os.path.isfile(here):
        return here, target_dir
    src = os.environ.get("AUGMENTUM_SELFEDIT_REPO", "/host-augmentum-src")
    return os.path.join(src, _AUDIT_SCRIPT), src


async def default_audit_runner(target_dir: str, *, timeout: float = 600.0) -> str:
    """Run the audit for ``target_dir`` (a candidate worktree audits itself; a
    tree without the script — the no-checkout clone — falls back to the live
    source). Returns its JSON. Skips deps/history for speed; ignores exit code
    (1 just means 'below baseline') — we parse stdout."""
    from augmentum.selfedit.sandbox import scrubbed_env
    script, cwd = _resolve_audit_target(target_dir)
    argv = [sys.executable, script, "--format=json", "--skip-deps", "--no-history"]
    # secret-scrubbed: the audit walks + imports candidate code (W11)
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        env=scrubbed_env(),
    )
    out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    return (out or b"").decode("utf-8", errors="replace")
