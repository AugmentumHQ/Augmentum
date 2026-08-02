"""The debt-paydown loop — the first dogfood, wired end to end.

Audit the live tree → triage the findings (``debt``) → for each mechanical
target, run one isolated self-edit attempt (``orchestrator``) verified against the
audit baseline (the finding must disappear with no regression). The score climbs,
one small reversible unit at a time, and — because fixing wiring/harness debt makes
every future edit safer — it compounds.

This module is the *orchestration*; it owns no agent and no audit of its own,
both are injected:

* ``live_audit_runner(repo_dir)`` → the known-good baseline + the target list.
* the orchestrator's ``driver`` → the editing agent.
* ``candidate_audit_runner`` (default: the candidate's own audit) → what the
  verifier diffs against the baseline.

So the whole loop is testable with a synthetic audit + a fake driver. ``dry_run``
returns the plan (what it *would* attempt) without touching anything — the basis
for a propose/preview.

Discipline (from the design): only the **mechanical** auto-lane runs here.
Structural / taste findings are surfaced for the human (P3), never auto-attempted.
Nothing is *promoted* by the loop (that's P4) — each attempt rests at ``gated``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.selfedit import debt, scanners
from augmentum.selfedit.debt import DebtTarget
from augmentum.selfedit.live import emit_progress
from augmentum.selfedit.orchestrator import (
    EditDriver,
    SelfEditOutcome,
    null_edit_driver,
    run_self_edit,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

AuditRunner = Callable[[str], Awaitable[str]]


@dataclass
class DebtLoopReport:
    baseline_score: float
    targets: list[DebtTarget] = field(default_factory=list)      # mechanical, considered
    attempted: list[SelfEditOutcome] = field(default_factory=list)
    deferred: int = 0                                            # targets past the cap
    structural: list[DebtTarget] = field(default_factory=list)   # surfaced for the human
    dry_run: bool = False

    @property
    def gated(self) -> list[SelfEditOutcome]:
        return [o for o in self.attempted if o.status == "gated"]

    def to_dict(self) -> dict:
        return {
            "baseline_score": self.baseline_score, "dry_run": self.dry_run,
            "deferred": self.deferred,
            "targets": [t.to_dict() for t in self.targets],
            "structural": [t.to_dict() for t in self.structural],
            "attempted": [o.to_dict() for o in self.attempted],
            "gated": len(self.gated),
        }


async def run_debt_loop(
    *, repo_dir: str, user_id: str, conn: Any, driver: EditDriver = null_edit_driver,
    live_audit_runner: AuditRunner | None = None,
    candidate_audit_runner: AuditRunner | None = None,
    max_attempts: int = 1, dry_run: bool = False,
    boot_runner: Any = None, run_health: Any = None, baseline_health: Any = None,
    worktrees_dir: str | None = None, preference_store: Any = None,
    rungs: list[Any] | None = None, evidence_tree: str | None = None,
    target_id: str = "", demand: list[Any] | None = None,
    max_heal_attempts: int = 0,
) -> DebtLoopReport:
    """Audit → triage → attempt the top ``max_attempts`` mechanical targets.

    Each attempt diffs the candidate's audit against the SAME baseline (the live
    tree before any edits), so independent findings don't interfere and any new
    debt a fix introduces is caught as a regression. The loop never promotes.

    ``demand`` (optional) is a list of demand-side ``DebtTarget``s — lived user
    friction read from ``signal_events`` (see ``selfedit/demand.py``). They join
    the STRUCTURAL (needs-you) lane beside the audit's structural findings; they
    are never mechanical and never auto-attempted. The reader lives at the route
    (it needs the main DB); the loop just merges the data."""
    live_audit_runner = live_audit_runner or scanners.default_audit_runner
    candidate_audit_runner = candidate_audit_runner or scanners.default_audit_runner

    text = await live_audit_runner(repo_dir)
    baseline = scanners.parse_audit_json(text)
    triage = debt.triage(baseline)
    targets = triage.mechanical
    # Optional target pin: tackle a specific scanner.metric (e.g. a tractable
    # "code_quality.silent_catches") instead of the top-by-count finding. Falls
    # back to the full list if the id doesn't match anything mechanical.
    if target_id:
        picked = [t for t in targets if f"{t.scanner}.{t.metric}" == target_id]
        if picked:
            targets = picked
            log.info("debt_loop_target_pinned", target=target_id)

    # Demand-side friction joins the needs-you lane, most-recurring first, ahead
    # of the audit's structural findings — lived user pain outranks latent debt
    # for the human's attention. Never enters the mechanical auto-lane.
    structural = list(demand or []) + triage.structural
    report = DebtLoopReport(
        baseline_score=baseline.score, targets=targets,
        structural=structural, dry_run=dry_run,
        deferred=max(0, len(targets) - max_attempts),
    )
    if report.deferred:
        log.info("debt_loop_deferred", deferred=report.deferred, cap=max_attempts,
                 total_mechanical=len(targets))
    if dry_run:
        return report

    # The verified skill graph (read-only routing hint): a confidently
    # failure-prone region starts HIGHER on the escalation ladder instead of
    # burning the cheap rung on a region the archive shows it can't land. Loaded
    # once; best-effort — never blocks the loop, never changes what gets promoted.
    # The hint is APPLIED only once the signal has *graduated* (its prequential
    # backtest clears the accuracy floor); until then it runs in shadow (computed
    # + logged, never acted on) — earned activation, not activation-on-existence.
    skill_graph = None
    calibration = None
    if rungs:
        try:
            from augmentum.selfedit import activation as _activation
            skill_graph, calibration = await _activation.load_graph_and_calibration(
                conn, user_id=user_id)
        except Exception as exc:  # noqa: BLE001 — advisory only
            log.warning("debt_loop_skill_graph_unavailable", error=repr(exc))

    for target in targets[:max_attempts]:
        emit_progress({
            "kind": "phase", "phase": "target",
            "target": f"{target.scanner}.{target.metric}",
            "title": getattr(target, "title", "") or target.metric,
            "count": getattr(target, "count", 0),
            "baseline_score": round(baseline.score, 1),
            "text": f"Target: {getattr(target, 'title', '') or target.metric}",
        })
        # Evidence-grounding: hand the agent the scanner's SPECIFIC findings + a
        # mechanical confirm oracle keyed to them, so it fixes a named item (and
        # the resolution is provable → verified), not a generic hunt.
        objective = target.objective
        target_verifiers: list = []
        grounded_files: list[str] = []
        if evidence_tree:
            from augmentum.selfedit.evidence import enrich_target
            enr = await enrich_target(evidence_tree, target.scanner, target.metric,
                                      target.objective)
            if enr.grounded:
                objective = enr.objective
                target_verifiers = enr.verifiers
                grounded_files = [f.file for f in enr.findings if getattr(f, "file", "")]
                log.info("debt_evidence_grounded", scanner=target.scanner,
                         metric=target.metric, findings=len(enr.findings))
                emit_progress({
                    "kind": "phase", "phase": "evidence",
                    "findings": [getattr(f, "symbol", "") or getattr(f, "key", "")
                                 for f in enr.findings][:12],
                    "count": len(enr.findings),
                    "text": f"Grounded in {len(enr.findings)} specific finding(s)",
                })
            else:
                # GATE: a target we can't ground (count-only metric, or no findings
                # in this tree) is NOT handed to the agent as a blind hunt — that's
                # the 40-step / zero-edit failure mode. Skip it; the advisor/human
                # handles ungrounded debt. Don't burn a rung exploring.
                log.info("debt_target_skipped_ungrounded", scanner=target.scanner,
                         metric=target.metric)
                emit_progress({
                    "kind": "phase", "phase": "skipped",
                    "target": f"{target.scanner}.{target.metric}",
                    "text": (f"Skipped {target.scanner}.{target.metric} — no specific "
                             "findings to ground on (count-only); left for review."),
                })
                report.deferred += 1
                continue

        target_class = f"{target.scanner}.{target.metric}"
        run_kwargs = dict(
            repo_dir=repo_dir, objective=objective, user_id=user_id, conn=conn,
            audit_baseline=baseline, run_audit=candidate_audit_runner,
            boot_runner=boot_runner, run_health=run_health, baseline_health=baseline_health,
            worktrees_dir=worktrees_dir, preference_store=preference_store,
            extra_verifiers=target_verifiers, target=target_class,
            max_heal_attempts=max_heal_attempts,
        )
        if rungs:
            # Escalation ladder: local does the groundwork, climb to a frontier
            # model on failure, carrying findings forward (lazy import keeps this
            # module free of the coder.external dependency).
            from augmentum.selfedit.escalate import run_self_edit_escalating
            start_index = 0
            if skill_graph is not None and grounded_files:
                from augmentum.selfedit import activation as _activation
                non_frontier = sum(1 for r in rungs if not getattr(r, "frontier", False))
                # blend per-file region AND per-debt-class history (the `target:`
                # atom) — the class signal transfers even when the specific files
                # are ones the graph hasn't seen before.
                score = skill_graph.score(
                    _activation.query_atoms(files=grounded_files, target=target_class))
                hint = _activation.recommend_start_rung(
                    score, len(rungs), max_skip=max(0, non_frontier - 1))
                graduated = bool(calibration and calibration.graduated)
                # earned activation: apply the skip only when the signal has proven
                # itself; otherwise compute + log it in shadow (never act on it).
                start_index = hint if graduated else 0
                if hint > 0:
                    log.info("debt_loop_route_skip", scanner=target.scanner,
                             metric=target.metric, hint=hint, applied=graduated,
                             score=round(score.score, 3), confidence=round(score.confidence, 3),
                             calibration=(calibration.rationale if calibration else ""))
                    emit_progress({
                        "kind": "phase", "phase": "route",
                        "target": f"{target.scanner}.{target.metric}",
                        "start_index": start_index, "hint": hint, "applied": graduated,
                        "score": round(score.score, 3),
                        "confidence": round(score.confidence, 3),
                        "text": (f"Skill graph: {score.rationale} → "
                                 + (f"starting at rung {hint + 1}" if graduated
                                    else f"would start at rung {hint + 1}, but signal "
                                         "not yet calibrated (shadow)")),
                    })
            outcome = await run_self_edit_escalating(
                rungs=rungs, start_index=start_index, **run_kwargs)
        else:
            outcome = await run_self_edit(driver=driver, **run_kwargs)
        if outcome is None:
            continue
        report.attempted.append(outcome)
        log.info("debt_loop_attempt", scanner=target.scanner, metric=target.metric,
                 status=outcome.status, tier=(outcome.verdict.tier if outcome.verdict else ""))

    return report
