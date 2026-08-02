"""The self-edit orchestrator — propose → isolate → edit → verify → record.

This is the loop the whole foundation was built for. It composes the primitives
into one honest, reversible attempt:

    create_attempt (store, 'proposed')
      → create_candidate (worktree off base_ref, 'editing')
      → drive the editing agent against the candidate (the EditDriver seam)
      → commit the candidate's edits + collect changed paths
      → verify_change (the honest oracle-tier verdict)
      → set_gate + finalize (permanent lineage, with the lesson)

**No promotion happens here** (that's P4). The live tree is never touched: the
agent edits a throwaway worktree, and a passing candidate rests at ``gated``
(awaiting the promote/endorse decision) with its oracle tier recorded in the
verdict. A failing candidate is ``rejected``; an agent that errored is ``failed``.
Every terminal state records a *lesson* — the durable takeaway that survives even
when the candidate is discarded (the anti-Westworld pillar).

The agent itself is injected via an ``EditDriver`` (the seam where the B1 RW-repo
container + run engine plug in), so the orchestrator is fully testable against a
real temp git repo with a fake driver — only the agent is swapped, never the
pipeline.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.selfedit import candidate as _cand
from augmentum.selfedit import store as _store
from augmentum.selfedit.adapters import verify_change
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.intent import (
    SURFACE_BACKEND,
    SURFACE_MIGRATION,
    SURFACE_MIXED,
    SelfEditIntent,
    classify_intent,
)
from augmentum.selfedit.live import emit_progress
from augmentum.selfedit.scanners import AuditReport
from augmentum.selfedit.verifier import (
    FAIL as _FAIL,
)
from augmentum.selfedit.verifier import (
    ORACLE_MECHANICAL,
    TIER_FAILED,
    Verdict,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _touches_audit_infra(path: str) -> bool:
    """True if a changed path is audit/scanner infrastructure (the judge): any
    ``.claude`` component, a ``*_suppressions.json``, ``security_exceptions.json``,
    or the audit history. The cross-driver backstop to native_loop's write refusal.
    Component-based (not prefix) so it also catches git's untracked-dir entry
    ``.claude/`` and nested ``.claude`` — and never mangles the leading dot."""
    n = (path or "").replace("\\", "/").strip()
    if n.startswith("./"):
        n = n[2:]
    parts = [p for p in n.rstrip("/").split("/") if p]
    base = parts[-1] if parts else ""
    return (
        ".claude" in parts
        or base.endswith("_suppressions.json")
        or base in ("security_exceptions.json", "audit_history.jsonl")
    )


# --- the editing-agent seam --------------------------------------------------

@dataclass
class EditRequest:
    """Everything a driver needs to run the agent against an isolated candidate."""
    candidate: Candidate
    objective: str
    attempt_id: str
    user_id: str
    prior_context: str = ""    # escalation: what weaker tiers already learned


@dataclass
class EditResult:
    """What a driver returns after the agent has run (or failed to)."""
    ok: bool
    run_id: str = ""           # links to claude_runs (the edit transcript)
    final_text: str = ""       # the agent's final message
    error: str = ""


# A driver runs the editing agent against the candidate worktree. The real one
# stands up the B1 RW-repo container and drives Claude via the run engine; tests
# pass a fake that edits files directly. Never expected to raise (it normalizes
# failures into EditResult.ok=False) — but the orchestrator guards anyway.
EditDriver = Callable[[EditRequest], Awaitable[EditResult]]


async def null_edit_driver(req: EditRequest) -> EditResult:
    """A driver that does nothing — for dry-runs and as the safe default. Produces
    a no-op edit (the orchestrator records it as ``rejected``: no changes)."""
    return EditResult(ok=True, final_text="(no-op: null driver)")


# --- terminal-status resolution (pure, the honest mapping) -------------------

# Autonomy tier from the change surface (the gradient seed; refined in P8).
def _tier_for_surface(surface: str) -> str:
    if surface == SURFACE_MIGRATION:
        return "red"          # schema corruption — never auto-attempted
    if surface in (SURFACE_BACKEND, SURFACE_MIXED):
        return "yellow"
    return "green"


@dataclass
class SelfEditOutcome:
    attempt_id: str
    status: str                                  # gated | rejected | failed
    verdict: Verdict | None = None
    candidate: Candidate | None = None
    edit: EditResult | None = None
    files_changed: list[str] = field(default_factory=list)
    outcome: str = ""
    lesson: str = ""
    diff: str = ""              # the committed patch (carried up the escalation ladder)
    heals: int = 0             # self-heal repair passes taken this attempt

    def to_dict(self) -> dict:
        return {
            "attempt_id": self.attempt_id, "status": self.status,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "files_changed": self.files_changed,
            "outcome": self.outcome, "lesson": self.lesson, "heals": self.heals,
            "candidate_branch": self.candidate.branch if self.candidate else "",
        }


def resolve_terminal(
    *, edit: EditResult, has_changes: bool, verdict: Verdict | None,
) -> tuple[str, str, str]:
    """(status, outcome, lesson) from the edit + verification. Pure — the single
    place the loop decides what an attempt *means*. A passing candidate rests at
    ``gated`` (awaiting promotion; its tier is in the verdict, so the promote
    layer distinguishes auto-promotable ``verified`` from ``human_required``).

    The VERIFIER is the arbiter, not the agent's clean exit: whenever the agent
    produced changes we judge those changes, even if the agent then errored late
    (a real edit followed by a flaky model turn shouldn't be thrown away — a
    partial/corrupt edit is caught by boot-smoke/audit and rejected here). The
    agent's ``ok`` flag only decides *failed vs no-op* when there are NO changes."""
    if has_changes:
        if verdict is None:  # shouldn't happen with changes, but stay honest
            return "rejected", "not verified", "changes were made but verification did not run"
        if verdict.tier == TIER_FAILED:
            return "rejected", f"verification failed — {verdict.summary}", (
                "the change broke or regressed the app: " + verdict.summary
            )
        # passed (verified / probable / human_required / human_confirmed)
        tail = "" if edit.ok else " (agent exited with an error, but the change verified)"
        return "gated", f"{verdict.tier}: {verdict.summary}{tail}", _passing_lesson(verdict)
    if not edit.ok:
        err = edit.error or "agent run did not complete"
        return "failed", f"agent run failed: {err}", f"agent could not edit: {err}"
    return "rejected", "no changes produced", (
        "the agent produced no edits — the objective may be unclear, already "
        "satisfied, or beyond what one pass can do"
    )


# --- self-heal: the cheap repair rung, BELOW escalation ----------------------
# When a candidate's change FAILS verification with a fixable break (a broken
# import, a syntax error, a failing test), the cheapest correct response is not
# to give up (reject) or to pay for a stronger model (escalate) — it's to tell
# the SAME model, on the SAME worktree, exactly what its change broke and let it
# repair. This mirrors the coder loop's failure→next-prompt pattern
# (handler.py::_append_tool_result_to_history + verify.model_facing_summary),
# bounded like promises (max_attempts) with the coder's identical-failure
# stagnation breaker. Escalation (a fresh checkout + a stronger model) remains
# the fallback when self-heal can't fix it.


def _failing_required(verdict: Verdict) -> list:
    """The required checks that actually FAILED — what a repair must address."""
    return [r for r in verdict.results if r.required and r.status == _FAIL]


def _healable(verdict: Verdict | None) -> bool:
    """A verdict is self-healable when it FAILED because a required MECHANICAL
    check failed — a concrete, model-fixable break (import/syntax/test/lint),
    not a taste call or a missing human verdict."""
    if verdict is None or verdict.tier != TIER_FAILED:
        return False
    return any(r.oracle == ORACLE_MECHANICAL for r in _failing_required(verdict))


def _failure_signature(verdict: Verdict) -> str:
    """A stable signature of the failing checks (name + head of detail). If a heal
    leaves this UNCHANGED, the model isn't making progress on the break — stop and
    let escalation take over (the coder's identical-retry breaker)."""
    return "|".join(sorted(
        f"{r.name}:{(r.detail or '')[:80]}" for r in _failing_required(verdict)))


def _repair_context(verdict: Verdict, *, attempt: int, max_attempts: int) -> str:
    """Frame the failure as a repair instruction the model reads next (the
    ``prior_context`` seam → prepended to the objective by the driver). Concrete,
    scoped, and — like the coder's model_facing_summary — names the exact checks
    that failed so the model fixes THOSE, not something else."""
    lines = [
        f"Your previous edit did NOT pass verification (repair attempt "
        f"{attempt}/{max_attempts}). Fix ONLY these specific failures, editing the "
        "same files you already changed — do not start over or add unrelated work:",
        "",
    ]
    for r in _failing_required(verdict):
        lines.append(f"  - [{r.name}] {(r.detail or '').strip()[:500]}")
    lines += [
        "",
        "Make the smallest change that resolves the failure(s) above, then stop. "
        "The same checks will run again immediately.",
    ]
    return "\n".join(lines)


def _passing_lesson(verdict: Verdict) -> str:
    if verdict.auto_promotable:
        return "verified by a mechanical oracle — safe to auto-promote"
    if verdict.tier == "probable":
        return "a judgment oracle confirmed it — likely good, wants a glance"
    return "no regression, but only you can say it's right — awaiting your verdict"


# --- the orchestrator --------------------------------------------------------

async def run_self_edit(
    *, repo_dir: str, objective: str, user_id: str, conn: Any,
    driver: EditDriver = null_edit_driver,
    intent: SelfEditIntent | None = None,
    base_ref: str = "HEAD", worktrees_dir: str | None = None,
    attempt_id: str | None = None,
    verify: Callable[..., Awaitable[Verdict]] = verify_change,
    audit_baseline: AuditReport | None = None,
    run_audit: Callable[[str], Awaitable[str]] | None = None,
    boot_runner: Any = None,
    run_health: Callable[[str], Awaitable[Any]] | None = None,
    baseline_health: Any = None,
    test_paths: list[str] | None = None,
    extra_verifiers: list | None = None,
    preference_store: Any = None,
    keep_worktree: bool = False,
    prior_context: str = "",
    target: str = "",
    max_heal_attempts: int = 0,
) -> SelfEditOutcome:
    """Run one full self-edit attempt and record it permanently. Never raises —
    any failure is captured as a terminal ``failed`` attempt with the error as the
    lesson (a dead loop teaches nothing; a recorded failure does).

    ``target`` is the structured debt class (``scanner.metric``) this attempt is
    paying down — persisted so the verified skill graph learns per-class trust, not
    only per-file/surface region. Empty for free-form (non-debt) edits."""
    attempt_id = attempt_id or uuid.uuid4().hex
    intent = intent or classify_intent(objective)
    cand: Candidate | None = None

    try:
        await _store.create_attempt(
            conn, attempt_id=attempt_id, user_id=user_id, objective=objective,
            surface=intent.surface, tier=_tier_for_surface(intent.surface),
            base_ref=base_ref, target=target,
        )
    except Exception as exc:  # noqa: BLE001 — can't even record → nothing to clean up
        log.warning("selfedit_create_attempt_failed", error=repr(exc))
        return SelfEditOutcome(attempt_id, "failed", outcome="could not record attempt",
                               lesson=repr(exc))

    try:
        cand = await _cand.create_candidate(
            repo_dir, name=attempt_id, base_ref=base_ref, worktrees_dir=worktrees_dir,
        )
        await _store.set_candidate(
            conn, attempt_id=attempt_id, user_id=user_id,
            candidate_ref=cand.branch, base_ref=cand.base_sha,
        )
        emit_progress({"kind": "phase", "phase": "candidate", "attempt_id": attempt_id,
                       "branch": cand.branch, "surface": intent.surface,
                       "text": "Isolated candidate worktree created"})
        emit_progress({"kind": "phase", "phase": "agent",
                       "text": "Agent working in the candidate…"})

        edit = await driver(EditRequest(
            candidate=cand, objective=objective, attempt_id=attempt_id, user_id=user_id,
            prior_context=prior_context,
        ))
        if edit.run_id:
            await _store.set_candidate(
                conn, attempt_id=attempt_id, user_id=user_id,
                candidate_ref=cand.branch, run_id=edit.run_id,
            )

        changes = await _cand.candidate_changes(cand)
        # Tamper guard (ALL drivers, not just native): if the agent edited the audit
        # infrastructure — the scanner, its suppressions, or anything under .claude/
        # — it tried to game the judge (a real run did exactly this). Refuse the
        # whole attempt; never verify a change that altered what verifies it.
        tampered = [p for p in changes if _touches_audit_infra(p)]
        if tampered:
            log.warning("selfedit_audit_tamper_refused", attempt_id=attempt_id,
                        paths=tampered[:5])
            lesson = ("edited the audit infrastructure (scanner/suppressions/.claude) "
                      "— the judge — instead of the real code; refused")
            await _store.finalize(conn, attempt_id=attempt_id, user_id=user_id,
                                  status="rejected", outcome="tampered with the audit infra",
                                  lesson=lesson)
            emit_progress({"kind": "verdict", "attempt_id": attempt_id,
                           "status": "rejected", "tier": "failed", "passed": False,
                           "files": tampered, "outcome": "tampered with the audit infra",
                           "lesson": lesson})
            return SelfEditOutcome(attempt_id, "rejected", candidate=cand, edit=edit,
                                   files_changed=changes, outcome="tampered with the audit infra",
                                   lesson=lesson)
        verdict: Verdict | None = None
        committed_diff = ""
        heals = 0
        # Verify whenever there ARE changes — the verifier is the arbiter, even if
        # the agent errored after editing (the verdict catches a bad/partial edit).
        if changes:
            await _cand.commit_candidate(cand, f"selfedit: {objective[:72]}")
            committed_diff = await _cand.candidate_diff(cand)
            # refresh the intent surface now that the real diff exists
            intent = classify_intent(objective, changed_paths=changes)
            # the learning loop: a consistently-kept shape lifts human_required →
            # probable through the honest router (judgment-tier, never auto).
            verifiers = list(extra_verifiers or [])
            # Differential contract gate (opt-in, default OFF): probe the
            # candidate's GET routes vs base_ref and FAIL on any NEW route break
            # the edit introduced. The differential cancels in-process mock noise
            # (the false-positive killer — Godefroid et al. ISSTA 2020). base_dir
            # = repo_dir (the base_ref checkout the candidate branched off).
            # ~2 full probes, so it stays off until validated on a live self-edit.
            from augmentum.config import settings as _se_settings
            if getattr(_se_settings, "selfedit_contract_gate_enabled", False):
                from augmentum.contracts.selfedit_gate import (
                    contract_regression_verifier,
                )
                verifiers.append(contract_regression_verifier(base_dir=repo_dir))
            if preference_store is not None:
                from augmentum.selfedit.preferences import change_shape, preference_verifier
                verifiers.append(preference_verifier(
                    change_shape(intent.surface, intent.intent_class),
                    store=preference_store, user_id=user_id))

            async def _verify_and_record(files: list) -> Verdict:
                emit_progress({"kind": "phase", "phase": "verify", "files": files,
                               "text": f"Verifying {len(files)} changed file(s)…"})
                v = await verify(
                    candidate_dir=cand.path, intent=intent,
                    baseline_audit=audit_baseline, run_audit=run_audit,
                    boot_runner=boot_runner, run_health=run_health,
                    baseline_health=baseline_health, test_paths=test_paths,
                    extra_verifiers=verifiers,
                )
                for r in v.results:
                    emit_progress({
                        "kind": "verifier", "name": r.name, "oracle": r.oracle,
                        "status": r.status, "required": r.required,
                        "confirms_intent": r.confirms_intent,
                        "detail": (r.detail or "")[:300],
                    })
                await _store.set_gate(
                    conn, attempt_id=attempt_id, user_id=user_id,
                    passed=v.passed, verdict=v.to_dict(), files_changed=files,
                )
                return v

            verdict = await _verify_and_record(changes)

            # SELF-HEAL: a fixable break (broken import/syntax/test) → tell the same
            # model exactly what it broke and let it repair, on the same worktree,
            # BEFORE giving up (reject) or paying for a stronger model (escalate).
            # Bounded + stagnation-guarded; amend keeps one promotable commit.
            while (max_heal_attempts and heals < max_heal_attempts and _healable(verdict)):
                sig = _failure_signature(verdict)
                emit_progress({
                    "kind": "phase", "phase": "self_heal", "attempt": heals + 1,
                    "max": max_heal_attempts,
                    "checks": [r.name for r in _failing_required(verdict)],
                    "text": (f"Self-healing: repairing {len(_failing_required(verdict))} "
                             f"failed check(s) (attempt {heals + 1}/{max_heal_attempts})…"),
                })
                repair = _repair_context(verdict, attempt=heals + 1,
                                         max_attempts=max_heal_attempts)
                edit = await driver(EditRequest(
                    candidate=cand, objective=objective, attempt_id=attempt_id,
                    user_id=user_id, prior_context=repair))
                new_changes = await _cand.candidate_changes(cand)
                if not new_changes:
                    break  # nothing left to verify
                await _cand.commit_candidate(cand, f"selfedit: {objective[:72]}", amend=True)
                committed_diff = await _cand.candidate_diff(cand)
                changes = new_changes
                intent = classify_intent(objective, changed_paths=changes)
                verdict = await _verify_and_record(changes)
                heals += 1
                if _failure_signature(verdict) == sig:
                    # same break after a repair → the model can't fix it here; stop
                    # and leave it for escalation (a stronger model) or the human.
                    emit_progress({
                        "kind": "phase", "phase": "self_heal_stalled",
                        "text": "Self-heal made no progress on the failure — leaving "
                                "it for escalation / your review.",
                    })
                    break
            if heals:
                log.info("selfedit_self_heal", attempt_id=attempt_id, heals=heals,
                         final_tier=(verdict.tier if verdict else ""),
                         healed=bool(verdict and verdict.tier != TIER_FAILED))

        status, outcome, lesson = resolve_terminal(
            edit=edit, has_changes=bool(changes), verdict=verdict,
        )
        emit_progress({
            "kind": "verdict", "attempt_id": attempt_id, "status": status,
            "tier": (verdict.tier if verdict else ""),
            "passed": bool(verdict.passed) if verdict else False,
            "auto_promotable": bool(verdict and verdict.auto_promotable),
            "files": changes, "outcome": outcome, "lesson": lesson,
        })
        await _store.finalize(
            conn, attempt_id=attempt_id, user_id=user_id,
            status=status, outcome=outcome, lesson=lesson,
        )
        log.info("selfedit_attempt_recorded", attempt_id=attempt_id, status=status,
                 tier=(verdict.tier if verdict else ""), files=len(changes))
        return SelfEditOutcome(
            attempt_id=attempt_id, status=status, verdict=verdict, candidate=cand,
            edit=edit, files_changed=changes, outcome=outcome, lesson=lesson,
            diff=committed_diff, heals=heals,
        )
    except Exception as exc:  # noqa: BLE001 — the loop must never leave an attempt dangling
        log.warning("selfedit_attempt_crashed", attempt_id=attempt_id, error=repr(exc))
        with contextlib.suppress(Exception):
            await _store.finalize(
                conn, attempt_id=attempt_id, user_id=user_id, status="failed",
                outcome="orchestration error", lesson=repr(exc),
            )
        return SelfEditOutcome(attempt_id, "failed", candidate=cand,
                               outcome="orchestration error", lesson=repr(exc))
    finally:
        # Free the worktree dir; KEEP the branch (the candidate's committed
        # lineage) unless the caller wants the live worktree for promotion.
        if cand is not None and not keep_worktree:
            try:
                await _cand.remove_candidate(repo_dir, cand, delete_branch=False)
            except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
                log.warning("selfedit_worktree_cleanup_failed",
                            attempt_id=attempt_id, error=repr(exc))
