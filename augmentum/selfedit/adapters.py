"""Adapters — register the codebase's existing verification primitives as
``Verifier``s, and the ``verify_change`` entrypoint that runs the whole honest
router over a candidate.

The verifier router is deliberately *not* new verification — we already have the
oracles, scattered across the app:

  * boot-smoke (selfedit/bootsmoke)          — mechanical no-regression
  * audit.py score + scanners (selfedit/scanners) — mechanical no-regression
  * health baseline-delta (selfedit/health)  — mechanical no-regression
  * targeted pytest (selfedit/gate)          — mechanical *confirm* (a reproducing
                                                test that now passes proves the fix)
  * build behavior gate (builds/verify)      — mechanical *confirm* (assert the
                                                requested behaviors in a real browser)
  * coder goal judge (coder/goal_judge)      — judgment (a model reads the work)

This module wraps each as a ``Verifier`` tagged by oracle + confirms_intent, and
``verify_change`` assembles the applicable pool for a given intent and returns the
honest ``Verdict``. That verdict is the P1 exit: it tells the rest of the system
whether a candidate is ``verified`` (auto-promotable), ``probable``, or
``human_required`` (didn't break, but only you can say it's good).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.selfedit import gate as _gate
from augmentum.selfedit import health as _health
from augmentum.selfedit.bootsmoke import boot_smoke_verifier
from augmentum.selfedit.intent import (
    CLASS_BUGFIX,
    CLASS_DEBT,
    CLASS_FEATURE,
    SelfEditIntent,
)
from augmentum.selfedit.scanners import AuditReport, audit_verifier
from augmentum.selfedit.verifier import (
    FAIL,
    ORACLE_MECHANICAL,
    PASS,
    SKIP,
    Verdict,
    Verifier,
    VerifierResult,
    judgment_verifier,
    verify,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Intent classes a *confirm* oracle (pytest/behavior gate) is meaningful for.
_CONFIRMABLE_CLASSES = (CLASS_BUGFIX, CLASS_FEATURE, CLASS_DEBT)


# --- Check → Verifier adapter (parallel to health.probe_from_check) ----------

def verifier_from_check(
    check: _gate.Check, *, oracle: str = ORACLE_MECHANICAL, confirms_intent: bool = False,
    intent_classes: tuple[str, ...] = ("*",), cost: int = 5, required: bool = True,
) -> Verifier:
    """Adapt a fitness-gate ``Check`` into a ``Verifier``. pass→PASS, fail→FAIL,
    skip→SKIP (never a false fail). The caller declares the oracle type and
    whether passing confirms the *intent* (a targeted test that reproduces the
    request → confirms_intent=True; a generic lint → False)."""
    async def _run(ctx: dict) -> VerifierResult:
        status, detail = await check.run()
        status = {"pass": PASS, "fail": FAIL, "skip": SKIP}.get(status, FAIL)
        return VerifierResult(check.name, oracle, status, confirms_intent=confirms_intent,
                              score=1.0 if status == PASS else 0.0,
                              required=required if status != SKIP else False,
                              detail=detail or "")
    return Verifier(check.name, oracle, _run, intent_classes, confirms_intent, cost, required)


def pytest_confirm_verifier(
    test_paths: list[str], *, cwd: str, intent_classes: tuple[str, ...] = _CONFIRMABLE_CLASSES,
    cost: int = 6, required: bool = True,
) -> Verifier:
    """Targeted pytest as a mechanical *confirm* oracle. A test written to
    reproduce the request that now passes is the strongest evidence the change
    did what was asked — so confirms_intent=True. No tests → SKIP."""
    return verifier_from_check(
        _gate.pytest_check(test_paths, cwd=cwd, required=required),
        oracle=ORACLE_MECHANICAL, confirms_intent=True,
        intent_classes=intent_classes, cost=cost, required=required,
    )


# --- health baseline-delta as a no-regression verifier -----------------------

# A health runner: given a target dir, return its HealthReport.
HealthRunner = Callable[[str], Awaitable[_health.HealthReport]]


def health_verifier(*, run_health: HealthRunner, baseline: _health.HealthReport | None,
                    required: bool = True, cost: int = 10) -> Verifier:
    """The app-health baseline-delta as a mechanical no-regression Verifier. FAILs
    when a required health dimension regressed into failure vs the known-good
    baseline. confirms_intent=False (health holding ≠ the change did what was
    asked)."""
    async def _run(ctx: dict) -> VerifierResult:
        target = ctx.get("candidate_dir") or "."
        try:
            current = await run_health(target)
        except Exception as exc:  # noqa: BLE001 — health unmeasurable → skip
            return VerifierResult("health", ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                                  required=required, detail=f"health unavailable: {exc!r}")
        delta = _health.compare(current, baseline)
        status = PASS if delta.ok else FAIL
        detail = f"score Δ{delta.score_delta:+.3f}"
        if delta.new_failures:
            detail += " | new failures: " + ", ".join(delta.new_failures[:8])
        elif delta.regressions:
            detail += " | regressions: " + ", ".join(delta.regressions[:8])
        return VerifierResult("health", ORACLE_MECHANICAL, status, confirms_intent=False,
                              score=1.0 if status == PASS else 0.0, required=required, detail=detail)
    return Verifier("health", ORACLE_MECHANICAL, _run, ("*",), confirms_intent=False,
                    cost=cost, required=required)


# --- build behavior gate as a mechanical-confirm verifier --------------------

def behaviors_passed(behaviors: list[dict]) -> tuple[bool, int, int]:
    """(all_checked_passed, n_pass, n_checked) over behavior dicts with a ``status``
    of pass/fail/untested. Untested behaviors don't count as failures (the gate
    couldn't check them) but a build with zero checked behaviors isn't confirmed."""
    checked = [b for b in behaviors if b.get("status") in ("pass", "fail")]
    n_pass = sum(1 for b in checked if b.get("status") == "pass")
    all_passed = bool(checked) and n_pass == len(checked)
    return all_passed, n_pass, len(checked)


def behavior_gate_verifier(behaviors: list[dict], *, intent_classes: tuple[str, ...] = _CONFIRMABLE_CLASSES,
                           cost: int = 12, required: bool = True) -> Verifier:
    """Turn an already-run build behavior gate (behaviors with per-behavior
    status, from ``builds/verify.run_behavior_gate``) into a mechanical *confirm*
    Verifier. The gate asserts the requested behaviors against a real browser DOM,
    so passing confirms the intent. The browser run itself happens in the
    orchestrator (it needs the workspace); this adapts its verdict to the router.
    Zero checked behaviors → SKIP (nothing was confirmed)."""
    async def _run(ctx: dict) -> VerifierResult:
        all_passed, n_pass, n_checked = behaviors_passed(behaviors)
        if n_checked == 0:
            return VerifierResult("behavior_gate", ORACLE_MECHANICAL, SKIP, confirms_intent=True,
                                  required=False, detail="no behaviors checked")
        status = PASS if all_passed else FAIL
        return VerifierResult("behavior_gate", ORACLE_MECHANICAL, status, confirms_intent=True,
                              score=1.0 if all_passed else 0.0, required=required,
                              detail=f"{n_pass}/{n_checked} behaviors passed")
    return Verifier("behavior_gate", ORACLE_MECHANICAL, _run, intent_classes, True, cost, required)


# --- coder goal judge as a judgment verifier ---------------------------------

# Default confidence for the goal judge: it returns a boolean, not a calibrated
# number, so we map a clear verdict to a confidence above the router floor (0.7)
# — enough to reach `probable`, never enough to masquerade as mechanical.
_GOAL_JUDGE_CONFIDENCE = 0.8


def make_goal_judge(judge_goal: Callable[..., Awaitable[Any]], **judge_kwargs: Any
                    ) -> Callable[[dict], Awaitable[tuple[bool, float, str]]]:
    """Adapt ``coder.goal_judge.judge_goal_satisfied`` (or any GoalVerdict-returning
    judge) into the ``(ok, confidence, detail)`` shape ``judgment_verifier`` wants.
    A no-signal verdict (``ok is None``) raises so the judgment verifier SKIPs it
    rather than counting a failed backend call as a failed intent."""
    async def _judge(ctx: dict) -> tuple[bool, float, str]:
        verdict = await judge_goal(**judge_kwargs)
        ok = getattr(verdict, "ok", None)
        reason = getattr(verdict, "reason", "")
        if ok is None:
            raise ValueError(f"goal judge returned no signal: {reason or 'unknown'}")
        return bool(ok), _GOAL_JUDGE_CONFIDENCE, reason
    return _judge


def goal_judge_verifier(judge_goal: Callable[..., Awaitable[Any]], *,
                        intent_classes: tuple[str, ...] = _CONFIRMABLE_CLASSES,
                        cost: int = 14, **judge_kwargs: Any) -> Verifier:
    """The coder goal judge as a judgment oracle (confidence-gated, never
    required). Confirms intent only as strongly as a model can — yields at most
    ``probable``."""
    return judgment_verifier(
        "goal_judge", make_goal_judge(judge_goal, **judge_kwargs),
        intent_classes=intent_classes, cost=cost, required=False,
    )


# --- the P1 entrypoint -------------------------------------------------------

async def verify_change(
    *, candidate_dir: str, intent: SelfEditIntent,
    baseline_audit: AuditReport | None = None,
    run_audit: Callable[[str], Awaitable[str]] | None = None,
    boot_runner: Any = None,
    run_health: HealthRunner | None = None, baseline_health: _health.HealthReport | None = None,
    test_paths: list[str] | None = None,
    extra_verifiers: list[Verifier] | None = None,
    extra_results: list[VerifierResult] | None = None,
    judgment_confidence_floor: float = 0.7,
) -> Verdict:
    """Run the honest verifier router over a candidate and return its ``Verdict``.

    Assembles the applicable oracle pool from whatever signals the caller can
    supply (boot-smoke always; audit/health/pytest/extras when available), filters
    by ``intent.intent_class``, runs cheap→expensive with short-circuit on the
    first required failure, and computes the oracle-tier verdict. A migration-
    surface change forces every confirm oracle off (the floor stays no-regression),
    because schema corruption is never auto-confirmed."""
    pool: dict[str, Verifier] = {}

    # Always: boot-smoke (cheapest required mechanical no-regression gate).
    boot = boot_smoke_verifier(boot_runner=boot_runner)
    pool[boot.name] = boot

    # Audit no-regression (needs a baseline to diff against).
    if baseline_audit is not None and run_audit is not None:
        av = audit_verifier(run_audit=run_audit, baseline=baseline_audit)
        pool[av.name] = av

    # Health baseline-delta no-regression.
    if run_health is not None:
        hv = health_verifier(run_health=run_health, baseline=baseline_health)
        pool[hv.name] = hv

    # Targeted pytest as a confirm oracle — only meaningful for confirmable,
    # non-migration intents.
    migration = intent.surface == "migration"
    if test_paths and not migration:
        pv = pytest_confirm_verifier(test_paths, cwd=candidate_dir)
        pool[pv.name] = pv

    for v in extra_verifiers or []:
        # A migration change may not carry confirm oracles — corruption is never
        # auto-confirmed; the floor is no-regression + human verdict.
        if migration and v.confirms_intent:
            continue
        pool[v.name] = v

    return await verify(
        {"candidate_dir": candidate_dir}, intent_class=intent.intent_class,
        verifiers=pool, extra_results=extra_results,
        judgment_confidence_floor=judgment_confidence_floor,
        # Fail-closed coverage: a migration is never mechanically confirmed, so no
        # confirm is expected there; otherwise the intent tells us whether a
        # mechanical oracle SHOULD have confirmed this change.
        expect_mechanical_confirm=(intent.mechanically_confirmable and not migration),
    )
