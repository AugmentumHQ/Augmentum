"""The self-edit Verifier — the system-level router over our verification
primitives, with an HONEST verdict that never lets "the code ran" masquerade as
"the edit is good."

We already have ~16 verification primitives across the codebase (build behavior
gate, sentinel command oracle, static audit, health/baseline-delta, LLM judges,
evidence convergence, …). This module is the *router*, not new verification: each
primitive registers as a ``Verifier`` tagged by **oracle type** (mechanical /
judgment / human) and — critically — whether passing it actually **confirms the
intent** (the change did what was asked) versus merely proving it **didn't
break**.

The output ``Verdict`` carries an explicit ``oracle_tier`` = the strongest oracle
that confirmed the *intent*:

  failed          — a required check failed / regressed
  verified        — a MECHANICAL oracle confirmed the intent (auto-promotable)
  human_confirmed — the user kept it
  probable        — a JUDGMENT oracle confirmed the intent (confidence-gated)
  human_required  — only no-regression / liveness passed; intent UNCONFIRMED
                    (e.g. "moved the CSS button" — runs fine, but only the user
                    can say it's good). Never marked good.

This is the actuation-gating primitive (architect's ``ActionResult.fulfilled``)
generalized to the whole self-edit system.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Oracle types — where the truth comes from.
ORACLE_MECHANICAL = "mechanical"   # environment answers definitively
ORACLE_JUDGMENT = "judgment"       # a model/heuristic estimates; can be wrong
ORACLE_HUMAN = "human"             # only the user can say

# Verifier statuses.
PASS, FAIL, SKIP = "pass", "fail", "skip"

# Verdict tiers, weakest→strongest confirmation of the *intent*.
TIER_FAILED = "failed"
TIER_HUMAN_REQUIRED = "human_required"
TIER_PROBABLE = "probable"
TIER_HUMAN_CONFIRMED = "human_confirmed"
TIER_VERIFIED = "verified"


@dataclass
class VerifierResult:
    name: str
    oracle: str                # mechanical | judgment | human
    status: str                # pass | fail | skip
    confirms_intent: bool      # passing PROVES the change did what was asked
    score: float = 1.0
    confidence: float = 1.0    # for judgment oracles
    required: bool = True       # a required FAIL sinks the verdict
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "oracle": self.oracle, "status": self.status,
            "confirms_intent": self.confirms_intent, "score": round(self.score, 4),
            "confidence": round(self.confidence, 4), "required": self.required,
            "detail": self.detail[:2000],
        }


# A verifier body: given a context dict, return a VerifierResult.
VerifierRun = Callable[[dict], Awaitable[VerifierResult]]


@dataclass
class Verifier:
    name: str
    oracle: str
    run: VerifierRun
    intent_classes: tuple[str, ...] = ("*",)  # which change-intents it applies to ("*"=any)
    confirms_intent: bool = False             # default: a no-regression/liveness check
    cost: int = 1                              # run order (cheap first); also short-circuit savings
    required: bool = True

    def applies_to(self, intent_class: str) -> bool:
        return "*" in self.intent_classes or intent_class in self.intent_classes


@dataclass
class Verdict:
    tier: str
    passed: bool                               # no required failure
    results: list[VerifierResult] = field(default_factory=list)
    intent_class: str = "*"
    summary: str = ""
    # Oracle COVERAGE — the fail-closed / anti-silent-no-op record (the OpenEvolve
    # "cascade that looks configured but falls through" footgun, generalized). Keys:
    #   expected_mechanical_confirm  a mechanical confirm-oracle was expected for
    #                                this intent-class (from intent.mechanically_confirmable)
    #   mechanical_confirm_ran       one actually produced a verdict (PASS/FAIL, not SKIP)
    #   complete                     not (expected and absent) — the gate really ran
    #   gap                          the honest "why it isn't fully trusted" when incomplete
    # Empty dict (direct-constructed verdicts) reads as fully-covered by default, so
    # this is backward-compatible with callers that build a Verdict by hand.
    coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier, "passed": self.passed, "intent_class": self.intent_class,
            "summary": self.summary, "coverage": self.coverage,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @property
    def coverage_complete(self) -> bool:
        """True unless an oracle that was EXPECTED to confirm this change silently
        didn't run. Absent coverage data (hand-built verdicts) → treated complete."""
        return bool(self.coverage.get("complete", True))

    @property
    def auto_promotable(self) -> bool:
        """Only objectively-confirmed changes with FULL oracle coverage may
        auto-promote. A ``verified`` tier whose expected mechanical oracle merely
        skipped can never auto-ship — fail closed, never over-claim. Everything
        else (probable / human_required / any coverage gap) goes through the
        human/endorsement path."""
        return self.tier == TIER_VERIFIED and self.coverage_complete


# --- registry (mirrors health.py) -----------------------------------------

_REGISTRY: dict[str, Verifier] = {}


def register_verifier(v: Verifier) -> None:
    _REGISTRY[v.name] = v


def registered_verifiers() -> dict[str, Verifier]:
    return dict(_REGISTRY)


def clear_registry() -> None:  # for tests
    _REGISTRY.clear()


# --- convenience constructors (wrap existing primitives) -------------------

def mechanical_verifier(name: str, run: VerifierRun, *, confirms_intent: bool,
                        intent_classes: tuple[str, ...] = ("*",), cost: int = 1,
                        required: bool = True) -> Verifier:
    return Verifier(name, ORACLE_MECHANICAL, run, intent_classes, confirms_intent, cost, required)


def judgment_verifier(name: str, judge: Callable[[dict], Awaitable[tuple[bool, float, str]]], *,
                      intent_classes: tuple[str, ...] = ("*",), cost: int = 5,
                      required: bool = False) -> Verifier:
    """Wrap an LLM/heuristic judge (returns (ok, confidence, detail)). A judgment
    oracle confirms intent only as strongly as its confidence allows."""
    async def _run(ctx: dict) -> VerifierResult:
        try:
            ok, confidence, detail = await judge(ctx)
        except Exception as exc:  # noqa: BLE001 — a crashing judge is a skip, not a block
            return VerifierResult(name, ORACLE_JUDGMENT, SKIP, confirms_intent=True,
                                  required=required, detail=f"judge error: {exc!r}")
        return VerifierResult(name, ORACLE_JUDGMENT, PASS if ok else FAIL,
                              confirms_intent=True, score=1.0 if ok else 0.0,
                              confidence=float(confidence), required=required, detail=detail)
    return Verifier(name, ORACLE_JUDGMENT, _run, intent_classes, True, cost, required)


def human_verdict(kept: bool, *, note: str = "") -> VerifierResult:
    """The captured user verdict for a change with no machine oracle (taste).
    Inject into ``verify(..., extra_results=[human_verdict(...)])`` once known."""
    return VerifierResult("human", ORACLE_HUMAN, PASS if kept else FAIL,
                          confirms_intent=True, score=1.0 if kept else 0.0,
                          required=True, detail=note)


# --- the router -----------------------------------------------------------

async def verify(ctx: dict, *, intent_class: str = "*",
                 verifiers: dict[str, Verifier] | None = None,
                 extra_results: list[VerifierResult] | None = None,
                 judgment_confidence_floor: float = 0.7,
                 expect_mechanical_confirm: bool = False) -> Verdict:
    """Run the applicable verifiers cheap→expensive and compute the honest
    oracle-tier verdict. Short-circuits on the first required failure (cheap
    checks gate the expensive ones). ``extra_results`` lets callers inject
    already-known results (e.g. a captured human verdict).

    ``expect_mechanical_confirm`` (from ``intent.mechanically_confirmable``) drives
    fail-closed COVERAGE: when a mechanical confirm-oracle was expected for this
    class but none actually ran, the verdict records the gap and can never be
    read as fully-trusted/auto-promotable — the gate is not allowed to silently
    no-op."""
    pool = registered_verifiers() if verifiers is None else verifiers
    applicable = sorted(
        (v for v in pool.values() if v.applies_to(intent_class)),
        key=lambda v: v.cost,
    )
    results: list[VerifierResult] = list(extra_results or [])
    for v in applicable:
        try:
            r = await v.run(ctx)
        except Exception as exc:  # noqa: BLE001 — a crashing verifier is a failed required check
            r = VerifierResult(v.name, v.oracle, FAIL, confirms_intent=v.confirms_intent,
                               score=0.0, required=v.required, detail=f"verifier crashed: {exc!r}")
        results.append(r)
        if r.required and r.status == FAIL:
            break  # short-circuit: don't burn expensive checks past a hard failure

    return _verdict(results, intent_class, floor=judgment_confidence_floor,
                    expect_mechanical_confirm=expect_mechanical_confirm)


def _coverage(results: list[VerifierResult], *, expect_mechanical_confirm: bool) -> dict:
    """The fail-closed oracle-coverage record. A mechanical confirm-oracle
    "actually ran" only if it produced a real verdict (PASS/FAIL) — a SKIP is
    the silent-no-op we must surface, not hide. When a confirm was expected for
    the class and none ran, ``complete=False`` and ``gap`` names it (which, for a
    confirmable class, is precisely the Oracle Foundry's demand signal: this
    class has no working oracle yet)."""
    mechanical_confirm_ran = any(
        r.oracle == ORACLE_MECHANICAL and r.confirms_intent and r.status in (PASS, FAIL)
        for r in results)
    complete = (not expect_mechanical_confirm) or mechanical_confirm_ran
    gap = ""
    if expect_mechanical_confirm and not mechanical_confirm_ran:
        gap = ("a mechanical oracle was expected to confirm this change but none "
               "ran (skipped/absent) — the verdict cannot be trusted as verified")
    return {
        "expected_mechanical_confirm": expect_mechanical_confirm,
        "mechanical_confirm_ran": mechanical_confirm_ran,
        "complete": complete, "gap": gap,
    }


def _verdict(results: list[VerifierResult], intent_class: str, *, floor: float,
             expect_mechanical_confirm: bool = False) -> Verdict:
    coverage = _coverage(results, expect_mechanical_confirm=expect_mechanical_confirm)
    required_fail = any(r.required and r.status == FAIL for r in results)
    if required_fail:
        bad = ", ".join(r.name for r in results if r.required and r.status == FAIL)
        return Verdict(TIER_FAILED, passed=False, results=results, intent_class=intent_class,
                       summary=f"FAILED — {bad}", coverage=coverage)

    passed = True
    mech_confirm = any(r.status == PASS and r.oracle == ORACLE_MECHANICAL and r.confirms_intent
                       for r in results)
    human_confirm = any(r.status == PASS and r.oracle == ORACLE_HUMAN for r in results)
    judge_confirm = any(r.status == PASS and r.oracle == ORACLE_JUDGMENT and r.confirms_intent
                        and r.confidence >= floor for r in results)

    # Strength order: objective-mechanical > explicit-human > model-judgment > unconfirmed.
    if mech_confirm:
        tier = TIER_VERIFIED
    elif human_confirm:
        tier = TIER_HUMAN_CONFIRMED
    elif judge_confirm:
        tier = TIER_PROBABLE
    else:
        # Everything green, but nothing CONFIRMED the intent — only "didn't break".
        tier = TIER_HUMAN_REQUIRED

    return Verdict(tier, passed=passed, results=results, intent_class=intent_class,
                   summary=_summary(tier, results, coverage), coverage=coverage)


def _summary(tier: str, results: list[VerifierResult], coverage: dict | None = None) -> str:
    n_pass = sum(1 for r in results if r.status == PASS)
    n_skip = sum(1 for r in results if r.status == SKIP)
    note = {
        TIER_VERIFIED: "intent confirmed by a mechanical oracle",
        TIER_HUMAN_CONFIRMED: "kept by the user",
        TIER_PROBABLE: "intent confirmed by a judgment oracle (probable)",
        TIER_HUMAN_REQUIRED: "no regression, but intent unconfirmed — needs a human verdict",
    }.get(tier, "")
    base = f"{tier} ({n_pass} passed, {n_skip} skipped) — {note}"
    if coverage and not coverage.get("complete", True):
        base += " · COVERAGE GAP: expected mechanical oracle did not run"
    return base
