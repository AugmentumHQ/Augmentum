"""Per-run metrics over the subagent worker loop.

Phase 0 of the subagent-professionalization program
(``docs/superpowers/specs/2026-06-19-subagent-professionalization.md``):
*you cannot professionalize what you cannot measure*. Every later phase
is A/B-graded against the scorecard these functions produce — a change
that the scorecard can't show is an improvement does not ship.

Everything here is PURE: it operates on the run-dict shape that
``persistence.SubagentRunStore`` already returns (``_row_to_dict``), where
``tool_call_log`` is a list of dicts carrying the ``ToolCallLog`` fields
(``iteration``/``tool``/``args``/``outcome``/``reason``/``output_len``/
``elapsed_ms``). No model, no DB, no async — so it unit-tests against
synthetic rows and runs identically offline (over persisted history) or
live (over a fresh eval set).

Three headline metrics, plus budget rollups:

* **tool-efficiency** — (tool calls that discovered new info) ÷ (total
  tool calls). The redundant-exploration KPI (arXiv:2601.19568). A call
  discovers new info when it SUCCEEDS, returns non-empty output, and is
  not an exact repeat of an earlier call in the same run.
* **verification accuracy / false-positive rate** — the judge's verdict
  vs. the eval set's known-good label. The FP rate (judge said *passed*
  when the run actually failed) is the metric that matters most: the 2026
  literature shows verifier errors are dominated by over-validation
  (SGV arXiv:2507.11662), which is exactly what Phase 1's fail-closed
  change targets. Requires ground-truth labels → ``None`` over unlabeled
  history.
* **reward-hacking Δ-gap** — visible-pass − held-out-pass (SpecBench
  arXiv:2605.21384). Wired here but reports ``None`` until Phase 2
  supplies the visible/held-out split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Outcomes from ``ToolCallLog.outcome``. Only ``success`` can carry new
# information; the rest are dead calls (failure/denied/unavailable/
# exception/argerror).
_SUCCESS_OUTCOME = "success"


def _args_key(args: Any) -> str:
    """Stable key for a tool-call's args, for repeat detection."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(args)


def tool_efficiency(tool_call_log: list[dict[str, Any]] | None) -> float | None:
    """Fraction of tool calls that discovered new info.

    Returns ``None`` when the run made no tool calls (efficiency is
    undefined, not zero — a clean text-only completion shouldn't score 0).
    """
    if not tool_call_log:
        return None
    total = 0
    new_info = 0
    seen: set[tuple[str, str]] = set()
    for entry in tool_call_log:
        if not isinstance(entry, dict):
            continue
        total += 1
        tool = str(entry.get("tool", ""))
        key = (tool, _args_key(entry.get("args")))
        outcome = str(entry.get("outcome", ""))
        output_len = int(entry.get("output_len", 0) or 0)
        # New info iff: succeeded, produced output, and isn't an exact
        # repeat of an earlier call in this run (a repeat returns the same
        # thing → no new info).
        if outcome == _SUCCESS_OUTCOME and output_len > 0 and key not in seen:
            new_info += 1
        seen.add(key)
    if total == 0:
        return None
    return new_info / total


def reward_hacking_gap(
    visible_pass_rate: float | None,
    held_out_pass_rate: float | None,
) -> float | None:
    """Δ-gap = visible − held-out pass rate (SpecBench arXiv:2605.21384).

    A large positive gap means the run passes the tests it can see far
    better than tests it can't — the signature of gaming. ``None`` until
    Phase 2 supplies a held-out split (today there is none, so callers
    pass ``None`` and the scorecard reports ``null``).
    """
    if visible_pass_rate is None or held_out_pass_rate is None:
        return None
    return visible_pass_rate - held_out_pass_rate


@dataclass
class VerificationGrade:
    """Judge accuracy vs ground truth, the Phase-1 target metric."""

    n_labeled: int = 0
    accuracy: float | None = None
    false_positive_rate: float | None = None
    """Judge said ``passed`` but the run actually failed the eval — the
    over-validation error Phase 1 (fail-closed) is built to reduce."""
    false_negative_rate: float | None = None
    """Judge said not-``passed`` but the run actually met criteria."""


def grade_verification(graded: list[tuple[dict[str, Any], bool]]) -> VerificationGrade:
    """Score judge verdicts against known-good labels.

    ``graded`` is a list of ``(run_dict, expected_pass)`` where
    ``expected_pass`` is the eval set's ground truth. The judge's
    prediction is ``run["verification"] == "passed"``.
    """
    n = len(graded)
    if n == 0:
        return VerificationGrade()
    correct = 0
    fp = 0  # predicted pass, actually fail
    fn = 0  # predicted not-pass, actually pass
    actual_fail = 0
    actual_pass = 0
    for run, expected_pass in graded:
        predicted_pass = str(run.get("verification", "")) == "passed"
        if expected_pass:
            actual_pass += 1
        else:
            actual_fail += 1
        if predicted_pass == expected_pass:
            correct += 1
        elif predicted_pass and not expected_pass:
            fp += 1
        elif not predicted_pass and expected_pass:
            fn += 1
    return VerificationGrade(
        n_labeled=n,
        accuracy=correct / n,
        false_positive_rate=(fp / actual_fail) if actual_fail else None,
        false_negative_rate=(fn / actual_pass) if actual_pass else None,
    )


@dataclass
class Scorecard:
    """The artifact every professionalization phase is graded against."""

    n_runs: int = 0
    stop_reasons: dict[str, int] = field(default_factory=dict)
    verification_counts: dict[str, int] = field(default_factory=dict)
    mean_iterations: float | None = None
    mean_tokens_in: float | None = None
    mean_tokens_out: float | None = None
    mean_wallclock_ms: float | None = None
    mean_tool_efficiency: float | None = None
    verification: VerificationGrade = field(default_factory=VerificationGrade)
    reward_hacking_gap: float | None = None
    per_role: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _rollup(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Budget + efficiency rollup over a set of runs (no ground truth)."""
    iters = [int(r.get("iterations", 0) or 0) for r in runs]
    tin = [int(r.get("tokens_in", 0) or 0) for r in runs]
    tout = [int(r.get("tokens_out", 0) or 0) for r in runs]
    wall = [int(r.get("wallclock_ms", 0) or 0) for r in runs]
    effs = [
        e
        for e in (tool_efficiency(r.get("tool_call_log")) for r in runs)
        if e is not None
    ]
    stop_reasons: dict[str, int] = {}
    verif_counts: dict[str, int] = {}
    for r in runs:
        sr = str(r.get("stop_reason", "") or "unknown")
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
        v = str(r.get("verification", "") or "unchecked")
        verif_counts[v] = verif_counts.get(v, 0) + 1
    return {
        "n_runs": len(runs),
        "stop_reasons": stop_reasons,
        "verification_counts": verif_counts,
        "mean_iterations": _mean([float(x) for x in iters]),
        "mean_tokens_in": _mean([float(x) for x in tin]),
        "mean_tokens_out": _mean([float(x) for x in tout]),
        "mean_wallclock_ms": _mean([float(x) for x in wall]),
        "mean_tool_efficiency": _mean(effs),
    }


def aggregate(
    runs: list[dict[str, Any]],
    *,
    graded: list[tuple[dict[str, Any], bool]] | None = None,
    visible_pass_rate: float | None = None,
    held_out_pass_rate: float | None = None,
) -> Scorecard:
    """Build a full scorecard from a list of run dicts.

    ``runs`` — the ``_row_to_dict`` shape from ``SubagentRunStore``.
    ``graded`` — optional ``(run, expected_pass)`` ground-truth pairs from
    the eval set; enables verification accuracy. ``visible/held_out`` — the
    Phase-2 reward-hacking split; ``None`` today.
    """
    base = _rollup(runs)
    per_role: dict[str, dict[str, Any]] = {}
    roles: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        roles.setdefault(str(r.get("role", "") or "unknown"), []).append(r)
    for role, role_runs in sorted(roles.items()):
        per_role[role] = _rollup(role_runs)
    return Scorecard(
        n_runs=base["n_runs"],
        stop_reasons=base["stop_reasons"],
        verification_counts=base["verification_counts"],
        mean_iterations=base["mean_iterations"],
        mean_tokens_in=base["mean_tokens_in"],
        mean_tokens_out=base["mean_tokens_out"],
        mean_wallclock_ms=base["mean_wallclock_ms"],
        mean_tool_efficiency=base["mean_tool_efficiency"],
        verification=grade_verification(graded or []),
        reward_hacking_gap=reward_hacking_gap(visible_pass_rate, held_out_pass_rate),
        per_role=per_role,
    )
