"""Intensity tiers for the shared LoopRunner.

The integrated coding nervous system spec calls for one loop that runs
everywhere at three intensities. Each :class:`Intensity` is a frozen
dataclass describing the loop's budget: how many iterations are allowed,
whether the verify-gate is mandatory, how much prior-turn memory is
folded in, and which set of soft breakers fires.

Naming note
-----------
The existing ``augmentum/modes/coder/intent.py`` already uses the word
"tier" for an orthogonal concept (REFLEX / SURGICAL / COMPOSED / PROJECT)
that shapes the priming tree. To avoid collision we use ``Intensity``
here; the spec uses "intensity tiers" interchangeably.

Tier presets
------------
LIGHT
    Just-act. Model is trusted to know what to do (think native loop).
    Short ceiling, no verify-gate, no prior-turn injection, minimal
    breakers. The escape valve for "stop second-guessing me, just run".

MEDIUM
    Plan + act + verify. The App Builder target. Verify-gate is
    mandatory before termination; soft breakers active; prior-turn
    memory bounded. This is the tier that structurally closes the
    "ships broken" complaint.

HEAVY
    Full coder mode. Hybrid-loop scope: full breaker suite, full
    prior-turn memory, mandatory verify-gate. The default for the
    Coder workspace when the model is opting in to a long-running
    multi-turn objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BreakerSet = Literal["minimal", "standard", "full"]
"""Which soft-breaker bundle :class:`BreakerRegistry` applies (PR-2.3).
- ``minimal`` — only the hard iteration ceiling.
- ``standard`` — adds same_validation_error_repeat, action_stagnation,
  failing_shell_nudge, termination quality gate.
- ``full`` — the complete coder set: test_failure_streak,
  inspection_loop_break, no_write_progress_break, etc."""


@dataclass(frozen=True)
class Intensity:
    """A frozen budget for one LoopRunner invocation.

    Instances are constants — use :data:`LIGHT`, :data:`MEDIUM`,
    :data:`HEAVY` rather than constructing your own. The dataclass is
    public so callers can introspect (e.g. UI surfaces showing
    "Medium intensity").
    """

    name: Literal["light", "medium", "heavy"]
    max_iterations: int
    """Hard ceiling on act-loop iterations. The runner raises a
    ``LoopBudgetExceeded`` (PR-2.4) when reached; per-iteration soft
    breakers may stop earlier."""

    verify_required: bool
    """When True the runner refuses to stream a terminating chunk
    until every outstanding Promise has been verified (PR-2.5). The
    structural fix for "ships broken"."""

    prior_turns_budget: int
    """How many prior-turn summaries to fold into the system block.
    0 = no cross-turn memory (Light); 10 = full coder memory (Heavy)."""

    breakers: BreakerSet
    """Which soft-breaker bundle to activate (see :data:`BreakerSet`)."""

    observation_ledger_enabled: bool
    """Whether the runner records validation errors / tool failures /
    tool call fingerprints. PR-2.2 moves this state out of CoderState
    and into a standalone ObservationLedger that the runner owns."""

    plan_phase_required: bool
    """Whether the runner expects an explicit Plan phase before Act.
    Light skips planning entirely; Medium plans for non-trivial goals;
    Heavy always plans (coder default)."""


# ── Tier presets ──────────────────────────────────────────────────────


LIGHT = Intensity(
    name="light",
    max_iterations=8,
    verify_required=False,
    prior_turns_budget=0,
    breakers="minimal",
    observation_ledger_enabled=False,
    plan_phase_required=False,
)
"""Just-act. Native-loop equivalent. Short ceiling so a misfire is
cheap; no verify-gate so the user-trust contract isn't second-guessed."""


MEDIUM = Intensity(
    name="medium",
    max_iterations=25,
    verify_required=True,
    prior_turns_budget=3,
    breakers="standard",
    observation_ledger_enabled=True,
    plan_phase_required=True,
)
"""Plan + Act + Verify. The App Builder retarget destination — see
PR-2.6. The verify-gate is the structural fix for the "ships broken"
complaint."""


HEAVY = Intensity(
    name="heavy",
    max_iterations=150,
    verify_required=True,
    prior_turns_budget=10,
    breakers="full",
    observation_ledger_enabled=True,
    plan_phase_required=True,
)
"""Full coder mode. The current default for a Coder workspace turn —
PR-2.4 retargets `_act_canonical` onto this tier without behavior
change."""


__all__ = [
    "HEAVY",
    "LIGHT",
    "MEDIUM",
    "BreakerSet",
    "Intensity",
]
