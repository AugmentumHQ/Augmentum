"""Insight — the L2 (fusion) output and L3 (judgment) input.

An ``Insight`` is one synthesized, would-be-surfaced thing: the product of
correlating ≥1 perceived signals into something potentially worth the user's
attention. It is NOT yet a delivery — the judgment gate (``judgment.py``) decides
whether it stays silent, files for pull, is spoken, or proposes an action.

This is the typed-entity discipline from the design (Apple App Intents lesson):
the pipeline reasons over meaning-bearing insights with an explicit evidence chain
and a regret-grouping ``shape`` — never raw rows. See
``docs/superpowers/specs/2026-06-25-sovereign-perception-pipeline-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Delivery channels the judgment gate can choose. String constants (not an enum)
# to match the codebase convention for small closed vocabularies (stakes, kinds).
SILENT = "silent"                  # filed for recall only — never surfaced unsolicited
FILE_FOR_PULL = "file_for_pull"    # lands in the Today digest, glanceable on open
SPEAK = "speak"                    # surfaced now (interruption or in-conversation)
ACT_WITH_CONSENT = "act_with_consent"  # propose an action via the gated-offer confirm

CHANNELS = frozenset({SILENT, FILE_FOR_PULL, SPEAK, ACT_WITH_CONSENT})


@dataclass(frozen=True, slots=True)
class Insight:
    """One synthesized candidate for the user's attention.

    ``value`` and ``confidence`` are the two [0,1] axes the gate multiplies into a
    base score; ``shape`` is the signature the regret loop damps per-user (so
    "social pressure during focus" can be learned-down without muting all social).
    """

    kind: str                                  # semantic kind, e.g. "logistics.flight_change"
    summary: str                               # the synthesized line she'd say/show
    shape: str = ""                            # regret-grouping signature (defaults to kind's head)
    evidence: list[str] = field(default_factory=list)  # the fused signals — the "why"
    value: float = 0.5                         # [0,1] how much this matters
    confidence: float = 0.5                    # [0,1] how sure the fusion is
    time_critical: bool = False                # does acting/knowing late lose value?
    suggested_action: str = ""                 # optional verb id or action description
    stakes: str = "trivial_reversible"         # gates ACT_WITH_CONSENT routing
    expires_at: float | None = None            # epoch seconds; past → the gate drops it

    def __post_init__(self) -> None:
        # shape defaults to the dotted head of kind ("logistics.flight_change" →
        # "logistics") so related insights share a regret bucket without each
        # call having to set it. frozen dataclass → object.__setattr__.
        if not self.shape:
            object.__setattr__(self, "shape", (self.kind.split(".", 1)[0] or self.kind))

    @property
    def base_score(self) -> float:
        """value × confidence, clamped to [0,1] — the regret-independent strength."""
        return max(0.0, min(1.0, self.value)) * max(0.0, min(1.0, self.confidence))


@dataclass(frozen=True, slots=True)
class DeliveryDecision:
    """The judgment gate's verdict for one insight."""

    channel: str                # one of CHANNELS
    reason: str                 # human-readable why (for logs + the Observatory)
    spent_budget: bool = False  # True iff this consumed an interruption-budget unit

    @property
    def is_interruption(self) -> bool:
        return self.spent_budget
