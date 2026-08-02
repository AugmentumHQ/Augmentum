"""Loop-health coordinator — one subsystem over the coder guard ladders.

Before 2026-07-07 the native act loop ran six-plus INDEPENDENT guards
(write-churn, duplicate-read, command-carousel, probe-signal, turn-
progress ledger, identical-call, silent-success, task-staleness,
code-intel adoption), each with its own counter, threshold, and direct
``messages.append(<nudge>)`` channel. Two failure modes followed:

* **No arbitration** — a struggling model could receive several
  simultaneous nudges in one iteration. For a 9B that's exactly the
  context noise that deepens a spiral instead of breaking it.
* **Drift** — every guard was a point-system wired by hand at each
  call site, so coverage silently diverged between loops (the
  recurring "hybrid wiring open" debt; hybrid has since been
  disconnected, which shrinks the problem to keeping ONE loop honest).

This module centralizes the cross-cutting concerns while leaving each
detector's logic in its own module (they remain independently
testable):

* **Construction** — :meth:`LoopHealthCoordinator.create` builds every
  tracker from the live-threshold resolver in one place.
* **Arbitration** — guards ``submit()`` nudges instead of appending
  messages; ``arbitrate()`` returns at most ONE nudge per iteration
  (highest priority wins). Structural interventions (reorientation
  repair, buddy escalation, break) registered via
  :meth:`note_intervention` suppress all nudges that iteration — the
  model is already getting a stronger corrective message.
* **Telemetry** — every event (fired, suppressed, intervention) lands
  in one counter set, exposed by :meth:`summary` for a single
  end-of-turn meta chunk instead of per-guard ad-hoc extras.

Suppressed nudges are DROPPED, not requeued: their bodies reference
"your last N calls"-style windows that go stale, and their trackers'
one-shot flags have already flipped. The suppression counter keeps the
loss visible in telemetry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from augmentum.coder.code_intel_nudge import CodeIntelAdoptionTracker
from augmentum.coder.command_carousel import CommandCarouselTracker
from augmentum.coder.duplicate_calls import DuplicateCallTracker
from augmentum.coder.probe_signal import ProbeSignalTracker
from augmentum.coder.task_spine import TaskSpineTracker
from augmentum.coder.turn_progress import TurnProgressLedger
from augmentum.coder.write_churn import WriteChurnTracker

# Highest-priority first. Corrective signals about an ACTIVE loop beat
# workflow hygiene, which beats advisory tool-adoption coaching. An
# unknown kind sorts last (defensive — a new guard should add itself
# here, but forgetting must not crash arbitration).
NUDGE_PRIORITY: tuple[str, ...] = (
    "flaky_test_nudge",
    "same_file_edit_nudge",
    "identical_result_nudge",
    "duplicate_call_nudge",
    "command_carousel_nudge",
    "probe_no_signal_nudge",
    "progress_stall_nudge",
    "silent_success_nudge",
    "task_stale_nudge",
    "symbol_grep_nudge",
    "single_read_nudge",
)


def _priority(kind: str) -> int:
    try:
        return NUDGE_PRIORITY.index(kind)
    except ValueError:
        return len(NUDGE_PRIORITY)


@dataclass
class PendingNudge:
    kind: str          # meta-chunk status, e.g. "same_file_edit_nudge"
    body: str          # message text WITHOUT the <nudge> wrapper
    extra: dict = field(default_factory=dict)


@dataclass
class LoopHealthCoordinator:
    """Per-turn coordinator. Guards submit; the loop arbitrates once
    per iteration and injects at most one nudge."""

    # Trackers — owned so thresholds/construction live in ONE place.
    task_spine: TaskSpineTracker
    write_churn: WriteChurnTracker
    probe_signal: ProbeSignalTracker
    command_carousel: CommandCarouselTracker
    progress_ledger: TurnProgressLedger
    duplicate_calls: DuplicateCallTracker
    code_intel: CodeIntelAdoptionTracker

    counters: Counter = field(default_factory=Counter)
    _pending: list[PendingNudge] = field(default_factory=list)
    _iter_interventions: int = 0

    @classmethod
    def create(
        cls,
        *,
        threshold: Callable[[str], int],
        tasks,
        tracked_read_tools,
    ) -> LoopHealthCoordinator:
        return cls(
            task_spine=TaskSpineTracker.start(tasks),
            write_churn=WriteChurnTracker(
                nudge_at=threshold("same_file_edit_nudge"),
                break_at=threshold("same_file_edit_break"),
            ),
            probe_signal=ProbeSignalTracker(
                nudge_at=threshold("probe_no_signal_nudge"),
            ),
            command_carousel=CommandCarouselTracker(
                nudge_at=threshold("command_carousel_nudge"),
                reorient_margin=threshold("command_carousel_reorient"),
            ),
            progress_ledger=TurnProgressLedger(
                stall_nudge=threshold("progress_stall_nudge"),
                stall_break=threshold("progress_stall_break"),
            ),
            duplicate_calls=DuplicateCallTracker(
                nudge_at=threshold("duplicate_call_nudge"),
                reorient_margin=threshold("duplicate_call_reorient"),
                tracked_tools=tracked_read_tools,
            ),
            code_intel=CodeIntelAdoptionTracker(),
        )

    # ── per-iteration protocol ───────────────────────────────────────

    def submit(self, kind: str, body: str, extra: dict | None = None) -> None:
        """Queue a nudge for this iteration's arbitration."""
        self._pending.append(PendingNudge(kind, body, dict(extra or {})))

    def note_intervention(self, kind: str) -> None:
        """Record a structural intervention (reorient / escalate /
        break) — suppresses all of this iteration's nudges."""
        self._iter_interventions += 1
        self.counters[kind] += 1

    def arbitrate(self) -> tuple[PendingNudge | None, list[PendingNudge]]:
        """Close the iteration: return ``(winner, suppressed)``.

        Winner is the single highest-priority pending nudge, or None
        when nothing is pending or an intervention already fired.
        """
        pending, self._pending = self._pending, []
        interventions, self._iter_interventions = self._iter_interventions, 0
        if not pending:
            return None, []
        pending.sort(key=lambda n: _priority(n.kind))
        if interventions:
            winner, suppressed = None, pending
        else:
            winner, suppressed = pending[0], pending[1:]
        if winner is not None:
            self.counters[winner.kind] += 1
        for s in suppressed:
            self.counters[f"suppressed:{s.kind}"] += 1
        return winner, suppressed

    # ── turn-end telemetry ───────────────────────────────────────────

    def summary(self) -> dict:
        """Counter snapshot for the end-of-turn meta chunk. Empty dict
        when the turn was healthy (nothing fired)."""
        return dict(self.counters)
