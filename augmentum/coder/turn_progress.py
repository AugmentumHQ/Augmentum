"""Turn-level progress ledger — the coarse no-progress backstop.

Motivation (2026-07-07, the same three Qwen3.6-35B runs that motivated
``command_carousel``: 150 / 147 / 147 iterations). Each narrow breaker
watches ONE signal — same file (``same_file_edit_break``), same tool
name (``action_stagnation_break``), same error (``same_validation_error
_repeat``), silent success. A carousel that varies its surface form
enough dodges each individual guard while globally accomplishing
nothing: the 150-iter run touched the same ~14 files over and over and
re-ran the same test suite, never growing the changed-file set and never
raising the passing-test count — yet no single narrow breaker tripped.

This ledger is the SUPERSET backstop. It answers one question the narrow
breakers don't: *"across the last N iterations, did ANY coarse measure
of progress move?"* — where progress is defined generously as either

  1. a NEW distinct file was mutated (the changed-file set grew), or
  2. the best passing-test count rose.

If neither moves for ``stall_nudge`` iterations it nudges once; for
``stall_break`` it stops. Because it resets on any genuine forward step,
a legitimately long build (many files, steadily greening tests) never
trips it — only a run that is measurably standing still does.

``command_carousel`` catches the common case early and cheaply; this is
the floor that makes runaway length structurally impossible even for a
carousel shape we didn't anticipate.

Pure bookkeeping — no I/O, no logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.coder.command_carousel import _passed_count


@dataclass
class TurnProgressLedger:
    """Rolling coarse-progress tracker for one turn.

    ``note`` is called once per iteration with that iteration's
    successful mutation paths and the signals of any verification
    commands it ran. It returns ``"" / "nudge" / "break"``.
    """

    stall_nudge: int
    stall_break: int
    files_changed: set[str] = field(default_factory=set)
    best_passed: int = -1
    last_progress_iter: int = 0
    nudged: bool = False

    def note(
        self,
        iteration: int,
        edited_paths: list[str],
        signals: list[str],
    ) -> str:
        progressed = False

        before = len(self.files_changed)
        self.files_changed.update(p for p in edited_paths if p)
        if len(self.files_changed) > before:
            progressed = True

        for sig in signals:
            p = _passed_count(sig)
            if p > self.best_passed:
                self.best_passed = p
                progressed = True

        if progressed:
            self.last_progress_iter = iteration
            self.nudged = False  # re-arm so a LATER stall is caught again
            return ""

        stalled = iteration - self.last_progress_iter
        if self.stall_break > 0 and stalled >= self.stall_break:
            return "break"
        if self.stall_nudge > 0 and stalled >= self.stall_nudge and not self.nudged:
            self.nudged = True
            return "nudge"
        return ""

    def reset_after_handoff(self, iteration: int) -> None:
        """Re-baseline the stall clock after a buddy handoff so the
        buddy is judged on ITS progress, not the looping model's — but
        keep the accumulated file/test state so a buddy that merely
        repeats the same non-progress trips the ceiling promptly."""
        self.last_progress_iter = iteration
        self.nudged = False


def progress_stall_nudge_body(stalled_iters: int, best_passed: int) -> str:
    state = (
        f" The passing-test count has been stuck at {best_passed} the whole time."
        if best_passed >= 0
        else ""
    )
    return (
        f"You have run {stalled_iters} iterations without any measurable "
        f"progress — no new file changed and no additional test started "
        f"passing.{state} You are very likely stuck in a loop. STOP the "
        "current approach: re-read the specific failure, write down in "
        "one line the single concrete thing blocking you, and either fix "
        "THAT or, if you cannot, summarize what you have done and what is "
        "blocking you and hand back to the user. Do not keep repeating "
        "the same class of action."
    )


__all__ = [
    "TurnProgressLedger",
    "progress_stall_nudge_body",
]
