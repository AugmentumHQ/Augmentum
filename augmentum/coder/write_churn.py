"""Same-file write-churn tracker — shared by the coder strategy loops.

Live failure this encodes (2026-07-06, 9B native run ctr_1b36af6c…):
the model got stuck on a bug it couldn't diagnose (a command registry
that stayed empty no matter what) and fell into rewriting the same
file wholesale — 20+ ``file_write`` calls to ``core/cli.py`` in one
turn — because every existing guard was blind to the shape:

- every write SUCCEEDS, so progress accounting kept resetting the
  continuation-nudge machinery (writes ARE the progress signal);
- the identical-call detector needs byte-identical args — each rewrite
  had different content;
- the silent-success detector only watches shell_exec, and the probe
  commands "succeeded" with the failure in their stdout;
- the stop-quality gates (TQG / goal judge / verify gate) only run
  when the model tries to stop — it never stopped.

Hybrid alone had a hard cap (``same_file_edit_break``, default 15,
no early nudge); native — the default strategy — had nothing. This
module is the single implementation for both (fix-the-class): an
early NUDGE at ``same_file_edit_nudge`` mutations of one path
(prescribing the recovery: re-read, hypothesis, surgical edit, a
check that can actually fail) and the existing hard BREAK threshold
after it.

Pure bookkeeping — no I/O, no logging. The loop owns message
appends, meta chunks, and the break.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WriteChurnTracker:
    """Per-turn, per-path mutation counter with a three-rung ladder:
    nudge → escalate → break.

    The escalate rung is the loop-CONFIRMATION signal: the nudge is
    the hypothesis ("you're churning"), and the same path advancing
    ``escalate_margin`` more mutations AFTER the one-shot nudge means
    the model demonstrably can't self-correct — hand the turn to the
    heavyweight buddy (loop-side) instead of waiting for the hard cap.
    The break stays as the backstop for installs with no buddy model.

    ``nudge_at`` / ``break_at`` are resolved by the loop (live
    settings override via ``_live_threshold``) so tests and deploys
    tune them in one place (loops/breakers.py).
    """

    nudge_at: int
    break_at: int
    escalate_margin: int = 3
    counts: dict[str, int] = field(default_factory=dict)
    nudged_paths: set[str] = field(default_factory=set)
    escalated_paths: set[str] = field(default_factory=set)

    def observe(self, edited_paths: list[str]) -> tuple[str, str, int]:
        """Record this iteration's successful mutation paths.

        Returns ``(action, path, count)`` where action is ``"break"``,
        ``"escalate"`` (nudge ignored — one-shot per path),
        ``"nudge"`` (one-shot per path) or ``""``. Higher rungs win
        when several trip in the same iteration.
        """
        for p in edited_paths:
            if p:
                self.counts[p] = self.counts.get(p, 0) + 1

        if self.break_at > 0:
            for p, n in self.counts.items():
                if n >= self.break_at:
                    return "break", p, n
        if self.nudge_at > 0 and self.escalate_margin > 0:
            for p, n in self.counts.items():
                if (
                    n >= self.nudge_at + self.escalate_margin
                    and p in self.nudged_paths
                    and p not in self.escalated_paths
                ):
                    self.escalated_paths.add(p)
                    return "escalate", p, n
        if self.nudge_at > 0:
            for p, n in self.counts.items():
                if n >= self.nudge_at and p not in self.nudged_paths:
                    self.nudged_paths.add(p)
                    return "nudge", p, n
        return "", "", 0

    def reset_counts(self) -> None:
        """Fresh ladder after a model handoff — the buddy deserves its
        own budget instead of inheriting the looping model's count
        (which would trip the hard break almost immediately)."""
        self.counts.clear()
        self.nudged_paths.clear()


def churn_nudge_body(path: str, count: int) -> str:
    """Prescriptive recovery text for the write-churn nudge."""
    return (
        f"You have now written or edited `{path}` {count} times this "
        "turn and your verification signal has not changed. Rewriting "
        "the file again will not fix it. Change approach NOW:\n"
        "1. Re-read the CURRENT file (file_read) — your mental copy is "
        "stale after this many rewrites.\n"
        "2. State a concrete hypothesis for WHY the failure persists "
        "(wrong module imported? two copies of the file? stale "
        "bytecode? state mutated at import time?). Test THE HYPOTHESIS "
        "with a read-only probe before touching the file.\n"
        "3. Make one surgical code_edit — not a full rewrite — and run "
        "a check that can actually FAIL (pytest, python3 -m py_compile, "
        "an assert in a one-liner).\n"
        "4. If you cannot form a hypothesis, dispatch "
        "task_dispatch(role=plan) with the failing output, or ask the "
        "user — don't keep rewriting."
    )


def escalation_handoff_body(
    *, previous_model: str, reason: str, detail: str,
) -> str:
    """Briefing injected when the turn is handed to the buddy model.

    Before 2026-07-06 escalation swapped ``request.model`` silently —
    the buddy woke up inside a transcript full of the weaker model's
    repeated failed actions with no signal that anything was wrong,
    so it tended to continue the same approach by momentum. This
    message is the context Matt asked for: you're here because the
    previous model was likely stuck; verify state before trusting the
    history; the original user goal is unchanged.
    """
    return (
        "<escalation_handoff>\n"
        f"You have just taken over this turn from a weaker model "
        f"({previous_model or 'the previous model'}) that appeared to "
        f"be stuck ({reason}"
        + (f": {detail}" if detail else "")
        + ").\n"
        "Treat the recent transcript with suspicion — it repeated the "
        "same kind of action without the outcome changing. Do NOT "
        "continue its last approach by momentum.\n"
        "1. Re-read the CURRENT contents of anything it was editing; "
        "its rewrites may not match what it believed it wrote.\n"
        "2. Diagnose WHY the repeated action never changed the "
        "failing signal before you act.\n"
        "3. Then continue the user's ORIGINAL request (the first user "
        "message of this turn) to completion, verifying as you go.\n"
        "</escalation_handoff>"
    )


__all__ = [
    "WriteChurnTracker",
    "churn_nudge_body",
    "escalation_handoff_body",
]
