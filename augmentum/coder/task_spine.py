"""Task-list plan spine — shared tracker for the coder strategy loops.

The ``task_list`` tool (Claude Code TodoWrite / Codex update_plan
equivalent) is only useful when the loop closes three feedback edges:

1. **Visibility** — the current list re-rendered every iteration via the
   sticky ``<system-reminder>`` (handler ``_inject_sticky_reminder``),
   so the model always sees its own plan state.
2. **Staleness** — a one-shot nudge when the list has open work but
   hasn't changed for ``TASK_STALE_NUDGE_AT`` iterations (the model is
   working without maintaining its plan, or stalled on it).
3. **Stop discipline** — a one-shot nudge when the model tries to stop
   while the list it engaged with THIS TURN still has unfinished items.

Hybrid grew (1) and (2) inline; native (the default strategy) had none
of the three — the model could call ``task_list`` but never saw the
state again. This module is the single implementation both loops use
(fix-the-class, 2026-07-06).

Cross-turn trap the ``engaged_this_turn`` flag exists for:
``CoderState.tasks`` persists per session, so a leftover list from a
prior turn must NOT block an unrelated quick question's stop. The stop
gate only fires when the model mutated the list during the current
turn; the staleness nudge is inherently per-turn (streak counts this
turn's iterations).

Pure bookkeeping — no I/O, no logging, no message mutation. The loop
owns appending nudge messages and emitting meta chunks so this stays
trivially testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from augmentum.loops.breakers import TASK_STALE_NUDGE_AT


def _signature(tasks: list) -> str:
    return json.dumps(tasks or [], sort_keys=True, default=str)


def _open_items(tasks: list) -> list[dict]:
    return [
        t for t in (tasks or [])
        if isinstance(t, dict) and t.get("status") != "completed"
    ]


@dataclass
class TaskSpineTracker:
    """Per-turn task-list tracker. Construct once at loop start with
    the session's current tasks; call :meth:`observe` once per
    iteration AFTER tool execution; call :meth:`stop_gate_nudge` when
    the loop is about to accept a model stop."""

    last_signature: str
    stale_streak: int = 0
    stale_nudge_fired: bool = False
    engaged_this_turn: bool = False
    stop_nudge_fired: bool = False

    @classmethod
    def start(cls, tasks: list) -> TaskSpineTracker:
        return cls(last_signature=_signature(tasks))

    def observe(self, tasks: list, *, nudge_enabled: bool = True) -> tuple[bool, str]:
        """Advance the staleness streak for this iteration.

        Returns ``(mutated, stale_nudge_body)`` — ``mutated`` is True
        when the list changed since the previous observation (callers
        use it to trigger the mid-turn state persist); the nudge body
        is non-empty at most once per stall (one-shot, re-armed by any
        real mutation).
        """
        sig = _signature(tasks)
        mutated = sig != self.last_signature
        self.last_signature = sig
        if mutated:
            self.engaged_this_turn = True
            self.stale_streak = 0
            # Re-arm the one-shot so a later re-stall in the same turn
            # can nudge again.
            self.stale_nudge_fired = False
            return True, ""

        if tasks and _open_items(tasks):
            self.stale_streak += 1
        else:
            self.stale_streak = 0

        if (
            not nudge_enabled
            or self.stale_streak < TASK_STALE_NUDGE_AT
            or self.stale_nudge_fired
        ):
            return False, ""

        self.stale_nudge_fired = True
        active = next(
            (
                t.get("content", "(unnamed)")
                for t in tasks
                if isinstance(t, dict) and t.get("status") == "in_progress"
            ),
            None,
        )
        if active:
            body = (
                f"You marked '{active}' as in_progress "
                f"{self.stale_streak} iterations ago and the task list "
                "hasn't changed since. If that task is now done, call "
                "task_list to mark it completed and promote the next "
                "pending one. If you're blocked, update the list to "
                "reflect what's actually happening."
            )
        else:
            body = (
                f"Your task list has been static for {self.stale_streak} "
                "iterations and no item is marked in_progress. Call "
                "task_list to promote the next pending task to "
                "in_progress, or mark completed work as done."
            )
        return False, body

    def stop_gate_nudge(self, tasks: list) -> str:
        """Nudge body when a stop is about to be accepted with open
        work on a list the model engaged with this turn; ``""`` to let
        the stop through. One-shot per turn — a model that stands by
        its stop after one nudge terminates normally (it may have a
        legitimate reason; the nudge asks it to say so on the list).
        """
        if self.stop_nudge_fired or not self.engaged_this_turn:
            return ""
        open_items = _open_items(tasks)
        if not open_items:
            return ""
        self.stop_nudge_fired = True
        listed = "; ".join(
            f"'{t.get('content', '(unnamed)')}'" for t in open_items[:5]
        )
        return (
            f"Your task list still shows {len(open_items)} unfinished "
            f"item(s): {listed}. Before finishing: either complete them, "
            "or call task_list to update their status with what actually "
            "happened (e.g. mark items you decided to skip and say why "
            "in your final answer). Don't leave the plan claiming work "
            "is pending that you consider done."
        )


__all__ = ["TaskSpineTracker"]
