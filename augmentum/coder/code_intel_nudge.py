"""Code-intel adoption nudges (find_symbol / file_outline / batch reads).

The batch-read zero-adoption postmortem (2026-07-03, NATIVE_META v2.2)
proved schema-only exposure gets ZERO adoption from local models — the
habit only shifts when the teaching lands in the system prompt AND a
result-time nudge fires when the model falls back to the old pattern.
find_symbol/file_outline shipped (279a3c4) with neither. This tracker is
the result-time half; the prompt teaching lives in NATIVE_SYSTEM v2.8.

Two one-shot detectors, both pure observation (no I/O, no state outside
the turn):

* **symbol grep** — a ``code_grep`` whose pattern is definition-shaped
  (``def foo`` / ``class Bar`` / ``function baz`` …) is a one-hop
  ``find_symbol`` done the slow way. Nudge after ``grep_nudge_at``
  such calls in one turn.
* **single-read streak** — ``streak_nudge_at`` consecutive iterations
  each containing exactly one single-path ``file_read`` is the
  chain-of-single-reads habit; nudge toward ``paths=[...]`` batching
  and ``file_outline`` triage.

Either detector disarms permanently for the turn once the model uses
the corresponding better tool on its own — a model already adopting
the tools should never be nagged about them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Definition-shaped grep patterns: an escaped-or-bare keyword+name hunt.
# Deliberately conservative — a plain identifier grep is often a
# *usage* hunt (find_symbol wouldn't help), so only keyword-prefixed
# patterns count.
_DEFINITION_GREP_RE = re.compile(
    r"^\s*\^?\s*(?:async\s+def|def|class|function|func|fn|interface|struct|"
    r"trait|impl|type|const|enum)(?:\\s|\\b|[\s(])",
)


def symbol_grep_nudge_body(pattern: str, count: int) -> str:
    return (
        f"You've run {count} definition-shaped code_grep calls "
        f"(latest: `{pattern.strip()[:80]}`). `find_symbol` answers "
        "\"where is X defined?\" in ONE hop from the workspace symbol "
        "index — no grep chain, no follow-up read to find the line. "
        "Use `find_symbol` with the bare name (optionally kind="
        "function/class/method) for definition lookups; keep code_grep "
        "for usages and free text."
    )


def single_read_nudge_body(streak: int) -> str:
    return (
        f"Your last {streak} iterations each issued a single one-file "
        "file_read. Two cheaper moves: (1) when you already know 2+ "
        "files you need, ONE file_read with `paths=[...]` replaces the "
        "chain; (2) when you only need a file's structure to decide "
        "whether it matters, `file_outline` (also batched via "
        "`paths=[...]`) returns classes/functions with line ranges "
        "without spending context on the body."
    )


@dataclass
class CodeIntelAdoptionTracker:
    """Per-turn tracker; feed it every successful tool call, then call
    :meth:`end_iteration` once per loop iteration."""

    grep_nudge_at: int = 2
    streak_nudge_at: int = 3

    _definition_greps: int = 0
    _last_definition_pattern: str = ""
    _grep_nudge_fired: bool = False
    _grep_disarmed: bool = False

    _single_read_streak: int = 0
    _iter_single_reads: int = 0
    _streak_nudge_fired: bool = False
    _streak_disarmed: bool = False

    _pending: list[tuple[str, str]] = field(default_factory=list)

    def observe(self, tool: str, tool_input: dict) -> None:
        """Record one successful tool call."""
        if tool == "find_symbol":
            self._grep_disarmed = True
        elif tool == "file_outline":
            self._streak_disarmed = True
        elif tool == "code_grep":
            pattern = str((tool_input or {}).get("pattern") or "")
            if _DEFINITION_GREP_RE.match(pattern):
                self._definition_greps += 1
                self._last_definition_pattern = pattern
                if (
                    not self._grep_disarmed
                    and not self._grep_nudge_fired
                    and self._definition_greps >= self.grep_nudge_at
                ):
                    self._grep_nudge_fired = True
                    self._pending.append((
                        "symbol_grep_nudge",
                        symbol_grep_nudge_body(pattern, self._definition_greps),
                    ))
        elif tool == "file_read":
            paths = (tool_input or {}).get("paths")
            if isinstance(paths, list | tuple) and len(paths) > 1:
                # A real batch read is adoption — disarm for the turn.
                self._streak_disarmed = True
            else:
                self._iter_single_reads += 1

    def end_iteration(self) -> list[tuple[str, str]]:
        """Close out one loop iteration; return pending (kind, body) nudges."""
        if self._iter_single_reads == 1:
            self._single_read_streak += 1
        elif self._iter_single_reads > 0:
            # Multiple reads in one iteration = the parallel-fanout
            # pattern, which is fine; reset.
            self._single_read_streak = 0
        # Iterations with no file_read at all are neutral (the model may
        # be editing / running tests) — the streak neither grows nor
        # resets, mirroring the silent-success detector's contract.
        if (
            not self._streak_disarmed
            and not self._streak_nudge_fired
            and self._single_read_streak >= self.streak_nudge_at
        ):
            self._streak_nudge_fired = True
            self._pending.append((
                "single_read_nudge",
                single_read_nudge_body(self._single_read_streak),
            ))
        self._iter_single_reads = 0
        pending, self._pending = self._pending, []
        return pending
