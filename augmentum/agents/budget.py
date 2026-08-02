"""Per-subagent budget accounting.

Three independent caps stacked, any of which can trip:

* ``max_iterations`` — turns of the subagent loop. Bounds reasoning loops.
* ``max_wallclock_seconds`` — real time elapsed since the subagent started.
  Bounds long tool calls (build, test, network).
* ``max_tokens`` — accumulated input+output tokens. Bounds cost directly.

OpenHands' lesson from #5480 / #6357: any single cap is insufficient — a
stuck agent can sit at 0 tokens/sec for the wall-clock duration, an
expensive agent can hit budget before the iteration cap, etc. All three
trip independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubagentBudget:
    """Hard caps for a single subagent run."""

    max_iterations: int = 30
    max_wallclock_seconds: float = 600.0
    max_tokens: int = 200_000


@dataclass
class BudgetTracker:
    """Mutable usage tracker bound to a ``SubagentBudget``.

    Token counts are split (in / out) so the per-run cost ledger can
    surface both without re-fetching from the response objects.
    """

    budget: SubagentBudget
    iterations: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    def record_iteration(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.iterations += 1
        self.tokens_in += max(0, tokens_in)
        self.tokens_out += max(0, tokens_out)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def exhausted(self) -> tuple[bool, str | None]:
        """Return (True, reason) if any cap has been crossed, else (False, None)."""
        if self.iterations >= self.budget.max_iterations:
            return True, "max_iterations"
        if self.elapsed_seconds() >= self.budget.max_wallclock_seconds:
            return True, "max_wallclock_seconds"
        if self.tokens_total >= self.budget.max_tokens:
            return True, "max_tokens"
        return False, None
