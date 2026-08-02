"""Interruption budget — the structural guarantee against nagging.

The hardest skill of a proactive assistant is deciding when to say *nothing*
(Google Now lesson). A bounded number of unsolicited interruptions per rolling
window is what makes that guarantee structural rather than a hope: once the budget
is spent, every further insight routes to pull, no matter how the model feels
about it. See the design spec (§4).

The core is PURE — ``decide`` takes the list of recent interruption timestamps and
returns whether one more is allowed — so it's testable without a clock or a store.
``InterruptionBudgetStore`` is a thin in-memory wrapper for the runtime; it is
ephemeral on purpose (a restart resets the window). Note the restart bias is
toward *more* room to interrupt, so a future brick should back this with a small
table if restarts become frequent; for the judgment-loop proof, in-memory is fine.
"""

from __future__ import annotations

from collections import OrderedDict

# 24h rolling window — "per day" without calendar-day edge effects.
WINDOW_S: float = 24 * 60 * 60.0
# Conservative default: at most this many unsolicited interruptions per window.
# Restraint-by-default — the budget should feel generous to NEVER hit in a calm
# day and protective on a noisy one. Overridable via settings.
DEFAULT_CAP: int = 3
# Outer-LRU cap on tracked users, mirroring AttentionStore.
_MAX_TRACKED_USERS = 256


def recent_in_window(timestamps: list[float], now: float, window_s: float = WINDOW_S) -> int:
    """How many of ``timestamps`` fall within ``window_s`` before ``now``."""
    cutoff = now - window_s
    return sum(1 for t in timestamps if t >= cutoff)


def remaining(timestamps: list[float], now: float, *, cap: int = DEFAULT_CAP,
              window_s: float = WINDOW_S) -> int:
    """Interruptions left in the current window (never negative)."""
    return max(0, cap - recent_in_window(timestamps, now, window_s))


def can_spend(timestamps: list[float], now: float, *, cap: int = DEFAULT_CAP,
              window_s: float = WINDOW_S) -> bool:
    """Pure gate: is there room for one more interruption right now?"""
    return remaining(timestamps, now, cap=cap, window_s=window_s) > 0


class InterruptionBudgetStore:
    """Per-user ring of recent interruption timestamps (in-memory, ephemeral).

    Single-process, asyncio-single-threaded — no locking. Each user's timestamp
    list is pruned to the window on write, so it stays bounded (~cap entries)
    without coupling its length to ``cap`` — which keeps ``set_cap`` free to change
    the cap at runtime (e.g. from settings) without losing recent spends."""

    def __init__(self, cap: int = DEFAULT_CAP, window_s: float = WINDOW_S) -> None:
        self.cap = max(0, int(cap))
        self.window_s = float(window_s)
        self._spent: OrderedDict[str, list[float]] = OrderedDict()

    def set_cap(self, cap: int) -> None:
        """Update the per-window cap (e.g. from companion_interruption_budget_per_day).
        Recent spends are preserved — only the threshold changes."""
        self.cap = max(0, int(cap))

    def _ring(self, user_id: str) -> list[float]:
        ring = self._spent.get(user_id)
        if ring is None:
            ring = []
            self._spent[user_id] = ring
            self._spent.move_to_end(user_id)
            while len(self._spent) > _MAX_TRACKED_USERS:
                self._spent.popitem(last=False)
        else:
            self._spent.move_to_end(user_id)
        return ring

    def remaining(self, user_id: str, now: float) -> int:
        if not user_id:
            return 0
        return remaining(self._ring(user_id), now, cap=self.cap, window_s=self.window_s)

    def can_spend(self, user_id: str, now: float) -> bool:
        return self.remaining(user_id, now) > 0

    def spend(self, user_id: str, now: float) -> bool:
        """Record one interruption. Returns False (and records nothing) if the
        budget is already exhausted — callers should have checked, but this makes
        over-spend impossible even on a logic slip."""
        if not user_id or not self.can_spend(user_id, now):
            return False
        ring = self._ring(user_id)
        # prune out-of-window entries so the list stays bounded regardless of cap
        cutoff = now - self.window_s
        ring[:] = [t for t in ring if t >= cutoff]
        ring.append(now)
        return True

    def reset(self) -> None:
        self._spent.clear()


# Process-wide shared store, mirroring presence_context.ATTENTION. The live
# adapter reads/charges this; the cap can be re-set from settings at startup
# (companion_interruption_budget_per_day).
BUDGET = InterruptionBudgetStore()
