"""Shared dispatch policy constants — used by both standing_tasks and
the management-verb dispatcher (event_bus.py).

Phase 1 catalog (docs/superpowers/working/companion_management_verb_
catalog.md) flagged ``standing_tasks._MAX_CONSECUTIVE_ERRORS = 5`` as a
constant the dispatcher would need to redeclare. Lifting it here so
both substrates import from one source. Per anti-duplication warning
#3 in the Phase 2 audit.

Future additions land here too: trust-tier defaults, default cost
envelopes per safety class, default cooldowns per dispatch class.
"""

from __future__ import annotations

# Maximum consecutive failures before the dispatcher auto-pauses a verb
# (or standing_tasks pauses a task). Same threshold both places so the
# operator's mental model — "Becca pauses things that fail 5x in a row"
# — applies uniformly.
DEFAULT_MAX_CONSECUTIVE_ERRORS: int = 5

# Maximum depth of a verb-fanout chain. A management verb that fires
# another management verb is depth=1. The dispatcher's chain_depth
# check is ``>=`` so depth N == limit N is skipped — the limit value
# is the count of permitted chained verbs.
#
# Sized for the Phase 3c three-verb path: emit_pad_if_delta (depth 0)
# → propose_action (depth 1) → enqueue_proposed_action (depth 2).
# A fourth chained verb would be depth 3 ≥ 3 and skip with
# CHAIN_DEPTH_EXCEEDED logged. Future taxonomy additions whose chain
# legitimately needs another hop should bump this, not work around it.
#
# Core verbs (notify, etc.) are OUTSIDE this count — a management verb
# can always invoke an allowlisted core verb regardless of depth.
DEFAULT_CHAIN_DEPTH_LIMIT: int = 3

# Coalesce window for burst events. Mirrors the existing
# TickLoop._COALESCE_WINDOW_S = 0.25 so PresenceBus → dispatcher
# fanout shares the same burst-handling assumption as the original
# tick-driven path being absorbed.
DEFAULT_COALESCE_WINDOW_S: float = 0.25

# Default cost envelope ceilings — applied when a verb declares no
# explicit envelope. Generous enough that most tick-aligned
# maintenance verbs run uncapped in practice; only outliers need to
# override.
DEFAULT_MAX_WALLCLOCK_MS: int = 5_000      # 5s per invocation
DEFAULT_MAX_DB_OPS: int = 1_000            # Per-invocation aiosqlite ops

__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_ERRORS",
    "DEFAULT_CHAIN_DEPTH_LIMIT",
    "DEFAULT_COALESCE_WINDOW_S",
    "DEFAULT_MAX_WALLCLOCK_MS",
    "DEFAULT_MAX_DB_OPS",
]
