"""Fusion (L1+L2) — turn perceived signals into candidate insights.

A **fuser** examines the current signal context and may emit zero or more
:class:`Insight` candidates. This is the plug-in seam of the whole pipeline: a new
data stream adds a fuser (and, when native, an L0 adapter that fills the context),
and inherits the judgment gate + regret budget for free. One signal read back is an
echo; a fuser that *correlates* signals into a meaning-bearing insight is the
difference (design spec §2).

Deliberately ships with NO built-in fusers. Real fusers arrive WITH their data
streams (notifications first) — "ship the judgment before the data." The framework
here is what they plug into, and is exercised end-to-end by synthetic fusers in the
tests.

``fuse`` dedups candidates by ``shape`` (the regret bucket), keeping the strongest —
so two fusers both firing on "social pressure" don't double-surface; the gate sees
one insight per shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.companion_runtime.perception.insight import Insight
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FusionContext:
    """Everything a fuser may read for one perception pass. All fields optional so
    a fuser only touches what it needs; ``snapshot`` is the presence snapshot
    (``presence_context.now_context``), ``signals`` is an open bag for L0 adapters
    to drop normalized entities into (notifications, calendar windows, …)."""

    user_id: str
    now: float
    in_conversation: bool = False
    snapshot: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)


# A fuser is a pure function: context → candidate insights. It must NOT raise — a
# misbehaving fuser is isolated by ``fuse`` so one bad stream can't blind the rest.
Fuser = Callable[[FusionContext], list[Insight]]

_FUSERS: list[tuple[str, Fuser]] = []


def register_fuser(name: str, fuser: Fuser) -> None:
    """Register a named fuser. Idempotent by name (re-registering replaces) so a
    module re-import in tests doesn't stack duplicates."""
    global _FUSERS
    _FUSERS = [(n, f) for (n, f) in _FUSERS if n != name]
    _FUSERS.append((name, fuser))


def clear_fusers() -> None:
    """Drop all registered fusers (tests + re-init)."""
    _FUSERS.clear()


def registered_fusers() -> list[str]:
    return [n for (n, _) in _FUSERS]


def _dedup_by_shape(insights: list[Insight]) -> list[Insight]:
    """Keep the strongest (highest base_score) insight per shape — so overlapping
    fusers collapse to one candidate per regret bucket."""
    best: dict[str, Insight] = {}
    for ins in insights:
        cur = best.get(ins.shape)
        if cur is None or ins.base_score > cur.base_score:
            best[ins.shape] = ins
    # Stable-ish order: strongest first (the orchestrator spends budget top-down).
    return sorted(best.values(), key=lambda i: -i.base_score)


def fuse(ctx: FusionContext, fusers: list[tuple[str, Fuser]] | None = None) -> list[Insight]:
    """Run every registered fuser over ``ctx`` and return deduped candidates.

    Each fuser is isolated: an exception is logged and skipped (a broken stream
    must never blind the others). ``fusers`` override is for tests."""
    active = fusers if fusers is not None else _FUSERS
    candidates: list[Insight] = []
    for name, fuser in active:
        try:
            out = fuser(ctx) or []
        except Exception:  # noqa: BLE001 — one bad fuser can't break perception
            log.warning("fuser_failed", fuser=name, exc_info=True)
            continue
        for ins in out:
            if isinstance(ins, Insight):
                candidates.append(ins)
    return _dedup_by_shape(candidates)
