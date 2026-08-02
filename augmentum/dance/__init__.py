"""Server-authoritative state for the companion widget dance timeline.

Contains:
  - ``DanceHistoryStore`` — ring buffer of recent playbacks (user-scoped)
  - ``DanceRatingsStore`` — per-animation curation (like/dislike/broken/
    longer), the substrate the conductor consults at selection time

Both are user-scoped per the multi-tenant data isolation contract. The
companion widget reads server-first with a localStorage cache fallback,
so curation follows the user across devices.
"""
from __future__ import annotations

from augmentum.dance.loops_store import DanceLoopsStore
from augmentum.dance.store import DanceHistoryStore, DanceRatingsStore

__all__ = ["DanceHistoryStore", "DanceLoopsStore", "DanceRatingsStore"]
