"""Signal aggregator substrate — see ``aggregator.py`` for the rationale.

Public surface intentionally narrow: callers import ``aggregate_all_users``
or ``aggregate_for_user``; the per-source helpers stay private.
"""
from __future__ import annotations

from augmentum.signals.aggregator import (
    aggregate_all_users,
    aggregate_for_user,
)

__all__ = ["aggregate_all_users", "aggregate_for_user"]
