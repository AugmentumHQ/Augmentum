"""Context-construction infrastructure — token budget governance.

Currently exports the :class:`BudgetTracker` from :mod:`.budget`. The
plan reserves space here for a future ContextPacker facade (Phase 2.1
of the build plan) that will unify the memory / dream / documents /
knowledge / cache subsystems behind a single recall API.
"""

from augmentum.context.budget import BudgetSnapshot, BudgetTracker

__all__ = ["BudgetSnapshot", "BudgetTracker"]
