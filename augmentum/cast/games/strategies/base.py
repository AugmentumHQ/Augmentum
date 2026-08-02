"""CastStrategy ABC + registry.

Mirrors :class:`augmentum.modes.base.ModeHandler` (ABC + template
method) and :class:`augmentum.game_stream.profiles.ProfileRegistry`
(declarative singleton registry).

A CastStrategy is a small plugin that knows how to:

  - decide whether it *can* serve a given title on a given host
    (``can_handle``)
  - turn the (title, profile) pair into a ``PreparedCast`` ready for
    library2 to ship to the receiver (``prepare``)

The classifier asks every registered strategy ``can_handle`` and picks
the cheapest one that says yes — see :class:`CastClassifier`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from augmentum.cast.games.models import (
    CastProfile,
    CastStrategyKind,
    HostCapabilities,
    PreparedCast,
)

if TYPE_CHECKING:
    pass


class CastStrategy(ABC):
    """ABC for a cast execution strategy.

    Subclasses MUST set ``id`` (matches the strategy slug stored in
    ``CastProfile.strategy``) and ``cost_rank`` (1 = cheapest; the
    classifier picks the lowest rank whose ``can_handle`` is true).
    """

    id: ClassVar[CastStrategyKind] = "shim"
    cost_rank: ClassVar[int] = 99

    @abstractmethod
    async def can_handle(
        self,
        title: dict[str, Any],
        host: HostCapabilities,
    ) -> bool:
        """True iff this strategy *could* serve ``title`` given the
        host's capabilities. Cheap check — no I/O. Used by the
        classifier to rank strategies for a given cast."""

    @abstractmethod
    async def prepare(
        self,
        title: dict[str, Any],
        profile: CastProfile,
    ) -> PreparedCast:
        """Build a PreparedCast describing how the receiver should mount
        the surface + which adapter chain to activate inside.

        Idempotent: prepare() is called once per cast attempt.
        Strategy implementations should NOT mutate the profile here —
        persistence belongs to the registry, not the strategy.
        """


class StrategyRegistry:
    """In-memory registry of installed strategies, indexed by ``id``.

    Lookup is also possible by cost_rank (``cheapest_first()``) so the
    classifier can iterate in preference order without re-sorting on
    every cast.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, CastStrategy] = {}

    def register(self, strategy: CastStrategy) -> None:
        sid = str(strategy.id)
        if not sid:
            raise ValueError("CastStrategy.id is required")
        self._strategies[sid] = strategy

    def get(self, strategy_id: str) -> CastStrategy | None:
        return self._strategies.get(strategy_id)

    def has(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies

    def list(self) -> list[CastStrategy]:
        return list(self._strategies.values())

    def cheapest_first(self) -> list[CastStrategy]:
        return sorted(self._strategies.values(), key=lambda s: s.cost_rank)


# Module-level singleton mirroring ``profile_registry`` in
# ``game_stream/profiles.py``. Tests instantiate their own.
strategy_registry = StrategyRegistry()
