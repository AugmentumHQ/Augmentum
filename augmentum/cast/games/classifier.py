"""CastClassifier — pick a strategy + profile per cast attempt.

Mirrors the parallel-leg pattern in
``augmentum.knowledge.packs.PackManager.search`` — every leg is asked
``can_handle`` in parallel; we then *rank* (not merge) and pick the
cheapest qualifying strategy. The registry's stored profile overrides
the rank when present (manual override / probe-classified).

The classifier is deliberately small + side-effect-free:

  1. Look up an existing profile in the registry (registry hit wins;
     respect ``failed_at`` to demote one rank if recent).
  2. Otherwise ask every strategy ``can_handle`` and pick the cheapest
     true one.
  3. Build a default profile so subsequent casts have a baseline; the
     caller decides whether to persist it (we don't write-on-read to
     avoid surprising the user with auto-classified rows they didn't
     ask for — Phase 4's probe does the write).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from augmentum.cast.games.models import (
    CLASSIFIED_DEFAULT,
    CastProfile,
    HostCapabilities,
    STRATEGY_PROXY,
    STRATEGY_SHIM,
)
# Importing strategies registers the defaults.
from augmentum.cast.games.strategies import (  # noqa: F401  (side-effect import)
    CastStrategy,
    StrategyRegistry,
    strategy_registry,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Telemetry-driven demotion window. If a profile was marked failed_at
# within this window, the classifier promotes by one cost_rank (the
# next-most-expensive strategy that can_handle).
DEMOTION_WINDOW_S = 30 * 60.0


@dataclass(slots=True)
class ClassifierResult:
    """Outcome of CastClassifier.classify — strategy + profile + source.

    ``source`` is one of:
      - ``registry``      — used the profile that was already persisted
      - ``registry_promoted`` — registry hit, but failed_at < DEMOTION_WINDOW_S
                                so promoted to next cost_rank
      - ``default``       — no registry hit; built a fresh default profile
    """

    strategy: CastStrategy
    profile: CastProfile
    source: str


class CastClassifier:
    """Picks the right strategy + profile for a given (title, host).

    The classifier holds references to the strategy + profile registries
    so the route handlers can construct one per request without
    re-wiring deps. Tests can pass a fresh ``StrategyRegistry`` to
    isolate.
    """

    def __init__(
        self,
        *,
        profile_registry: CastProfileRegistry,
        strategies: StrategyRegistry | None = None,
        demotion_window_s: float = DEMOTION_WINDOW_S,
    ) -> None:
        self._profiles = profile_registry
        self._strategies = strategies or strategy_registry
        self._demotion_window = float(demotion_window_s)

    async def classify(
        self,
        title: dict[str, Any],
        host: HostCapabilities | None = None,
        *,
        user_id: str = "",
    ) -> ClassifierResult:
        """Return a strategy + profile to use for this cast."""
        title_id = str(title.get("id") or title.get("title_id") or "")
        host = host or HostCapabilities()

        existing = await self._profiles.get(title_id, user_id=user_id)
        if existing is not None:
            strategy, source = await self._pick_for_profile(existing, title, host)
            return ClassifierResult(strategy=strategy, profile=existing, source=source)

        # No persisted profile — rank strategies + build a default one.
        strategy = await self._rank_pick(title, host)
        profile = self._build_default_profile(title_id, strategy, title)
        return ClassifierResult(
            strategy=strategy,
            profile=profile,
            source="default",
        )

    # ── internals ────────────────────────────────────────────────

    async def _pick_for_profile(
        self,
        profile: CastProfile,
        title: dict[str, Any],
        host: HostCapabilities,
    ) -> tuple[CastStrategy, str]:
        """Resolve the strategy named in ``profile``. If the profile is
        recently-failed, promote one rank to the next qualifying
        strategy."""
        named = self._strategies.get(profile.strategy)
        recent_fail = (
            profile.failed_at > 0
            and (time.time() - profile.failed_at) < self._demotion_window
        )
        if named is not None and not recent_fail:
            return named, "registry"

        # Demotion path: pick the next-cheapest strategy that can_handle.
        ranked = self._strategies.cheapest_first()
        anchor_rank = named.cost_rank if named is not None else 0
        for s in ranked:
            if s.cost_rank <= anchor_rank:
                continue
            if await s.can_handle(title, host):
                return s, "registry_promoted"
        # Nothing more expensive qualifies — fall back to whatever was
        # named (or the cheapest registered) so we don't block the cast.
        fallback = named or (ranked[0] if ranked else None)
        if fallback is None:
            raise RuntimeError("CastClassifier: no strategies registered")
        return fallback, "registry"

    async def _rank_pick(
        self,
        title: dict[str, Any],
        host: HostCapabilities,
    ) -> CastStrategy:
        """Parallel ``can_handle`` over registered strategies, pick the
        cheapest True — mirrors PackManager's parallel-leg dispatch.
        """
        ranked = self._strategies.cheapest_first()
        if not ranked:
            raise RuntimeError("CastClassifier: no strategies registered")

        async def _check(s: CastStrategy) -> tuple[CastStrategy, bool]:
            try:
                ok = await s.can_handle(title, host)
            except Exception:
                log.warning(
                    "cast_strategy_can_handle_raised",
                    strategy=s.id,
                    exc_info=True,
                )
                ok = False
            return s, ok

        results = await asyncio.gather(*(_check(s) for s in ranked))
        # cheapest_first() ordered the input; take the first True
        for s, ok in results:
            if ok:
                return s
        # Nothing qualified — return the cheapest so the route can fail
        # gracefully (prepare() will raise if it really can't proceed).
        return ranked[0]

    def _build_default_profile(
        self,
        title_id: str,
        strategy: CastStrategy,
        title: dict[str, Any],
    ) -> CastProfile:
        meta = title.get("metadata") if isinstance(title.get("metadata"), dict) else {}
        embed = str(meta.get("embed_url") or title.get("embed_url") or "")
        return CastProfile(
            title_id=title_id,
            strategy=strategy.id,
            embed_url=embed,
            input_chain=("gamepad_api",),
            classified_by=CLASSIFIED_DEFAULT,
            classified_at=0.0,
        )
