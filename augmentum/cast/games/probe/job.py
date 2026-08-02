"""Fire-and-forget probe coordinator — runs on the first cast of an
unknown title and persists the resulting profile for next time.

Wired into the ``/classify`` route: when classify returns a ``default``
profile (no persisted row) and the title carries an ``embed_url``, the
route calls :meth:`CastProbeCoordinator.maybe_probe`, which schedules a
background probe. The current cast is NOT blocked — it proceeds on the
cheap shim default; the probe's result lands in the registry so the NEXT
cast of that title is pre-classified.

Safety:
  - Deduped per ``(user_id, title_id)`` so a burst of casts triggers one
    probe.
  - Strategy is chosen conservatively: ``proxy`` only when the embed is
    cross-origin to the server AND the proxy strategy is actually
    serviceable (``can_handle`` True) — so a probe-written profile can
    never make ``prepare()`` raise. Otherwise ``shim`` (the telemetry
    demotion remains the safety net for strategy escalation).
  - Every failure is swallowed; a probe can improve a cast, never break
    one.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from augmentum.cast.games.models import (
    STRATEGY_PROXY,
    STRATEGY_SHIM,
    HostCapabilities,
)
from augmentum.cast.games.probe.playwright_probe import (
    PlaywrightProbe,
    build_probe_profile,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.cast.games.registry import CastProfileRegistry
    from augmentum.cast.games.strategies.base import StrategyRegistry

log = get_logger(__name__)


def _same_origin(a: str, b: str) -> bool:
    """True iff two URLs share scheme+host+port (default-port aware)."""
    if not a or not b:
        return False
    try:
        pa, pb = urlparse(a), urlparse(b)
    except ValueError:
        return False
    defaults = {"http": 80, "https": 443}
    pa_port = pa.port or defaults.get(pa.scheme.lower())
    pb_port = pb.port or defaults.get(pb.scheme.lower())
    return (
        pa.scheme.lower() == pb.scheme.lower()
        and (pa.hostname or "").lower() == (pb.hostname or "").lower()
        and pa_port == pb_port
    )


class CastProbeCoordinator:
    """Schedules + applies probes without blocking the cast."""

    def __init__(
        self,
        *,
        probe: PlaywrightProbe,
        profile_registry: CastProfileRegistry,
        strategy_registry: StrategyRegistry,
        server_origin: str = "",
    ) -> None:
        self._probe = probe
        self._profiles = profile_registry
        self._strategies = strategy_registry
        self._server_origin = server_origin
        self._inflight: set[tuple[str, str]] = set()

    def maybe_probe(
        self, *, title_id: str, user_id: str, embed_url: str,
    ) -> bool:
        """Schedule a background probe for an unknown title. Returns True
        if a probe was scheduled (False = no embed, already in flight, or
        a row already exists is the caller's check). Never raises."""
        if not title_id or not embed_url:
            return False
        key = (user_id, title_id)
        if key in self._inflight:
            return False
        self._inflight.add(key)
        try:
            asyncio.create_task(self._run(title_id, user_id, embed_url))
        except RuntimeError:
            # No running loop (shouldn't happen from a route) — bail clean.
            self._inflight.discard(key)
            return False
        return True

    async def _run(self, title_id: str, user_id: str, embed_url: str) -> None:
        key = (user_id, title_id)
        try:
            result = await self._probe.probe(embed_url)
            if result is None:
                log.info("cast_probe_no_result", title_id=title_id)
                return
            strategy = await self._choose_strategy(
                title_id, user_id, embed_url,
            )
            profile = build_probe_profile(
                title_id=title_id,
                user_id=user_id,
                embed_url=embed_url,
                result=result,
                strategy=strategy,
            )
            await self._profiles.upsert(profile, user_id=user_id)
            log.info(
                "cast_probe_profile_written",
                title_id=title_id, strategy=strategy,
                input_chain=list(profile.input_chain),
                responded=result.responded,
            )
        except Exception:
            log.warning("cast_probe_run_failed", title_id=title_id, exc_info=True)
        finally:
            self._inflight.discard(key)

    async def _choose_strategy(
        self, title_id: str, user_id: str, embed_url: str,
    ) -> str:
        """Pick a SAFE strategy for the probed profile.

        proxy only when (a) we know the server origin, (b) the embed is
        cross-origin to it, and (c) the proxy strategy reports it can
        serve this title — otherwise shim. The can_handle gate guarantees
        a probe-written ``proxy`` profile won't make prepare() raise.
        """
        if not self._server_origin or _same_origin(embed_url, self._server_origin):
            return STRATEGY_SHIM
        proxy = self._strategies.get(STRATEGY_PROXY)
        if proxy is None:
            return STRATEGY_SHIM
        title: dict[str, Any] = {
            "id": title_id, "title_id": title_id,
            "embed_url": embed_url, "user_id": user_id,
        }
        try:
            if await proxy.can_handle(title, HostCapabilities(has_network_egress=True)):
                return STRATEGY_PROXY
        except Exception:
            log.debug("cast_probe_proxy_can_handle_raised", exc_info=True)
        return STRATEGY_SHIM
