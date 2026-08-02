"""Telemetry-driven strategy demotion (Phase 4, server side).

The universal input adapter loader inside a cast surface emits a periodic
``augmentum.input_telemetry`` postMessage; the TV shell relays it as a
generic ``surface_event`` (see ``cast_routes`` WS loop). This module
consumes those events and decides whether the *active* cast strategy is
actually reaching the game — and if not, stamps the title's profile so
the :class:`CastClassifier` promotes to a costlier strategy on the next
cast.

Why not the dispatch ratio? The spec proposed ``dispatches_per_frame < 1``
as the failure signal, but ``adapters/gamepad_api.js`` increments the
dispatch counter once *per received frame* regardless of whether the shim
reached a realm — so for the main cross-origin case frames and dispatches
climb 1:1 and the ratio never dips. The honest, positive signal is an
**unreachable cross-origin iframe with input flowing**: the loader's
``_scanTargets()`` reports ``unreachable_targets`` by probing each iframe's
``contentWindow.document`` (a SecurityError means cross-origin). A
cheap-``shim`` cast whose game sits behind that boundary is exactly what
the origin-proxy strategy fixes.

The demoter is deliberately small + deterministic: it accumulates the
loader's own per-tick ``window_ms`` (no wall-clock dependency) so a sparse
or chatty receiver both converge on the same threshold, and it only acts
on ``shim`` casts (the one strategy with a costlier registered fallback —
``proxy``; ``containerized`` is Phase 5 and unbuilt, so a failing proxy has
nowhere to go and we log rather than thrash).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from augmentum.cast.games.models import (
    CLASSIFIED_TELEMETRY,
    STRATEGY_SHIM,
    CastProfile,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Defaults — conservative so we never demote a working same-origin cast.
# A 15s active window + 20 input frames means the user genuinely tried to
# play before we conclude the chain isn't landing.
DEFAULT_MIN_FRAMES = 20
DEFAULT_MIN_WINDOW_MS = 15_000.0


@dataclass(slots=True)
class _Accum:
    """Per-(user, title) accumulator across telemetry ticks."""

    frames: int = 0
    window_ms: float = 0.0
    unreachable: int = 0
    strategy: str = ""
    demoted: bool = False  # latched so we record at most once per surface

    def reset(self) -> None:
        self.frames = 0
        self.window_ms = 0.0
        self.unreachable = 0


class TelemetryDemoter:
    """Evaluates input telemetry + demotes under-reaching cast strategies.

    Stateful but single-event-loop (one process). One accumulator per
    ``(user_id, title_id)``; entries are cheap and self-reset, so we don't
    bother reaping them — a long-running server accrues at most one tiny
    record per distinct game a user has cast.
    """

    def __init__(
        self,
        *,
        profile_registry: CastProfileRegistry,
        min_frames: int = DEFAULT_MIN_FRAMES,
        min_window_ms: float = DEFAULT_MIN_WINDOW_MS,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._profiles = profile_registry
        self._min_frames = int(min_frames)
        self._min_window_ms = float(min_window_ms)
        self._now = now
        self._accum: dict[tuple[str, str], _Accum] = {}

    async def on_telemetry(self, user_id: str, payload: dict[str, Any]) -> str:
        """Fold one telemetry tick in. Returns one of:

          - ``"accumulating"`` — window not yet long enough to judge
          - ``"healthy"``      — judged reaching the game; accumulator reset
          - ``"demoted"``      — recorded a failure so the next cast promotes
          - ``"skipped"``      — not actionable (no title, wrong strategy, …)

        The string return is for tests + structured logging; callers in the
        WS loop ignore it.
        """
        title_id = str(payload.get("title_id") or "").strip()
        if not title_id:
            return "skipped"
        strategy = str(payload.get("strategy") or "").strip().lower()

        key = (user_id, title_id)
        accum = self._accum.get(key)
        if accum is None:
            accum = _Accum()
            self._accum[key] = accum

        accum.frames += int(payload.get("frames_received") or 0)
        accum.window_ms += float(payload.get("window_ms") or 0.0)
        accum.unreachable = int(payload.get("unreachable_targets") or 0)
        if strategy:
            accum.strategy = strategy

        if accum.window_ms < self._min_window_ms:
            return "accumulating"

        # Window elapsed — judge, then reset the rolling counters.
        reaching = not (
            accum.frames >= self._min_frames and accum.unreachable > 0
        )
        if reaching:
            accum.reset()
            return "healthy"

        # Not reaching. We only have a costlier registered strategy for
        # the shim → proxy step today; a failing proxy has nowhere to go
        # (containerized is Phase 5), so log it and stop thrashing.
        if accum.strategy and accum.strategy != STRATEGY_SHIM:
            if not accum.demoted:
                log.info(
                    "cast_telemetry_strategy_stuck",
                    user_id=user_id, title_id=title_id,
                    strategy=accum.strategy, frames=accum.frames,
                    unreachable=accum.unreachable,
                )
                accum.demoted = True
            accum.reset()
            return "skipped"

        if accum.demoted:
            accum.reset()
            return "skipped"

        await self._record_demotion(user_id, title_id, accum.strategy or STRATEGY_SHIM)
        accum.demoted = True
        accum.reset()
        return "demoted"

    async def _record_demotion(
        self, user_id: str, title_id: str, strategy: str,
    ) -> None:
        """Stamp ``failed_at`` so the classifier promotes next cast.

        If a profile row exists we just mark it failed; otherwise we
        persist a minimal telemetry-classified row carrying the failing
        strategy so the classifier has an anchor to promote *from*.
        """
        now = self._now()
        try:
            existing = await self._profiles.get(title_id, user_id=user_id)
            if existing is not None:
                await self._profiles.mark_failed(
                    title_id, user_id=user_id, when=now,
                )
            else:
                await self._profiles.upsert(
                    CastProfile(
                        title_id=title_id,
                        user_id=user_id,
                        strategy=strategy,
                        classified_by=CLASSIFIED_TELEMETRY,
                        classified_at=now,
                        failed_at=now,
                        notes="auto-demoted: input not reaching game",
                    ),
                    user_id=user_id,
                )
            log.info(
                "cast_telemetry_demotion_recorded",
                user_id=user_id, title_id=title_id, from_strategy=strategy,
            )
        except Exception:
            # A demotion that fails to persist must not break the WS loop;
            # the next window will try again.
            log.warning(
                "cast_telemetry_demotion_failed",
                user_id=user_id, title_id=title_id, exc_info=True,
            )
