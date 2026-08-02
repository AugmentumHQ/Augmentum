"""Periodic healthcheck loop for the SearXNG proxy manager.

A long-running asyncio task that wakes every N minutes (configured by
``search_proxy_healthcheck_interval_minutes``), probes every configured
proxy, and asks the manager to reconcile (re-pick + write settings.yml
+ restart SearXNG) if the active choice changed.

Spawned from the server lifespan startup if a manager exists; cancelled
on shutdown. Idempotent — multiple starts replace the prior task. The
loop checks the live ``settings`` object on every tick so toggling
``search_proxy_rotation_enabled`` or changing the interval takes effect
at the next wake-up without needing a restart.

Not in :mod:`augmentum.jobs` because the jobs primitive is for queue-
driven, restart-survivable, user-triggered work. This is a system loop:
it has no input payload, runs as long as the process does, and its
state is reconstructed from settings on each tick.
"""

from __future__ import annotations

import asyncio
from typing import Any

from augmentum.search.proxy_manager import SearxngProxyManager
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# How often we wake up to RE-CHECK the settings even when rotation is
# disabled. Cheap (no probes), just lets a settings change resume the
# loop without a restart.
_DISABLED_POLL_SECONDS = 30.0


class ProxyHealthcheckLoop:
    """One-instance-per-process supervisor for the healthcheck task."""

    def __init__(
        self,
        *,
        manager: SearxngProxyManager,
        settings_provider: Any,  # callable returning the live Settings
    ) -> None:
        self._manager = manager
        self._settings_provider = settings_provider
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        """Start (or restart) the loop. Idempotent."""
        self.stop()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="searxng_proxy_healthcheck")
        log.info("searxng_proxy_healthcheck_started")

    def stop(self) -> None:
        """Signal the loop to exit. Idempotent."""
        if self._task is not None and not self._task.done():
            self._stop_event.set()
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        # First tick: short sleep so the server finishes coming up
        # before we hit the network.
        await self._sleep_or_stop(5.0)
        while not self._stop_event.is_set():
            try:
                settings = self._settings_provider()
                if not getattr(settings, "search_proxy_rotation_enabled", False):
                    # Rotation disabled — just sleep and re-check the
                    # setting. Don't burn network on probes nobody asked
                    # for. We still tick frequently so re-enabling takes
                    # effect quickly without a restart.
                    await self._sleep_or_stop(_DISABLED_POLL_SECONDS)
                    continue
                # Re-parse the proxy list each tick — handles the case
                # where the user added/removed proxies since the last
                # cycle without explicitly poking the manager.
                await self._manager.update_proxy_list(
                    getattr(settings, "search_proxies", "") or ""
                )
                await self._manager.healthcheck_all()
                fallback = bool(
                    getattr(settings, "search_proxy_fallback_direct_enabled", True)
                )
                await self._manager.reconcile(fallback_to_direct=fallback)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                log.warning("searxng_proxy_healthcheck_iter_failed", error=str(exc))
            interval_min = max(
                1,
                int(
                    getattr(
                        self._settings_provider(),
                        "search_proxy_healthcheck_interval_minutes",
                        5,
                    )
                ),
            )
            await self._sleep_or_stop(interval_min * 60.0)

    async def _sleep_or_stop(self, seconds: float) -> None:
        """Sleep that wakes early on shutdown signal."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except TimeoutError:
            return
