"""Service health registry for graceful degradation."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Awaitable

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class HealthStatus(str, Enum):
    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class ServiceState:
    """Health state of a single service."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0.0
    last_success: float = 0.0
    last_error: str = ""
    consecutive_failures: int = 0
    check_fn: Callable[[], Awaitable[bool]] | None = None


class ServiceHealthRegistry:
    """Tracks health of all external dependencies.

    Usage:
        registry = ServiceHealthRegistry()
        registry.register("searxng", check_fn=check_searxng)
        registry.register("executor", check_fn=check_executor)

        # Start background health checks
        await registry.start(interval=30)

        # Check before using a service
        if registry.is_available("searxng"):
            results = await search(query)
        else:
            log.warning("searxng_unavailable_skipping_search")
    """

    def __init__(self) -> None:
        self._services: dict[str, ServiceState] = {}
        self._task: asyncio.Task | None = None
        self._running = False
        self._check_interval = 30.0

    def register(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Register a service to monitor."""
        self._services[name] = ServiceState(name=name, check_fn=check_fn)

    def is_available(self, name: str) -> bool:
        """Check if a service is available (UP or DEGRADED)."""
        svc = self._services.get(name)
        if not svc:
            return True  # Unknown services assumed available
        return svc.status in (HealthStatus.UP, HealthStatus.DEGRADED, HealthStatus.UNKNOWN)

    def get_status(self, name: str) -> HealthStatus:
        """Get the current status of a service."""
        svc = self._services.get(name)
        return svc.status if svc else HealthStatus.UNKNOWN

    def mark_success(self, name: str) -> None:
        """Mark a service as healthy (call on successful interaction)."""
        svc = self._services.get(name)
        if svc:
            was_down = svc.status == HealthStatus.DOWN
            svc.status = HealthStatus.UP
            svc.last_success = time.monotonic()
            svc.consecutive_failures = 0
            svc.last_error = ""
            if was_down:
                log.info("service_recovered", service=name)

    def mark_failure(self, name: str, error: str = "") -> None:
        """Mark a service interaction as failed."""
        svc = self._services.get(name)
        if not svc:
            return
        svc.consecutive_failures += 1
        svc.last_error = error

        if svc.consecutive_failures >= 3:
            if svc.status != HealthStatus.DOWN:
                log.warning(
                    "service_down", service=name, error=error,
                    failures=svc.consecutive_failures,
                )
            svc.status = HealthStatus.DOWN
        elif svc.consecutive_failures >= 1:
            svc.status = HealthStatus.DEGRADED

    def snapshot(self) -> dict:
        """Get current health state of all services."""
        return {
            name: {
                "status": svc.status.value,
                "last_error": svc.last_error,
                "consecutive_failures": svc.consecutive_failures,
                "seconds_since_success": (
                    round(time.monotonic() - svc.last_success, 1)
                    if svc.last_success > 0 else None
                ),
            }
            for name, svc in self._services.items()
        }

    async def start(self, interval: float = 30.0) -> None:
        """Start background health check loop."""
        self._check_interval = interval
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        log.info(
            "health_checks_started", services=list(self._services.keys()),
            interval_s=interval,
        )

    async def stop(self) -> None:
        """Stop background health checks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _check_loop(self) -> None:
        """Periodically check all services with registered check functions."""
        while self._running:
            for name, svc in self._services.items():
                if svc.check_fn is None:
                    continue
                try:
                    ok = await asyncio.wait_for(svc.check_fn(), timeout=10)
                    svc.last_check = time.monotonic()
                    if ok:
                        self.mark_success(name)
                    else:
                        self.mark_failure(name, "health check returned False")
                except asyncio.TimeoutError:
                    svc.last_check = time.monotonic()
                    self.mark_failure(name, "health check timed out")
                except Exception as exc:
                    svc.last_check = time.monotonic()
                    self.mark_failure(name, str(exc))

            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
