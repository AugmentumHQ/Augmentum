"""Tests for augmentum/utils/service_health.py — health registry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from augmentum.utils.service_health import HealthStatus, ServiceHealthRegistry, ServiceState


class TestServiceState:
    """Verify ServiceState defaults."""

    def test_default_status_is_unknown(self):
        state = ServiceState(name="test")
        assert state.status == HealthStatus.UNKNOWN

    def test_default_consecutive_failures_zero(self):
        state = ServiceState(name="test")
        assert state.consecutive_failures == 0


class TestServiceHealthRegistry:
    """Verify health registry behavior."""

    def test_register_service(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        assert "searxng" in reg._services

    def test_unknown_service_is_available(self):
        reg = ServiceHealthRegistry()
        assert reg.is_available("nonexistent") is True

    def test_registered_service_starts_unknown(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        assert reg.get_status("searxng") == HealthStatus.UNKNOWN

    def test_mark_success(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        reg.mark_success("searxng")
        assert reg.get_status("searxng") == HealthStatus.UP
        assert reg.is_available("searxng") is True

    def test_single_failure_marks_degraded(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        reg.mark_failure("searxng", "timeout")
        assert reg.get_status("searxng") == HealthStatus.DEGRADED

    def test_three_failures_marks_down(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        reg.mark_failure("searxng", "err1")
        reg.mark_failure("searxng", "err2")
        reg.mark_failure("searxng", "err3")
        assert reg.get_status("searxng") == HealthStatus.DOWN
        assert reg.is_available("searxng") is False

    def test_recovery_after_down(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        for _ in range(3):
            reg.mark_failure("searxng", "err")
        assert reg.get_status("searxng") == HealthStatus.DOWN
        reg.mark_success("searxng")
        assert reg.get_status("searxng") == HealthStatus.UP

    def test_snapshot(self):
        reg = ServiceHealthRegistry()
        reg.register("searxng")
        reg.mark_success("searxng")
        snap = reg.snapshot()
        assert "searxng" in snap
        assert snap["searxng"]["status"] == "up"

    def test_snapshot_includes_failure_info(self):
        reg = ServiceHealthRegistry()
        reg.register("executor")
        reg.mark_failure("executor", "connection refused")
        snap = reg.snapshot()
        assert snap["executor"]["last_error"] == "connection refused"
        assert snap["executor"]["consecutive_failures"] == 1

    async def test_check_loop_calls_check_fn(self):
        reg = ServiceHealthRegistry()
        check_fn = AsyncMock(return_value=True)
        reg.register("searxng", check_fn=check_fn)
        # Start with very short interval then stop immediately
        await reg.start(interval=0.05)
        await asyncio.sleep(0.15)
        await reg.stop()
        assert check_fn.call_count >= 1
        assert reg.get_status("searxng") == HealthStatus.UP

    async def test_failing_check_fn(self):
        reg = ServiceHealthRegistry()
        check_fn = AsyncMock(return_value=False)
        reg.register("executor", check_fn=check_fn)
        await reg.start(interval=0.05)
        await asyncio.sleep(0.15)
        await reg.stop()
        assert reg.get_status("executor") in (HealthStatus.DEGRADED, HealthStatus.DOWN)
