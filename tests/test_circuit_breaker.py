"""Tests for ToolCircuitBreaker — per-tool failure tracking and recovery."""

from __future__ import annotations

import time

from augmentum.tools.circuit_breaker import ToolCircuitBreaker


class TestCircuitBreakerClosed:
    """Normal (closed) state — all calls pass through."""

    def test_starts_closed(self):
        cb = ToolCircuitBreaker(threshold=3, cooldown=60.0)
        assert cb.is_open("web_search") is False

    def test_single_failure_stays_closed(self):
        cb = ToolCircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is False

    def test_below_threshold_stays_closed(self):
        cb = ToolCircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is False


class TestCircuitBreakerOpen:
    """Open state — calls are blocked after threshold failures."""

    def test_opens_at_threshold(self):
        cb = ToolCircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is True

    def test_opens_above_threshold(self):
        cb = ToolCircuitBreaker(threshold=2, cooldown=60.0)
        cb.record_failure("executor")
        cb.record_failure("executor")
        cb.record_failure("executor")
        assert cb.is_open("executor") is True

    def test_independent_tools(self):
        cb = ToolCircuitBreaker(threshold=2, cooldown=60.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        # web_search is open
        assert cb.is_open("web_search") is True
        # calculator is still closed
        assert cb.is_open("calculator") is False


class TestCircuitBreakerReset:
    """Breaker resets after success."""

    def test_success_resets_failures(self):
        cb = ToolCircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_success("web_search")
        # Back to zero failures — needs 3 more to trip
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is False

    def test_success_closes_open_breaker(self):
        cb = ToolCircuitBreaker(threshold=2, cooldown=60.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is True
        # Force past cooldown
        cb._states["web_search"].opened_at = time.monotonic() - 120
        # Half-open: is_open returns False, allowing one attempt
        assert cb.is_open("web_search") is False
        cb.record_success("web_search")
        assert cb.is_open("web_search") is False


class TestCircuitBreakerHalfOpen:
    """Half-open state — allows one attempt after cooldown."""

    def test_transitions_to_half_open_after_cooldown(self):
        cb = ToolCircuitBreaker(threshold=2, cooldown=1.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is True

        # Simulate cooldown expiry
        cb._states["web_search"].opened_at = time.monotonic() - 2.0
        # After cooldown, is_open returns False (half-open allows one call)
        assert cb.is_open("web_search") is False
        assert cb._states["web_search"].half_open is True

    def test_half_open_failure_reopens(self):
        cb = ToolCircuitBreaker(threshold=2, cooldown=1.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        # Simulate cooldown
        cb._states["web_search"].opened_at = time.monotonic() - 2.0
        cb.is_open("web_search")  # triggers half-open
        # The attempt fails
        cb.record_failure("web_search")
        assert cb.is_open("web_search") is True

    def test_half_open_success_closes(self):
        cb = ToolCircuitBreaker(threshold=2, cooldown=1.0)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb._states["web_search"].opened_at = time.monotonic() - 2.0
        cb.is_open("web_search")  # triggers half-open
        cb.record_success("web_search")
        assert cb.is_open("web_search") is False
        assert cb._states["web_search"].failures == 0


class TestCircuitBreakerConfig:
    """Configurable threshold and cooldown."""

    def test_threshold_one(self):
        cb = ToolCircuitBreaker(threshold=1, cooldown=60.0)
        cb.record_failure("test")
        assert cb.is_open("test") is True

    def test_high_threshold(self):
        cb = ToolCircuitBreaker(threshold=10, cooldown=60.0)
        for _ in range(9):
            cb.record_failure("test")
        assert cb.is_open("test") is False
        cb.record_failure("test")
        assert cb.is_open("test") is True
