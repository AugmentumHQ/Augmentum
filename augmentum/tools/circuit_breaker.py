"""Circuit breaker for tool execution — prevents hammering broken services."""

from __future__ import annotations

import time
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _ToolState:
    failures: int = 0
    opened_at: float = 0.0
    half_open: bool = False
    # Why it broke. "temporarily unavailable (too many recent failures)"
    # is a symptom, not a cause — without carrying the original error the
    # breaker turns a diagnosable bug into a mystery for both the model
    # and whoever reads the transcript.
    last_error: str = ""


class ToolCircuitBreaker:
    """Per-tool circuit breaker with configurable threshold and cooldown.

    States:
    - **closed** (healthy): tool executes normally.
    - **open** (broken): tool is skipped.  Transitions to *half-open*
      after *cooldown* seconds.
    - **half-open**: allows one attempt.  Success → closed; failure → open.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 60.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._states: dict[str, _ToolState] = {}

    def _get(self, tool_name: str) -> _ToolState:
        if tool_name not in self._states:
            self._states[tool_name] = _ToolState()
        return self._states[tool_name]

    def is_open(self, tool_name: str) -> bool:
        """Return True if the tool should be skipped (breaker open)."""
        state = self._get(tool_name)
        if state.failures < self.threshold:
            return False

        elapsed = time.monotonic() - state.opened_at
        if elapsed >= self.cooldown:
            # Transition to half-open: allow one attempt
            state.half_open = True
            return False

        return True

    def record_success(self, tool_name: str) -> None:
        """Reset the breaker after a successful execution."""
        state = self._get(tool_name)
        if state.failures > 0:
            log.info("circuit_breaker_closed", tool=tool_name)
        state.failures = 0
        state.opened_at = 0.0
        state.half_open = False
        state.last_error = ""

    def last_error(self, tool_name: str) -> str:
        """The most recent underlying failure for ``tool_name`` (may be empty)."""
        return self._get(tool_name).last_error

    def record_failure(self, tool_name: str, error: str = "") -> None:
        """Record a failure.  Opens the breaker at threshold."""
        state = self._get(tool_name)
        state.failures += 1
        if error:
            state.last_error = error
        state.half_open = False
        if state.failures >= self.threshold:
            state.opened_at = time.monotonic()
            log.warning(
                "circuit_breaker_opened",
                tool=tool_name,
                failures=state.failures,
                cooldown=self.cooldown,
            )
