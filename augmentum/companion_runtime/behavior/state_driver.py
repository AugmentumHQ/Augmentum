"""State driver — moves CompanionState's primary axis based on observed signals.

The state machine in ``state.py`` is mechanically complete (transition_state,
cooldowns, persistence, listeners), but nothing was calling
``transition_state`` from any tick or observer path — state stayed at
``dormant`` forever even when the user was actively chatting. This module
is the missing driver.

It reads three inputs every tick:

1. ``runtime.observed_state`` — last_chat_at, last_tool_at, last_chat_mode
   (maintained by :class:`BeccaObserver`)
2. Wall-clock time vs. ``companion_quiet_hours_start`` / ``_end``
3. The current state-axis value

…and computes a target state from a small rule set:

    quiet-hours window AND no activity in the last QUIET_INACTIVITY_S
      → asleep

    activity within PRESENCE_WINDOW_S
      → present  (overrides quiet hours: she wakes if you ping her)

    otherwise
      → dormant

If the target differs from current, it calls ``transition_state`` with
``reason="state_driver:<rule>"``. Transitions are subject to the existing
2.0s cooldown so a chatty bus can't oscillate the axis.

Wired into :func:`TickLoop._tick` as the first step, before role_channel
+ activity_selector. Cheap (no DB I/O on the read side; transition_state
itself writes a single UPDATE row when it does fire).

Design spec: docs/superpowers/specs/2026-05-14-companion-runtime-design-v2.md §4.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from augmentum.companion_runtime.state import AttentionState
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Activity within this many seconds keeps her in ``present``.
PRESENCE_WINDOW_S: float = 90.0

# Inactivity longer than this during quiet hours drops her to ``asleep``.
# Short enough that a real bedtime resolves within minutes; long enough
# that brief mid-night activity doesn't bounce.
QUIET_INACTIVITY_S: float = 300.0


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    """Parse "HH:MM" / "24:00" into (hour, minute). None on garbage."""
    if not s:
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 24 and 0 <= m < 60):
        return None
    return (h, m)


def _in_quiet_hours(now_local: datetime, start_s: str, end_s: str) -> bool:
    """Is the current local wall-clock inside the quiet-hours window?

    Handles wrap-around (e.g., 22:00 → 07:00). "24:00" is treated as
    "the window starts at midnight" — equivalent to 00:00 for the
    purpose of the inclusion test.
    """
    start = _parse_hhmm(start_s)
    end = _parse_hhmm(end_s)
    if start is None or end is None:
        return False
    start_min = (start[0] % 24) * 60 + start[1]
    end_min = (end[0] % 24) * 60 + end[1]
    now_min = now_local.hour * 60 + now_local.minute
    if start_min == end_min:
        return False  # zero-length window
    if start_min < end_min:
        # Non-wrapping window (e.g., 02:00 → 07:00)
        return start_min <= now_min < end_min
    # Wrapping window (e.g., 22:00 → 07:00)
    return now_min >= start_min or now_min < end_min


def _target_state(
    *,
    observed_state: dict,
    now_wall: float,
    now_local: datetime,
    quiet_start: str,
    quiet_end: str,
) -> AttentionState:
    """Pure decision function — easy to unit test."""
    last_chat = float(observed_state.get("last_chat_at") or 0.0)
    last_tool = float(observed_state.get("last_tool_at") or 0.0)
    most_recent = max(last_chat, last_tool)
    activity_age_s = (now_wall - most_recent) if most_recent > 0 else float("inf")

    if activity_age_s <= PRESENCE_WINDOW_S:
        return AttentionState.PRESENT

    if (
        _in_quiet_hours(now_local, quiet_start, quiet_end)
        and activity_age_s >= QUIET_INACTIVITY_S
    ):
        return AttentionState.ASLEEP

    return AttentionState.DORMANT


async def drive_once(runtime: "CompanionRuntime") -> bool:
    """Compute target state, transition if changed. Returns True on fire."""
    observed_state = getattr(runtime, "observed_state", None)
    if not observed_state:
        # Observer hasn't initialized — common during the first ~250ms
        # of runtime.start() before the observer task runs. Skip silently.
        return False

    from augmentum.config import settings
    quiet_start = getattr(settings, "companion_quiet_hours_start", "24:00")
    quiet_end = getattr(settings, "companion_quiet_hours_end", "07:00")

    target = _target_state(
        observed_state=observed_state,
        now_wall=time.time(),
        now_local=datetime.now(),
        quiet_start=quiet_start,
        quiet_end=quiet_end,
    )

    current = runtime.state.get_state()
    if target == current:
        return False

    # The transition_state call honors its own 2s cooldown — repeated
    # drives during a burst won't thrash the axis.
    try:
        fired = await runtime.state.transition_state(
            target,
            reason=f"state_driver:{current.value}->{target.value}",
        )
    except Exception:
        log.warning("state_driver_transition_failed", exc_info=True)
        return False
    if fired:
        log.debug(
            "state_driver_transitioned",
            companion_id=runtime.companion_id,
            from_state=current.value,
            to_state=target.value,
        )
    return fired


__all__ = [
    "drive_once",
    "PRESENCE_WINDOW_S",
    "QUIET_INACTIVITY_S",
    "_target_state",  # exported for tests
    "_in_quiet_hours",
]
