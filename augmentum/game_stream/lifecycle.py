"""Streaming-session state machine.

A streaming session goes through a small handful of well-defined
states. Transitions are validated by ``GameStreamLifecycle.transition``
to keep the runtime honest -- a container can't go ``stopped`` ->
``connected`` without first passing through ``starting`` and ``ready``.

```
                      ┌──────────────┐
                      │   stopped    │ <─────────────┐
                      └──────┬───────┘               │
                             │ start                 │ exit clean
                             ▼                       │ exit crash → 'crashed'
                      ┌──────────────┐               │
                      │   starting   │ ──────────────┤
                      └──────┬───────┘               │
                             │ container ready       │
                             ▼                       │
                      ┌──────────────┐               │
                  ┌──>│    ready     │               │
                  │   └──────┬───────┘               │
       reconnect  │          │ client connected      │
                  │          ▼                       │
                  │   ┌──────────────┐               │
                  └───┤  connected   │ ──┐           │
                      └──────┬───────┘   │           │
                             │ client    │ idle      │
                             │ dropped   │ timeout   │
                             ▼           ▼           │
                      ┌──────────────┐               │
                      │     idle     │ ──────────────┘
                      └──────────────┘
```
"""

from __future__ import annotations

from enum import Enum

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class SessionStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    CONNECTED = "connected"
    IDLE = "idle"
    PAUSED = "paused"
    STOPPING = "stopping"
    CRASHED = "crashed"


# Allowed transitions. (from, to) -> True if legal.
_LEGAL_TRANSITIONS: frozenset[tuple[SessionStatus, SessionStatus]] = frozenset({
    # cold start
    (SessionStatus.STOPPED, SessionStatus.STARTING),
    # spin-up complete
    (SessionStatus.STARTING, SessionStatus.READY),
    # client connects
    (SessionStatus.READY, SessionStatus.CONNECTED),
    # client drops (container stays alive briefly for reconnect)
    (SessionStatus.CONNECTED, SessionStatus.IDLE),
    # reconnect from idle
    (SessionStatus.IDLE, SessionStatus.READY),
    (SessionStatus.IDLE, SessionStatus.CONNECTED),
    # pause (cgroup freeze) — accepted from any actively-running state
    (SessionStatus.CONNECTED, SessionStatus.PAUSED),
    (SessionStatus.IDLE, SessionStatus.PAUSED),
    (SessionStatus.READY, SessionStatus.PAUSED),
    # resume (cgroup thaw) — back to a connect-eligible state
    (SessionStatus.PAUSED, SessionStatus.READY),
    (SessionStatus.PAUSED, SessionStatus.CONNECTED),
    # graceful shutdown
    (SessionStatus.READY, SessionStatus.STOPPING),
    (SessionStatus.CONNECTED, SessionStatus.STOPPING),
    (SessionStatus.IDLE, SessionStatus.STOPPING),
    (SessionStatus.PAUSED, SessionStatus.STOPPING),
    (SessionStatus.STOPPING, SessionStatus.STOPPED),
    # failures
    (SessionStatus.STARTING, SessionStatus.CRASHED),
    (SessionStatus.READY, SessionStatus.CRASHED),
    (SessionStatus.CONNECTED, SessionStatus.CRASHED),
    (SessionStatus.IDLE, SessionStatus.CRASHED),
    (SessionStatus.PAUSED, SessionStatus.CRASHED),
    # cleanup of crashed rows
    (SessionStatus.CRASHED, SessionStatus.STOPPED),
})


# Statuses that imply a running container (paused counts — the
# container exists, it's just frozen by the cgroup freezer).
RUNNING_STATUSES: frozenset[SessionStatus] = frozenset({
    SessionStatus.STARTING,
    SessionStatus.READY,
    SessionStatus.CONNECTED,
    SessionStatus.IDLE,
    SessionStatus.PAUSED,
})


# Statuses that should not count against per-user concurrency caps.
TERMINAL_STATUSES: frozenset[SessionStatus] = frozenset({
    SessionStatus.STOPPED,
    SessionStatus.CRASHED,
})


class LifecycleTransitionError(ValueError):
    """Raised on an illegal status transition."""


class GameStreamLifecycle:
    """Pure state-machine helpers.

    No I/O happens here -- callers (the runtime) coordinate with the
    store, port pool, and Docker. This class just answers "is X -> Y
    legal" and exposes idle-detection helpers.
    """

    @staticmethod
    def can_transition(
        from_status: str | SessionStatus,
        to_status: str | SessionStatus,
    ) -> bool:
        try:
            f = SessionStatus(from_status)
            t = SessionStatus(to_status)
        except ValueError:
            return False
        return (f, t) in _LEGAL_TRANSITIONS

    @staticmethod
    def transition(
        from_status: str | SessionStatus,
        to_status: str | SessionStatus,
    ) -> SessionStatus:
        """Validate a transition and return the new status.

        Raises ``LifecycleTransitionError`` for illegal transitions.
        """
        if not GameStreamLifecycle.can_transition(from_status, to_status):
            raise LifecycleTransitionError(
                f"illegal transition {from_status!s} -> {to_status!s}"
            )
        return SessionStatus(to_status)

    @staticmethod
    def is_running(status: str | SessionStatus) -> bool:
        try:
            return SessionStatus(status) in RUNNING_STATUSES
        except ValueError:
            return False

    @staticmethod
    def is_terminal(status: str | SessionStatus) -> bool:
        try:
            return SessionStatus(status) in TERMINAL_STATUSES
        except ValueError:
            return False
