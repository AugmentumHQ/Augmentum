"""Action-handler registry for notification action buttons.

When the UI clicks an action button on a notification, the route
layer needs to know what to do — accept a call, retry a failed
job, dismiss a memory candidate. Each subsystem registers its own
handler against a ``channel_pattern`` (glob over channel_id) at app
startup; the route layer dispatches.

This keeps the route file dumb: it persists the click + looks up the
handler + calls it + persists the result. Subsystem-specific logic
(routing an accept through ConnectHub, requeueing a job, etc.) lives
in the subsystem's own module.

Pattern matching uses ``fnmatch`` so ``"connect.call.*"`` matches
``"connect.call.incoming"`` and ``"connect.call.missed"``.

The registry is process-global because handlers are stateless function
references — no per-app instance state. Re-registering an existing
pattern replaces the prior handler (so tests can swap in mocks
cleanly).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.notifications.store import Notification


log = get_logger(__name__)


# An action handler receives the notification, the action id, and the
# request (so it can reach app.state if it needs the ConnectHub, etc.).
# It returns a dict the route layer renders to JSON.
ActionHandler = Callable[
    ["Notification", str, "Request"], Awaitable[dict[str, Any]],
]


@dataclass
class _HandlerEntry:
    pattern: str
    handler: ActionHandler


# Process-global registry. Order matters: the first matching pattern
# wins (so a specific pattern registered before a wildcard takes
# precedence). New registrations append; replaces collapse to the
# same slot to preserve order.
_REGISTRY: list[_HandlerEntry] = []


def register_action_handler(
    channel_pattern: str, handler: ActionHandler,
) -> None:
    """Register a handler for a channel glob.

    Re-registering the same pattern replaces the prior handler in
    place (preserves order). ``channel_pattern`` is matched via
    ``fnmatch``: ``"connect.call.*"`` matches ``connect.call.incoming``;
    ``"*"`` matches anything.
    """

    if not channel_pattern:
        raise ValueError("channel_pattern is required")
    for entry in _REGISTRY:
        if entry.pattern == channel_pattern:
            entry.handler = handler
            return
    _REGISTRY.append(_HandlerEntry(pattern=channel_pattern, handler=handler))


def unregister_action_handler(channel_pattern: str) -> bool:
    """Remove a registered pattern. Returns whether it was present."""

    for i, entry in enumerate(_REGISTRY):
        if entry.pattern == channel_pattern:
            _REGISTRY.pop(i)
            return True
    return False


def resolve_handler(channel_id: str) -> ActionHandler | None:
    """First registered pattern that matches ``channel_id``."""

    for entry in _REGISTRY:
        if entry.pattern == "*" or fnmatch.fnmatchcase(channel_id, entry.pattern):
            return entry.handler
    return None


def reset_registry() -> None:
    """Drop all registered handlers. For tests only."""

    _REGISTRY.clear()


def registered_patterns() -> list[str]:
    """Snapshot of patterns currently registered, in registration order."""

    return [entry.pattern for entry in _REGISTRY]
