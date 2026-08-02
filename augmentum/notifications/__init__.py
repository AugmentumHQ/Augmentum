"""Notification substrate — unified attention-worthy event store.

OS-tier primitive: any subsystem publishes events via ``store.publish``;
any surface (UI, voice, cast, future phone APK) subscribes via
``notification_subscriptions``. Persistence + dedup + dismiss state +
mute happens once, here, instead of being reinvented in every
subsystem that needs to surface something.

See ``docs/superpowers/specs/2026-06-01-notification-substrate-design.md``
for the design rationale + the seven scattered mechanisms this
replaces.

Phase 1 scope (this module): publish-side primitives only — channels,
catalog, store with publish/list/mark-read/dismiss/mute/expire.
HTTP surface, WS fan-out, and migrating the existing surfacers are
follow-on tasks tracked separately.
"""

from __future__ import annotations

from .actions import (
    ActionHandler,
    register_action_handler,
    registered_patterns,
    reset_registry,
    resolve_handler,
    unregister_action_handler,
)
from .catalog import (
    DEFAULT_CHANNELS,
    IMPORTANCE_CRITICAL,
    IMPORTANCE_DEFAULT,
    IMPORTANCE_HIGH,
    IMPORTANCE_LOW,
    IMPORTANCE_MIN,
    catalog_channel,
    catalog_channel_ids,
)
from .hub import NotificationHub, publish_and_dispatch
from .store import (
    Notification,
    NotificationAction,
    NotificationChannel,
    NotificationStore,
)

__all__ = [
    "ActionHandler",
    "DEFAULT_CHANNELS",
    "IMPORTANCE_CRITICAL",
    "IMPORTANCE_DEFAULT",
    "IMPORTANCE_HIGH",
    "IMPORTANCE_LOW",
    "IMPORTANCE_MIN",
    "Notification",
    "NotificationAction",
    "NotificationChannel",
    "NotificationHub",
    "NotificationStore",
    "catalog_channel",
    "catalog_channel_ids",
    "publish_and_dispatch",
    "register_action_handler",
    "registered_patterns",
    "reset_registry",
    "resolve_handler",
    "unregister_action_handler",
]
