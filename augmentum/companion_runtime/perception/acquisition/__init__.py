"""L0 acquisition — the native/device data streams that feed perception.

Each module here is a thin adapter for one stream: it normalizes raw device
events into typed entities, persists them user-scoped, and exposes a recent-read
the live pass drops into the :class:`FusionContext` signal bag so the (pure)
fusers can correlate without doing I/O.

The sovereignty contract (``project_android_data_sovereignty_surface``): read
on-device, aggregate on-device, leave only via the user's own server pull. Every
store fn is user-scoped (multi-tenant rule) and every stream is gated OFF by
default — data only lands once the user has granted both the Android special
access AND the matching ``companion_perception_acquire_*`` setting.

- ``notifications`` — the all-app notification stream (the richest single grant;
  the first stream to land, per the spec build order).
"""

from __future__ import annotations

from augmentum.companion_runtime.perception.acquisition.notifications import (
    NotificationObservation,
    prune_notifications,
    recent_notifications,
    record_notifications,
)

__all__ = [
    "NotificationObservation",
    "record_notifications",
    "recent_notifications",
    "prune_notifications",
]
