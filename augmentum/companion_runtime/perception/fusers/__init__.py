"""Built-in fusers — the L1/L2 correlators that turn acquired signals into
insights. Each fuser is a pure ``(FusionContext) -> list[Insight]`` plugged into
``fusion.register_fuser``; it reads the signal bag the live pass filled (never
the DB) and emits zero or more meaning-bearing insights.

A new data stream ships its acquisition adapter (``perception/acquisition/``) AND
its fuser here, then inherits the judgment gate + interruption budget for free.

``register_builtin_fusers`` is idempotent (registration is keyed by name) so the
live adapter can call it on every pass cheaply, or once at startup.
"""

from __future__ import annotations

from augmentum.companion_runtime.perception import fusion
from augmentum.companion_runtime.perception.fusers.notifications import (
    NOTIFICATION_FUSER_NAME,
    fuse_notifications,
)


def register_builtin_fusers() -> None:
    """Register every shipped fuser. Idempotent — safe to call repeatedly."""
    fusion.register_fuser(NOTIFICATION_FUSER_NAME, fuse_notifications)


__all__ = ["register_builtin_fusers", "fuse_notifications", "NOTIFICATION_FUSER_NAME"]
