"""Add-ons — container-delivered extensions of Augmentum's own capabilities.

An **add-on** is not a service. The distinction is the whole reason this
package exists rather than another ``install_via`` on the service path:

    A service is a product you open.
    An add-on is a capability Augmentum gains.

Nobody builds Open WebUI instead of pulling it, because for a service the
container *is* the thing the user wants: it runs continuously, owns a port,
and "Open" leaves Augmentum for its own UI. The service standard
(``marketplace/manifest.py``) is correct for that and is untouched here.

An add-on inverts nearly every one of those properties:

* **Install means "make a capability available", not "start something."**
  There is no long-running container, no port, no healthcheck. The image is
  inventory; ``GameStreamRuntime`` (or the cast layer) spawns it per session
  and removes it at teardown. An add-on with nothing running is working
  correctly, so its state is *available / absent*, never *running / stopped*.
* **Provenance is pull-or-build, and both are pinned.** A service pins an
  image tag; an add-on that builds locally pins its BUILD ARGS
  (``DOLPHIN_VERSION``, ``PCSX2_VERSION``, ``SELKIES_VERSION``) — the same
  anti-drift guarantee expressed where the version actually lives. This is
  the field ``_validate_image_pinned`` cannot express, which is why bending
  that rule was the wrong move and adding a category was the right one.
* **The user is the builder.** This is what keeps the project's distribution
  exposure at zero: Dolphin is GPLv2 and Chrome's ToS forbids
  redistribution, so we ship *recipes*, not binaries. The person who
  compiles the emulator is the person who clicked Install. Add-ons may
  therefore declare license terms to acknowledge before building — a step no
  service needs, because pulling someone's published image carries none of
  those obligations.
* **Its surface is an existing Augmentum surface.** "Open" deep-links into
  Library → Games, not an allocated 6800-6809 front door.

Resource semantics are two-part rather than one. A service declares
steady-state RAM and is gated on it at install. An add-on has a *build cost*
(minutes, peak RAM, GBs of disk) that is paid once, and a separate *spawn
cost* that applies only while a session is live. That split is already
documented in ``compose.game-stream.yaml`` — the ``mem_limit`` on each
build-only marker bounds the marker, never what the runtime spawns.

INSTALLED STATE IS THE ANCHOR. ``scripts/bootstrap/ensure_game_stream_anchors
.py`` created a zero-cost ``created`` container per image so a host
``docker image prune -a`` could not sweep images that nothing references
between sessions (the 2026-07-25 failure: a per-title 404 months after the
sweep). That anchor is also the natural install record — it is the only
artifact that is *both* the prune protection and the answer to "is this
installed?", so this package treats the two as one thing rather than
maintaining a second source of truth that could disagree with the daemon.

Modules:
    catalog   — the add-on specs (what exists, what it costs, what it needs)
    registry  — availability, anchors, capability resolution, teardown
    builder   — narrow build context + streamed build progress
"""

from __future__ import annotations

from augmentum.addons.catalog import (
    ADDONS,
    AddonSpec,
    addon_by_id,
    addon_for_capability,
    resolve_build_order,
    user_facing_addons,
)
from augmentum.addons.registry import (
    AddonState,
    anchor_name_for,
    capability_available,
    capability_image,
    get_state,
    list_states,
    remove_addon,
)

__all__ = [
    "ADDONS",
    "AddonSpec",
    "AddonState",
    "addon_by_id",
    "addon_for_capability",
    "anchor_name_for",
    "capability_available",
    "capability_image",
    "get_state",
    "list_states",
    "remove_addon",
    "resolve_build_order",
    "user_facing_addons",
]
