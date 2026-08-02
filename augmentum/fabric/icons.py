"""Curated emoji set used as visual identifiers for fabric peers.

Each paired remote box gets one of these (or a freeform emoji) at
pair time. The icon shows in:

  - Peer list rows (next to hostname)
  - Capability matrix column headers
  - Chat model dropdown items that resolve to a peer-served model
  - Small badges on chat turns served by that peer

The local-box self-icon comes from ``settings.local_fabric_icon`` and
is picked the first time fabric is enabled.

Twenty entries, themed around speed / animals / vehicles / weather so
a small fleet can be labelled distinctively without operator fatigue.
The list is deliberately fixed (not user-extensible at this layer);
the UI offers an "Other…" escape hatch that accepts any emoji.

Order is intentional: speed metaphors first (racecar, rocket, snail
— matches the operator vocabulary), then mammals, then aquatic,
then atmospheric. Newest additions append; the grid renders in the
order defined here so the UX stays stable across upgrades.
"""

from __future__ import annotations

# Source of truth — single list, no synonyms / aliases.
PEER_ICONS: tuple[str, ...] = (
    "🏎",   # racecar (fastest box)
    "🚀",   # rocket
    "⚡",   # lightning
    "🐢",   # tortoise (steady)
    "🐌",   # snail (slow but reliable)
    "🦊",   # fox
    "🐻",   # bear
    "🐉",   # dragon
    "🦉",   # owl (night/background worker)
    "🐙",   # octopus (many capabilities)
    "🦅",   # eagle
    "🐺",   # wolf
    "🐳",   # whale
    "🌋",   # volcano
    "⛰",    # mountain
    "🌊",   # wave
    "❄",    # snowflake
    "🔥",   # fire
    "🌟",   # star
    "🛰",    # satellite
)

# Fallback when a peer has no icon set (legacy rows from before the
# 170 migration, or freshly-paired peers that haven't been re-iconed).
DEFAULT_PEER_ICON = "🔗"


def is_curated(icon: str) -> bool:
    """True iff ``icon`` is one of the curated set. Free-form emoji
    from the "Other…" picker return False but are still accepted.
    """
    return icon in PEER_ICONS
