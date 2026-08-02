"""Canonical schema + defaults for per-receiver display preferences.

Single source of truth for what a TV-receiver pref bag may contain.
Both the GET route (when filling in missing keys with defaults) and
the PUT route (when validating incoming writes) consult this module
so the server can't drift from the client.

Adding a new toggle:
  1. Add it to ``DEFAULT_PREFS`` with the spec default
  2. Add validation in ``coerce_prefs`` (typing + value range)
  3. Document it in the README block at the top of this file

Removing a toggle: leave the key out of ``DEFAULT_PREFS`` and add it
to ``DEPRECATED_KEYS`` for a release cycle so old clients writing it
don't error — we just drop the value silently.
"""

from __future__ import annotations

from typing import Any

from augmentum.cast.rail_catalog import KNOWN_RAILS

# ── Pref bag schema ────────────────────────────────────────────────
#
# Each key has:
#   - a default (used when GET sees a missing key)
#   - a coercer (used at PUT to validate + normalise the incoming value)
#
# Booleans are the dominant shape today — checkbox-style toggles in the
# settings sheet. The ``rails_visible`` key is a sub-bag because the
# rail set is open-ended and we want to add new rails without
# expanding the top-level keyspace each time.
#
# KNOWN_RAILS is re-exported from rail_catalog so this module's coerce
# /defaults logic stays in lockstep with what cast_library_home can
# actually render — no risk of a prefs key existing without a backing
# section (the "music ghost" gap before the catalog refactor).
__all__ = ["KNOWN_RAILS", "DEFAULT_PREFS", "DEPRECATED_KEYS",
           "coerce_prefs", "with_defaults"]


DEFAULT_PREFS: dict[str, Any] = {
    # Per-rail visibility on the cast-home idle surface. Missing rail
    # keys default to True so a newly-added rail appears without
    # needing every existing user to explicitly enable it.
    "rails_visible": {rail: True for rail in KNOWN_RAILS},
    # Backdrop image cycling on cast-home. Off = calm static gradient.
    "backdrop_cycle": True,
    # Default subtitle state when a new video casts to this TV.
    # The controller can still toggle subs mid-playback regardless.
    "subtitle_default": False,
    # Whether the controller's "Follow on TV" toggle is allowed to
    # take effect for this receiver. Some TVs in a group setting
    # might want to stay locked to idle (e.g. a back-room TV that
    # shouldn't broadcast everyone's browsing).
    "follow_mode_allowed": True,
}


# Keys we used to accept but no longer write. Tolerated on incoming
# PUTs so old clients don't 400; their values are dropped silently.
DEPRECATED_KEYS: frozenset[str] = frozenset()


def coerce_prefs(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate + normalise an incoming prefs dict. Unknown top-level
    keys are dropped. Type-incorrect values fall back to the default
    rather than raising — we don't want a single bad checkbox to
    poison the whole bag.

    Returns a clean dict ready to merge into the stored JSON.
    """
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out

    # rails_visible: dict of str → bool, restricted to KNOWN_RAILS
    rv = raw.get("rails_visible")
    if isinstance(rv, dict):
        cleaned: dict[str, bool] = {}
        for rail, visible in rv.items():
            if not isinstance(rail, str) or rail not in KNOWN_RAILS:
                continue
            cleaned[rail] = bool(visible)
        if cleaned:
            out["rails_visible"] = cleaned

    # Simple booleans.
    for key in ("backdrop_cycle", "subtitle_default", "follow_mode_allowed"):
        if key in raw:
            out[key] = bool(raw[key])

    return out


def with_defaults(stored: dict[str, Any] | None) -> dict[str, Any]:
    """Merge stored prefs onto the default schema so the client always
    sees a fully-populated bag. Missing top-level keys take the
    default; for ``rails_visible``, missing rail keys default to True
    so newly-introduced rails are visible by default."""
    merged: dict[str, Any] = {
        "rails_visible": {rail: True for rail in KNOWN_RAILS},
        "backdrop_cycle": DEFAULT_PREFS["backdrop_cycle"],
        "subtitle_default": DEFAULT_PREFS["subtitle_default"],
        "follow_mode_allowed": DEFAULT_PREFS["follow_mode_allowed"],
    }
    if not isinstance(stored, dict):
        return merged

    rv = stored.get("rails_visible")
    if isinstance(rv, dict):
        for rail in KNOWN_RAILS:
            if rail in rv:
                merged["rails_visible"][rail] = bool(rv[rail])

    for key in ("backdrop_cycle", "subtitle_default", "follow_mode_allowed"):
        if key in stored:
            merged[key] = bool(stored[key])

    return merged
