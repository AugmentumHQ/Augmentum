"""Presence mode — the user-facing autonomy dial.

Sprint 5 Piece 14, Aletheia × Augmentum arc.

Three modes determine how present the companion is:

  silent  — substrate exists but produces no user-visible output.
            Wondering generator suppresses writes. Revisit_thread
            respects the gate. Pip never appears. Pre-context never
            injects. Safe default for new users.

  gentle  — wonderings + revisits write; pip visible; pre-context
            stays off. The standard "she's here, occasionally has
            something to share" register.

  engaged — gentle + pre-context injection at session start +
            affect-tinted UI accents. The "she's been carrying
            things" register.

The setting lives at ``companion_presence_mode`` in global settings.
Future enhancement: per-user override via user_settings (Sprint α-style
refinement; substrate already supports per-user iso).
"""

from __future__ import annotations

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


MODE_SILENT = "silent"
MODE_GENTLE = "gentle"
MODE_ENGAGED = "engaged"

VALID_MODES = (MODE_SILENT, MODE_GENTLE, MODE_ENGAGED)

DEFAULT_MODE: str = MODE_SILENT


def get_presence_mode() -> str:
    """Read the current presence_mode setting. Returns the validated
    value, falling back to DEFAULT_MODE on missing/invalid input.

    Reads from settings on every call to honor admin changes without
    a restart (consistent with how other companion flags work).
    """
    try:
        from augmentum.config import settings
        raw = (getattr(settings, "companion_presence_mode", "") or "").strip().lower()
    except Exception:
        return DEFAULT_MODE
    if raw in VALID_MODES:
        return raw
    return DEFAULT_MODE


def autonomy_allowed() -> bool:
    """True when the runtime is allowed to write autonomous entries
    (wonderings, synthesized noticings). False when ``silent``."""
    return get_presence_mode() != MODE_SILENT


def pre_context_allowed() -> bool:
    """True when pre-context injection is allowed (``engaged`` only)."""
    return get_presence_mode() == MODE_ENGAGED


def pip_allowed() -> bool:
    """True when the note pip should be visible to the user.

    ``silent`` suppresses the pip entirely. UI module reads this via
    the presence_mode setting; this helper exists for backend code
    that wants to know whether to mark new entries quiet_share_ready.
    """
    return get_presence_mode() != MODE_SILENT


__all__ = [
    "MODE_SILENT",
    "MODE_GENTLE",
    "MODE_ENGAGED",
    "VALID_MODES",
    "DEFAULT_MODE",
    "get_presence_mode",
    "autonomy_allowed",
    "pre_context_allowed",
    "pip_allowed",
]
