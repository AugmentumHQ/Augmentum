"""Architect — companion-as-orchestrator dispatch layer.

The architect is Becca's command surface. It sits between the verbal
command (voice/chat/cast) and the existing intent registry, adding two
things the bare intent path doesn't have:

  1. **Surface filtering** — Action.surfaces gates whether an action
     is callable on this client surface ('voice'/'chat'/'cast'/'xr').
  2. **Defaults inference** — Action.arg_inferrer fills missing args
     from observation history (device_play_history, image_generations,
     browse_history, ReferentCache). The novel UX piece: "play jazz"
     picks a track from the user's favourites instead of asking.

The architect does NOT re-implement matching, the registry, or the
handler contract — those stay in augmentum/intent/. This module is a
thin coordinator + the inference layer + new architect-callable
primitives.

Architecturally Becca is **adjacent to** the user's active mode, not
above it. She doesn't drive the user's turn; she watches, and when
called (or when initiative fires) she dispatches a primitive that the
owning surface would have run anyway.

Wiring contract:

  - Gated by ``architect_dispatch_enabled`` setting (default False).
  - Primitives import from this package; the module's __init__ imports
    builtin primitives so @register_action decorators fire at startup.
  - Voice and chat routes call ``dispatch_architect_command`` BEFORE
    falling through to the LLM/UARF path.

See docs/superpowers/specs/2026-05-28-companion-architect-design.md.
"""

from __future__ import annotations

from augmentum.architect.dispatch import (
    ArchitectResult,
    dispatch_architect_command,
)
from augmentum.architect.inference import infer_args

# Import builtin architect primitives so @register_action runs at
# process startup. Order matters: the matcher returns the FIRST hit
# across registered actions, so primitives that should win
# disambiguation must register first.
#
# Priority order (specificity high → low — more specific structures
# claim utterances they uniquely match BEFORE the generic ones get
# a chance):
#   1. time.set_timer  — "set a N minute timer" / "remind me in N"
#                        — verb+number-anchored, never collides.
#   2. media.resume    — "resume my X" / "continue my book" — narrow.
#   3. browse.find     — "find that article about X" — has the
#                        "article|page|thing" keyword anchor that
#                        distinguishes it from a bare web search.
#   4. discovery.show  — "show me a movie about X" — kind-keyword
#                        anchored. Beats grove.play_matching's
#                        `play {query}` if user says "show me a song".
#   5. web.search      — "search for X" / "look up X" / "google X"
#                        — generic search; loses to history-aware
#                        browse.find when both match.
#   6. image.generate  — generate/create/draw + prompt slot.
#   7. grove.play      — most permissive {query} slot — registered
#                        LAST so other primitives have first claim
#                        on overlapping utterances.
from augmentum.architect.primitives import time_timer as _time_timer  # noqa: F401
from augmentum.architect.primitives import media_resume as _media_resume  # noqa: F401
from augmentum.architect.primitives import browse_find as _browse_find  # noqa: F401
from augmentum.architect.primitives import discovery_show as _discovery_show  # noqa: F401
from augmentum.architect.primitives import web_search as _web_search  # noqa: F401
from augmentum.architect.primitives import image_defaults as _image_defaults  # noqa: F401
from augmentum.architect.primitives import grove_match as _grove_match  # noqa: F401
# Second-wave primitives (media transport control, local file lookup,
# Today reflection). Registered AFTER the original seven so registration
# order keeps the specificity-high-to-low policy intact — these are all
# narrower than grove.play_matching's permissive {query} slot, but they
# don't overlap with each other.
from augmentum.architect.primitives import media_control as _media_control  # noqa: F401
from augmentum.architect.primitives import files_find as _files_find  # noqa: F401
from augmentum.architect.primitives import companion_today as _companion_today  # noqa: F401

__all__ = [
    "ArchitectResult",
    "dispatch_architect_command",
    "infer_args",
]
