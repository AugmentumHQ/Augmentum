"""Action registry — voice + text intent dispatch surface.

One ``register_action(...)`` definition produces:

  * A Tier 1 regex matcher (compiled from ``patterns`` + ``examples``)
  * A Tier 2 embedding-similarity match target (precomputed elsewhere)
  * A Tier 3 LLM tool schema (auto-generated from ``arg_schema``)
  * A Discovery UI entry (Settings panel listing)

The same registry serves every entry point a user can address Becca
through — voice WebSocket transcripts, browse/note Ask bars, cast
voice, XR voice. Frontend fast-path mirror loads just the Tier 1
patterns for conversation-repair actions (stop/repeat/bye/slower) so
those land sub-100ms without a server round-trip.

Misses fall through to today's UARF pipeline — registering an action
narrows ambiguity, it never removes capability.
"""

from __future__ import annotations

from augmentum.intent.action import (
    Action,
    ActionFanout,
    ActionResult,
    IntentMatch,
    ReferentCache,
    SessionContext,
)
from augmentum.intent.dispatch import dispatch, get_referent_cache, serialize_action_event
from augmentum.intent.tool_adapter import ActionTool, register_action_tools
from augmentum.intent.matcher import match_intent
from augmentum.intent.registry import (
    REGISTRY,
    list_actions,
    register_action,
)

# Eager-import builtins so their @register_action decorators run.
# Adding a new module under augmentum.intent.builtin requires adding
# it here too — explicit import beats import-side-effect discovery
# magic when the package is consumed across multiple workers.
# Import order matters: matcher walks REGISTRY in registration order
# and returns the first hit. Specific-qualifier modules (search,
# notes) come before generic catch-alls (media) so that
# "search my notes for X" lands on search.knowledge rather than a
# broader media verb. (media.search itself was retired 2026-06-11 —
# it duplicated search.local; see builtin/media.py.)
from augmentum.intent.builtin import calendar as _calendar  # noqa: F401
from augmentum.intent.builtin import chat as _chat  # noqa: F401
from augmentum.intent.builtin import control as _control  # noqa: F401
from augmentum.intent.builtin import navigation as _navigation  # noqa: F401
from augmentum.intent.builtin import notes as _notes  # noqa: F401
from augmentum.intent.builtin import search as _search  # noqa: F401
from augmentum.intent.builtin import media as _media  # noqa: F401
from augmentum.intent.builtin import games as _games  # noqa: F401
from augmentum.intent.builtin import livetv as _livetv  # noqa: F401
from augmentum.intent.builtin import trail as _trail  # noqa: F401
from augmentum.intent.builtin import coder as _coder  # noqa: F401
from augmentum.intent.builtin import app_act as _app_act  # noqa: F401
from augmentum.intent.builtin import weather as _weather  # noqa: F401
from augmentum.intent.builtin import notify as _notify  # noqa: F401
from augmentum.intent.builtin import memory_admin as _memory_admin  # noqa: F401
from augmentum.intent.builtin import my_data as _my_data  # noqa: F401
from augmentum.intent.builtin import introspect as _introspect  # noqa: F401
from augmentum.intent.builtin import manage as _manage  # noqa: F401
from augmentum.intent.builtin import device as _device  # noqa: F401
# reshape last: its patterns are specific so registration precedence is moot, and
# it must not shadow earlier verbs. (Deliberate order — do NOT isort this block.)
from augmentum.intent.builtin import reshape as _reshape  # noqa: F401

# --- Synthesized verbs (capability synthesis — augmentum/selfedit/capabilities)
# Machine-authored from CapabilitySpecs; each is template-rendered (bounded
# behavior) with a passing smoke-test oracle. Keep at the end. ---


__all__ = [
    "Action",
    "ActionFanout",
    "ActionResult",
    "ActionTool",
    "IntentMatch",
    "ReferentCache",
    "SessionContext",
    "REGISTRY",
    "dispatch",
    "get_referent_cache",
    "list_actions",
    "match_intent",
    "register_action",
    "register_action_tools",
    "serialize_action_event",
]
