"""Capability self-model — a compact, always-present summary of what the
assistant can actually do, so the shell (the LLM) never confabulates a denial
of a capability the kernel has ("I don't have a scheduler / I can't manage that").

Distinct from the per-turn tool roster (``prompt_compose`` Layer 6), which only
lists the tools SELECTED this turn. This digest is CATEGORY-level and ALWAYS
present: it tells the model a capability EXISTS even when its specific tool
isn't loaded this turn, so it offers/confirms instead of denying.

Derived from what's actually registered — a group drops out when none of its
probe capabilities are present (subsystem off) — the introspection seed for the
capability-OS architecture (docs/.../2026-06-18-capability-os-architecture.md).
"""
from __future__ import annotations

from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# (label, probe ids/names). A group is included only if at least one probe is
# registered, so the digest reflects what THIS install can really do rather than
# promising a disabled subsystem. Probes span both kernels: Tool names AND
# intent-action ids.
_CAPABILITY_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("set up and manage scheduled briefings, reminders, deadline countdowns, "
     "and price/page/feed watches (create, list, cancel)",
     ("schedule_briefing", "watch_for", "schedule_deadline", "schedule_reminder")),
    ("take, append to, and recall notes", ("note.create", "note.append")),
    ("remember things for the user and recall them later",
     ("memory.save", "memory.recall", "memory_recall")),
    ("research and search the web, read pages, and pull YouTube transcripts",
     ("research", "web_search", "web", "youtube")),
    ("do exact calculations and verify arithmetic by running code "
     "(python_exec) instead of computing in your head",
     ("python_exec", "calculator", "math_verify")),
    ("generate images", ("image_generation",)),
    ("play and control media", ("media.play", "media.pause")),
    ("launch and recommend games from the user's library",
     ("game.play", "game.recommend")),
    ("tune to and browse live TV channels from the user's Emby/Jellyfin server",
     ("livetv.play", "livetv.browse")),
    ("open and navigate app surfaces", ("navigate.open_surface",)),
    ("check the weather", ("weather.today",)),
]


def build_capability_digest(capability_ids: Any) -> str:
    """Compact capability self-model from the set of registered capability
    ids/names (tool names ∪ intent-action ids). Empty string when nothing
    matches (so a fully-stripped install adds no line)."""
    present = set(capability_ids or ())
    have = [
        label for label, probes in _CAPABILITY_GROUPS
        if any(p in present for p in probes)
    ]
    if not have:
        return ""
    body = "; ".join(have)
    return (
        "CAPABILITIES — things you can actually do for the user right now "
        f"(ask and the tool is provided): {body}. When the user asks for one "
        "of these, do it or confirm the details — never claim you lack a "
        "capability listed here, and offer them when genuinely relevant."
    )


# Cached: the registries are populated at startup and don't change per turn, so
# the digest is computed once (the first time it resolves to a non-empty value).
_cached_digest: str | None = None


def companion_capability_block(app_state: Any) -> str:
    """The capability self-model from the live registries (intent REGISTRY ∪
    ToolRegistry). Cached after the first non-empty resolution. Never raises —
    a broken registry read just yields an empty line."""
    global _cached_digest
    if _cached_digest:
        return _cached_digest

    ids: set[str] = set()
    try:
        from augmentum.intent.registry import REGISTRY
        ids.update(a.id for a in REGISTRY.all())
    except Exception:
        log.debug("capability_digest_intent_read_failed", exc_info=True)
    try:
        reg = getattr(app_state, "tool_registry", None)
        if reg is not None:
            ids.update(t.name for t in reg.list_tools())
    except Exception:
        log.debug("capability_digest_tool_read_failed", exc_info=True)

    digest = build_capability_digest(ids)
    if digest:
        _cached_digest = digest
    return digest
