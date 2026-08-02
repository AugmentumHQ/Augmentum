"""Voice-tool manifest — single source of truth for what tools the
companion may invoke through the voice path.

Two consumers read from this module:

* ``augmentum/proxy/voice_routes.py`` — composes the LLM tool list
  per turn, filtered by user policy (full / safe / minimal / custom)
  and surface context (ambient widget vs foreground call).
* ``augmentum/architect/voice_router.py`` — uses the same names as a
  confidence signal when classifying address.

Before this module existed, ``_VOICE_TOOLS`` in voice_routes.py was a
hard-coded frozenset that had to be edited every time a new primitive
landed. New primitives also had to be cross-referenced against the
architect's action-verb signal list — and the two drifted (e.g.
``web.search`` was listed in voice_tools but never registered as an
intent action, so it was both broken and apparently working depending
on which code path you read).

Now both code paths derive from the buckets below. To expose a new
primitive to voice:

  1. Register it via ``register_action`` like normal.
  2. Add its id to the appropriate bucket below:
     ``VOICE_TOOLS_CORE``       — always safe, no resource cost,
                                   reversible. Notes, memory, recall,
                                   navigation, reference lookups.
     ``VOICE_TOOLS_INTERACTIVE`` — opens a surface, user attention
                                   required. Web search, discovery.
     ``VOICE_TOOLS_DISRUPTIVE``  — interrupts current state. Media
                                   playback control, timers.
     ``VOICE_TOOLS_COSTLY``     — meaningful resource burn. Image
                                   generation.

The bucket choice drives the ambient-mode policy filter below. Pick
the most conservative bucket the primitive fits — false-positive
"safe" classifications are noisier than false-positive "disruptive"
ones (the latter just hides a capability behind a setting toggle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentum.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Registry binding (Phase 1 unified primitive layer)
# ---------------------------------------------------------------------------
#
# The static frozensets below cover **intent-action verbs** (note.create,
# navigate.open_surface, …) — primitives that live in the intent registry
# and are not ``Tool`` subclasses. ``Tool`` primitives that opt into voice
# via ``Tool.surfaces.voice`` are unioned in at call time through the
# registry binding below. This lets the chat/coder ``ToolRegistry`` be
# the single source of truth for what is voice-reachable, without
# duplicating tool names into a hand-curated allowlist that drifts.
#
# Call :func:`bind_registry` once after the registry is populated
# (server.py lifespan). If left unbound, ``_runtime_voice_tools`` returns
# empty and the behavior collapses to the pre-Phase-1 static-only union,
# which is what the standalone unit tests exercise.

_registry: ToolRegistry | None = None


def bind_registry(registry: ToolRegistry) -> None:
    """Bind a populated ToolRegistry so voice tool unions include
    Tools that declare ``surfaces.voice``. Idempotent."""
    global _registry
    _registry = registry


def _runtime_voice_tools(level: str) -> frozenset[str]:
    """Names of registered Tools whose ``surfaces.voice`` matches level."""
    if _registry is None:
        return frozenset()
    return frozenset(
        t.name for t in _registry.list_tools() if t.surfaces.voice == level
    )


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

# Always safe — reversible, no resource burn, no playback interruption.
# Lookups, internal navigation, the user's own notes/memory.
VOICE_TOOLS_CORE: frozenset[str] = frozenset({
    # Intent registry — notes + memory + navigation
    "note.create",
    "note.append",
    "note.show_sticky",
    "note.start_capture",
    "note.end_capture",
    "memory.save",
    "memory.recall",
    # Memory hygiene (wiring program Phase 2) — forget is confirm-
    # gated inside the handler (recall-then-confirm); tier is
    # reversible with an audit trail.
    "memory.forget",
    "memory.tier",
    # Ask-about-your-own-data (Phase 3) — pure SELECT wrappers
    "my.taste",
    "my.playtime",
    "my.jobs",
    "my.calls",
    "system.health",
    "system.signals",
    # Her interiority (Phase 4) — one introspection plane, pure read
    "companion.introspect",
    # Content management (Phase 5) — reversible edits to the user's
    # own lists/titles (playlist.delete is DISRUPTIVE, below)
    "playlist.create",
    "playlist.rename",
    "file.favorite",
    "chat.rename",
    "library.rename",
    "navigate.open_surface",
    "navigate.back",
    # Reference lookups — pure read, never side-effecting
    "calculator",
    "datetime",
    "unit_converter",
    "wikipedia",
    "memory_recall",
    # Perception pull door — full detail behind the index/digest tiers
    # (open page text, full note, play position). Pure read.
    "context_peek",
    # Direct keyless data source — typed weather, no search round-trip
    "weather.today",
    # Companion reflection — opens a surface but doesn't burn or interrupt
    "companion.today_recap",
    # Local file lookup — pure index read
    "files.find",
    # Content inventory — read-only, informational. Tells the user what
    # they have without changing state.
    "game.recommend",
    "livetv.browse",
    # Browse panel search — opens a search results surface
    "browse.find",
    # Notification hygiene — reversible feed/channel state, no burn
    "notify.mute",
    "notify.dismiss",
})

# Mid-stakes — opens external content surface; user attention required
# but no resource burn or playback interruption.
VOICE_TOOLS_INTERACTIVE: frozenset[str] = frozenset({
    "web_search",
    "web_fetch",
    "web",
    "web.search",
    "discovery.show",
    # Creation tools (Phase 7) — produce artifacts the user will look
    # at; explicit-ask material, not ambient-widget material.
    "create_document",
    "create_spreadsheet",
    "create_presentation",
    "create_chart",
    "convert_document",
    "youtube",
    "image_search",
})

# Disruptive — changes current state (playback, alarms). Should be
# explicitly invoked, not auto-fired by a passive companion in the
# background.
VOICE_TOOLS_DISRUPTIVE: frozenset[str] = frozenset({
    "media.resume",
    "media.pause",
    "media.next",
    "media.previous",
    # Mid-experience controls (wiring program Phase 1) — change live
    # playback state, same stakes shape as the transport verbs.
    "media.volume",
    "media.speed",
    "media.sleep_timer",
    "grove.play_matching",
    "time.set_timer",
    # Phone device verbs (phone-as-capability-provider) — fire a clock
    # alarm/timer, open the dialer/SMS composer/contact editor pre-filled,
    # or launch an app on the paired phone. State-changing on the phone,
    # so disruptive — but consent-by-construction: dial/text/contact only
    # OPEN a composer the user taps to confirm. Gated server-side by
    # companion_device_tools_enabled. See intent/builtin/device.py.
    "device.set_alarm",
    "device.set_timer",
    "device.dial",
    "device.compose_text",
    "device.add_contact",
    "device.launch_app",
    # Destructive-with-confirm (Phase 5) — kept out of ambient policy
    "playlist.delete",
    # Content launch — opens a game player or live TV HLS overlay.
    # Changes current state same tier as media.resume.
    "game.play",
    "livetv.play",
})

# Costly — meaningful resource consumption (GPU image gen, paid API).
VOICE_TOOLS_COSTLY: frozenset[str] = frozenset({
    "image_generation",
    "image.generate_with_defaults",
    # rembg inference — real compute per call (Phase 7)
    "remove_background",
    # Delegate a background coding run — real GPU/compute burn on the coder
    # stack. Costly (not just disruptive) so it stays out of the default-safe
    # ambient widget policy: it fires from a foreground voice call or chat,
    # never silently from the always-on companion. See intent/builtin/coder.py.
    "coder.delegate",
})

# Per-tool capability description — surfaced in the system prompt so
# refusal-prone local models can't lazily claim a capability they have.
# Keyed by tool name; missing entries are silently skipped (action
# primitives don't need a line — their name + summary suffice when the
# tool schema is rendered).
VOICE_TOOL_CAPABILITIES: dict[str, str] = {
    "image_generation": "generate an image (image_generation)",
    "image.generate_with_defaults": "generate an image (image.generate_with_defaults)",
    "web_search": "search the web (web_search)",
    "web_fetch": "fetch a web page (web_fetch)",
    "web": "search or fetch the web (web)",
    "web.search": "search the web (web.search)",
    "wikipedia": "look up Wikipedia (wikipedia)",
    "calculator": "do math (calculator)",
    "datetime": "check the time or date (datetime)",
    "unit_converter": "convert units (unit_converter)",
    "memory_recall": "recall what the user has said before (memory_recall)",
    "context_peek": (
        "see what's on their screen in full, or list everything you "
        "can do (context_peek slot 'abilities')"
    ),
    "memory.recall": "recall what the user has said before (memory.recall)",
    "companion.today_recap": "summarize today's reflection (companion.today_recap)",
    "files.find": "find a local file (files.find)",
    "browse.find": "search the browse panel (browse.find)",
    "coder.delegate": "delegate a coding task to a background agent (coder.delegate)",
}

# Allowed policy values for the ambient tool filter. Kept here so the
# config layer can validate without importing every consumer.
AMBIENT_POLICIES: frozenset[str] = frozenset({"full", "safe", "minimal", "custom"})

# Default policy when the operator hasn't explicitly chosen one. ``safe``
# excludes disruptive playback and costly generation from the always-on
# widget — the companion can still do those things when explicitly
# invoked through a foreground voice call, just not silently from the
# background widget.
DEFAULT_AMBIENT_POLICY = "safe"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def all_voice_tools() -> frozenset[str]:
    """Return the universe of tools the voice path may expose.

    Equivalent to the old hard-coded ``_VOICE_TOOLS`` set in voice_routes.py.
    Used as the outer filter for tool-list resolution — the
    ``_resolve_passthrough_tools`` chain layer may include tools that
    don't make sense in voice (file_ops, python_exec, etc.), and this
    set is the final allowlist.

    Includes both the static intent-verb sets and any registered Tool
    whose ``surfaces.voice`` is set (see :func:`bind_registry`).
    """
    return (
        VOICE_TOOLS_CORE
        | VOICE_TOOLS_INTERACTIVE
        | VOICE_TOOLS_DISRUPTIVE
        | VOICE_TOOLS_COSTLY
        | _runtime_voice_tools("core")
        | _runtime_voice_tools("interactive")
        | _runtime_voice_tools("disruptive")
        | _runtime_voice_tools("costly")
    )


def voice_tools_for(
    *,
    ambient: bool,
    policy: str = DEFAULT_AMBIENT_POLICY,
    custom_allowlist: list[str] | tuple[str, ...] | None = None,
) -> frozenset[str]:
    """Return the allowed tool set for a voice surface.

    Args:
        ambient: True when the surface is the passive always-on companion
            widget (``persona_id == 'becca'``). False for a foreground
            voice call modal, which gets the full set regardless of
            policy — the user has actively engaged a voice session and
            their attention is on it.
        policy: Ambient policy. One of ``AMBIENT_POLICIES``. Unknown
            values fall back to ``DEFAULT_AMBIENT_POLICY``.
        custom_allowlist: Used only when ``policy == 'custom'``. Tool
            names not in :func:`all_voice_tools` are silently dropped.

    Returns:
        Frozen allowlist. Always a subset of :func:`all_voice_tools`.
    """
    if not ambient:
        # Foreground voice call — user is actively driving the session,
        # no need to gate tools behind a passive-mode policy.
        return all_voice_tools()

    p = (policy or DEFAULT_AMBIENT_POLICY).strip().lower()
    if p not in AMBIENT_POLICIES:
        p = DEFAULT_AMBIENT_POLICY

    if p == "full":
        return all_voice_tools()
    if p == "minimal":
        return VOICE_TOOLS_CORE | _runtime_voice_tools("core")
    if p == "custom":
        if not custom_allowlist:
            # Empty custom list collapses to minimal — never expose
            # everything by accident from a misconfigured allowlist.
            return VOICE_TOOLS_CORE | _runtime_voice_tools("core")
        universe = all_voice_tools()
        return frozenset(name for name in custom_allowlist if name in universe)
    # "safe" — core + interactive, no disruption or burn
    return (
        VOICE_TOOLS_CORE
        | VOICE_TOOLS_INTERACTIVE
        | _runtime_voice_tools("core")
        | _runtime_voice_tools("interactive")
    )


def capability_line(name: str) -> str:
    """Return the one-line capability description for a tool name.

    Returns an empty string when the tool has no registered description.
    Falls back to the registered Tool's ``surfaces.voice_capability_line``
    when not in the static table.
    """
    line = VOICE_TOOL_CAPABILITIES.get(name, "")
    if line:
        return line
    if _registry is not None:
        tool = _registry.get(name)
        if tool is not None and tool.surfaces.voice_capability_line:
            return tool.surfaces.voice_capability_line
    return ""
