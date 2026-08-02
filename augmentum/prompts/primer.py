"""The native-model primer — the train==serve guarantee.

The companion model is trained with its protocol in the WEIGHTS, so production
ships a tiny PRIMER (the "key") instead of a full system prompt (the F7 design).
The hard requirement: the system context the model trains on MUST match the
context it's served, byte-for-byte in FORMAT (the values differ — synthetic at
train time, real runtime state at serve time). A format mismatch silently
degrades the model (the Open-WebUI lesson).

This module is the SINGLE constructor for that primer. Both callers use it:
  - training-row assembly (`build_primer(...)` with the scenario's values)
  - the inference request path (`build_primer(...)` with live runtime state)
So they cannot drift. One function, two callers = the guarantee.

The primer is surface-specific and MINIMAL (~100-200 tokens). The tag activates
everything in the weights; the primer only carries TODAY's variable context.

Canonical shape (lines present only when their value is set)::

    :B
    [companion: Becca | user: Matt]
    [CHARACTER: ...]            # narrative only
    [STATE: ...]               # narrative world-state, or coder workspace
    [MEMORY: ...]              # narrative established facts
    [state: energy 0.4 | drives: curiosity 0.7, connection 0.8]   # companion runtime
    [now: 2026-06-26 21:15 | last seen: 2026-06-23 19:40]
    tools: web_search, memory.recall, media.play

Generic (non-native) models get the full system prompt elsewhere; this is the
fast path only.
"""

from __future__ import annotations

from collections.abc import Iterable

# Surfaces and their canonical short tags. The tag is line 1 of every primer.
SURFACE_TAGS = {
    "passthrough": ":C", "chat": ":C",
    "becca_direct": ":B", "companion": ":B",
    "voice": ":V", "phone": ":Vp",
    "coder": ":-", "analytical": ":A",
    # Every classifier Mode value MUST appear here or ``tag_for`` silently
    # falls through to the ``:C`` default and mis-tags that mode's turns as
    # chat. agentic (5th core handler, multi-step plan→execute — where doc/app
    # builds classify) and direct (verbatim external-harness pass-through, NOT
    # companion behavior) were both missing. ``:T`` = Task; ``:D`` = Direct,
    # kept separable so external-harness turns never pollute companion data.
    "agentic": ":T", "task": ":T",
    "direct": ":D",
    "narrative": ":N", "builder": ":W", "artifact": ":W",
    "game": ":G", "stream": ":L", "live": ":L",
    "xr": ":X", "cast": ":R", "system": ":S",
    "knowledge": ":K",
}


def tag_for(surface: str) -> str:
    """Resolve a mode/surface name to its primer tag. Pass-through if already a tag."""
    if surface.startswith(":"):
        return surface
    return SURFACE_TAGS.get(surface, ":C")


def build_primer(
    surface: str,
    *,
    companion: str = "",
    user: str = "",
    character: str = "",
    world_state: str = "",
    memory: str = "",
    state: str = "",
    now: str = "",
    last_seen: str = "",
    tools: Iterable[str] = (),
    extra: str = "",
) -> str:
    """Construct the native-model primer for a surface.

    Every field is optional — a line appears only when its value is non-empty,
    so the primer stays minimal. This is the ONE place the primer shape is
    defined; train-time assembly and serve-time request-building both call it.

    Args:
        surface: mode name ("companion") or tag (":B").
        companion / user: identity (set by the character card / persona at serve;
            synthetic or empty at train).
        character / world_state / memory: narrative (:N) — the character card
            persona, [STATE] scene, [MEMORY] established facts.
        state: a compact runtime state line (companion drives/energy, coder
            workspace summary, etc.).
        now / last_seen: timestamps for time-grounding (deltas computed by the
            model). last_seen only meaningful for relational surfaces.
        tools: active tool NAMES (not schemas — the schemas are in the weights).
        extra: any surface-specific trailing line.
    """
    tag = tag_for(surface)
    lines: list[str] = [tag]

    ident = []
    if companion:
        ident.append(f"companion: {companion}")
    if user:
        ident.append(f"user: {user}")
    if ident:
        lines.append("[" + " | ".join(ident) + "]")

    if character:
        lines.append(f"[CHARACTER: {character}]")
    if world_state:
        lines.append(f"[STATE: {world_state}]")
    if memory:
        lines.append(f"[MEMORY: {memory}]")
    if state:
        lines.append(f"[state: {state}]")

    if now:
        t = f"[now: {now}"
        if last_seen:
            t += f" | last seen: {last_seen}"
        lines.append(t + "]")

    tool_list = [t for t in tools if t]
    if tool_list:
        lines.append("tools: " + ", ".join(tool_list))

    if extra:
        lines.append(extra.strip())

    return "\n".join(lines)


def is_primer(system_value: str) -> bool:
    """Heuristic: True if a system message looks like a native primer (starts
    with a known tag on its own first line) rather than a full system prompt."""
    if not system_value:
        return False
    first = system_value.split("\n", 1)[0].strip()
    return first in set(SURFACE_TAGS.values()) or (
        first.startswith(":") and len(first) <= 4
    )
