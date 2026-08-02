"""World-system native tools — ``world.track.shift`` / ``world.roll`` /
``world.lookup``.

Exposed to the model ONLY when the session's world manifest declares the
matching module (spec: 2026-07-15-world-system-manifest-design.md).
Follows the lorebook_native_schemas contract: dispatchers return a result
STRING (never raise), and mutation dicts ride back for UI sync.

The model narrates outcomes; the engine owns the numbers. Failed
validations return instructive errors the model can recover from
(e.g. band-step limits, user-lock messages).
"""
from __future__ import annotations

import json
from typing import Any

from augmentum.modes.narrative.world_system import (
    WorldManifest,
    WorldStore,
    lookup_table,
    roll_dice,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


WORLD_NATIVE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "world.track.shift",
            "description": (
                "Move a world tracker after the story establishes a change "
                "(injury, spending, exposure...). Band trackers move one "
                "band per call; include reason='force: <major event>' for "
                "larger jumps. The current values in [World State] are "
                "authoritative — narrate FROM them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracker": {"type": "string", "description": "Tracker id."},
                    "to": {"description": "Target band/value (band, flag, scalar)."},
                    "delta": {"type": "number", "description": "Counter change (+/-)."},
                    "owner": {"type": "string", "description": "Character name (character-scope trackers). Omit for the player character or world scope."},
                    "reason": {"type": "string", "description": "One line: what in the story caused this."},
                },
                "required": ["tracker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "world.roll",
            "description": (
                "Roll real dice for a check the story calls for (e.g. "
                "'d20+3', '2d6'). Use the returned total to decide the "
                "outcome, then narrate it. Never invent roll results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "NdM(+/-K), e.g. 'd20+2'."},
                    "check": {"type": "string", "description": "What is being tested (shown to the player)."},
                    "dc": {"type": "number", "description": "Difficulty to beat, if applicable."},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "world.lookup",
            "description": (
                "Look up exact world data (prices, exchange rates, ranks) "
                "instead of inventing numbers. Returns matching rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table id."},
                    "query": {"type": "string", "description": "Filter text (optional)."},
                },
                "required": ["table"],
            },
        },
    },
]

WORLD_NATIVE_TOOL_NAMES: frozenset[str] = frozenset(
    s["function"]["name"] for s in WORLD_NATIVE_TOOL_SCHEMAS
)
WORLD_NATIVE_MUTATING_TOOLS: frozenset[str] = frozenset({"world.track.shift"})


def schemas_for_manifest(manifest: WorldManifest) -> list[dict[str, Any]]:
    """Only the tools whose module the manifest declares."""
    out: list[dict[str, Any]] = []
    for s in WORLD_NATIVE_TOOL_SCHEMAS:
        name = s["function"]["name"]
        if name == "world.track.shift" and not manifest.has("trackers"):
            continue
        if name == "world.roll" and not manifest.has("dice"):
            continue
        if name == "world.lookup" and not manifest.has("tables"):
            continue
        out.append(s)
    return out


def _parse_args(raw_arguments: Any) -> dict:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        out = json.loads(raw_arguments or "{}")
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
        return {}


def dispatch_world_native_tool(
    store: WorldStore,
    *,
    turn: int,
    tool_name: str,
    raw_arguments: Any,
) -> tuple[str, list[dict]]:
    """Execute a world tool. Returns (result_text, ui_events).

    Never raises — errors come back as instructive strings so the model
    can self-correct within the same turn.
    """
    args = _parse_args(raw_arguments)
    events: list[dict] = []

    if tool_name == "world.track.shift":
        ok, msg, value = store.shift(
            str(args.get("tracker") or ""),
            owner=str(args.get("owner") or ""),
            turn=turn,
            to=args.get("to"),
            delta=args.get("delta"),
            by="model",
            reason=str(args.get("reason") or ""),
        )
        if ok:
            events.append({
                "kind": "tracker_shift",
                "tracker": args.get("tracker"),
                "owner": args.get("owner") or "",
                "value": value,
                "reason": (args.get("reason") or "")[:200],
            })
        return msg, events

    if tool_name == "world.roll":
        result = roll_dice(str(args.get("expression") or ""))
        if result is None:
            return (
                "Invalid dice expression. Use NdM(+/-K), e.g. 'd20+2' or "
                "'2d6'.", events,
            )
        dc = args.get("dc")
        outcome = ""
        if isinstance(dc, int | float):
            outcome = "success" if result["total"] >= dc else "failure"
        events.append({
            "kind": "roll",
            "check": (args.get("check") or "")[:120],
            "expression": result["expression"],
            "rolls": result["rolls"],
            "modifier": result["modifier"],
            "total": result["total"],
            "dc": dc,
            "outcome": outcome,
        })
        text = (
            f"Rolled {result['expression']}: {result['rolls']} "
            f"{'+' if result['modifier'] >= 0 else ''}{result['modifier']} "
            f"= {result['total']}"
        )
        if outcome:
            text += f" vs DC {dc} -> {outcome.upper()}"
        return text, events

    if tool_name == "world.lookup":
        return lookup_table(
            store.manifest, str(args.get("table") or ""),
            str(args.get("query") or ""),
        ), events

    return f"Unknown world tool '{tool_name}'.", events
