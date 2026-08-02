"""OpenAI-format tool schemas for the narrative recall verbs.

Exposes the data-layer recall functions (``recall.py``) to LLMs as
OpenAI-compatible function definitions. The model can call these
mid-turn to fetch specific narrative state instead of receiving the
full STATE/LEDGER/ARCHIVE injection every turn.

This is the second half of the substrate-as-lookup-layer thesis:
the data layer landed first as plain async functions + HTTP routes
(both already in production); this module wraps them as tools the
narrative tool-execution loop can dispatch.

Design notes
------------

* **Schemas are concise.** Each description names the verb, when to
  use it, and the one or two args that matter. A small model can fit
  all 5 tools in <500 tokens of schema budget.
* **Dispatcher is pure.** Takes the persistence handle, session_id,
  user_id, and a parsed tool_call dict. Returns a string suitable for
  pasting into the tool_result content. No side effects.
* **Errors are content, not exceptions.** Malformed args / unknown
  tool name / persistence failure all return a clear error STRING
  the model can read and recover from. Never raises.
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.modes.narrative.recall import (
    list_entities,
    recall_archive,
    recall_entity,
    recall_facts,
    recall_plot_thread,
)
from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.state.narrative_state import EntityType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# The full schema set — fed into ``InternalChatRequest.tools`` when the
# narrative_recall_tools_enabled setting is on.
RECALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recall_entity",
            "description": (
                "Look up one specific character, location, item, or "
                "faction by exact name or known alias. Use when you "
                "need precise current state (location, emotional "
                "state, inventory, relationships) for an entity you "
                "remember the name of."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name or alias of the entity to look up.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "Enumerate tracked entities; optionally filter by type. "
                "Use for discovery — 'who is present in this scene' / "
                "'what items have been introduced' — when you don't "
                "remember a specific name yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["character", "location", "item", "faction"],
                        "description": "Filter to one entity type. Omit to list all.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": (
                "Search established facts by keyword or tag. Use to "
                "verify continuity, find supporting evidence, or "
                "recall details about an event. Superseded (stale) "
                "facts are automatically excluded. Empty query returns "
                "the most recently established facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring or tag to search for. Multiple terms must all match.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Max number of facts to return (capped at 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_plot_thread",
            "description": (
                "Look up a plot thread by id or title substring. "
                "Returns full thread description and status. Closed/"
                "abandoned threads remain recallable so you can "
                "callback to resolved arcs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Plot thread id or part of the title.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_archive",
            "description": (
                "Semantic search over compacted narrative archive. "
                "Use to retrieve exact past dialogue or scene details "
                "from earlier in this session that have rolled out of "
                "the active context window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the scene or detail to find.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                        "description": "Max number of archive entries to return (capped at 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# Set of tool names — used by the loop to detect whether a tool_call
# from the model is a recall verb (vs. some other tool the request
# might also carry, e.g. UARF/passthrough tools).
RECALL_TOOL_NAMES: frozenset[str] = frozenset(
    schema["function"]["name"] for schema in RECALL_TOOL_SCHEMAS
)


async def dispatch_recall_tool(
    persistence: NarrativePersistence,
    session_id: str,
    *,
    user_id: str,
    tool_name: str,
    raw_arguments: str | dict[str, Any] | None,
) -> str:
    """Execute one recall tool_call.

    ``raw_arguments`` is whatever the model emitted — usually a JSON
    string for OpenAI tool calls. Parse defensively; on failure return
    an actionable error string so the model can correct its call
    rather than the loop bailing.
    """
    args: dict[str, Any]
    if raw_arguments is None or raw_arguments == "":
        args = {}
    elif isinstance(raw_arguments, dict):
        args = raw_arguments
    else:
        try:
            args = json.loads(raw_arguments)
            if not isinstance(args, dict):
                return (
                    f"Tool '{tool_name}' arguments must be a JSON object. "
                    f"Got: {type(args).__name__}"
                )
        except (json.JSONDecodeError, TypeError) as exc:
            return f"Tool '{tool_name}' arguments are not valid JSON: {exc}"

    if tool_name == "recall_entity":
        name = str(args.get("name") or "").strip()
        if not name:
            return "recall_entity requires a 'name' argument."
        result = await recall_entity(
            persistence, session_id, user_id=user_id, name=name,
        )
        return result.summary

    if tool_name == "list_entities":
        type_arg = args.get("type")
        etype: EntityType | None = None
        if type_arg:
            try:
                etype = EntityType(str(type_arg))
            except ValueError:
                valid = ", ".join(t.value for t in EntityType)
                return (
                    f"list_entities 'type' must be one of: {valid}. "
                    f"Got: {type_arg!r}"
                )
        result = await list_entities(
            persistence, session_id, user_id=user_id, entity_type=etype,
        )
        return result.summary

    if tool_name == "recall_facts":
        query = str(args.get("query") or "").strip()
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        result = await recall_facts(
            persistence, session_id, user_id=user_id, query=query, limit=limit,
        )
        return result.summary

    if tool_name == "recall_plot_thread":
        query = str(args.get("query") or "").strip()
        if not query:
            return "recall_plot_thread requires a 'query' argument."
        result = await recall_plot_thread(
            persistence, session_id, user_id=user_id, query=query,
        )
        return result.summary

    if tool_name == "recall_archive":
        query = str(args.get("query") or "").strip()
        if not query:
            return "recall_archive requires a 'query' argument."
        try:
            limit = int(args.get("limit", 3))
        except (TypeError, ValueError):
            limit = 3
        result = await recall_archive(
            persistence, session_id, user_id=user_id, query=query, limit=limit,
        )
        return result.summary

    return (
        f"Unknown recall tool '{tool_name}'. Available: "
        f"{', '.join(sorted(RECALL_TOOL_NAMES))}."
    )
