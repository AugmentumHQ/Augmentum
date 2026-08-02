"""OpenAI-format tool schemas for lorebook management verbs.

Gives the model native access to the session's lorebook — search existing
entries, create new ones as the story evolves, update content or keywords,
and remove entries that are no longer relevant.

This is F1 + F5 from the companion-model training design: lorebook-as-tool
(grounded retrieval instead of blind pre-injection) and lorebook authoring
from narrative (the model proactively records world details it establishes).

Follows the same pattern as ``recall_schemas.py``: pure schema list, pure
dispatcher, no side effects beyond the LoreEngine mutation. The recall loop
handles orchestration.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from augmentum.modes.narrative.lore_engine import LoreEngine
from augmentum.state.narrative_state import LorebookEntry, LorebookPosition
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


LOREBOOK_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lorebook_search",
            "description": (
                "Search the lorebook (world info) for entries matching a "
                "query. Use before writing about established lore — locations, "
                "factions, items, rules — to stay consistent. Returns matching "
                "entries with their keywords and full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search term to match against entry keywords, "
                            "names, and content."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Max entries to return.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lorebook_create",
            "description": (
                "Create a new lorebook entry to record world details "
                "established in the narrative — a new location, faction, "
                "rule, item, or other persistent world fact. The entry will "
                "be keyword-triggered in future turns when those keywords "
                "appear in conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short display name for the entry.",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Trigger keywords. Entry activates when any "
                            "keyword appears in recent messages."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The lore text injected into context when "
                            "triggered. Write as concise reference material."
                        ),
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100,
                        "description": (
                            "Higher priority entries are injected first "
                            "when budget is tight."
                        ),
                    },
                    "constant": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, always injected regardless of "
                            "keyword matches."
                        ),
                    },
                },
                "required": ["name", "keywords", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lorebook_update",
            "description": (
                "Update an existing lorebook entry's content, keywords, or "
                "settings. Use when established lore changes — a location is "
                "destroyed, a faction's allegiance shifts, an item's "
                "properties are revealed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": (
                            "ID of the entry to update (from lorebook_search "
                            "results)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (replaces existing).",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New keywords (replaces existing).",
                    },
                    "name": {
                        "type": "string",
                        "description": "New display name.",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Set false to disable without deleting.",
                    },
                },
                "required": ["entry_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lorebook_delete",
            "description": (
                "Remove a lorebook entry that is no longer relevant to the "
                "narrative. Prefer lorebook_update with enabled=false for "
                "entries that might become relevant again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": (
                            "ID of the entry to delete (from lorebook_search "
                            "results)."
                        ),
                    },
                },
                "required": ["entry_id"],
            },
        },
    },
]

LOREBOOK_TOOL_NAMES: frozenset[str] = frozenset(
    schema["function"]["name"] for schema in LOREBOOK_TOOL_SCHEMAS
)

LOREBOOK_MUTATING_TOOLS: frozenset[str] = frozenset(
    {"lorebook_create", "lorebook_update", "lorebook_delete"}
)


def _format_entry(entry: LorebookEntry) -> str:
    """Render one entry as a compact text block for the model."""
    keywords = ", ".join(entry.keywords) if entry.keywords else "(none)"
    status_parts: list[str] = []
    if not entry.enabled:
        status_parts.append("DISABLED")
    if entry.constant:
        status_parts.append("constant")
    status = f" [{', '.join(status_parts)}]" if status_parts else ""
    name = entry.comment or entry.id
    return (
        f"[{name}] (id: {entry.id}){status}\n"
        f"  keywords: {keywords}\n"
        f"  priority: {entry.priority}\n"
        f"  content: {entry.content}"
    )


def _search_entries(
    lore_engine: LoreEngine,
    query: str,
    limit: int = 5,
) -> list[LorebookEntry]:
    """Find entries matching a query across keywords, name, and content.

    Tokenized per-term (ANY, scored) so a phrasal query surfaces entries
    that hit any word — shared behavior with the native surface."""
    from augmentum.modes.narrative.recall import _query_terms
    terms = _query_terms(query) or [query.lower().strip()]
    terms = [t for t in terms if t]
    scored: list[tuple[int, LorebookEntry]] = []

    for entry in lore_engine.entries.values():
        score = 0
        kws = [kw.lower() for kw in entry.keywords]
        name = (entry.comment or "").lower()
        content = (entry.content or "").lower()
        for term in terms:
            for kw in kws:
                if term == kw:
                    score += 10
                elif term in kw or kw in term:
                    score += 6
            if name and term in name:
                score += 4
            if term in content:
                score += 2
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: (-x[0], x[1].priority))
    return [entry for _, entry in scored[:limit]]


def dispatch_lorebook_tool(
    lore_engine: LoreEngine,
    session_id: str,
    *,
    tool_name: str,
    raw_arguments: str | dict[str, Any] | None,
) -> tuple[str, list[dict] | None]:
    """Execute one lorebook tool call.

    Returns (result_text, mutations) where mutations is a list of
    {action, entry_data} dicts for mutating tools (so the UI can sync),
    or None for read-only tools.
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
                    f"Got: {type(args).__name__}",
                    None,
                )
        except (json.JSONDecodeError, TypeError) as exc:
            return f"Tool '{tool_name}' arguments are not valid JSON: {exc}", None

    if tool_name == "lorebook_search":
        return _handle_search(lore_engine, args), None

    if tool_name == "lorebook_create":
        return _handle_create(lore_engine, session_id, args)

    if tool_name == "lorebook_update":
        return _handle_update(lore_engine, args)

    if tool_name == "lorebook_delete":
        return _handle_delete(lore_engine, args)

    return (
        f"Unknown lorebook tool '{tool_name}'. Available: "
        f"{', '.join(sorted(LOREBOOK_TOOL_NAMES))}.",
        None,
    )


def _handle_search(lore_engine: LoreEngine, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "lorebook_search requires a 'query' argument."
    try:
        limit = int(args.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 20))

    matches = _search_entries(lore_engine, query, limit)
    if not matches:
        total = len(lore_engine.entries)
        return f"No lorebook entries match '{query}'. ({total} entries total)"

    parts = [f"Found {len(matches)} matching entries:"]
    for entry in matches:
        parts.append(_format_entry(entry))
    return "\n\n".join(parts)


def _handle_create(
    lore_engine: LoreEngine, session_id: str, args: dict,
) -> tuple[str, list[dict]]:
    name = str(args.get("name") or "").strip()
    keywords = args.get("keywords")
    content = str(args.get("content") or "").strip()

    if not name:
        return "lorebook_create requires a 'name' argument.", None
    if not keywords or not isinstance(keywords, list):
        return "lorebook_create requires a 'keywords' array.", None
    if not content:
        return "lorebook_create requires a 'content' argument.", None

    clean_keywords = [str(k).strip() for k in keywords if str(k).strip()]
    if not clean_keywords:
        return "lorebook_create: keywords array must contain non-empty strings.", None

    # Read-before-write duplicate guard — shared with the native surface so
    # both lorebook tool families dedup identically (fix-the-class).
    from augmentum.modes.narrative.lorebook_native_schemas import (
        _find_duplicate_candidate,
    )
    dup = _find_duplicate_candidate(lore_engine, clean_keywords)
    if dup is not None:
        log.info("lorebook_tool_duplicate_blocked", session_id=session_id,
                 existing_id=dup.id, keywords=clean_keywords)
        preview = (dup.content or "").strip()
        if len(preview) > 240:
            preview = preview[:240].rstrip() + "…"
        return (
            f"A similar entry already exists — \"{dup.comment or (dup.keywords[0] if dup.keywords else dup.id)}\" "
            f"(entry_id: {dup.id}):\n  {preview}\n\n"
            f"If this records the SAME subject, call lorebook_update with "
            f"entry_id=\"{dup.id}\" to refine it — do NOT create a duplicate. "
            f"If it is genuinely a DIFFERENT subject, call lorebook_create "
            f"again with more distinctive keywords.",
            None,
        )

    entry_id = f"llm_{uuid.uuid4().hex[:12]}"
    raw_priority = args.get("priority", 100)
    try:
        priority = int(raw_priority)
    except (TypeError, ValueError):
        priority = 100
    constant = bool(args.get("constant", False))

    entry = LorebookEntry(
        id=entry_id,
        session_id=session_id,
        keywords=clean_keywords,
        content=content,
        priority=max(1, min(priority, 1000)),
        source="llm_authored",
        enabled=True,
        constant=constant,
        position=LorebookPosition.BEFORE_CHAR,
        comment=name,
    )
    lore_engine.add_entry(entry)
    log.info("lorebook_tool_created", entry_id=entry_id, name=name,
             keywords=clean_keywords)

    mutation = {
        "action": "create",
        "entry": {
            "id": entry_id,
            "keys": clean_keywords,
            "content": content,
            "name": name,
            "priority": entry.priority,
            "enabled": True,
            "constant": constant,
            "position": "before_char",
            "source": "llm_authored",
            "comment": name,
        },
    }
    return (
        f"Created lorebook entry '{name}' (id: {entry_id}) with keywords: "
        f"{', '.join(clean_keywords)}",
        [mutation],
    )


def _handle_update(
    lore_engine: LoreEngine, args: dict,
) -> tuple[str, list[dict] | None]:
    entry_id = str(args.get("entry_id") or "").strip()
    if not entry_id:
        return "lorebook_update requires an 'entry_id' argument.", None

    entries = lore_engine.entries
    entry = entries.get(entry_id)
    if entry is None:
        return (
            f"No lorebook entry with id '{entry_id}'. Use lorebook_search "
            f"to find the correct id.",
            None,
        )

    changes: list[str] = []
    mutation_data: dict[str, Any] = {"id": entry_id}

    if "content" in args and args["content"] is not None:
        entry.content = str(args["content"])
        mutation_data["content"] = entry.content
        changes.append("content")

    if "keywords" in args and isinstance(args["keywords"], list):
        entry.keywords = [str(k).strip() for k in args["keywords"] if str(k).strip()]
        mutation_data["keys"] = entry.keywords
        changes.append("keywords")

    if "name" in args and args["name"] is not None:
        entry.comment = str(args["name"]).strip()
        mutation_data["name"] = entry.comment
        changes.append("name")

    if "enabled" in args and args["enabled"] is not None:
        entry.enabled = bool(args["enabled"])
        mutation_data["enabled"] = entry.enabled
        changes.append(f"enabled={entry.enabled}")

    if not changes:
        return "lorebook_update: no fields to update were provided.", None

    log.info("lorebook_tool_updated", entry_id=entry_id, changes=changes)
    mutation = {"action": "update", "entry": mutation_data}
    name = entry.comment or entry_id
    return f"Updated lorebook entry '{name}': {', '.join(changes)}", [mutation]


def _handle_delete(
    lore_engine: LoreEngine, args: dict,
) -> tuple[str, list[dict] | None]:
    entry_id = str(args.get("entry_id") or "").strip()
    if not entry_id:
        return "lorebook_delete requires an 'entry_id' argument.", None

    entries = lore_engine.entries
    if entry_id not in entries:
        return (
            f"No lorebook entry with id '{entry_id}'. Use lorebook_search "
            f"to find the correct id.",
            None,
        )

    name = entries[entry_id].comment or entry_id
    lore_engine.remove_entry(entry_id)
    log.info("lorebook_tool_deleted", entry_id=entry_id, name=name)
    mutation = {"action": "delete", "entry": {"id": entry_id}}
    return f"Deleted lorebook entry '{name}' (id: {entry_id})", [mutation]
