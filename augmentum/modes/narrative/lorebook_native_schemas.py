"""Native dot-named lorebook tools — ``lorebook.check`` / ``lorebook.create``.

These are the F1/F5 tools from ``docs/companion-model-training-design.md``:
the exact tool names the narrative training rows emit, so a trained model's
tool calls land on real handlers at inference instead of phantom-calling.

Why a second lorebook tool module (alongside ``lorebook_schemas.py``):

* ``lorebook_schemas.py`` exposes the underscore family
  (``lorebook_search`` / ``lorebook_create`` / ``lorebook_update`` /
  ``lorebook_delete``) — a richer authoring surface that mutates the live
  ``LoreEngine`` only (no user_id, no explicit branch tag, no DB source
  contract).
* This module implements the two **dot-named** verbs the training data
  actually uses, with the F1/F5 semantics nailed down:
    - ``lorebook.check`` — grounded mid-scene retrieval (read-only).
    - ``lorebook.create`` — records a newly-established detail as
      **session lore** with ``source="narrative_established"`` (NEVER
      touches the source character card) and the current ``branch_id``
      (migration 304), so branch retrieval scopes it correctly.

Both are dispatched by the narrative recall loop. Mutations land in the
``LoreEngine`` (the live source of truth); the engine's ``sync_to_state``
mirrors entries back into ``state.lorebook`` and the user-scoped
``save_session_state`` path writes them to SQLite — so every insert is
scoped by ``user_id`` + ``session_id`` and never reaches the anon row.

Design contract (mirrors recall_schemas.py):
* **Errors are content, not exceptions.** Bad args / unknown verb return
  a clear error STRING the model can read and recover from. Never raises.
* **Empty check is not an error.** No matching lore → a clear "nothing
  established" summary so the model knows it's free to invent.
* **Pure-ish.** ``check`` is read-only; ``create`` mutates only the
  LoreEngine (the one intended side effect) and reports a mutation dict
  for UI sync.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from augmentum.modes.narrative.lore_engine import LoreEngine
from augmentum.state.narrative_state import LorebookEntry, LorebookPosition
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# The category enum from the F5 spec. Stored on the entry's ``comment``
# field as a ``[category]`` prefix (there is no dedicated column) so the
# promotion UI / future retrieval can read it back without a migration.
LOREBOOK_CATEGORIES: tuple[str, ...] = (
    "character",
    "location",
    "item",
    "event",
    "rule",
    "faction",
    "lore",
)

# Source tag for model-authored, session-scoped lore. Distinct from
# "character_book" (imported card copies) so the promotion flow can tell
# what the model established this session vs. what came from the card —
# and so we NEVER mutate the source card.
NARRATIVE_SOURCE = "narrative_established"


LOREBOOK_NATIVE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lorebook.check",
            "description": (
                "Query the world knowledge base for established lore "
                "relevant to what you're about to describe. Use when "
                "writing about a location, character, item, or event "
                "that might have established details, so you ground the "
                "description instead of inventing a contradiction. Empty "
                "result means no established lore — you're free to "
                "establish new details (and should record them with "
                "lorebook.create)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What you need to know — a location, character "
                            "name, event, item, or concept."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Brief context for why you're checking (helps "
                            "retrieval relevance). Optional."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lorebook.create",
            "description": (
                "Record a newly established detail as session world lore. "
                "Use when you've introduced something significant — a new "
                "character, location feature, rule, faction, or event — "
                "that should stay consistent for the rest of this session. "
                "Records to THIS session only; never modifies the source "
                "character card. Check first with lorebook.check to avoid "
                "duplicating something already established."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Keywords that should trigger this entry in "
                            "future turns (names, places, concepts)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The established detail to record.",
                    },
                    "category": {
                        "type": "string",
                        "enum": list(LOREBOOK_CATEGORIES),
                        "description": "What kind of world detail this is.",
                    },
                },
                "required": ["keywords", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lorebook.update",
            "description": (
                "Update an existing lorebook entry when established facts "
                "change — a character dies, a location is destroyed, an "
                "alliance shifts, new information is revealed. Search first "
                "with lorebook.check to find the entry, then update. Do not "
                "create a duplicate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": (
                            "ID of the entry to update (from lorebook.check "
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
                    "enabled": {
                        "type": "boolean",
                        "description": (
                            "Set false to disable without deleting."
                        ),
                    },
                },
                "required": ["entry_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lorebook.delete",
            "description": (
                "Remove a lorebook entry that is no longer relevant. Prefer "
                "lorebook.update with enabled=false for entries that might "
                "become relevant again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": (
                            "ID of the entry to delete (from lorebook.check "
                            "results)."
                        ),
                    },
                },
                "required": ["entry_id"],
            },
        },
    },
]


# Canonical dotted names PLUS their underscore-sanitized forms. OpenAI-
# compatible function names forbid dots, so llama-server / the model often
# rewrites ``lorebook.check`` → ``lorebook_check`` in the response stream
# (see ``tools/registry.py::_normalise``). The recall loop matches tool
# names by exact membership in the internal-names set, so we register BOTH
# spellings — otherwise the model's correctly-chosen verb comes back
# sanitized and is rejected as "Unknown tool". ``_canonical_name`` folds
# either spelling back to the dotted id the dispatcher switches on.
_CANONICAL_BY_SANITIZED: dict[str, str] = {}
for _schema in LOREBOOK_NATIVE_TOOL_SCHEMAS:
    _dotted = _schema["function"]["name"]
    _CANONICAL_BY_SANITIZED[_dotted] = _dotted
    _CANONICAL_BY_SANITIZED[_dotted.replace(".", "_")] = _dotted

LOREBOOK_NATIVE_TOOL_NAMES: frozenset[str] = frozenset(_CANONICAL_BY_SANITIZED)

# Which of the native verbs write state (so the loop / handler can treat
# their results as needing a UI sync + persist).
LOREBOOK_NATIVE_MUTATING_TOOLS: frozenset[str] = frozenset(
    {"lorebook.create", "lorebook_create",
     "lorebook.update", "lorebook_update",
     "lorebook.delete", "lorebook_delete"}
)


def _canonical_name(tool_name: str) -> str:
    """Fold a possibly-sanitized tool name back to its dotted canonical id."""
    return _CANONICAL_BY_SANITIZED.get(tool_name, tool_name)


def _format_entry(entry: LorebookEntry) -> str:
    """Render one entry as a compact text block for the model to ground on."""
    keywords = ", ".join(entry.keywords) if entry.keywords else "(none)"
    name = entry.comment or entry.id
    return (
        f"[{name}] (id: {entry.id}, keywords: {keywords})\n"
        f"  {entry.content}"
    )


def _search_entries(
    lore_engine: LoreEngine,
    query: str,
    limit: int,
) -> list[LorebookEntry]:
    """Score entries against a query across keywords, name, and content.

    Keywords serve as a relevance signal for ranking — entries with more
    keyword matches appear first. Entries with no keyword match but a
    name or content match still appear, just ranked lower. All entries
    are searchable regardless of source or enabled state.
    """
    # Tokenize so a phrasal query ("Luna spirit dragon bond") matches
    # entries hitting ANY term, ranked by cumulative score — not as one
    # opaque substring that misses everything. Fall back to the raw query
    # when it's all stopwords/punctuation.
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

    scored.sort(key=lambda x: (-x[0], -x[1].priority))
    return [entry for _, entry in scored[:limit]]


def _find_duplicate_candidate(
    lore_engine: LoreEngine,
    keywords: list[str],
) -> LorebookEntry | None:
    """Return an existing entry the new keywords likely duplicate, else None.

    Read-before-write guard: models routinely skip the advisory
    ``lorebook.check`` and mint a fresh UUID for a subject already on file,
    so duplicates pile up. We detect the overlap mechanically here.

    Signal: keyword-set overlap, weighted by whether the *label* keyword
    (the first, which becomes the entry name) is shared — that's the
    strongest "same subject" tell. A Jaccard floor keeps unrelated entries
    that happen to share one generic keyword ("castle") from tripping it.
    """
    new_kw = {k.lower() for k in keywords if k.strip()}
    if not new_kw:
        return None
    label = keywords[0].lower()
    best: LorebookEntry | None = None
    best_score = 0.0
    for entry in lore_engine.entries.values():
        ent_kw = {k.lower() for k in (entry.keywords or []) if k.strip()}
        if not ent_kw:
            continue
        overlap = new_kw & ent_kw
        if not overlap:
            continue
        jaccard = len(overlap) / len(new_kw | ent_kw)
        ent_label = entry.keywords[0].lower() if entry.keywords else ""
        label_match = label in ent_kw or (ent_label and ent_label in new_kw)
        score = jaccard + (0.5 if label_match else 0.0)
        if score > best_score:
            best_score = score
            best = entry
    # Strong signal: shared label keyword (>=0.5 via the bonus), or half the
    # combined keyword set overlaps. Below that, treat as genuinely distinct.
    return best if best is not None and best_score >= 0.5 else None


def _parse_args(
    tool_name: str, raw_arguments: str | dict[str, Any] | None,
) -> dict[str, Any] | str:
    """Parse tool arguments defensively. Returns a dict, or an error string."""
    if raw_arguments is None or raw_arguments == "":
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        parsed = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"Tool '{tool_name}' arguments are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return (
            f"Tool '{tool_name}' arguments must be a JSON object. "
            f"Got: {type(parsed).__name__}"
        )
    return parsed


def dispatch_lorebook_native_tool(
    lore_engine: LoreEngine,
    session_id: str,
    *,
    user_id: str,
    branch_id: str = "main",
    tool_name: str,
    raw_arguments: str | dict[str, Any] | None,
) -> tuple[str, list[dict] | None]:
    """Execute one native lorebook tool call.

    ``user_id`` is required for the mutating verb — a model-authored
    entry must be attributable to a user so the user-scoped persist path
    never writes into the anon row. ``branch_id`` tags created entries so
    branch retrieval scopes them (migration 304).

    Returns ``(result_text, mutations)`` where ``mutations`` is a list of
    ``{action, entry}`` dicts for the mutating verb (so the UI can sync),
    or ``None`` for the read-only verb.
    """
    args = _parse_args(tool_name, raw_arguments)
    if isinstance(args, str):  # parse error
        return args, None

    canonical = _canonical_name(tool_name)

    if canonical == "lorebook.check":
        return _handle_check(lore_engine, args), None

    if canonical == "lorebook.create":
        return _handle_create(
            lore_engine, session_id, args,
            user_id=user_id, branch_id=branch_id,
        )

    if canonical == "lorebook.update":
        return _handle_update(lore_engine, args)

    if canonical == "lorebook.delete":
        return _handle_delete(lore_engine, args)

    return (
        f"Unknown lorebook tool '{tool_name}'. Available: "
        f"{', '.join(sorted(LOREBOOK_NATIVE_TOOL_NAMES))}.",
        None,
    )


def _handle_check(lore_engine: LoreEngine, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "lorebook.check requires a 'query' argument."

    matches = _search_entries(lore_engine, query, limit=5)
    if not matches:
        return (
            f"No established lore matching '{query}'. You're free to "
            f"establish new details — record significant ones with "
            f"lorebook.create."
        )

    parts = [f"Established lore for '{query}':"]
    for entry in matches:
        parts.append(_format_entry(entry))
    return "\n\n".join(parts)


def _handle_create(
    lore_engine: LoreEngine,
    session_id: str,
    args: dict,
    *,
    user_id: str,
    branch_id: str,
) -> tuple[str, list[dict] | None]:
    if not user_id:
        # Guard: never author session lore without an owner — it would
        # otherwise be persisted into the anon row by the user-scoped path.
        return (
            "lorebook.create: no user context — cannot record session lore.",
            None,
        )

    keywords = args.get("keywords")
    content = str(args.get("content") or "").strip()
    category = str(args.get("category") or "").strip().lower()

    if not keywords or not isinstance(keywords, list):
        return "lorebook.create requires a 'keywords' array.", None
    if not content:
        return "lorebook.create requires a 'content' argument.", None

    clean_keywords = [str(k).strip() for k in keywords if str(k).strip()]
    if not clean_keywords:
        return "lorebook.create: keywords array must contain non-empty strings.", None

    if category and category not in LOREBOOK_CATEGORIES:
        valid = ", ".join(LOREBOOK_CATEGORIES)
        return (
            f"lorebook.create 'category' must be one of: {valid}. "
            f"Got: {category!r}",
            None,
        )

    # Read-before-write: block a likely duplicate and offer update instead
    # of minting a second entry for the same subject. Not a create — returns
    # the existing entry + both resolution paths so the model can decide.
    dup = _find_duplicate_candidate(lore_engine, clean_keywords)
    if dup is not None:
        log.info(
            "lorebook_native_duplicate_blocked",
            session_id=session_id,
            existing_id=dup.id,
            keywords=clean_keywords,
        )
        preview = (dup.content or "").strip()
        if len(preview) > 240:
            preview = preview[:240].rstrip() + "…"
        return (
            f"A similar entry already exists — \"{dup.comment or (dup.keywords[0] if dup.keywords else dup.id)}\" "
            f"(entry_id: {dup.id}):\n  {preview}\n\n"
            f"If this records the SAME subject, call lorebook.update with "
            f"entry_id=\"{dup.id}\" to refine it — do NOT create a duplicate. "
            f"If it is genuinely a DIFFERENT subject, call lorebook.create "
            f"again with more distinctive keywords.",
            None,
        )

    # The entry's display/memo field doubles as the category carrier
    # ("[location] Ashwander") since there is no dedicated category column.
    # Use the first keyword as a human label.
    label = clean_keywords[0]
    comment = f"[{category}] {label}" if category else label

    entry_id = f"narr_{uuid.uuid4().hex[:12]}"
    entry = LorebookEntry(
        id=entry_id,
        session_id=session_id,
        keywords=clean_keywords,
        content=content,
        priority=100,
        source=NARRATIVE_SOURCE,
        enabled=True,
        constant=False,
        position=LorebookPosition.BEFORE_CHAR,
        comment=comment,
        branch_id=branch_id or "main",
    )
    lore_engine.add_entry(entry)
    log.info(
        "lorebook_native_created",
        entry_id=entry_id,
        session_id=session_id,
        category=category or "(none)",
        branch_id=branch_id or "main",
        keywords=clean_keywords,
    )

    mutation = {
        "action": "create",
        "entry": {
            "id": entry_id,
            "keys": clean_keywords,
            "content": content,
            "name": comment,
            "category": category,
            "priority": entry.priority,
            "enabled": True,
            "constant": False,
            "position": "before_char",
            "source": NARRATIVE_SOURCE,
            "branch_id": entry.branch_id,
            "comment": comment,
        },
    }
    return (
        f"Recorded session lore '{label}'"
        + (f" [{category}]" if category else "")
        + f" with keywords: {', '.join(clean_keywords)}.",
        [mutation],
    )


def _handle_update(
    lore_engine: LoreEngine, args: dict,
) -> tuple[str, list[dict] | None]:
    entry_id = str(args.get("entry_id") or "").strip()
    if not entry_id:
        return "lorebook.update requires an 'entry_id' argument.", None

    entries = lore_engine.entries
    entry = entries.get(entry_id)
    if entry is None:
        return (
            f"No lorebook entry with id '{entry_id}'. Use lorebook.check "
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

    if "enabled" in args and args["enabled"] is not None:
        entry.enabled = bool(args["enabled"])
        mutation_data["enabled"] = entry.enabled
        changes.append(f"enabled={entry.enabled}")

    if not changes:
        return "lorebook.update: no fields to update were provided.", None

    log.info("lorebook_native_updated", entry_id=entry_id, changes=changes)
    name = entry.comment or entry_id
    return (
        f"Updated lorebook entry '{name}': {', '.join(changes)}",
        [{"action": "update", "entry": mutation_data}],
    )


def _handle_delete(
    lore_engine: LoreEngine, args: dict,
) -> tuple[str, list[dict] | None]:
    entry_id = str(args.get("entry_id") or "").strip()
    if not entry_id:
        return "lorebook.delete requires an 'entry_id' argument.", None

    entries = lore_engine.entries
    if entry_id not in entries:
        return (
            f"No lorebook entry with id '{entry_id}'. Use lorebook.check "
            f"to find the correct id.",
            None,
        )

    name = entries[entry_id].comment or entry_id
    lore_engine.remove_entry(entry_id)
    log.info("lorebook_native_deleted", entry_id=entry_id, name=name)
    return (
        f"Deleted lorebook entry '{name}' (id: {entry_id})",
        [{"action": "delete", "entry": {"id": entry_id}}],
    )
