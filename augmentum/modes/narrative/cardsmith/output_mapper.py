"""Map accumulated Cardsmith session fields into save-ready outputs.

The Cardsmith conversation produces a heterogeneous bag of field emissions
(scalars, arrays, JSON objects). This module translates that bag into:

  - A ``ui_characters.data`` JSON blob ready for the existing _upsert_char path
  - A list of ``regex_scripts`` rows for the regex_scripts table (character-scoped)
  - An ``avatar_prompt`` hint for downstream background image generation

Phase 1 keeps it simple: avatars are not generated automatically, so the
prompt is stashed in extensions.augmentum.cardsmith for now and a later phase
can drive an image-generation job from it.

Defensive: every text field is hard-capped at a sane upper bound and unknown
keys (model drift) are logged as warnings rather than silently dropped.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from augmentum.utils.logging import get_logger

from .state import CardsmithSession

log = get_logger(__name__)


# Hard caps per field. Above these we truncate at a word boundary and append
# an ellipsis. Sized to be larger than any reasonable card field but small
# enough to defend against pathological model output.
_FIELD_CAPS: dict[str, int] = {
    "name": 200,
    "description": 12000,
    "personality": 1500,
    "scenario": 2000,
    "greeting": 4000,
    "examples": 6000,
    "visualTraits": 600,
    "imageStyle": 50,
    "voice": 100,
    "systemPrompt": 4000,
    "postHistoryInstructions": 2000,
    "depthPrompt": 1000,
    "creatorNotes": 2000,
    "backgroundImage": 1000,
    "desc_physical": 4000,
    "desc_personality": 4000,
    "desc_depth": 4000,
}

# Recognized fields. Any session.fields key not in this set is logged as
# model drift so we can spot prompt regressions in production.
_KNOWN_SCALAR_KEYS: frozenset[str] = frozenset({
    "name", "description", "personality", "scenario", "greeting", "examples",
    "visualTraits", "imageStyle", "voice", "systemPrompt",
    "postHistoryInstructions", "depthPrompt", "depthPromptDepth",
    "creatorNotes", "backgroundImage",
    "desc_physical", "desc_personality", "desc_depth",
    # Ensemble-specific scalars
    "group_dynamic", "generation_mode",
})
_KNOWN_ARRAY_KEYS: frozenset[str] = frozenset({
    "tags", "alternateGreetings", "lorebook", "regex_scripts", "avatar_prompt",
    # Ensemble-specific arrays
    "members", "relationships",
    # Control / agentic — consumed by routes layer, not persisted on card
    "fetch_targets",
})


def _cap(text: str, key: str) -> str:
    """Hard-cap text at the configured limit for `key`, with word-boundary trim."""
    cap = _FIELD_CAPS.get(key, 8000)
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[:cap].rsplit(" ", 1)[0] + "…"

# Augmentum image_style enum — anything outside this set falls back to "".
_VALID_IMAGE_STYLES: frozenset[str] = frozenset({
    "anime", "painterly", "photorealistic", "watercolor", "pixel", "comic",
    "dark", "fantasy", "scifi", "ukiyoe", "noir", "cozy",
})


# Canonical empty character payload — matches the shape used by createCharacter()
# in narrative/index.js.
def _empty_character() -> dict[str, Any]:
    return {
        "id": "",
        "name": "New Character",
        "description": "",
        "personality": "",
        "scenario": "",
        "greeting": "",
        "alternateGreetings": [],
        "examples": "",
        "systemPrompt": "",
        "postHistoryInstructions": "",
        "depthPrompt": "",
        "depthPromptDepth": 4,
        "creatorNotes": "",
        "tags": [],
        "avatar": None,
        "backgroundImage": None,
        "visualTraits": "",
        "imageStyle": "",
        "voice": "",
        "autoCollapseNarrativePanels": True,
        "lorebook": [],
        "createdAt": int(time.time() * 1000),
    }


def _new_char_id() -> str:
    """Match the id format used by ui/scripts/narrative/index.js::createCharacter."""
    ts = format(int(time.time() * 1000), "x")
    return f"ch_{ts}_{uuid.uuid4().hex[:5]}"


def _normalize_lorebook_entry(raw: Any) -> dict[str, Any] | None:
    """Coerce a Cardsmith-emitted lorebook entry into character_book shape.

    Accepts JSON objects with at least ``keys`` and ``content``. Drops malformed
    entries silently rather than blowing up the save.
    """
    if not isinstance(raw, dict):
        return None
    keys = raw.get("keys")
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    if not isinstance(keys, list) or not keys:
        return None
    content = raw.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    entry = {
        "keys": [str(k) for k in keys],
        "content": content,
        "enabled": True,
        "priority": int(raw.get("priority", 100)),
        "position": str(raw.get("position", "before_char")),
        "constant": bool(raw.get("constant", False)),
    }
    # Optional World Info V2 fields — pass through when present
    for opt in (
        "secondary_keywords", "selective", "selective_logic",
        "group_name", "group_override", "group_weight",
        "probability", "use_probability", "ignore_budget",
        "match_whole_words", "exclude_recursion", "prevent_recursion",
        "delay_until_recursion", "match_persona",
        "match_char_description", "match_char_personality",
        "match_scenario", "match_creator_notes", "delay_turns",
        "scan_depth", "case_sensitive", "sticky_turns", "cooldown_turns",
        "comment", "outlet_name",
    ):
        if opt in raw:
            entry[opt] = raw[opt]
    return entry


def _normalize_regex_script(raw: Any, char_name: str) -> dict[str, Any] | None:
    """Coerce a Cardsmith-emitted regex script into a regex_scripts row."""
    if not isinstance(raw, dict):
        return None
    find = raw.get("find") or raw.get("pattern")
    if not isinstance(find, str) or not find.strip():
        return None
    placement = str(raw.get("placement", "output")).strip().lower()
    if placement not in ("input", "output", "both"):
        placement = "output"
    return {
        "id": "rgx_" + uuid.uuid4().hex[:12],
        "name": str(raw.get("name") or "Cardsmith script"),
        "find_regex": find,
        "replace_string": str(raw.get("replace") or raw.get("replace_string") or ""),
        "placement": placement,
        "enabled": True,
        "order_num": int(raw.get("order") or raw.get("order_num") or 100),
        "character_name": char_name,
    }


def build_character_payload(session: CardsmithSession) -> dict[str, Any]:
    """Build a save-ready character payload from a Cardsmith session.

    Dispatches on ``session.card_type``. Single returns the canonical solo
    card shape. Ensemble adds a ``character_group`` dict in the payload that
    the route persists to the character_groups table.

    Returns a dict with:
      - char_id, name: identifiers for _upsert_char
      - data: full ui_characters.data JSON blob
      - avatar: base64 data URI or ""
      - regex_scripts: list of regex_scripts table rows to insert
      - avatar_prompt: str (saved into extensions.augmentum)
      - character_group: dict of character_groups columns (ensemble only)
    """
    if session.card_type == "ensemble":
        return _build_ensemble_payload(session)
    return _build_single_payload(session)


def _build_single_payload(session: CardsmithSession) -> dict[str, Any]:
    fields = session.fields
    char = _empty_character()
    char_id = _new_char_id()
    char["id"] = char_id

    # ── Drift detection ───────────────────────────────────────────────────
    # Log any keys the model committed that aren't in our schema. Helps spot
    # prompt regressions where the model invents new field names.
    known = _KNOWN_SCALAR_KEYS | _KNOWN_ARRAY_KEYS
    unknown = [k for k in fields if k not in known]
    if unknown:
        log.warning(
            "cardsmith_unknown_fields",
            session_id=session.session_id,
            keys=sorted(unknown),
        )

    # ── Scalars ────────────────────────────────────────────────────────────
    raw_name = (fields.get("name") or "").strip()
    name = _cap(raw_name, "name") if raw_name else "New Character"
    char["name"] = name

    for path in (
        "personality", "scenario", "greeting", "examples",
        "systemPrompt", "postHistoryInstructions", "creatorNotes",
        "depthPrompt", "visualTraits", "voice", "backgroundImage",
    ):
        v = fields.get(path)
        if isinstance(v, str) and v.strip():
            char[path] = _cap(v, path)

    # description: prefer the slot composition (desc_physical / personality /
    # depth) — emitted once-each by the Cardsmith. Fall back to a directly-
    # emitted `description` if the model bypassed the slots. Each slot is
    # capped individually before joining; the composed result is capped
    # again to the description limit.
    slot_parts: list[str] = []
    for slot in ("desc_physical", "desc_personality", "desc_depth"):
        v = fields.get(slot)
        if isinstance(v, str) and v.strip():
            slot_parts.append(_cap(v, slot))
    if slot_parts:
        char["description"] = _cap("\n\n".join(slot_parts), "description")
    else:
        v = fields.get("description")
        if isinstance(v, str) and v.strip():
            char["description"] = _cap(v, "description")

    # imageStyle: only accept known enum values
    style = (fields.get("imageStyle") or "").strip().lower()
    if style in _VALID_IMAGE_STYLES:
        char["imageStyle"] = style

    # depthPromptDepth: coerce int, clamp 0–10
    raw_depth = fields.get("depthPromptDepth")
    if raw_depth is not None:
        try:
            depth = int(raw_depth)
            char["depthPromptDepth"] = max(0, min(10, depth))
        except (TypeError, ValueError):
            pass

    # ── Arrays ─────────────────────────────────────────────────────────────
    tags = fields.get("tags") or []
    if isinstance(tags, list):
        # Dedupe while preserving order, drop empties, cap reasonable length
        seen: set[str] = set()
        clean: list[str] = []
        for t in tags:
            s = str(t).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                clean.append(s)
            if len(clean) >= 30:
                break
        char["tags"] = clean

    alts = fields.get("alternateGreetings") or []
    if isinstance(alts, list):
        char["alternateGreetings"] = [
            str(g).strip() for g in alts if isinstance(g, str) and g.strip()
        ]

    # Lorebook entries embed in the card data blob (TavernCard character_book
    # round-trip; existing engine seeds session-scoped lorebook_entries from
    # this array on first chat).
    lorebook_raw = fields.get("lorebook") or []
    if isinstance(lorebook_raw, list):
        char["lorebook"] = [
            e for e in (_normalize_lorebook_entry(x) for x in lorebook_raw)
            if e is not None
        ]

    # ── Augmentum extensions metadata ──────────────────────────────────────
    avatar_prompt = ""
    avatar_prompts = fields.get("avatar_prompt") or []
    if isinstance(avatar_prompts, list) and avatar_prompts:
        last = avatar_prompts[-1]
        if isinstance(last, str) and last.strip():
            avatar_prompt = last.strip()

    char["extensions"] = {
        "augmentum": {
            "cardsmith": {
                "source": session.source,
                "card_type": session.card_type,
                "created_at": int(session.created_at * 1000),
                "avatar_prompt": avatar_prompt,
                "seed_prompt": session.meta.get("seed_prompt", ""),
            },
        },
    }

    # ── Regex scripts (separate table — not embedded) ──────────────────────
    regex_raw = fields.get("regex_scripts") or []
    regex_scripts: list[dict[str, Any]] = []
    if isinstance(regex_raw, list):
        regex_scripts = [
            r for r in (_normalize_regex_script(x, name) for x in regex_raw)
            if r is not None
        ]

    return {
        "char_id": char_id,
        "name": name,
        "data": char,
        "avatar": "",
        "regex_scripts": regex_scripts,
        "avatar_prompt": avatar_prompt,
    }


# ── Ensemble payload ───────────────────────────────────────────────────────

_VALID_GENERATION_MODES: frozenset[str] = frozenset({
    "round_robin", "random", "manual", "llm_decide",
})


def _merge_members(raw: Any) -> list[dict[str, str]]:
    """Merge member entries with roster-supersede semantics.

    The Cardsmith establishes the roster first (placeholder entries: name +
    role only, no content fields), then fills per-member during the loop.
    Sometimes the model hallucinates a roster (e.g. "Seraphina, Lorien…")
    before the user supplies real names — without supersede semantics the
    hallucinated members carry over into the saved card.

    Strategy: walk ``raw`` from the end backwards, find the latest
    contiguous run of *placeholder-shaped* entries (name set, content
    fields empty). That defines the canonical roster. Earlier rosters are
    dropped entirely. Any filled entries (with content) merge into the
    canonical roster, ignored when their name isn't part of it.

    Falls back to old by-name-merge behavior when no placeholder block is
    detected (i.e. model emitted only filled entries — no roster
    declaration phase).
    """
    if not isinstance(raw, list):
        return []
    from collections import OrderedDict

    def _is_placeholder(entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        name = (entry.get("name") or "").strip()
        if not name:
            return False
        # Placeholder: name set, no substantive content yet.
        for key in ("summary", "physical", "voice_hint"):
            v = entry.get(key)
            if isinstance(v, str) and v.strip():
                return False
        return True

    # Find the last contiguous run of placeholder entries (the canonical
    # roster declaration). Scan backwards from end. We require at least 2
    # placeholder entries to consider this a "roster declaration" — a
    # single trailing placeholder is more likely an in-progress fill than
    # a deliberate roster reset, and we don't want to drop earlier filled
    # members based on it.
    last_idx = -1
    for i in range(len(raw) - 1, -1, -1):
        if _is_placeholder(raw[i]):
            last_idx = i
            break

    canonical_names: list[str] | None = None
    if last_idx >= 0:
        # Walk backwards from last_idx collecting contiguous placeholder
        # entries — that's the latest roster batch.
        names_in_reverse: list[str] = []
        seen: set[str] = set()
        i = last_idx
        while i >= 0 and _is_placeholder(raw[i]):
            name = (raw[i].get("name") or "").strip()
            if name and name not in seen:
                names_in_reverse.append(name)
                seen.add(name)
            i -= 1
        if len(names_in_reverse) >= 2:
            canonical_names = list(reversed(names_in_reverse))

    if canonical_names is None:
        # No placeholder block — model emitted only filled entries. Fall
        # back to legacy by-name-merge (every name is canonical).
        by_name: OrderedDict[str, dict[str, str]] = OrderedDict()
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            if name not in by_name:
                by_name[name] = {
                    "name": name, "role": "", "summary": "",
                    "physical": "", "voice_hint": "",
                }
            member = by_name[name]
            for key in ("role", "summary", "physical", "voice_hint"):
                v = entry.get(key)
                if isinstance(v, str) and v.strip():
                    member[key] = v.strip()
        return list(by_name.values())

    # Canonical roster path: pre-create the result with the canonical names,
    # then merge fills (and any role-set updates) from anywhere in raw —
    # ignoring entries whose name isn't in the canonical set.
    result: OrderedDict[str, dict[str, str]] = OrderedDict(
        (n, {
            "name": n, "role": "", "summary": "",
            "physical": "", "voice_hint": "",
        })
        for n in canonical_names
    )
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        if not name or name not in result:
            continue
        for key in ("role", "summary", "physical", "voice_hint"):
            v = entry.get(key)
            if isinstance(v, str) and v.strip():
                result[name][key] = v.strip()
    return list(result.values())


def _merge_relationships(raw: Any) -> list[dict[str, Any]]:
    """Merge relationship entries by (source, target). Floats clamped to
    valid ranges (trust/affection ∈ [-1, 1], tension ∈ [0, 1]).
    """
    if not isinstance(raw, list):
        return []
    from collections import OrderedDict
    by_pair: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        src = (entry.get("source") or "").strip()
        tgt = (entry.get("target") or "").strip()
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key not in by_pair:
            by_pair[key] = {
                "source": src,
                "target": tgt,
                "trust": 0.0,
                "affection": 0.0,
                "tension": 0.0,
                "label": "",
            }
        rel = by_pair[key]
        for axis, lo, hi in (("trust", -1.0, 1.0), ("affection", -1.0, 1.0), ("tension", 0.0, 1.0)):
            v = entry.get(axis)
            if isinstance(v, int | float):
                rel[axis] = max(lo, min(hi, float(v)))
        label = entry.get("label")
        if isinstance(label, str) and label.strip():
            rel["label"] = label.strip()
    return list(by_pair.values())


def _compose_ensemble_visual_traits(members: list[dict[str, str]]) -> str:
    """Build the ``<Name> traits`` format the image distiller's regex expects."""
    parts: list[str] = []
    for m in members:
        name = m.get("name", "").strip()
        physical = m.get("physical", "").strip()
        if name and physical:
            parts.append(f"<{name}> {physical}")
    return " ".join(parts)


def _compose_ensemble_description(group_dynamic: str, members: list[dict[str, str]]) -> str:
    """Compose group description: dynamic paragraph + roster bullet list."""
    chunks: list[str] = []
    if group_dynamic.strip():
        chunks.append(group_dynamic.strip())
    if members:
        roster_lines = ["Members:"]
        for m in members:
            name = m.get("name", "")
            role = m.get("role", "")
            summary = m.get("summary", "")
            label = f"{name}" + (f" ({role})" if role else "")
            line = f"- {label}: {summary}" if summary else f"- {label}"
            roster_lines.append(line)
        chunks.append("\n".join(roster_lines))
    return "\n\n".join(chunks)


def _build_ensemble_payload(session: CardsmithSession) -> dict[str, Any]:
    """Ensemble equivalent of _build_single_payload.

    Adds a ``character_group`` dict to the returned payload — the route
    INSERTs that into the character_groups table during finalize. Members
    live in member_summaries (JSON map); the group's name field on
    ui_characters MUST equal character_groups.name for downstream lookups
    to align (the engine matches by name).
    """
    fields = session.fields
    char = _empty_character()
    char_id = _new_char_id()
    char["id"] = char_id

    # Drift detection
    known = _KNOWN_SCALAR_KEYS | _KNOWN_ARRAY_KEYS
    unknown = [k for k in fields if k not in known]
    if unknown:
        log.warning(
            "cardsmith_unknown_fields_ensemble",
            session_id=session.session_id,
            keys=sorted(unknown),
        )

    raw_name = (fields.get("name") or "").strip()
    name = _cap(raw_name, "name") if raw_name else "New Group"
    char["name"] = name

    # Members + relationships (merged across emissions)
    members = _merge_members(fields.get("members"))
    relationships = _merge_relationships(fields.get("relationships"))

    # Description: group_dynamic paragraph + roster bullet list
    group_dynamic = (fields.get("group_dynamic") or "").strip()
    if isinstance(fields.get("description"), str) and fields["description"].strip() and not group_dynamic:
        # Backward compat: model bypassed group_dynamic and emitted description.
        group_dynamic = fields["description"].strip()
    composed_desc = _compose_ensemble_description(group_dynamic, members)
    if composed_desc:
        char["description"] = _cap(composed_desc, "description")

    # visualTraits: <Name> tokens format the image distiller already understands
    composed_vt = _compose_ensemble_visual_traits(members)
    if composed_vt:
        char["visualTraits"] = _cap(composed_vt, "visualTraits")

    # Standard scalars
    for path in (
        "personality", "scenario", "greeting", "examples",
        "systemPrompt", "postHistoryInstructions", "creatorNotes",
        "voice", "backgroundImage",
    ):
        v = fields.get(path)
        if isinstance(v, str) and v.strip():
            char[path] = _cap(v, path)

    # imageStyle enum
    style = (fields.get("imageStyle") or "").strip().lower()
    if style in _VALID_IMAGE_STYLES:
        char["imageStyle"] = style

    # Tags
    tags = fields.get("tags") or []
    if isinstance(tags, list):
        seen: set[str] = set()
        clean: list[str] = []
        for t in tags:
            s = str(t).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                clean.append(s)
            if len(clean) >= 30:
                break
        char["tags"] = clean

    alts = fields.get("alternateGreetings") or []
    if isinstance(alts, list):
        char["alternateGreetings"] = [
            str(g).strip() for g in alts if isinstance(g, str) and g.strip()
        ]

    # Lorebook entries — same shape as single
    lorebook_raw = fields.get("lorebook") or []
    if isinstance(lorebook_raw, list):
        char["lorebook"] = [
            e for e in (_normalize_lorebook_entry(x) for x in lorebook_raw)
            if e is not None
        ]

    # generation_mode enum, defaults to llm_decide
    gen_mode = (fields.get("generation_mode") or "").strip().lower()
    if gen_mode not in _VALID_GENERATION_MODES:
        gen_mode = "llm_decide"

    # Augmentum extensions — stash members + relationships for future
    # session-time seeding into character_relationships.
    avatar_prompt = ""
    avatar_prompts = fields.get("avatar_prompt") or []
    if isinstance(avatar_prompts, list) and avatar_prompts:
        last = avatar_prompts[-1]
        if isinstance(last, str) and last.strip():
            avatar_prompt = last.strip()

    char["extensions"] = {
        "augmentum": {
            "cardsmith": {
                "source": session.source,
                "card_type": "ensemble",
                "created_at": int(session.created_at * 1000),
                "avatar_prompt": avatar_prompt,
                "seed_prompt": session.meta.get("seed_prompt", ""),
                "members": members,
                "relationships": relationships,
                "generation_mode": gen_mode,
            },
        },
    }

    # Regex scripts — character-scoped to the GROUP name (engine resolves
    # group regex like single-character regex).
    regex_raw = fields.get("regex_scripts") or []
    regex_scripts: list[dict[str, Any]] = []
    if isinstance(regex_raw, list):
        regex_scripts = [
            r for r in (_normalize_regex_script(x, name) for x in regex_raw)
            if r is not None
        ]

    # character_groups row
    member_summaries = {
        m["name"]: m.get("summary") or ""
        for m in members
        if m.get("name")
    }
    member_names = [m["name"] for m in members if m.get("name")]
    character_group = {
        "name": name,
        "description": _cap(group_dynamic, "scenario") if group_dynamic else "",
        "member_names": member_names,
        "member_summaries": member_summaries,
        "generation_mode": gen_mode,
        "avatar": "",
        "muted_names": [],
    }

    return {
        "char_id": char_id,
        "name": name,
        "data": char,
        "avatar": "",
        "regex_scripts": regex_scripts,
        "avatar_prompt": avatar_prompt,
        "character_group": character_group,
    }
