"""Load a :class:`CompanionPersona` from a stored character card.

The route layer turns ``StartSessionBody.character_id`` into a
``CompanionPersona`` via this helper. Kept here (rather than in the
route file) so a unit test can exercise the parsing logic with just
an aiosqlite-shaped fake instead of mounting the FastAPI app.

Data sources
------------
1. ``ui_characters.data`` — the JSON blob the chat UI persists. May
   contain top-level ``name`` / ``personality`` / ``description`` /
   ``scenario`` / ``voice`` keys, OR a nested ``data`` field (V2 card
   format), OR a ``system_prompt`` we can run through CardParser.
2. ``ui_characters.name`` — authoritative display name.
3. Voice is a top-level ``voice`` field in the data JSON. Cards that
   were created before the companion picker existed have no voice
   field; the persona falls back to the bridge default.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from augmentum.game_agent.companion import CompanionPersona
from augmentum.modes.narrative.card_parser import CardParser

log = structlog.get_logger(__name__)


# Max chars from each source field; persona prompt budget is tight.
_PERSONA_FIELD_LIMIT = 240
_PERSONA_TOTAL_LIMIT = 600


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    # Cut on a word boundary if one is near.
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit - 60:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _persona_summary_from(card_data: dict[str, Any]) -> str:
    """Build a compact persona paragraph from whichever card fields exist.

    Before:
    - card_data = {"personality": "Curious...", "scenario": "Playing Pokémon."}
    After:
    - returns "Curious... | Scenario: Playing Pokémon."

    Returns ``""`` if no usable fields were present.
    """

    parts: list[str] = []
    for key in ("personality", "description"):
        v = card_data.get(key, "")
        if isinstance(v, str) and v.strip():
            parts.append(_truncate(v, _PERSONA_FIELD_LIMIT))
            break  # one of personality OR description, not both
    scenario = card_data.get("scenario", "")
    if isinstance(scenario, str) and scenario.strip():
        parts.append("Scenario: " + _truncate(scenario, _PERSONA_FIELD_LIMIT))
    joined = " | ".join(parts)
    return _truncate(joined, _PERSONA_TOTAL_LIMIT)


def _flatten_card_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize V2 ``{"data": {...}}`` nesting so callers see one flat dict.

    Before:
    - raw = {"spec": "chara_card_v2", "data": {"name": "Aria", ...}}
    After:
    - returns {"name": "Aria", ...}

    Before:
    - raw = {"name": "Aria", "personality": "..."}
    After:
    - returns the same dict (already flat)
    """

    nested = raw.get("data")
    if isinstance(nested, dict) and ("name" in nested or "personality" in nested):
        merged = dict(nested)
        # Top-level voice / image fields win when both exist — the
        # client UI tends to set those at the top level.
        for k in ("voice", "image_model", "image_style", "avatar"):
            if k in raw and k not in merged:
                merged[k] = raw[k]
        return merged
    return raw


async def load_persona(
    conn: Any, character_id: str, user_id: str | None,
) -> CompanionPersona | None:
    """Look up a character card and turn it into a :class:`CompanionPersona`.

    Use when:
    - The game-agent route is constructing an :class:`Orchestrator` and
      the client supplied a ``character_id``. The conn is the SQLite
      backend connection used everywhere else in the proxy.

    Expects:
    - ``conn`` exposes ``execute()`` returning an awaitable cursor with
      ``fetchone()`` — matches the aiosqlite shape Augmentum uses.
    - When ``user_id`` is non-empty, the lookup is scoped to that user
      (cross-tenant isolation, matches the persona_routes pattern).

    Returns:
    - A :class:`CompanionPersona` on a successful lookup. ``None`` when
      the character does not exist (caller falls back to anonymous
      companion mode).
    """

    if not character_id:
        return None

    q = "SELECT name, data FROM ui_characters WHERE id = ?"
    params: list[Any] = [character_id]
    if user_id:
        q += " AND user_id = ?"
        params.append(user_id)

    try:
        cursor = await conn.execute(q, params)
        row = await cursor.fetchone()
    except Exception as exc:  # noqa: BLE001
        log.warning("game_agent.persona.lookup_failed", char_id=character_id, error=str(exc))
        return None

    if not row:
        return None

    name_col = row[0] or ""
    try:
        raw = json.loads(row[1] or "{}")
    except json.JSONDecodeError:
        raw = {}
    card_data = _flatten_card_data(raw) if isinstance(raw, dict) else {}

    name = (card_data.get("name") or name_col or "").strip()
    persona_summary = _persona_summary_from(card_data)

    # Cards may have a ``system_prompt`` (SillyTavern / V2 format) we
    # can parse for extra context when the direct fields are empty.
    if not persona_summary:
        sysp = card_data.get("system_prompt") or card_data.get("description") or ""
        if isinstance(sysp, str) and sysp.strip():
            parsed = CardParser().parse(sysp)
            if parsed is not None and parsed.personality:
                persona_summary = _truncate(parsed.personality, _PERSONA_TOTAL_LIMIT)

    voice = ""
    raw_voice = card_data.get("voice", "")
    if isinstance(raw_voice, str):
        voice = raw_voice.strip()

    return CompanionPersona(name=name, persona=persona_summary, voice=voice)


__all__ = ["load_persona"]
