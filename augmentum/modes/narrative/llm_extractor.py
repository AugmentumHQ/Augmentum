"""LLM-based semantic extraction for narrative mode.

Extracts structured state changes from narrative text using an LLM call,
providing deeper understanding than regex-based trackers. Results are merged
back into the existing tracker system (LLM wins on conflicts).

Runs asynchronously after each response — never blocks the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

_MAX_INPUT_CHARS = 3000

_SYSTEM_PROMPT = """\
You are a narrative state extraction system. Analyze the given roleplay/story \
text and extract structured state changes. Rules:
- Extract character emotions, actions, locations, and relationship shifts.
- Extract world state changes (location, time, weather, atmosphere).
- Extract plot signals (new threads, progressions, resolutions).
- Extract new facts established in the narrative.
- Use the character names exactly as given in the known characters list.
- Only extract what is explicitly stated or strongly implied — do not invent.
- Return valid JSON matching the schema below. Return {"characters":[],"world":{},"plots":{},"facts":[]} if nothing notable.

JSON Schema:
{
  "characters": [
    {
      "name": "CharName",
      "emotion": "happy|sad|angry|afraid|surprised|calm|excited|confused|loving|disgusted|null",
      "emotion_confidence": 0.8,
      "physical_state": "sitting|standing|fighting|injured|null",
      "location": "the tavern|null",
      "inventory_add": ["sword"],
      "inventory_remove": ["gold pouch"],
      "relationship_changes": {"OtherChar": "growing trust"}
    }
  ],
  "world": {
    "location": "the tavern|null",
    "time_of_day": "dawn|morning|noon|afternoon|evening|night|null",
    "weather": "raining|snowing|stormy|foggy|clear|cloudy|windy|null",
    "atmosphere": "tense|peaceful|chaotic|null"
  },
  "plots": {
    "new_threads": ["A mysterious stranger arrives with a sealed letter"],
    "progressions": ["The group moves closer to finding the artifact"],
    "resolutions": ["The tavern dispute is settled peacefully"]
  },
  "facts": [
    {"content": "Lyra is left-handed", "confidence": 0.9}
  ]
}
"""

_USER_PROMPT = """\
Known characters: {characters}

Narrative text to analyze:
{text}
"""


@dataclass
class NarrativeExtraction:
    """Structured result from LLM narrative extraction."""

    characters: list[CharacterExtraction] = field(default_factory=list)
    world: WorldExtraction | None = None
    plots: PlotExtraction | None = None
    facts: list[FactExtraction] = field(default_factory=list)


@dataclass
class CharacterExtraction:
    name: str = ""
    emotion: str | None = None
    emotion_confidence: float = 0.5
    physical_state: str | None = None
    location: str | None = None
    inventory_add: list[str] = field(default_factory=list)
    inventory_remove: list[str] = field(default_factory=list)
    relationship_changes: dict[str, str] = field(default_factory=dict)


@dataclass
class WorldExtraction:
    location: str | None = None
    time_of_day: str | None = None
    weather: str | None = None
    atmosphere: str | None = None


@dataclass
class PlotExtraction:
    new_threads: list[str] = field(default_factory=list)
    progressions: list[str] = field(default_factory=list)
    resolutions: list[str] = field(default_factory=list)


@dataclass
class FactExtraction:
    content: str = ""
    confidence: float = 0.8


def _parse_extraction(raw: str) -> NarrativeExtraction | None:
    """Parse LLM JSON output into structured extraction."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        log.debug("narrative_llm_extract_parse_failed", raw_len=len(raw))
        return None

    if not isinstance(data, dict):
        return None

    result = NarrativeExtraction()

    # Parse characters
    for char_data in data.get("characters", []):
        if not isinstance(char_data, dict):
            continue
        name = char_data.get("name", "")
        if not name:
            continue
        result.characters.append(CharacterExtraction(
            name=name,
            emotion=char_data.get("emotion") if char_data.get("emotion") != "null" else None,
            emotion_confidence=float(char_data.get("emotion_confidence", 0.5)),
            physical_state=char_data.get("physical_state") if char_data.get("physical_state") != "null" else None,
            location=char_data.get("location") if char_data.get("location") != "null" else None,
            inventory_add=char_data.get("inventory_add", []) or [],
            inventory_remove=char_data.get("inventory_remove", []) or [],
            relationship_changes=char_data.get("relationship_changes", {}) or {},
        ))

    # Parse world
    world_data = data.get("world", {})
    if isinstance(world_data, dict) and any(
        world_data.get(k) and world_data.get(k) != "null"
        for k in ("location", "time_of_day", "weather", "atmosphere")
    ):
        result.world = WorldExtraction(
            location=world_data.get("location") if world_data.get("location") != "null" else None,
            time_of_day=world_data.get("time_of_day") if world_data.get("time_of_day") != "null" else None,
            weather=world_data.get("weather") if world_data.get("weather") != "null" else None,
            atmosphere=world_data.get("atmosphere") if world_data.get("atmosphere") != "null" else None,
        )

    # Parse plots
    plots_data = data.get("plots", {})
    if isinstance(plots_data, dict):
        new_threads = plots_data.get("new_threads", []) or []
        progressions = plots_data.get("progressions", []) or []
        resolutions = plots_data.get("resolutions", []) or []
        if new_threads or progressions or resolutions:
            result.plots = PlotExtraction(
                new_threads=[t for t in new_threads if isinstance(t, str) and t],
                progressions=[p for p in progressions if isinstance(p, str) and p],
                resolutions=[r for r in resolutions if isinstance(r, str) and r],
            )

    # Parse facts
    for fact_data in data.get("facts", []):
        if not isinstance(fact_data, dict):
            continue
        content = fact_data.get("content", "")
        if content:
            result.facts.append(FactExtraction(
                content=content,
                confidence=float(fact_data.get("confidence", 0.8)),
            ))

    return result


async def extract_narrative_state(
    backend: ModelBackend,
    text: str,
    known_characters: list[str],
    model: str = "",
) -> NarrativeExtraction | None:
    """Call the LLM to extract structured narrative state from text.

    Args:
        model: Override model name. Empty string uses the backend default.

    Returns None on any failure (parse error, LLM error, etc.).
    """
    from augmentum.models.base import InternalChatRequest, Message

    truncated = text[:_MAX_INPUT_CHARS]
    char_list = ", ".join(known_characters) if known_characters else "(none known yet)"

    user_content = _USER_PROMPT.format(characters=char_list, text=truncated)

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
        stream=False,
        temperature=0.1,
        # Run on a non-chat slot when multi-slot is enabled so extraction
        # doesn't block the user's next turn.
        is_background_task=True,
    )

    try:
        response = await backend.chat(request)
        if not response.message or not response.message.content:
            return None
        return _parse_extraction(response.message.content)
    except Exception:
        log.debug("narrative_llm_extraction_failed", exc_info=True)
        return None
