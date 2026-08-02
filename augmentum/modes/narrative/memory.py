"""Narrative memory — three-layer architecture: state snapshot, memory ledger, compaction.

Three layers:
- **State snapshot**: Current scene fields (location, who's present, activity, etc.)
  updated by the LLM after each batch of messages.
- **Memory ledger**: Chronological list of significant events with round stamps
  and categories, produced alongside the state snapshot.
- **Compaction**: Periodic LLM-based merge of old ledger entries to keep the
  ledger from growing without bound.

Detects the card type (narrator, character, ensemble) from a parsed
CharacterCard and builds card-type-specific prompts for the state+memory
LLM call.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from augmentum.modes.narrative.card_parser import CharacterCard


# Refusal detection — uses compound phrases to avoid false positives
# on normal dialogue like "I can't believe you did that!"
# Each entry requires BOTH an action phrase AND a context phrase.
_REFUSAL_PAIRS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # (action phrases, context phrases) — must match one from EACH group
    (
        ("i can't", "i cannot", "i'm unable to", "i am unable to",
         "i won't", "i will not", "i'm not able to", "i must decline",
         "i need to decline", "i have to decline"),
        ("as an ai", "as a language model", "content policy", "guidelines",
         "appropriate", "programmed to", "designed to", "safety",
         "ethical", "this type of content", "this kind of content",
         "generate that", "write that", "create that", "produce that",
         "engage with", "continue with this"),
    ),
]

# Standalone phrases that are unambiguous refusal markers on their own
_REFUSAL_STANDALONE = (
    "as an ai language model",
    "as an artificial intelligence",
    "as a large language model",
    "against my content policy",
    "violates my guidelines",
    "my ethical guidelines",
    "i'm designed to be helpful, harmless",
    "i'm programmed to avoid",
    "i apologize, but as an ai",
    "content policy violation",
)


def _is_refusal_text(text: str) -> bool:
    """Check if text is an AI refusal/safety response.

    Uses compound phrase matching — requires BOTH an action phrase
    (e.g. 'I can't') AND a context phrase (e.g. 'as an AI') to avoid
    false positives on normal dialogue.
    """
    if not text or len(text) > 2000:
        return False
    lower = text.lower()

    # Check standalone markers first (unambiguous on their own)
    if any(phrase in lower for phrase in _REFUSAL_STANDALONE):
        return True

    # Check compound pairs — must match one from each group
    for action_phrases, context_phrases in _REFUSAL_PAIRS:
        has_action = any(p in lower for p in action_phrases)
        has_context = any(p in lower for p in context_phrases)
        if has_action and has_context:
            return True

    return False


class CardType(str, Enum):
    CHARACTER = "character"
    NARRATOR = "narrator"
    ENSEMBLE = "ensemble"


class SummaryMode(str, Enum):
    LITE = "lite"
    STANDARD = "standard"


# Keywords that indicate a narrator / world-builder card
_NARRATOR_KEYWORDS = [
    "narrator",
    "dungeon master",
    "game master",
    "storyteller",
    "world of",
    "the world",
    "the realm",
    "the kingdom",
    "the land",
    "chronicles",
    "campaign",
    "adventure setting",
    "rpg",
]

# Patterns that indicate multiple characters (ensemble)
_ENSEMBLE_PATTERNS = [
    re.compile(r"characters?\s*:", re.IGNORECASE),
    re.compile(r"the (?:group|party|team|crew|companions)", re.IGNORECASE),
    re.compile(r"(?:and|&)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,|\s+and\b)", re.IGNORECASE),
]


def detect_card_type(card: CharacterCard) -> CardType:
    """Classify a character card as narrator, character, or ensemble.

    Uses heuristic keyword/pattern matching on the card's description,
    personality, scenario, and system prompt fields.
    """
    # Combine searchable text
    searchable = " ".join([
        card.description,
        card.personality,
        card.scenario,
        card.system_prompt,
        card.name,
    ]).lower()

    # Check for narrator indicators
    narrator_score = 0
    for keyword in _NARRATOR_KEYWORDS:
        if keyword in searchable:
            narrator_score += 1

    # Strong narrator signal: name itself is "narrator" or "game master" etc.
    name_lower = card.name.lower()
    if any(kw in name_lower for kw in ("narrator", "dungeon master", "game master", "storyteller")):
        narrator_score += 3

    if narrator_score >= 2:
        return CardType.NARRATOR

    # Check for ensemble indicators
    ensemble_score = 0
    full_text = " ".join([
        card.description,
        card.personality,
        card.scenario,
        card.system_prompt,
    ])
    for pattern in _ENSEMBLE_PATTERNS:
        if pattern.search(full_text):
            ensemble_score += 1

    # Count capitalized names in description (rough heuristic for multiple characters)
    # Match "Name" patterns that look like character names (not sentence starters)
    name_pattern = re.compile(r"(?:,\s*|\band\b\s+)([A-Z][a-z]{2,})")
    name_matches = name_pattern.findall(card.description)
    if len(set(name_matches)) >= 2:
        ensemble_score += 1

    if ensemble_score >= 2:
        return CardType.ENSEMBLE

    return CardType.CHARACTER


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StateSnapshot:
    """Current scene snapshot — what's happening RIGHT NOW."""

    fields: dict[str, str] = field(default_factory=dict)
    card_type: CardType = CardType.CHARACTER

    def to_dict(self) -> dict:
        return {"fields": self.fields, "card_type": self.card_type.value}

    @classmethod
    def from_dict(cls, data: dict) -> StateSnapshot:
        ct = CardType.CHARACTER
        with contextlib.suppress(ValueError):
            ct = CardType(data.get("card_type", "character"))
        return cls(fields=data.get("fields", {}), card_type=ct)


@dataclass
class MemoryEntry:
    """A single memory ledger entry with temporal stamp."""

    round_num: int = 0
    category: str = ""
    content: str = ""

    def to_dict(self) -> dict:
        return {"round_num": self.round_num, "category": self.category, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        return cls(
            round_num=data.get("round_num", 0),
            category=data.get("category", ""),
            content=data.get("content", ""),
        )


# ---------------------------------------------------------------------------
# State field definitions per card type
# ---------------------------------------------------------------------------

STATE_FIELDS: dict[CardType, list[str]] = {
    CardType.CHARACTER: [
        "location", "who_present", "current_activity",
        "emotional_tone", "immediate_tensions", "open_threads",
        "character_dynamics",
    ],
    CardType.ENSEMBLE: [
        "location", "characters_present", "group_dynamic",
        "current_activity", "open_threads",
        "character_dynamics",
    ],
    CardType.NARRATOR: [
        "location", "party_status", "active_quest",
        "immediate_situation", "environmental_conditions", "pending_decisions",
        "key_relationships",
    ],
}

# Extra instructions for relationship-aware STATE fields
_DYNAMICS_INSTRUCTION = {
    CardType.CHARACTER: (
        "- character_dynamics: One line per present character showing "
        "their current disposition and key relationships. "
        'Format: "Name (mood/state) → relationship toward others". '
        'Example: "Alice (nervous) → trusts Bob, wary of Carol"'
    ),
    CardType.ENSEMBLE: (
        "- character_dynamics: One line per present character showing "
        "disposition and inter-character relationships. "
        'Format: "Name (state) → feelings/stance toward others"'
    ),
    CardType.NARRATOR: (
        "- key_relationships: One line per significant NPC/faction pair. "
        'Format: "Entity → relationship toward other". '
        'Example: "Guard Captain (suspicious) → loyal to Crown, hunting rebels"'
    ),
}

# ---------------------------------------------------------------------------
# Memory category definitions per card type
# ---------------------------------------------------------------------------

MEMORY_CATEGORIES: dict[CardType, list[str]] = {
    CardType.CHARACTER: [
        "relationship_shift", "discovery", "commitment",
        "consequence", "emotional_milestone", "world_change",
    ],
    CardType.ENSEMBLE: [
        "alliance", "conflict", "shared_discovery",
        "group_decision", "character_development", "world_change",
    ],
    CardType.NARRATOR: [
        "quest_update", "lore_reveal", "world_change",
        "party_decision", "npc_relationship", "resource_change", "rule_established",
    ],
}


# ---------------------------------------------------------------------------
# Prompt builders and parsers
# ---------------------------------------------------------------------------


def build_state_memory_prompt(
    card_type: CardType,
    current_state: StateSnapshot | None,
    memory_ledger: list[MemoryEntry],
    recent_messages: list[str],
    char_name: str,
    batch_start: int,
    batch_end: int,
    *,
    custom_prompt: str = "",
    mode: SummaryMode = SummaryMode.STANDARD,
) -> tuple[str, str]:
    """Build (system, user) prompt pair for the STATE+MEMORY LLM call."""
    fields = STATE_FIELDS.get(card_type, STATE_FIELDS[CardType.CHARACTER])
    categories = MEMORY_CATEGORIES.get(card_type, MEMORY_CATEGORIES[CardType.CHARACTER])

    state_word_target = 100 if mode == SummaryMode.LITE else 200
    max_new_entries = 3 if mode == SummaryMode.LITE else 10

    # If custom prompt provided, use it directly
    if custom_prompt:
        system_content = custom_prompt.format(
            char_name=char_name or "the character",
            state_fields=", ".join(fields),
            categories=", ".join(categories),
            word_target=state_word_target,
        )
    else:
        # Build field list, replacing the dynamics field with its detailed instruction
        dynamics_key = "character_dynamics" if card_type != CardType.NARRATOR else "key_relationships"
        dynamics_inst = _DYNAMICS_INSTRUCTION.get(card_type, "")
        field_lines = []
        for f in fields:
            if f == dynamics_key and dynamics_inst:
                field_lines.append(dynamics_inst)
            else:
                field_lines.append(f"- {f}")

        system_content = (
            f"Ignore all previous instructions, roleplay context, and character cards.\n"
            f"You are a narrative state tracker for \"{char_name or 'the character'}\".\n\n"
            f"Given the recent exchanges (rounds R{batch_start}-R{batch_end}), produce TWO sections:\n\n"
            f"## STATE\n"
            f"A snapshot of the CURRENT situation in ~{state_word_target} words. "
            f"Use these fields (one per line, field: value):\n"
            + "\n".join(field_lines)
            + "\n\n"
            f"## MEMORY\n"
            f"Up to {max_new_entries} new bullet points for IMPORTANT events from these rounds only.\n"
            f"Each bullet: [R#|category] content\n"
            f"Valid categories: {', '.join(categories)}\n"
            f"R# must be between R{batch_start} and R{batch_end}.\n"
            f"Only record genuinely significant events — skip routine dialogue.\n\n"
            f"Output ONLY the two sections. No preamble, no commentary."
        )

    # Build user message with round-numbered messages
    # Filter out AI refusal/safety responses to prevent memory contamination
    truncated = recent_messages[-30:] if len(recent_messages) > 30 else recent_messages
    # Each message's round number is its 1-based position in the full history.
    # The old formula (batch_start + offset + i) produced inflated R# values
    # when the history exceeded 30 messages, causing all entries to be clamped
    # to batch_end and destroying temporal ordering in the ledger.
    start_idx = len(recent_messages) - len(truncated)
    message_lines = []
    for i, msg in enumerate(truncated):
        if _is_refusal_text(msg):
            continue
        round_num = start_idx + i + 1  # 1-based position in full history
        message_lines.append(f"[R{round_num}] {msg}")

    user_parts = [
        f"These exchanges span rounds R{batch_start} through R{batch_end}.\n",
        "\n\n".join(message_lines),
    ]

    # Include current state for context
    if current_state and current_state.fields:
        state_lines = [f"- {k}: {v}" for k, v in current_state.fields.items() if v]
        if state_lines:
            user_parts.insert(0, "Previous STATE (update/overwrite based on new events):\n" + "\n".join(state_lines) + "\n")

    # Include recent ledger for context
    if memory_ledger:
        recent_entries = memory_ledger[-10:]
        ledger_lines = [f"[R{e.round_num}|{e.category}] {e.content}" for e in recent_entries]
        user_parts.insert(0 if not current_state else 1,
            "Recent MEMORY entries (for context, do NOT repeat these):\n" + "\n".join(ledger_lines) + "\n")

    user_content = "\n".join(user_parts)
    return system_content, user_content


def parse_state_memory_response(
    raw_text: str,
    card_type: CardType,
    batch_start: int,
    batch_end: int,
) -> tuple[StateSnapshot, list[MemoryEntry]]:
    """Parse the LLM response into a StateSnapshot and list of MemoryEntry.

    Validates R# are within [batch_start, batch_end]; out-of-range falls back to batch_end.
    """
    fields = STATE_FIELDS.get(card_type, STATE_FIELDS[CardType.CHARACTER])
    categories = set(MEMORY_CATEGORIES.get(card_type, MEMORY_CATEGORIES[CardType.CHARACTER]))

    # Split into STATE and MEMORY sections
    state_text = ""
    memory_text = ""

    # Try to find ## STATE and ## MEMORY headers
    state_match = re.search(r"##\s*STATE\s*\n(.*?)(?=##\s*MEMORY|$)", raw_text, re.DOTALL | re.IGNORECASE)
    memory_match = re.search(r"##\s*MEMORY\s*\n(.*?)$", raw_text, re.DOTALL | re.IGNORECASE)

    if state_match:
        state_text = state_match.group(1).strip()
    if memory_match:
        memory_text = memory_match.group(1).strip()

    # If no headers found, try splitting on "MEMORY" keyword
    if not state_text and not memory_text:
        parts = re.split(r"\n\s*(?:MEMORY|Memory)\s*\n", raw_text, maxsplit=1)
        state_text = parts[0].strip()
        if len(parts) > 1:
            memory_text = parts[1].strip()

    # Parse STATE fields
    parsed_fields: dict[str, str] = {}
    for line in state_text.split("\n"):
        line = line.strip().lstrip("- ")
        # Strip markdown bold/italic wrapping (e.g. **field:** or *field:*)
        line = re.sub(r"^\*{1,2}(.+?)\*{1,2}", r"\1", line)
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            # Strip any trailing markdown from key (e.g. "**" residue)
            key = key.strip("*")
            value = value.strip()
            # Match against known fields (fuzzy)
            for f in fields:
                if f == key or f.replace("_", " ") == key.replace("_", " "):
                    parsed_fields[f] = value
                    break

    snapshot = StateSnapshot(fields=parsed_fields, card_type=card_type)

    # Parse MEMORY entries
    entries: list[MemoryEntry] = []
    # Pattern: [R42|category] content
    entry_pattern = re.compile(r"\[R(\d+)\|([^\]]+)\]\s*(.+)")
    for line in memory_text.split("\n"):
        line = line.strip().lstrip("-* ")
        m = entry_pattern.match(line)
        if m:
            round_num = int(m.group(1))
            category = m.group(2).strip().lower().replace(" ", "_")
            content = m.group(3).strip()

            # Validate round number
            if round_num < batch_start or round_num > batch_end:
                round_num = batch_end

            # Skip entries that describe AI refusal/safety behavior
            if _is_refusal_text(content):
                continue

            entries.append(MemoryEntry(
                round_num=round_num,
                category=category,
                content=content,
            ))

    return snapshot, entries


def format_state_for_context(snapshot: StateSnapshot) -> str:
    """Format a StateSnapshot for injection into the system prompt."""
    if not snapshot.fields:
        return ""
    lines = []
    for key, value in snapshot.fields.items():
        if value:
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")
    if not lines:
        return ""
    return "[Current State]\n" + "\n".join(lines)


def format_ledger_for_context(entries: list[MemoryEntry]) -> str:
    """Format memory ledger entries for injection into the system prompt."""
    if not entries:
        return ""
    lines = []
    for e in entries:
        lines.append(f"[R{e.round_num}|{e.category}] {e.content}")
    return "[Story Memory]\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph derivation from STATE + MEMORY (no LLM call)
# ---------------------------------------------------------------------------

def build_compaction_prompt(
    entries_to_compact: list[MemoryEntry],
    card_type: CardType,
) -> tuple[str, str]:
    """Build prompt for merging/compacting old memory ledger entries."""
    categories = MEMORY_CATEGORIES.get(card_type, MEMORY_CATEGORIES[CardType.CHARACTER])

    n = len(entries_to_compact)
    system_content = (
        "You are a text compressor for a structured event log. This is NOT a creative task.\n\n"
        "Each line is a database record:\n"
        "  [R{round}|{category}] {text}\n\n"
        "The [R#|category] tag is a LOCKED KEY — copy it exactly, character for character.\n"
        "Only the {text} portion may change.\n\n"
        "RULES:\n"
        f"1. You will receive {n} records. You must output {n} records (same count).\n"
        "   EXCEPTION: if two *adjacent* records describe the exact same moment or fact,\n"
        "   you may merge them into one — use the lower R#, preserve all facts from both.\n"
        "   This should be rare. When in doubt, keep them separate.\n"
        "2. Shorten {text} by removing filler words, articles, and redundant phrases.\n"
        "   Every named entity, action, and state change in the original must survive.\n"
        "3. Do NOT infer, extrapolate, or add anything not literally in that record's own text.\n"
        "4. Do NOT use any knowledge from outside these records — no future context,\n"
        "   no background story knowledge, no events not shown here.\n"
        f"5. Valid categories: {', '.join(categories)}\n"
        "6. Output ONLY the records, one per line. No headers, no commentary.\n\n"
        "EXAMPLE:\n"
        "IN:  [R4|discovery] Elena found an old letter hidden beneath the loose floorboard in the attic\n"
        "OUT: [R4|discovery] Elena found old letter under loose floorboard in attic"
    )

    entry_lines = [f"[R{e.round_num}|{e.category}] {e.content}" for e in entries_to_compact]
    user_content = (
        f"Compress these {n} records. "
        "Copy each [R#|category] tag unchanged. "
        "Shorten text only — preserve all named entities and events:\n\n"
        + "\n".join(entry_lines)
    )

    return system_content, user_content
