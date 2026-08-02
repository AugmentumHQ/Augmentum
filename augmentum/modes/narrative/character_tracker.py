"""Character tracker — maintains character state with confidence-dampened updates.

Inspired by BetterSimTracker's approach: confidence-scaled deltas with dampening
to prevent wild swings in emotional/relationship state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from augmentum.state.narrative_state import Entity, EntityType, StateDelta
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Dampening factor — higher = more sensitive to confidence
_DAMPENING = 0.6

# Emotional state keywords mapped to categories
_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "happy": ["happy", "joyful", "delighted", "pleased", "cheerful", "elated", "laughing", "smiled", "grinning"],
    "sad": ["sad", "sorrowful", "melancholy", "crying", "tearful", "grieving", "depressed", "mourning"],
    "angry": ["angry", "furious", "enraged", "irritated", "annoyed", "seething", "livid", "wrathful"],
    "afraid": ["afraid", "scared", "terrified", "frightened", "anxious", "nervous", "trembling", "worried"],
    "surprised": ["surprised", "shocked", "astonished", "stunned", "amazed", "startled"],
    "disgusted": ["disgusted", "revolted", "repulsed", "sickened", "nauseated"],
    "calm": ["calm", "serene", "peaceful", "composed", "relaxed", "tranquil"],
    "excited": ["excited", "enthusiastic", "eager", "thrilled", "animated", "energetic"],
    "confused": ["confused", "puzzled", "bewildered", "perplexed", "baffled"],
    "loving": ["loving", "affectionate", "tender", "warm", "caring", "adoring"],
}

# Flattened for quick lookup
_EMOTION_LOOKUP: dict[str, str] = {}
for category, words in _EMOTION_KEYWORDS.items():
    for word in words:
        _EMOTION_LOOKUP[word] = category

# Action patterns that indicate physical state changes
# These match on inner action text (without surrounding asterisks)
_PHYSICAL_PATTERNS = [
    (re.compile(r"(?:sits|sat|sitting)\s+(?:down|on|in)", re.IGNORECASE), "sitting"),
    (re.compile(r"(?:stands|stood|standing)\s+(?:up|tall|firm)", re.IGNORECASE), "standing"),
    (re.compile(r"(?:lies|lay|lying)\s+(?:down|on|in)", re.IGNORECASE), "lying down"),
    (re.compile(r"(?:kneels?|kneeling|knelt)", re.IGNORECASE), "kneeling"),
    (re.compile(r"(?:walks?|walking|walked)\s", re.IGNORECASE), "walking"),
    (re.compile(r"(?:runs?|running|ran)\s", re.IGNORECASE), "running"),
    (re.compile(r"(?:fights?|fighting|fought|attacks?|attacking)", re.IGNORECASE), "fighting"),
    (re.compile(r"(?:sleeps?|sleeping|slept)", re.IGNORECASE), "sleeping"),
    (re.compile(r"(?:wounded|injured|hurt|bleeding)", re.IGNORECASE), "injured"),
]

# Location change patterns — match on inner action text (without asterisks)
_LOCATION_PATTERNS = [
    re.compile(r"(?:enters?|entering|entered|walks?\s+into|stepped\s+into)\s+(?:the\s+)?(.+?)$", re.IGNORECASE),
    re.compile(r"(?:arrives?\s+at|arriving\s+at|arrived\s+at)\s+(?:the\s+)?(.+?)$", re.IGNORECASE),
    re.compile(r"(?:moves?\s+to|moving\s+to|moved\s+to)\s+(?:the\s+)?(.+?)$", re.IGNORECASE),
    re.compile(r"(?:goes?\s+to|going\s+to|went\s+to)\s+(?:the\s+)?(.+?)$", re.IGNORECASE),
]


@dataclass
class CharacterUpdate:
    """Extracted state change for a character from a message."""

    name: str = ""
    emotional_state: str | None = None
    emotional_confidence: float = 0.5
    physical_state: str | None = None
    location: str | None = None
    inventory_add: list[str] | None = None
    inventory_remove: list[str] | None = None
    relationship_updates: dict[str, str] | None = None


class CharacterTracker:
    """Tracks character state across messages with confidence-dampened updates."""

    def extract_updates(
        self,
        text: str,
        known_characters: list[Entity],
    ) -> list[CharacterUpdate]:
        """Extract character state changes from a message.

        Uses heuristic extraction — no LLM call needed.
        """
        updates: list[CharacterUpdate] = []

        # Build name→entity lookup
        char_names = {}
        for entity in known_characters:
            if entity.entity_type == EntityType.CHARACTER:
                char_names[entity.name.lower()] = entity.name
                for alias in entity.aliases:
                    char_names[alias.lower()] = entity.name

        # Extract action blocks (text between *asterisks*)
        action_blocks = re.findall(r"\*([^*]+)\*", text)

        # Process each known character
        for _alias, char_name in char_names.items():
            update = CharacterUpdate(name=char_name)
            has_changes = False

            # Check emotional state from the full text
            emotion = self._detect_emotion(text, char_name, char_names)
            if emotion:
                update.emotional_state = emotion[0]
                update.emotional_confidence = emotion[1]
                has_changes = True

            # Check physical state from action blocks
            for action in action_blocks:
                physical = self._detect_physical_state(action, char_name, char_names)
                if physical:
                    update.physical_state = physical
                    has_changes = True
                    break

            # Check location changes
            for action in action_blocks:
                location = self._detect_location_change(action, char_name, char_names)
                if location:
                    update.location = location
                    has_changes = True
                    break

            if has_changes:
                updates.append(update)

        return updates

    def apply_update(
        self,
        entity: Entity,
        update: CharacterUpdate,
        message_index: int,
    ) -> StateDelta:
        """Apply a character update to an entity with confidence dampening.

        Returns the delta that was applied.
        """
        delta: dict = {}

        if update.emotional_state:
            # Confidence-dampened emotional update
            # "Stickiness" — low confidence preserves previous mood
            confidence = update.emotional_confidence
            scale = (1 - _DAMPENING) + confidence * _DAMPENING

            if scale >= 0.5 or not entity.state.emotional_state:
                delta["emotional_state"] = update.emotional_state

        if update.physical_state:
            delta["physical_state"] = update.physical_state

        if update.location:
            delta["location"] = update.location

        if update.inventory_add or update.inventory_remove:
            delta["inventory"] = {
                "add": update.inventory_add or [],
                "remove": update.inventory_remove or [],
            }

        if update.relationship_updates:
            delta["relationships"] = update.relationship_updates

        # Apply delta to entity
        if delta:
            entity.state = entity.state.apply_delta(delta)
            log.debug(
                "character_updated",
                name=entity.name,
                message_index=message_index,
                delta_keys=list(delta.keys()),
            )

        return StateDelta(
            entity_id=entity.id,
            message_index=message_index,
            delta=delta,
            branch_id=entity.branch_id,
        )

    def _detect_emotion(
        self,
        text: str,
        char_name: str,
        all_chars: dict[str, str],
    ) -> tuple[str, float] | None:
        """Detect emotional state for a specific character from text."""
        text_lower = text.lower()

        # Look for emotion words near the character's name or in their actions
        # Priority: action blocks attributed to character > general text
        best_emotion = None
        best_confidence = 0.0

        # Check action blocks attributed to this character
        char_action_pattern = re.compile(
            rf"(?:{re.escape(char_name)}[^*]*?\*([^*]+)\*|\*([^*]*?{re.escape(char_name)}[^*]*?)\*)",
            re.IGNORECASE,
        )
        attributed_actions = char_action_pattern.findall(text)
        action_text = " ".join(
            a[0] or a[1] for a in attributed_actions
        ).lower() if attributed_actions else ""

        # Search for emotion keywords
        for word, category in _EMOTION_LOOKUP.items():
            if word in action_text:
                confidence = 0.8  # High confidence for attributed actions
                if confidence > best_confidence:
                    best_emotion = category
                    best_confidence = confidence
            elif word in text_lower:
                confidence = 0.4  # Lower for general text
                if confidence > best_confidence:
                    best_emotion = category
                    best_confidence = confidence

        if best_emotion:
            return (best_emotion, best_confidence)
        return None

    def _detect_physical_state(
        self,
        action: str,
        char_name: str,
        all_chars: dict[str, str],
    ) -> str | None:
        """Detect physical state changes from an action block."""
        # Check if action references this character
        if char_name.lower() not in action.lower():
            return None

        for pattern, state in _PHYSICAL_PATTERNS:
            if pattern.search(action):
                return state
        return None

    def _detect_location_change(
        self,
        action: str,
        char_name: str,
        all_chars: dict[str, str],
    ) -> str | None:
        """Detect location changes from an action block."""
        if char_name.lower() not in action.lower():
            return None

        for pattern in _LOCATION_PATTERNS:
            match = pattern.search(action)
            if match:
                location = match.group(1).strip().rstrip(".")
                if len(location) < 100:  # Sanity check
                    return location
        return None
