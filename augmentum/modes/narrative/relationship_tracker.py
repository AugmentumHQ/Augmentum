"""Character relationship tracker — trust/affection/tension scores between characters.

Maintains a directed graph of relationships. Each edge carries three dimensions:
- trust:     -1.0 (betrayal) to +1.0 (absolute trust)
- affection: -1.0 (hatred) to +1.0 (love)
- tension:    0.0 (none) to +1.0 (explosive)

Updates are confidence-dampened — small signals nudge slowly, explicit signals shift faster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Dampening factor — higher = more responsive to new signals
_DAMPENING = 0.3

# Clamp all scores to [-1.0, 1.0] (tension to [0.0, 1.0])
_MIN_SCORE = -1.0
_MAX_SCORE = 1.0


@dataclass
class Relationship:
    """A directed relationship from source to target."""

    source: str
    target: str
    trust: float = 0.0
    affection: float = 0.0
    tension: float = 0.0
    label: str = ""
    last_updated_at: int = 0


@dataclass
class RelationshipDelta:
    """A change to a relationship."""

    source: str
    target: str
    trust_delta: float = 0.0
    affection_delta: float = 0.0
    tension_delta: float = 0.0
    label: str = ""
    confidence: float = 0.5


# Regex-based signal detection for relationship changes
_TRUST_POSITIVE = [
    (re.compile(r"\b(?:trust(?:s|ed|ing)?|confid(?:e[sd]?|ing)|reli(?:es?|ed) on)\b", re.IGNORECASE), 0.15),
    (re.compile(r"\b(?:sav(?:es?|ed|ing)|protect(?:s|ed|ing)?|defend(?:s|ed|ing)?)\b", re.IGNORECASE), 0.1),
    (re.compile(r"\b(?:honest(?:ly)?|truthful(?:ly)?|loyal(?:ty)?)\b", re.IGNORECASE), 0.1),
]

_TRUST_NEGATIVE = [
    (re.compile(r"\b(?:betray(?:s|ed|al|ing)?|deceiv(?:es?|ed|ing)|li(?:es?|ed|ying) to)\b", re.IGNORECASE), -0.2),
    (re.compile(r"\b(?:distrust(?:s|ed)?|suspicio(?:us|n)|doubt(?:s|ed|ing)?)\b", re.IGNORECASE), -0.1),
]

_AFFECTION_POSITIVE = [
    (re.compile(r"\b(?:lov(?:es?|ed|ing)|ador(?:es?|ed|ing))\b", re.IGNORECASE), 0.2),
    (re.compile(r"\b(?:kiss(?:es|ed|ing)?|embrac(?:es?|ed|ing)|hug(?:s|ged|ging)?)\b", re.IGNORECASE), 0.15),
    (re.compile(r"\b(?:caring|tender(?:ly)?|gentle|warmth)\b", re.IGNORECASE), 0.1),
    (re.compile(r"\b(?:friend(?:s|ship)?|companion|ally|allies)\b", re.IGNORECASE), 0.08),
]

_AFFECTION_NEGATIVE = [
    (re.compile(r"\b(?:hat(?:es?|ed|ing)|despi(?:ses?|sed))\b", re.IGNORECASE), -0.2),
    (re.compile(r"\b(?:repuls(?:ed|ion)|revuls(?:ed|ion)|disgust(?:ed)?)\b", re.IGNORECASE), -0.15),
    (re.compile(r"\b(?:cold(?:ly)?|dismiss(?:es|ed|ive)?|ignor(?:es?|ed|ing))\b", re.IGNORECASE), -0.08),
]

_TENSION_INCREASE = [
    (re.compile(r"\b(?:argu(?:es?|ed|ing|ment)|quarrel(?:s|ed|ing)?|fight(?:s|ing)?|fought)\b", re.IGNORECASE), 0.15),
    (re.compile(r"\b(?:threaten(?:s|ed|ing)?|confront(?:s|ed|ing)?|challeng(?:es?|ed|ing))\b", re.IGNORECASE), 0.12),
    (re.compile(r"\b(?:tension|hostil(?:e|ity)|antagoni(?:sm|stic|ze))\b", re.IGNORECASE), 0.1),
    (re.compile(r"\b(?:glare[sd]?|snarl(?:s|ed)?|growl(?:s|ed)?)\b", re.IGNORECASE), 0.08),
]

_TENSION_DECREASE = [
    (re.compile(r"\b(?:reconcil(?:es?|ed|iation)|forgiv(?:es?|en|ing)|apologiz(?:es?|ed|ing))\b", re.IGNORECASE), -0.15),
    (re.compile(r"\b(?:peac(?:e|eful)|calm(?:ed|ing)?|resolution)\b", re.IGNORECASE), -0.08),
]


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


class RelationshipTracker:
    """Tracks directed relationships between characters."""

    def __init__(self) -> None:
        self._relationships: dict[tuple[str, str], Relationship] = {}

    @property
    def relationships(self) -> list[Relationship]:
        return list(self._relationships.values())

    def get(self, source: str, target: str) -> Relationship | None:
        return self._relationships.get((source, target))

    def get_all_for(self, character: str) -> list[Relationship]:
        """Get all relationships where character is source or target."""
        return [
            r for r in self._relationships.values()
            if r.source == character or r.target == character
        ]

    def _get_or_create(self, source: str, target: str) -> Relationship:
        key = (source, target)
        if key not in self._relationships:
            self._relationships[key] = Relationship(source=source, target=target)
        return self._relationships[key]

    def apply_delta(
        self,
        delta: RelationshipDelta,
        message_index: int,
    ) -> Relationship:
        """Apply a relationship change with confidence dampening."""
        rel = self._get_or_create(delta.source, delta.target)
        scale = _DAMPENING * delta.confidence

        if delta.trust_delta:
            rel.trust = _clamp(rel.trust + delta.trust_delta * scale, _MIN_SCORE, _MAX_SCORE)
        if delta.affection_delta:
            rel.affection = _clamp(rel.affection + delta.affection_delta * scale, _MIN_SCORE, _MAX_SCORE)
        if delta.tension_delta:
            rel.tension = _clamp(rel.tension + delta.tension_delta * scale, 0.0, _MAX_SCORE)
        if delta.label:
            rel.label = delta.label

        rel.last_updated_at = message_index

        log.debug(
            "relationship_updated",
            source=delta.source,
            target=delta.target,
            trust=round(rel.trust, 3),
            affection=round(rel.affection, 3),
            tension=round(rel.tension, 3),
        )
        return rel

    def extract_signals(
        self,
        text: str,
        character_names: list[str],
    ) -> list[RelationshipDelta]:
        """Extract relationship signals from text using regex heuristics.

        Finds pairs of characters mentioned near relationship-relevant keywords.
        """
        if len(character_names) < 2:
            return []

        deltas: list[RelationshipDelta] = []
        text_lower = text.lower()

        # For each pair of characters, check for signals
        for i, source in enumerate(character_names):
            for target in character_names[i + 1:]:
                # Check if both characters are mentioned
                if source.lower() not in text_lower or target.lower() not in text_lower:
                    continue

                trust_d = 0.0
                affection_d = 0.0
                tension_d = 0.0

                for pattern, score in _TRUST_POSITIVE + _TRUST_NEGATIVE:
                    if pattern.search(text):
                        trust_d += score

                for pattern, score in _AFFECTION_POSITIVE + _AFFECTION_NEGATIVE:
                    if pattern.search(text):
                        affection_d += score

                for pattern, score in _TENSION_INCREASE + _TENSION_DECREASE:
                    if pattern.search(text):
                        tension_d += score

                if trust_d or affection_d or tension_d:
                    # Bidirectional — both characters experience the shift
                    deltas.append(RelationshipDelta(
                        source=source,
                        target=target,
                        trust_delta=trust_d,
                        affection_delta=affection_d,
                        tension_delta=tension_d,
                        confidence=0.5,
                    ))
                    deltas.append(RelationshipDelta(
                        source=target,
                        target=source,
                        trust_delta=trust_d,
                        affection_delta=affection_d,
                        tension_delta=tension_d,
                        confidence=0.5,
                    ))

        return deltas

    def merge_llm_relationships(
        self,
        character_name: str,
        relationship_changes: dict[str, str],
        message_index: int,
    ) -> None:
        """Merge LLM-extracted relationship labels into the graph.

        LLM provides qualitative labels like "growing trust", "romantic tension",
        which we convert into dimension deltas.
        """
        _LABEL_SIGNALS: dict[str, tuple[float, float, float]] = {
            # label keyword → (trust_d, affection_d, tension_d)
            "trust": (0.3, 0.0, 0.0),
            "distrust": (-0.3, 0.0, 0.1),
            "friend": (0.2, 0.2, 0.0),
            "enemy": (-0.2, -0.2, 0.3),
            "love": (0.1, 0.4, 0.0),
            "hate": (-0.1, -0.4, 0.2),
            "romantic": (0.0, 0.3, 0.1),
            "rivalry": (-0.1, 0.0, 0.3),
            "tension": (0.0, 0.0, 0.3),
            "respect": (0.2, 0.1, 0.0),
            "fear": (-0.1, -0.1, 0.2),
            "loyalty": (0.3, 0.1, 0.0),
            "betrayal": (-0.4, -0.2, 0.3),
            "warmth": (0.1, 0.2, -0.1),
            "cold": (-0.1, -0.2, 0.1),
        }

        for target, label in relationship_changes.items():
            label_lower = label.lower()

            # Find best matching signal
            trust_d, affection_d, tension_d = 0.0, 0.0, 0.0
            for keyword, (t, a, te) in _LABEL_SIGNALS.items():
                if keyword in label_lower:
                    trust_d += t
                    affection_d += a
                    tension_d += te

            # If no keyword match, just store the label
            if not (trust_d or affection_d or tension_d):
                rel = self._get_or_create(character_name, target)
                rel.label = label
                rel.last_updated_at = message_index
                continue

            delta = RelationshipDelta(
                source=character_name,
                target=target,
                trust_delta=trust_d,
                affection_delta=affection_d,
                tension_delta=tension_d,
                label=label,
                confidence=0.8,  # LLM signals are higher confidence
            )
            self.apply_delta(delta, message_index)

    def get_context_summary(self, max_chars: int = 500) -> str:
        """Build a text summary of relationships for context injection."""
        if not self._relationships:
            return ""

        parts = ["Character relationships:"]
        remaining = max_chars - len(parts[0])

        for rel in sorted(
            self._relationships.values(),
            key=lambda r: abs(r.trust) + abs(r.affection) + r.tension,
            reverse=True,
        ):
            dims = []
            if abs(rel.trust) >= 0.1:
                dims.append(f"trust:{rel.trust:+.1f}")
            if abs(rel.affection) >= 0.1:
                dims.append(f"affection:{rel.affection:+.1f}")
            if rel.tension >= 0.1:
                dims.append(f"tension:{rel.tension:.1f}")
            if not dims:
                continue

            line = f"- {rel.source} → {rel.target}: {', '.join(dims)}"
            if rel.label:
                line += f" ({rel.label})"
            if len(line) > remaining:
                break
            parts.append(line)
            remaining -= len(line)

        return "\n".join(parts) if len(parts) > 1 else ""

    def to_dict_list(self) -> list[dict]:
        """Serialize all relationships for persistence."""
        return [
            {
                "source": r.source,
                "target": r.target,
                "trust": r.trust,
                "affection": r.affection,
                "tension": r.tension,
                "label": r.label,
                "last_updated_at": r.last_updated_at,
            }
            for r in self._relationships.values()
        ]

    def load_from_dict_list(self, data: list[dict]) -> None:
        """Load relationships from persistence."""
        self._relationships.clear()
        for item in data:
            rel = Relationship(
                source=item["source"],
                target=item["target"],
                trust=item.get("trust", 0.0),
                affection=item.get("affection", 0.0),
                tension=item.get("tension", 0.0),
                label=item.get("label", ""),
                last_updated_at=item.get("last_updated_at", 0),
            )
            self._relationships[(rel.source, rel.target)] = rel
