"""World tracker — delta-compressed world state tracking.

Inspired by Talemate's approach: store incremental changes per message,
reconstruct state at any point by replaying deltas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Environment descriptors
_TIME_PATTERNS = [
    (re.compile(r"\b(?:dawn|sunrise|early morning)\b", re.IGNORECASE), "dawn"),
    (re.compile(r"\b(?:morning|mid-morning)\b", re.IGNORECASE), "morning"),
    (re.compile(r"\b(?:noon|midday|high sun)\b", re.IGNORECASE), "noon"),
    (re.compile(r"\b(?:afternoon|late afternoon)\b", re.IGNORECASE), "afternoon"),
    (re.compile(r"\b(?:dusk|sun\s*set|twilight|evening)\b", re.IGNORECASE), "evening"),
    (re.compile(r"\b(?:night|midnight|nighttime|moonlit|starlit)\b", re.IGNORECASE), "night"),
]

_WEATHER_PATTERNS = [
    (re.compile(r"\b(?:rain(?:ing|s)?|rainstorm|downpour|drizzl(?:e|ing))\b", re.IGNORECASE), "raining"),
    (re.compile(r"\b(?:snow(?:ing|s)?|snowfall|blizzard)\b", re.IGNORECASE), "snowing"),
    (re.compile(r"\b(?:storm(?:y|ing)?|thunder(?:storm)?|lightning)\b", re.IGNORECASE), "stormy"),
    (re.compile(r"\b(?:fog(?:gy)?|mist(?:y)?|haz(?:e|y))\b", re.IGNORECASE), "foggy"),
    (re.compile(r"\b(?:clear\s+sk(?:y|ies)|sunny|bright\s+sun)\b", re.IGNORECASE), "clear"),
    (re.compile(r"\b(?:cloud(?:y|s)|overcast)\b", re.IGNORECASE), "cloudy"),
    (re.compile(r"\b(?:wind(?:y)?|gust(?:s|y)?|breez(?:e|y))\b", re.IGNORECASE), "windy"),
]

_LOCATION_PATTERNS = [
    re.compile(r"\*[^*]*(?:enter(?:s|ed)?|arrive[sd]?\s+(?:at|in)|reach(?:es|ed)?)\s+(?:the\s+)?([^*]{3,60})\*", re.IGNORECASE),
    re.compile(r"(?:The\s+scene\s+(?:is\s+set|takes\s+place|shifts?)\s+(?:in|to|at)\s+)(?:the\s+)?(.{3,60}?)[\.\n]", re.IGNORECASE),
]


@dataclass
class SceneState:
    """Current scene / environment state."""

    location: str = ""
    time_of_day: str = ""
    weather: str = ""
    atmosphere: str = ""
    present_characters: list[str] = field(default_factory=list)
    active_events: list[str] = field(default_factory=list)
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "atmosphere": self.atmosphere,
            "present_characters": self.present_characters,
            "active_events": self.active_events,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneState:
        return cls(
            location=data.get("location", ""),
            time_of_day=data.get("time_of_day", ""),
            weather=data.get("weather", ""),
            atmosphere=data.get("atmosphere", ""),
            present_characters=data.get("present_characters", []),
            active_events=data.get("active_events", []),
            custom=data.get("custom", {}),
        )

    def apply_delta(self, delta: dict) -> SceneState:
        """Return a new SceneState with the delta applied."""
        new = SceneState.from_dict(self.to_dict())
        for key, value in delta.items():
            if hasattr(new, key):
                current = getattr(new, key)
                if isinstance(current, list) and isinstance(value, dict):
                    adds = value.get("add", [])
                    removes = value.get("remove", [])
                    new_list = [item for item in current if item not in removes] + adds
                    setattr(new, key, new_list)
                elif isinstance(current, dict) and isinstance(value, dict):
                    setattr(new, key, {**current, **value})
                else:
                    setattr(new, key, value)
        return new


@dataclass
class WorldDelta:
    """A change to the world state at a specific message."""

    message_index: int = 0
    delta: dict = field(default_factory=dict)
    branch_id: str = "main"


class WorldTracker:
    """Tracks world/scene state across messages using delta compression."""

    def __init__(self) -> None:
        self._deltas: list[WorldDelta] = []
        self._current_state = SceneState()

    @property
    def state(self) -> SceneState:
        return self._current_state

    @property
    def deltas(self) -> list[WorldDelta]:
        return list(self._deltas)

    def extract_world_changes(self, text: str) -> dict:
        """Extract world state changes from a message (heuristic, no LLM call)."""
        delta: dict = {}

        # Time of day
        for pattern, time_value in _TIME_PATTERNS:
            if pattern.search(text):
                if time_value != self._current_state.time_of_day:
                    delta["time_of_day"] = time_value
                break

        # Weather
        for pattern, weather_value in _WEATHER_PATTERNS:
            if pattern.search(text):
                if weather_value != self._current_state.weather:
                    delta["weather"] = weather_value
                break

        # Location changes
        for pattern in _LOCATION_PATTERNS:
            match = pattern.search(text)
            if match:
                location = match.group(1).strip().rstrip(".,;:")
                if location and location != self._current_state.location:
                    delta["location"] = location
                break

        return delta

    def apply_delta(self, delta: dict, message_index: int, branch_id: str = "main") -> WorldDelta:
        """Apply a world state delta and track it."""
        if not delta:
            return WorldDelta(message_index=message_index, branch_id=branch_id)

        world_delta = WorldDelta(
            message_index=message_index,
            delta=delta,
            branch_id=branch_id,
        )
        self._deltas.append(world_delta)
        self._current_state = self._current_state.apply_delta(delta)

        log.debug(
            "world_state_updated",
            message_index=message_index,
            delta_keys=list(delta.keys()),
        )

        return world_delta

    def reconstruct_at(self, message_index: int, branch_id: str = "main") -> SceneState:
        """Reconstruct world state at a specific message index by replaying deltas."""
        state = SceneState()
        for d in self._deltas:
            if d.branch_id == branch_id and d.message_index <= message_index:
                state = state.apply_delta(d.delta)
        return state

    def rollback_to(self, message_index: int, branch_id: str = "main") -> None:
        """Roll back world state to a specific message index."""
        self._deltas = [
            d for d in self._deltas
            if d.branch_id != branch_id or d.message_index <= message_index
        ]
        self._current_state = self.reconstruct_at(message_index, branch_id)
        log.info("world_state_rolled_back", to_index=message_index, branch=branch_id)

    def set_state(self, state: SceneState) -> None:
        """Set the current state directly (used when loading from DB)."""
        self._current_state = state

    def set_deltas(self, deltas: list[WorldDelta]) -> None:
        """Set deltas directly (used when loading from DB)."""
        self._deltas = deltas
