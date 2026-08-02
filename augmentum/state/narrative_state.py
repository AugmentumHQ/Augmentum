"""Narrative state data models — in-memory representations of session state."""

from __future__ import annotations

import enum
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentum.modes.narrative.memory_settings import SessionMemorySettings

# --- Enums ---


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    ITEM = "item"
    FACTION = "faction"


class PlotStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"
    PAUSED = "paused"


class ContradictionSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class LorebookPosition(str, Enum):
    BEFORE_CHAR = "before_char"
    AFTER_CHAR = "after_char"
    AT_DEPTH = "at_depth"
    EM_TOP = "em_top"
    EM_BOTTOM = "em_bottom"
    OUTLET = "outlet"


class SelectiveLogic(enum.IntEnum):
    """Logic for secondary keyword matching (matches SillyTavern)."""

    AND_ANY = 0   # Primary matches AND any secondary matches
    NOT_ALL = 1   # Primary matches AND NOT all secondaries match
    NOT_ANY = 2   # Primary matches AND no secondary matches
    AND_ALL = 3   # Primary matches AND all secondaries match


# --- Data models ---


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def content_hash(text: str) -> str:
    """SHA256 hash of message content for branch detection."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


@dataclass
class TrackedMessage:
    """A message in the session DAG."""

    id: int | None = None
    session_id: str = ""
    parent_id: int | None = None
    role: str = ""
    content: str = ""
    content_hash: str = ""
    message_index: int = 0
    branch_id: str = "main"
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash and self.content:
            self.content_hash = content_hash(self.content)


@dataclass
class Fact:
    """An established truth within the narrative."""

    id: str = field(default_factory=_new_id)
    session_id: str = ""
    content: str = ""
    source: str = "extracted"
    confidence: float = 0.8
    domain: str = "general"
    established_at: int = 0
    superseded_by: str | None = None
    branch_id: str = "main"
    tags: list[str] = field(default_factory=list)


@dataclass
class EntityState:
    """Current state of a tracked entity."""

    location: str = ""
    emotional_state: str = ""
    physical_state: str = ""
    inventory: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "emotional_state": self.emotional_state,
            "physical_state": self.physical_state,
            "inventory": self.inventory,
            "relationships": self.relationships,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EntityState:
        return cls(
            location=data.get("location", ""),
            emotional_state=data.get("emotional_state", ""),
            physical_state=data.get("physical_state", ""),
            inventory=data.get("inventory", []),
            relationships=data.get("relationships", {}),
            custom=data.get("custom", {}),
        )

    def apply_delta(self, delta: dict) -> EntityState:
        """Return a new state with the delta applied."""
        new_data = self.to_dict()
        for key, value in delta.items():
            if key in new_data:
                if isinstance(new_data[key], dict) and isinstance(value, dict):
                    new_data[key] = {**new_data[key], **value}
                elif isinstance(new_data[key], list) and isinstance(value, dict):
                    # Handle list operations: {"add": [...], "remove": [...]}
                    adds = value.get("add", [])
                    removes = value.get("remove", [])
                    new_data[key] = [
                        item for item in new_data[key] if item not in removes
                    ] + adds
                else:
                    new_data[key] = value
        return EntityState.from_dict(new_data)


@dataclass
class Entity:
    """A tracked entity (character, location, item, faction)."""

    id: str = field(default_factory=_new_id)
    session_id: str = ""
    entity_type: EntityType = EntityType.CHARACTER
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    state: EntityState = field(default_factory=EntityState)
    branch_id: str = "main"

    def to_db_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "aliases": json.dumps(self.aliases),
            "state": json.dumps(self.state.to_dict()),
            "branch_id": self.branch_id,
        }


@dataclass
class StateDelta:
    """A change to an entity's state at a specific message."""

    entity_id: str = ""
    message_index: int = 0
    delta: dict = field(default_factory=dict)
    branch_id: str = "main"


@dataclass
class PlotThread:
    """A narrative arc or plot thread."""

    id: str = field(default_factory=_new_id)
    session_id: str = ""
    title: str = ""
    description: str = ""
    status: PlotStatus = PlotStatus.ACTIVE
    established_at: int = 0
    resolved_at: int | None = None
    branch_id: str = "main"
    state: dict = field(default_factory=dict)


@dataclass
class Contradiction:
    """A detected inconsistency in the narrative."""

    session_id: str = ""
    message_index: int = 0
    contradiction_type: str = ""
    description: str = ""
    severity: ContradictionSeverity = ContradictionSeverity.MINOR
    resolution: str | None = None
    fact_ids: list[str] = field(default_factory=list)
    branch_id: str = "main"


@dataclass
class LorebookEntry:
    """A world info / lorebook entry for context injection."""

    id: str = field(default_factory=_new_id)
    session_id: str = ""
    keywords: list[str] = field(default_factory=list)
    content: str = ""
    priority: int = 100
    source: str = "character_book"
    enabled: bool = True
    constant: bool = False
    position: LorebookPosition = LorebookPosition.BEFORE_CHAR
    scan_depth: int = 5
    case_sensitive: bool = False
    sticky_turns: int = 0
    cooldown_turns: int = 0
    last_triggered_at: int | None = None
    trigger_count: int = 0

    # Secondary keywords
    secondary_keywords: list[str] = field(default_factory=list)
    selective: bool = True
    selective_logic: SelectiveLogic = SelectiveLogic.AND_ANY

    # Inclusion groups
    group: str = ""
    group_override: bool = False
    group_weight: int = 100

    # Probability
    probability: int = 100
    use_probability: bool = True

    # Budget
    ignore_budget: bool = False

    # Matching options (None = inherit global setting)
    match_whole_words: bool | None = None
    use_group_scoring: bool | None = None

    # Recursion control
    exclude_recursion: bool = False
    prevent_recursion: bool = False
    delay_until_recursion: int = 0

    # Scanning scope flags
    match_persona: bool = False
    match_char_description: bool = False
    match_char_personality: bool = False
    match_scenario: bool = False
    match_creator_notes: bool = False

    # Timed effects
    delay_turns: int = 0

    # Outlet
    outlet_name: str = ""

    # Comment/memo
    comment: str = ""

    # At-depth injection controls (only meaningful when position == AT_DEPTH).
    # Depth 0 = appended after the newest message; N = N turns back from the end.
    # Same-depth same-role entries are joined with "\n" into a single message.
    injection_depth: int = 4
    injection_role: str = "system"  # "system" | "user" | "assistant"

    # Branch tag (migration 304) — which narrative branch this entry was
    # authored on. Mirrors facts/entities. Character-book imports and any
    # pre-migration rows default to "main" (globally visible). Model-created
    # entries (source="narrative_established") carry the branch they were
    # established on so branch retrieval can scope them correctly.
    branch_id: str = "main"


@dataclass
class NarrativeSessionState:
    """Complete in-memory state for a narrative session."""

    session_id: str = ""
    branch_id: str = "main"
    message_count: int = 0
    character_card_name: str = ""
    entities: dict[str, Entity] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    plot_threads: list[PlotThread] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    lorebook: list[LorebookEntry] = field(default_factory=list)
    # LoreEngine timed-effect counters (sticky/cooldown/delay/turn_count) —
    # snapshot so long chats don't re-arm timers after a server bounce.
    lorebook_runtime_state: dict = field(default_factory=dict)
    scene_context: dict = field(default_factory=dict)

    # Long-term memory
    memory_summary: str = ""
    card_type: str = "character"
    last_summary_at: int = 0

    # Overflow persistence
    overflow_summaries: list[str] = field(default_factory=list)
    archived_messages: list[str] = field(default_factory=list)

    # Three-layer memory (STATE + MEMORY + ARCHIVE)
    state_snapshot_data: dict = field(default_factory=dict)
    memory_ledger_data: list[dict] = field(default_factory=list)

    # Relationship tracker data (serialized as list of dicts)
    relationships: list[dict] = field(default_factory=list)

    # Per-branch state cache for branch swap restore
    branch_states_data: dict[str, dict] = field(default_factory=dict)

    # Message history for summary refresh after restart
    message_history_data: list[str] = field(default_factory=list)

    # Compaction flag (resume interrupted compaction)
    needs_compaction: bool = False

    # Request logs (context viewer history, persisted for inspector)
    request_logs: list[dict] = field(default_factory=list)
    last_request_log: dict = field(default_factory=dict)  # backward compat

    # Group chat state
    group_id: str = ""
    group_speaker_index: int = 0

    # World-system tracker state (migration 312; values/history/locks —
    # see modes/narrative/world_system.py). The manifest itself is not
    # stored: it re-parses from the character card on load.
    world_state: dict = field(default_factory=dict)

    # Per-session LTM settings (None = use global defaults)
    memory_settings: SessionMemorySettings | None = None

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Find entity by name or alias (case-insensitive)."""
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                return entity
            if any(a.lower() == name_lower for a in entity.aliases):
                return entity
        return None

    def get_active_characters(self) -> list[Entity]:
        """Get all character entities."""
        return [
            e for e in self.entities.values()
            if e.entity_type == EntityType.CHARACTER
        ]

    def get_active_plots(self) -> list[PlotThread]:
        """Get active plot threads."""
        return [p for p in self.plot_threads if p.status == PlotStatus.ACTIVE]

    def get_recent_facts(self, limit: int = 20) -> list[Fact]:
        """Get the most recently established facts."""
        active = [f for f in self.facts if f.superseded_by is None]
        return sorted(active, key=lambda f: f.established_at, reverse=True)[:limit]
