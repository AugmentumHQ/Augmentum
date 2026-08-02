"""Memory data models."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class MemoryType(str, enum.Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    ENTITY = "entity"
    NARRATIVE = "narrative"
    ANALYSIS = "analysis"
    SKILL = "skill"
    RELATIONSHIP = "relationship"
    PROCEDURAL = "procedural"  # learned coding conventions/workflows (harness scope)


class Durability(str, enum.Enum):
    """LLM-emitted classification of how stable a fact is over time.

    - DURABLE: identity, relationships, allergies, durable preferences —
      things that are still true months or years from now.
    - TRANSIENT: in-progress work, current projects, today's mood —
      true now, likely false in weeks.
    - UNKNOWN: the model couldn't tell (also the default when missing).
      Treated as DURABLE-shaped for tier routing.

    Routed in MemoryStore: TRANSIENT facts land in PROVISIONAL with a
    longer-than-default TTL regardless of confidence; DURABLE / UNKNOWN
    follow the existing confidence-based tier choice. EXPLICIT-source
    facts ignore durability entirely (user phrasing = user intent).
    """
    DURABLE = "durable"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class SourceType(str, enum.Enum):
    EXTRACTED = "extracted"
    EXPLICIT = "explicit"
    USER_MANUAL = "user_manual"
    SYSTEM = "system"


class MemoryTier(str, enum.Enum):
    CORE = "core"            # Highest importance, always-in-context candidates
    ACTIVE = "active"        # Default tier, recalled via search
    ARCHIVE = "archive"      # Old/low-access, candidates for compaction
    PROVISIONAL = "provisional"  # Unvalidated, never injected, 7-day TTL


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    user_id: str
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    confidence: float = 0.8
    session_id: str | None = None
    embedding: list[float] | None = None
    valid_from: str = ""
    valid_until: str | None = None
    superseded_by: str | None = None
    source_type: SourceType | None = None
    source_context: str | None = None
    access_count: int = 0
    last_accessed: str | None = None
    created_at: str = ""
    updated_at: str = ""
    scope: str | None = None
    tier: str | MemoryTier = MemoryTier.ACTIVE
    last_compacted_at: str | None = None
    provisional_expires_at: str | None = None  # Phase 2: TTL for PROVISIONAL tier
    evidence: str = ""                         # Phase 1B: extraction evidence quote
    retrieval_count: int = 0                   # Phase 3: Hebbian retrieval frequency
    last_accessed_at: str | None = None        # Phase 3: recency for scoring
    source_memory_ids: str = "[]"              # Phase 4: JSON array of IDs that seeded a reflection


@dataclass
class ExtractedFact:
    """A fact extracted from conversation (pre-storage)."""

    content: str
    type: MemoryType = MemoryType.FACT
    importance: float = 0.5
    confidence: float = 0.8
    is_explicit: bool = False
    source_context: dict = field(default_factory=dict)
    evidence: str = ""
    # LLM-emitted lifecycle judgment. UNKNOWN is the safe default for
    # heuristic/regex extractions and any LLM output that omits the
    # field. See Durability for routing semantics.
    durability: Durability = Durability.UNKNOWN
