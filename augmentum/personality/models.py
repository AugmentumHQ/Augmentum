"""Personality facet dataclasses — pure data, no I/O."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FacetCategory(str, Enum):
    """Coarse grouping for facets. Used for filtered queries + UI grouping,
    not for runtime behavior — cooccurrence ignores category."""

    AFFECT = "affect"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    STANCE = "stance"
    ENERGY = "energy"


class FacetActivationSource(str, Enum):
    """How a facet activation was detected.

    - SELF_LABEL: model labeled its own response post-hoc (default path)
    - CLASSIFIER: dedicated small classifier (future, faster, narrower)
    - MANUAL: explicit user/test write (debug + test fixtures)
    """

    SELF_LABEL = "self_label"
    CLASSIFIER = "classifier"
    MANUAL = "manual"


@dataclass
class Facet:
    name: str
    description: str
    category: FacetCategory
    created_at: str = ""


@dataclass
class FacetActivation:
    id: int
    user_id: str
    companion_id: str
    facet: str
    intensity: float
    source: FacetActivationSource
    activated_at: str
    session_id: str | None = None
    turn_id: str | None = None


@dataclass
class FacetCooccurrence:
    user_id: str
    companion_id: str
    facet_a: str
    facet_b: str
    count: int
    last_updated: str


@dataclass
class MemoryFacetAssociation:
    user_id: str
    companion_id: str
    memory_id: str
    facet: str
    count: int
    last_updated: str
