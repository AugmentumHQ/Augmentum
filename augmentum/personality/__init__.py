"""Personality facet substrate.

Hebbian cooccurrence graph over facet activations plus cross-table memory
associations. Implements the substrate for commitment #7 (companion-with-
owner is its own being) — drift in the relationship-specific personality
emerges from facet co-activation patterns specific to a given user.

Design rationale:
  - docs/superpowers/specs/2026-05-14-companion-design-commitments.md
  - Mischel & Shoda, Psychological Review (1995): CAPS — Cognitive-Affective
    Personality System. Personality as a network of facets whose if-then
    activation signatures stabilize per relationship.

Implementation mirrors `augmentum.memory.store` cooccurrence pattern
(migration 050). The cross-table `personality_memory_associations` is the
elegant integration point: when memory retrieval surfaces certain memories,
the facets historically associated with them are pulled into composition.

Public API:

    PersonalityStore        — schema-level CRUD + Hebbian ops
    SEED_FACETS, Facet,     — vocabulary
    FacetCategory,
    FacetActivation,
    FacetActivationSource

    compose_facet_affects   — pre-prompt: predict active facets for this turn
    update_after_response   — post-response: write activations + memory links

    build_labeler_messages  — pure prompt construction (no I/O)
    parse_labeler_response  — pure JSON parsing (graceful failure)
    label_response          — high-level: build → call → parse (caller-supplied LLM)

Runtime integration target: the CompanionRuntime persona-kernel digester
at `augmentum/companion_runtime/identity.py` (owned by a separate agent).
The digester calls `compose_facet_affects(...)` during persona composition;
the response pipeline calls `update_after_response(...)` post-generation,
typically wrapping `label_response(...)` to produce the input.
"""
from __future__ import annotations

from augmentum.personality.graph import (
    CONTRIBUTION_CEILING,
    COOCCURRENCE_WEIGHT,
    MEMORY_ASSOC_WEIGHT,
    RECENT_WEIGHT,
    compose_facet_affects,
    update_after_response,
)
from augmentum.personality.labeler import (
    build_labeler_messages,
    label_response,
    parse_labeler_response,
)
from augmentum.personality.models import (
    Facet,
    FacetActivation,
    FacetActivationSource,
    FacetCategory,
    FacetCooccurrence,
    MemoryFacetAssociation,
)
from augmentum.personality.store import (
    COOCCURRENCE_FLOOR,
    DECAY_FACTOR,
    PersonalityStore,
)
from augmentum.personality.vocabulary import SEED_FACETS

__all__ = [
    # constants
    "CONTRIBUTION_CEILING",
    "COOCCURRENCE_FLOOR",
    "COOCCURRENCE_WEIGHT",
    "DECAY_FACTOR",
    "MEMORY_ASSOC_WEIGHT",
    "RECENT_WEIGHT",
    # models
    "Facet",
    "FacetActivation",
    "FacetActivationSource",
    "FacetCategory",
    "FacetCooccurrence",
    "MemoryFacetAssociation",
    # store
    "PersonalityStore",
    "SEED_FACETS",
    # graph
    "compose_facet_affects",
    "update_after_response",
    # labeler
    "build_labeler_messages",
    "label_response",
    "parse_labeler_response",
]
