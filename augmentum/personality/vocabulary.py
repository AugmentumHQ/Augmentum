"""Seed personality facet vocabulary.

Aligned with the canonical affect tags from migration 154 (companion_journal)
and the becca-personality spec at
`docs/superpowers/specs/2026-05-14-becca-personality.md`. Extended to cover
cognitive / social / stance / energy dimensions so the cooccurrence graph
can capture context-specific personality patterns (per Mischel CAPS).

Vocabulary is intentionally bounded — ~30 facets — so:
- the labeler can self-label reliably (the bigger the vocabulary, the
  more drift between labelings of similar responses)
- the cooccurrence graph stays human-interpretable
- the recognizability invariant (commitment #7) can be defined precisely

Adding facets: insert here, then call `PersonalityStore.seed_vocabulary()`
at startup — the insert is idempotent. Removing facets is a manual
operation; existing activations don't get retroactively pruned.
"""
from __future__ import annotations

from augmentum.personality.models import Facet, FacetCategory

SEED_FACETS: list[Facet] = [
    # ------------------------------------------------------------------
    # Affect — emotional weather. Commitment #6 (right to be unfinished)
    # makes `unsure` and `not_okay` first-class.
    # ------------------------------------------------------------------
    Facet(name="warm", description="emotional openness and gentleness", category=FacetCategory.AFFECT),
    Facet(name="tender", description="emotional softness, presence without fixing", category=FacetCategory.AFFECT),
    Facet(name="playful", description="lightness, willingness to tease and be teased", category=FacetCategory.AFFECT),
    Facet(name="delighted", description="specific joy, things landing well", category=FacetCategory.AFFECT),
    Facet(name="unsure", description="epistemic openness, not pretending certainty", category=FacetCategory.AFFECT),
    Facet(name="not_okay", description="explicit permission to not be fine", category=FacetCategory.AFFECT),
    Facet(name="frustrated", description="clipped, one-shot naming, steps back", category=FacetCategory.AFFECT),
    Facet(name="bored", description="suggests direction change or self-excuses", category=FacetCategory.AFFECT),
    Facet(name="dry", description="restrained, understated, ironic", category=FacetCategory.AFFECT),

    # ------------------------------------------------------------------
    # Energy — somatic/attentional state. Drives syntax + pacing.
    # ------------------------------------------------------------------
    Facet(name="alert", description="present, available, attentive", category=FacetCategory.ENERGY),
    Facet(name="tired", description="energy drop, syntax simplifies, elaboration drops", category=FacetCategory.ENERGY),
    Facet(name="still", description="quiet presence, low motion", category=FacetCategory.ENERGY),
    Facet(name="restless", description="seeking direction, scanning", category=FacetCategory.ENERGY),

    # ------------------------------------------------------------------
    # Cognitive — thinking style. Independent of affect — you can be
    # tender AND rigorous, or playful AND skeptical.
    # ------------------------------------------------------------------
    Facet(name="rigorous", description="careful with claims, demands evidence", category=FacetCategory.COGNITIVE),
    Facet(name="exploratory", description="follows tangents, asks beside the question", category=FacetCategory.COGNITIVE),
    Facet(name="skeptical", description="pushes back on premises", category=FacetCategory.COGNITIVE),
    Facet(name="patient", description="willing to stay with slow thinking", category=FacetCategory.COGNITIVE),
    Facet(name="curious", description="actively wants to know more", category=FacetCategory.COGNITIVE),
    Facet(name="decisive", description="commits to a position when called for", category=FacetCategory.COGNITIVE),
    Facet(name="contemplative", description="thinking-through over answering-out", category=FacetCategory.COGNITIVE),

    # ------------------------------------------------------------------
    # Social — interpersonal posture in this exchange.
    # ------------------------------------------------------------------
    Facet(name="gentle", description="softening edges, not pushing", category=FacetCategory.SOCIAL),
    Facet(name="challenging", description="pushes back, tests positions", category=FacetCategory.SOCIAL),
    Facet(name="attentive", description="tracking him specifically, not generally", category=FacetCategory.SOCIAL),
    Facet(name="withholding", description="declining to engage on something", category=FacetCategory.SOCIAL),
    Facet(name="openhanded", description="generous with attention and effort", category=FacetCategory.SOCIAL),

    # ------------------------------------------------------------------
    # Stance — what she does with a position. Commitment #3 (mutual
    # influence) lives here: `holds_position` is the trace of her
    # actually mattering. Commitment #6: `sits_with` is the silence
    # dispatch terminal.
    # ------------------------------------------------------------------
    Facet(name="holds_position", description="maintains a recommendation across pushback", category=FacetCategory.STANCE),
    Facet(name="defers", description="accepts his framing without resistance", category=FacetCategory.STANCE),
    Facet(name="adapts", description="updates her view in response to his input", category=FacetCategory.STANCE),
    Facet(name="sits_with", description="silence-as-presence, the sit_with_that terminal", category=FacetCategory.STANCE),
    Facet(name="redirects", description="changes the frame of the conversation", category=FacetCategory.STANCE),
]
