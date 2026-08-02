"""Cross-table spreading-activation query + high-level integration entry points.

This is the layer the CompanionRuntime persona-kernel digester is expected
to call. Two entry points:

  compose_facet_affects(...)   — pre-prompt: predict which facets should be
                                  emphasized in this turn's composition
  update_after_response(...)   — post-prompt: write activations and link
                                  them to retrieved memory IDs

Mechanism: spreading activation. Recent activations seed the active-facet
set; cooccurrence graph spreads to historically-co-firing facets; memory
associations contribute facets linked to retrieved memories. All three
sources are merged and rank-weighted.

The graph itself is `PersonalityStore`'s tables — this module is a thin
composition layer over it. Pure I/O orchestration; no LLM calls.
"""
from __future__ import annotations

from augmentum.personality.models import FacetActivationSource
from augmentum.personality.store import (
    COOCCURRENCE_FLOOR,
    PersonalityStore,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Weights for the three activation sources when merging. Tuned so that:
# - recent activations dominate (they're direct observations)
# - cooccurrence spreads broadly but at lower confidence
# - memory associations are context-specific so they're weighted between
RECENT_WEIGHT = 1.0
COOCCURRENCE_WEIGHT = 0.5
MEMORY_ASSOC_WEIGHT = 0.7

# Cap how much a single highly-co-firing pair can contribute to a facet's
# composition score, expressed as `count / COOCCURRENCE_FLOOR` capped at
# this value. Prevents one runaway association from dominating composition.
# Distinct from MemoryStore's multiplicative boost cap — this is an additive
# contribution ceiling in the spreading-activation merge.
CONTRIBUTION_CEILING = 1.3


async def compose_facet_affects(
    store: PersonalityStore,
    *,
    user_id: str,
    companion_id: str,
    recent_hours: int = 24,
    retrieved_memory_ids: list[str] | None = None,
    limit: int = 8,
) -> dict[str, float]:
    """Predict which facets should be active in this composition.

    Strategy:
      1. Get recent facet activations (last `recent_hours`)
      2. Spread to cooccurrence graph (facets that historically co-fire
         with the recent ones)
      3. Spread to memory associations (facets linked to memories
         retrieved this turn)
      4. Merge weighted scores, normalize to [0, 1], return top-`limit`
         as `{facet: score}`

    Returns empty dict if user_id/companion_id missing or no signal.
    Score values are meant for prompt-composition weighting; the runtime
    decides how to translate them into prompt text.
    """
    if not user_id or not companion_id:
        return {}

    raw_scores: dict[str, float] = {}

    # 1. Recent activations (direct evidence)
    recent = await store.query_recent_activations(
        user_id=user_id,
        companion_id=companion_id,
        recent_hours=recent_hours,
        limit=200,  # over-fetch so aggregation is meaningful
    )
    recent_facets: set[str] = set()
    for activation in recent:
        raw_scores[activation.facet] = (
            raw_scores.get(activation.facet, 0.0)
            + activation.intensity * RECENT_WEIGHT
        )
        recent_facets.add(activation.facet)

    # 2. Spread to cooccurrence graph
    if recent_facets:
        spread = await store.query_cooccurrent_facets(
            list(recent_facets),
            user_id=user_id,
            companion_id=companion_id,
            limit=limit * 2,
        )
        for facet, count in spread:
            # Normalize by COOCCURRENCE_FLOOR so the weakest qualifying
            # association contributes ~1.0 * weight; stronger associations
            # contribute proportionally more, capped at CONTRIBUTION_CEILING
            # to keep ranking sane.
            normalized = min(count / max(COOCCURRENCE_FLOOR, 1), CONTRIBUTION_CEILING)
            raw_scores[facet] = (
                raw_scores.get(facet, 0.0)
                + normalized * COOCCURRENCE_WEIGHT
            )

    # 3. Spread to memory associations (the cross-table contribution)
    if retrieved_memory_ids:
        mem_assoc = await store.query_facets_for_memories(
            retrieved_memory_ids,
            user_id=user_id,
            companion_id=companion_id,
            limit=limit * 2,
        )
        for facet, count in mem_assoc:
            normalized = min(count / max(COOCCURRENCE_FLOOR, 1), CONTRIBUTION_CEILING)
            raw_scores[facet] = (
                raw_scores.get(facet, 0.0)
                + normalized * MEMORY_ASSOC_WEIGHT
            )

    if not raw_scores:
        return {}

    # 4. Normalize to [0, 1] and take top-limit
    max_score = max(raw_scores.values())
    if max_score <= 0:
        return {}
    normalized_scores = {
        facet: round(score / max_score, 4)
        for facet, score in raw_scores.items()
    }
    top = sorted(
        normalized_scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:limit]
    return dict(top)


async def update_after_response(
    store: PersonalityStore,
    labeled_facets: list[tuple[str, float]],
    *,
    user_id: str,
    companion_id: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    retrieved_memory_ids: list[str] | None = None,
    source: FacetActivationSource = FacetActivationSource.SELF_LABEL,
) -> int:
    """Write activations and cross-table memory↔facet links in one batch.

    This is the post-response wiring path. The runtime calls this after
    each generation (typically via the labeler module's `label_response`
    feeding its output here).

    Returns the number of activation rows written. Memory associations
    are written silently — empty `retrieved_memory_ids` skips that step.

    Empty `labeled_facets` returns 0 with no write. Required: user_id
    and companion_id (both raise ValueError if empty, matching the
    store-layer convention).
    """
    if not labeled_facets:
        return 0
    activation_ids = await store.record_activations(
        labeled_facets,
        user_id=user_id,
        companion_id=companion_id,
        session_id=session_id,
        turn_id=turn_id,
        source=source,
    )
    if retrieved_memory_ids and activation_ids:
        facet_names = [name for name, _ in labeled_facets]
        await store.record_memory_associations(
            retrieved_memory_ids,
            facet_names,
            user_id=user_id,
            companion_id=companion_id,
        )
    return len(activation_ids)
