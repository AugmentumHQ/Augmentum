"""DPO-style preference retrieval.

Sprint 4b. At dispatch time, for each candidate subagent, we ask:
"in past decisions whose context was similar to this one, did
choosing this subagent end well?" The answer becomes a preference
delta added to the heuristic utility from Sprint 3.

This is *not* fine-tuning. No model weights change. We're using
retrieval over past outcomes to influence ranking, which preserves
server-resident portability and zero training infrastructure (sprint
plan §10, decision: "DPO-style retrieval, not fine-tuning").

Maths:
- Pull k=8 nearest archive rows (by intent embedding) — see
  :func:`skill_archive.nearest`.
- For each candidate subagent, compute:
    pref_delta = sum(outcome_signal_i * similarity_i for rows where
                     chosen_subagent == candidate)
               / max(1, count_for_candidate)
- Scale by a small weight (default 0.05 — feature is one of many)
  and add to the candidate's heuristic utility.
- Flag-gated by ``companion_skill_archive_enabled``. Off → returns
  zeros so dispatch behaves exactly as Sprint 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent

log = get_logger(__name__)


PREFERENCE_WEIGHT: float = 0.05   # one feature among dispatch's ~9
K_NEAREST: int = 8


async def preference_delta(
    runtime: CompanionRuntime,
    intent: Intent,
    candidate_names: list[str],
) -> dict[str, float]:
    """Return a per-candidate preference delta in the same units as
    dispatch utility. Empty dict (= no deltas) when the flag is off
    or there's no data to learn from.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_skill_archive_enabled", False):
        return {}

    from augmentum.companion_runtime import skill_archive
    try:
        import asyncio
        from augmentum.memory.embeddings import EmbeddingService
        query_embedding = await asyncio.to_thread(
            EmbeddingService.embed_one, intent.text,
        )
    except Exception:
        return {}
    if not query_embedding:
        return {}

    rows = await skill_archive.nearest(
        runtime, query_embedding, k=K_NEAREST, user_id=intent.user_id,
    )
    if not rows:
        return {}

    # Compute mean outcome per candidate seen in the k-NN window.
    # Similarity is implicit in the order — we already filtered to
    # the closest rows, so treat all of them with weight 1.
    sums: dict[str, float] = {n: 0.0 for n in candidate_names}
    counts: dict[str, int] = {n: 0 for n in candidate_names}
    for r in rows:
        if r.chosen_subagent in sums:
            sums[r.chosen_subagent] += r.outcome_signal
            counts[r.chosen_subagent] += 1

    deltas: dict[str, float] = {}
    for name in candidate_names:
        c = counts.get(name, 0)
        if c == 0:
            continue
        deltas[name] = (sums[name] / c) * PREFERENCE_WEIGHT
    return deltas


__all__ = ["K_NEAREST", "PREFERENCE_WEIGHT", "preference_delta"]
