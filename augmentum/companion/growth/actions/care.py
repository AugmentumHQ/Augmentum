"""Care / maintenance — flag memory consolidation candidates.

Catalog category G (care / maintenance). See
``docs/superpowers/specs/2026-05-31-companion-action-catalog.md`` §G.

What this does (Phase 1 scope):

  1. Take ``ctx.target_ref`` as a query topic (or the empty string to
     scan recent memories broadly).
  2. Pull a candidate batch from memory recall.
  3. Cluster candidates by snippet-prefix overlap — a stand-in for
     embedding similarity. The action surfaces *candidates for
     consolidation*; it does NOT merge memories in place.
  4. Return a structured report of clusters (count + representative
     snippets). The growth log records this; Phase 2's verifier turns
     candidate clusters into actual merges with dry-run + rollback.

This action is intentionally read-only in Phase 1 — actually merging
memories is a Tier 2 action that needs preview + rollback before it
can land. Surfacing the candidates earns the substrate value (the
user can see Becca is paying attention to her own debt) without the
blast-radius risk.

Cost / tier: mana 4.0, tier 0 (read-only audit).

Reward signal (Phase 5 wires this): user approves a flagged cluster
for merge = +25 per cluster; ignored = 0; user marks a cluster as
wrong = -10 (calibrates the clustering heuristic).
"""

from __future__ import annotations

import time
from typing import Any

from augmentum.companion.growth.actions import (
    ActionContext,
    ActionResult,
    register,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# How wide to cast for candidates. Larger = more clusters found but
# also more noise. 50 is a working starting point — Phase 5 tunes.
_CANDIDATE_LIMIT = 50

# Minimum number of overlapping snippet-prefix words for two memories
# to be considered cluster-mates. The prefix-overlap heuristic is a
# coarse stand-in for embedding similarity; real cosine-distance
# clustering lands in Phase 2 alongside the actual merge primitive.
_MIN_OVERLAP_WORDS = 4

# How many tokens of the snippet prefix to compare. Long enough to
# distinguish "the cat sat on the mat" from "the cat caught a mouse",
# short enough to be cheap and dialect-tolerant.
_PREFIX_TOKENS = 12

# A cluster needs at least this many members to be worth surfacing —
# a 2-member cluster is usually just rephrasing, not duplication.
_MIN_CLUSTER_SIZE = 2


class FlagConsolidationCandidates:
    """G — Care / maintenance (Phase 1: read-only audit variant)."""

    action_type = "care_consolidate"
    mana_cost = 4.0
    tier = 0

    async def run(self, ctx: ActionContext) -> ActionResult:
        if ctx.memory_store is None:
            return ActionResult(
                ok=False,
                error="care_consolidate: memory_store not provided",
            )

        # Empty target_ref is allowed — scans the most-recently-recalled
        # set as a "what's in scope for consolidation right now" sweep.
        # Most callers will pass a topic so the audit is bounded.
        query = ctx.target_ref or "recent"
        try:
            hits = await ctx.memory_store.recall(
                query=query,
                user_id=ctx.user_id,
                limit=_CANDIDATE_LIMIT,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "growth.care.memory_failed",
                user_id=ctx.user_id, error=str(exc)[:200],
            )
            return ActionResult(
                ok=False, error=f"care_consolidate_failed: {exc!s}",
            )

        memories = [_unwrap(h) for h in (hits or [])]
        memories = [m for m in memories if m is not None]

        clusters = _cluster_by_prefix_overlap(memories)
        eligible = [c for c in clusters if len(c) >= _MIN_CLUSTER_SIZE]
        if not eligible:
            return ActionResult(
                ok=False,
                error=(
                    f"care_consolidate: no consolidation candidates "
                    f"({len(memories)} memories scanned)"
                ),
            )

        # Surface a digest, not the full memory bodies — the panel can
        # request the full member list per cluster on demand.
        cluster_digests = [
            {
                "size": len(cluster),
                "representative_snippet": _snippet(cluster[0])[:240],
                "member_ids": [_id(m) for m in cluster],
            }
            for cluster in eligible
        ]
        surface_event = {
            "topic": "growth.care.consolidation_candidates",
            "payload": {
                "scanned_count": len(memories),
                "cluster_count": len(eligible),
                "clusters": cluster_digests,
                "target_ref": ctx.target_ref,
                "rationale": ctx.rationale,
                "surfaced_at": int(time.time()),
            },
        }
        return ActionResult(
            ok=True,
            payload={
                "scanned_count": len(memories),
                "cluster_count": len(eligible),
            },
            surface_event=surface_event,
            ledger_delta={"consolidation_candidates_flagged": len(eligible)},
            continue_loop=False,
        )


# ── Clustering ───────────────────────────────────────────────────────


def _cluster_by_prefix_overlap(memories: list[Any]) -> list[list[Any]]:
    """Greedy single-pass clustering by snippet-prefix word overlap.

    O(n²) — fine for the 50-memory cap; Phase 2 swaps in embedding
    cosine similarity with a real clustering algorithm.
    """
    clusters: list[list[Any]] = []
    for memory in memories:
        prefix = _prefix_words(memory)
        if not prefix:
            continue
        placed = False
        for cluster in clusters:
            head_prefix = _prefix_words(cluster[0])
            if not head_prefix:
                continue
            overlap = len(prefix & head_prefix)
            if overlap >= _MIN_OVERLAP_WORDS:
                cluster.append(memory)
                placed = True
                break
        if not placed:
            clusters.append([memory])
    return clusters


def _prefix_words(memory: Any) -> set[str]:
    snippet = _snippet(memory)
    if not snippet:
        return set()
    words = [w.lower().strip(".,!?;:\"'`()[]{}") for w in snippet.split()]
    words = [w for w in words if w and len(w) > 2]
    return set(words[:_PREFIX_TOKENS])


def _snippet(memory: Any) -> str:
    for n in ("text", "content", "body", "snippet", "summary"):
        v = getattr(memory, n, None)
        if v is None and isinstance(memory, dict):
            v = memory.get(n)
        if v:
            return str(v)
    return ""


def _unwrap(hit: Any) -> Any | None:
    if isinstance(hit, tuple) and hit:
        return hit[0]
    return hit


def _id(memory: Any) -> str:
    for n in ("id", "memory_id", "uuid"):
        v = getattr(memory, n, None)
        if v is None and isinstance(memory, dict):
            v = memory.get(n)
        if v:
            return str(v)
    return ""


register(FlagConsolidationCandidates())
