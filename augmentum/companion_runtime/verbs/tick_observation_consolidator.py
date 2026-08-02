"""tick_observation_consolidator — fifth Phase 3b management verb.

Subscribes to ``time.tick(5min)`` and runs the L0 maintenance pass
on the owner's ``bom_observations_exact`` rows: apply exponential
decay to ``decay_weight`` based on elapsed time since last_seen, then
prune rows that have fallen below the relevance floor.

Phase A of the broader observation substrate shipped L0; L1
(token-type abstractions) and L2 (logit fingerprints) are not yet
built. The "consolidator" name reflects the eventual promotion
intent. Today the verb keeps L0 honest — without it, decay_weight
only grows toward 1.0 (capped at observe time) and stale rows
accumulate indefinitely.

Cadence is 5min because the L0 ranking is queried by the lookup-
cache exporter on demand; staleness measured in hours is fine.
Cheap pass — one UPDATE + one DELETE bounded by the user's slice,
typically < 50ms even at 100k rows.

No-ops when there's no owner (multi-tenant cache export gating is a
deeper concern documented in the observation substrate spec).
"""

from __future__ import annotations

from augmentum.companion_runtime.event_bus import (
    CostEnvelope,
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.observation import consolidator
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_OBSERVATION_COOLDOWN_MS = 290_000  # 4 m 50 s

# The decay+prune pass is a full table walk per user. On a 100k-row L0
# set, the DELETE alone can take several seconds — the prune predicate
# (observation_count * decay_weight) is a computed expression with no
# straightforward index. 30s is the honest envelope for a 5min-cadence
# maintenance pass; below that the dispatcher legitimately catches
# budget_exceeded and we should fix the substrate, not the verb.
_OBSERVATION_COST = CostEnvelope(max_wallclock_ms=30_000, max_db_ops=20)


@verb(
    "time.tick(5min)",
    name="tick_observation_consolidator",
    reads=("bom_observations_exact",),
    writes=("bom_observations_exact",),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_OBSERVATION_COOLDOWN_MS,
    cost_envelope=_OBSERVATION_COST,
)
async def tick_observation_consolidator(event, ctx) -> None:
    """Decay-then-prune L0 observation rows for the bound owner."""
    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    backend = runtime.backend
    result = await consolidator.consolidate(backend.conn, user_id=owner)
    ctx.cite("bom_observations_exact", row_id=owner)
    ctx.db_ops += 2  # 1 UPDATE + 1 DELETE
    log.debug(
        "tick_observation_consolidator_ran",
        user_id=owner,
        decayed=result.decayed_rows,
        pruned=result.pruned_rows,
    )


VerbRegistry.register(tick_observation_consolidator)
