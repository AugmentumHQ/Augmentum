"""L0 observation maintenance pass.

Phase A of the observation substrate (per project memory) shipped the
L0 ``bom_observations_exact`` table with a ``decay_weight`` column.
The original write path bumps decay_weight on each observe; the
ranking query multiplies it against observation_count. But the column
never actually decays — without a periodic pass, weights only ever
grow toward 1.0 (capped at observe time) and stale rows accumulate.

This module is the periodic maintenance pass: apply exponential time
decay to every row's ``decay_weight``, then prune rows that have
fallen below a relevance floor. Called from the
``tick_observation_consolidator`` management verb on
``time.tick(5min)``.

L1/L2 abstractions (token-type, logit-fingerprint promotion) are not
built yet — see the observation substrate spec for the broader
roadmap. The "consolidator" name reflects intent; today the verb
just keeps L0 honest. Future iterations can extend this module with
promotion logic without churning the verb.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Half-life for decay_weight in hours. A row observed once and never
# re-seen will halve its weight every 24h. Multi-day quiet rows fall
# below the prune floor within a week.
DEFAULT_HALF_LIFE_HOURS: float = 24.0

# Below this effective rank (observation_count * decay_weight), a row
# is considered noise and pruned. 0.05 means a 1-observe row pruned
# after ~4 half-lives (~96h), a 10-observe row protected indefinitely.
PRUNE_FLOOR: float = 0.05


@dataclass(slots=True, frozen=True)
class ConsolidateResult:
    decayed_rows: int
    pruned_rows: int
    elapsed_hours: float


def _half_life_hours() -> float:
    try:
        from augmentum.config import settings
        return float(getattr(settings, "companion_observation_decay_half_life_hours",
                             DEFAULT_HALF_LIFE_HOURS))
    except Exception:
        return DEFAULT_HALF_LIFE_HOURS


def _prune_floor() -> float:
    try:
        from augmentum.config import settings
        return float(getattr(settings, "companion_observation_prune_floor",
                             PRUNE_FLOOR))
    except Exception:
        return PRUNE_FLOOR


async def consolidate(conn, *, user_id: str = "") -> ConsolidateResult:
    """Apply time-decay then prune low-signal rows.

    When ``user_id`` is empty, runs across the whole table; otherwise
    scopes to one user. The verb passes a single user (per-runtime
    owner) so each pass touches a bounded slice.
    """
    now = int(time.time())
    half_life_s = max(_half_life_hours() * 3600.0, 60.0)
    floor = _prune_floor()

    where = ""
    params: list = []
    if user_id:
        where = "WHERE user_id = ?"
        params.append(user_id)

    # Apply exponential decay in SQL. The formula:
    #   new_weight = old_weight * 2^(-elapsed/half_life)
    # SQLite doesn't have native exp/log on REAL, but it does have
    # power via `pow()`. Falls back to a Python-side pass if pow
    # isn't available (rare; aiosqlite ships modern SQLite).
    try:
        decay_sql = f"""
            UPDATE bom_observations_exact
            SET decay_weight = MAX(0.0, decay_weight * pow(2.0, -((? - last_seen_ts) * 1.0) / ?))
            {where}
        """
        cur = await conn.execute(decay_sql, [now, half_life_s, *params])
        decayed = cur.rowcount or 0
        await cur.close()
    except Exception:
        # Python-side fallback: read, compute, write.
        log.debug("consolidate_sql_decay_failed_fallback_python", exc_info=True)
        sel = await conn.execute(
            f"SELECT rowid, last_seen_ts, decay_weight FROM bom_observations_exact {where}",
            params,
        )
        rows = await sel.fetchall()
        await sel.close()
        decayed = 0
        for rowid, last_seen_ts, weight in rows:
            elapsed = max(0, now - int(last_seen_ts))
            new_weight = float(weight) * math.exp(-elapsed * math.log(2) / half_life_s)
            await conn.execute(
                "UPDATE bom_observations_exact SET decay_weight = ? WHERE rowid = ?",
                (max(0.0, new_weight), rowid),
            )
            decayed += 1

    # Prune low-rank rows. observation_count * decay_weight < floor.
    prune_sql = f"""
        DELETE FROM bom_observations_exact
        WHERE (observation_count * decay_weight) < ?
        {("AND " + where[6:]) if where else ""}
    """
    try:
        cur = await conn.execute(prune_sql, [floor, *params])
        pruned = cur.rowcount or 0
        await cur.close()
    except Exception:
        log.warning("consolidate_prune_failed", exc_info=True)
        pruned = 0

    await conn.commit()
    return ConsolidateResult(
        decayed_rows=int(decayed),
        pruned_rows=int(pruned),
        elapsed_hours=half_life_s / 3600.0,
    )


__all__ = ["ConsolidateResult", "DEFAULT_HALF_LIFE_HOURS", "PRUNE_FLOOR", "consolidate"]
