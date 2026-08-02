"""Companion energy axis — per-(user, companion) "how depleted am I."

Companion verbs architecture, Phase 3b. A minimal substrate that
``tick_energy`` maintains: one scalar ``energy_level`` in [0, 1]
that exponentially decays toward a per-row ``baseline_level`` and
gets spent when activity_selector picks a non-trivial action.

Distinct from:

* **Drives** (``drives.py``) — drives are appetites (curiosity,
  competence, connection, rest); energy is the capacity-to-act that
  gates them.
* **Economy** (``companion_economy``) — economy is earned berry/mana
  motivation; energy is physiological regeneration.

Shape mirrors ``drives.py`` so the verb is structurally analogous
to ``tick_drive``. The decay function is the verb's only writer in
Phase 3b; ``spend`` is provided as the seam for a future
``apply_signal(energy)`` verb that wires activity_selector to the
substrate. Until that lands, the level just decays toward baseline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Floor / ceiling — energy never reaches 0 or 1 (would freeze the
# system in either direction). Same epsilon-slack pattern as drives.
LEVEL_FLOOR: float = 0.05
LEVEL_CEILING: float = 0.95

# Default baseline new rows are seeded with — at rest, she sits at
# 60% energy and exponentially decays toward that floor from either
# direction.
DEFAULT_BASELINE: float = 0.6

# Half-life for decay toward baseline. After this many hours, the
# distance from baseline has halved. Tuneable via
# ``companion_energy_decay_half_life_hours``.
DEFAULT_HALF_LIFE_HOURS: float = 6.0

# Default amount spent per non-trivial activity (when the future
# apply_signal(energy) verb lands). Exposed for tests + that verb.
SPEND_AMOUNT: float = 0.10


@dataclass(slots=True)
class EnergyState:
    """Per-(user, companion) energy snapshot."""
    user_id: str
    companion_id: str
    level: float = DEFAULT_BASELINE
    baseline: float = DEFAULT_BASELINE
    last_decay_at: str | None = None
    last_spend_at: str | None = None

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "baseline": self.baseline,
            "last_decay_at": self.last_decay_at,
            "last_spend_at": self.last_spend_at,
        }


def _clamp(v: float) -> float:
    return max(LEVEL_FLOOR, min(LEVEL_CEILING, v))


def _half_life_hours() -> float:
    """Configurable half-life from settings."""
    try:
        from augmentum.config import settings
        return float(getattr(settings, "companion_energy_decay_half_life_hours",
                             DEFAULT_HALF_LIFE_HOURS))
    except Exception:
        return DEFAULT_HALF_LIFE_HOURS


async def load(runtime: CompanionRuntime, *, user_id: str) -> EnergyState:
    """Read EnergyState from the DB. Provisions defaults on first load."""
    user_id = user_id or ""
    backend = runtime.backend
    try:
        cur = await backend.conn.execute(
            "SELECT energy_level, baseline_level, last_decay_at, last_spend_at "
            "FROM companion_energy_state "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("energy_load_failed", user_id=user_id, exc_info=True)
        return EnergyState(
            user_id=user_id, companion_id=runtime.companion_id,
        )

    if row is None:
        try:
            await backend.conn.execute(
                "INSERT OR IGNORE INTO companion_energy_state "
                "(user_id, companion_id) VALUES (?, ?)",
                (user_id, runtime.companion_id),
            )
            await backend.conn.commit()
        except Exception:
            log.warning("energy_state_seed_failed", exc_info=True)
        return EnergyState(
            user_id=user_id, companion_id=runtime.companion_id,
        )

    return EnergyState(
        user_id=user_id,
        companion_id=runtime.companion_id,
        level=float(row[0] or DEFAULT_BASELINE),
        baseline=float(row[1] or DEFAULT_BASELINE),
        last_decay_at=row[2],
        last_spend_at=row[3],
    )


async def decay(runtime: CompanionRuntime, *, user_id: str) -> EnergyState:
    """Apply exponential decay toward baseline based on elapsed time
    since ``last_decay_at``. Idempotent on short intervals — early
    returns when less than a few seconds have passed.
    """
    state = await load(runtime, user_id=user_id)
    if state.last_decay_at is None:
        return state

    try:
        last_t = datetime.strptime(
            str(state.last_decay_at).replace("T", " ").split(".", 1)[0],
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=UTC).timestamp()
    except (ValueError, AttributeError):
        return state

    elapsed_hours = max(0.0, (time.time() - last_t) / 3600.0)
    if elapsed_hours < 0.001:
        return state

    half_life = max(_half_life_hours(), 0.1)
    decay_fraction = 1.0 - math.exp(-elapsed_hours * math.log(2) / half_life)
    new_level = _clamp(state.level + (state.baseline - state.level) * decay_fraction)

    try:
        await runtime.backend.conn.execute(
            "UPDATE companion_energy_state "
            "SET energy_level = ?, last_decay_at = datetime('now') "
            "WHERE user_id = ? AND companion_id = ?",
            (new_level, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("energy_decay_persist_failed", user_id=user_id, exc_info=True)
        return state

    state.level = new_level
    return state


async def spend(
    runtime: CompanionRuntime, *, user_id: str,
    amount: float = SPEND_AMOUNT,
) -> None:
    """Reduce the energy level by ``amount``. Bounded by LEVEL_FLOOR.

    Provided as the seam for a future apply_signal(energy) verb wiring
    activity_selector to this substrate. Not yet called in Phase 3b.
    """
    state = await load(runtime, user_id=user_id)
    new_level = _clamp(state.level - amount)
    try:
        await runtime.backend.conn.execute(
            "UPDATE companion_energy_state "
            "SET energy_level = ?, last_spend_at = datetime('now') "
            "WHERE user_id = ? AND companion_id = ?",
            (new_level, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("energy_spend_persist_failed", user_id=user_id, exc_info=True)


__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_HALF_LIFE_HOURS",
    "EnergyState",
    "LEVEL_CEILING",
    "LEVEL_FLOOR",
    "SPEND_AMOUNT",
    "decay",
    "load",
    "spend",
]
