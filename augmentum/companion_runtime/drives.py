"""Drives — appetite-shaped motivators that satiate and decay.

Sprint 6, Aletheia × Augmentum arc Piece 4.

Four drives, each [0, 1]:

  curiosity   — appetite for novelty / open questions
  competence  — appetite for finishing things / mastery
  connection  — appetite for user presence / shared attention
  rest        — appetite for low-activity periods

Each drive has a current ``level`` and a ``last_satiated_at``. When
the matching activity performs (revisit_thread satisfies curiosity,
reach_out satisfies connection, etc.) the drive is satiated — its
level drops. Between satiations, levels decay back toward a baseline
at the configured half-life (default 4h).

Drive urgency at scoring time:

    urgency = level × (1 - predicted_satiation_from_last_action)

Connection drive overrides on user interaction: when the user is
co-present (role_channel returns ``companion`` role), connection
urgency is forced high regardless of stored level. This is what makes
her yield to the user — present, not just polite.

Persistence: one row per (user, companion) in ``companion_drive_state``
(migration 184). All reads/writes are scoped to the calling user.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Drive names — fixed set (changing requires migration + code change).
CURIOSITY = "curiosity"
COMPETENCE = "competence"
CONNECTION = "connection"
REST = "rest"

DRIVE_NAMES = (CURIOSITY, COMPETENCE, CONNECTION, REST)

# Default levels at provision time. Per Becca's personality doc §13
# (settled-curious baseline): curiosity + connection slightly above
# midline; rest below midline (she's "leaned back but paying attention").
DEFAULT_LEVELS: dict[str, float] = {
    CURIOSITY: 0.6,
    COMPETENCE: 0.5,
    CONNECTION: 0.6,
    REST: 0.4,
}

# Half-life for decay back to baseline. After this many hours, a drive's
# elevation has decayed by half. Tuning: companion_drive_decay_half_life_hours.
DEFAULT_HALF_LIFE_HOURS: float = 4.0

# Floor / ceiling — drives never reach exactly 0 or 1 (would freeze
# the system). Soft bounds with epsilon slack.
LEVEL_FLOOR: float = 0.05
LEVEL_CEILING: float = 0.95

# Satiation amount when a matching activity performs.
SATIATION_AMOUNT: float = 0.30


@dataclass(slots=True)
class DriveState:
    """Per-user-companion drive snapshot."""
    user_id: str
    companion_id: str
    levels: dict[str, float] = field(default_factory=dict)
    satiated_at: dict[str, str | None] = field(default_factory=dict)
    last_decay_at: str | None = None

    def urgency(self, name: str) -> float:
        """Current urgency for a named drive — what the activity_selector
        multiplies candidate scores by."""
        level = float(self.levels.get(name, DEFAULT_LEVELS.get(name, 0.5)))
        # Recency penalty: a drive satiated in the last 30s shouldn't fire
        # again immediately; it's a soft cooldown.
        last = self.satiated_at.get(name)
        recency_dampening = 0.0
        if last:
            try:
                t = datetime.strptime(
                    str(last).replace("T", " ").split(".", 1)[0],
                    "%Y-%m-%d %H:%M:%S",
                ).replace(tzinfo=UTC).timestamp()
                age_s = max(0.0, time.time() - t)
                # Within 30s, dampening up to 0.5; falls off exponentially.
                recency_dampening = 0.5 * math.exp(-age_s / 30.0)
            except (ValueError, AttributeError):
                pass
        return max(0.0, level * (1.0 - recency_dampening))

    def as_dict(self) -> dict:
        return {
            "levels": dict(self.levels),
            "satiated_at": dict(self.satiated_at),
            "urgency": {n: self.urgency(n) for n in DRIVE_NAMES},
        }

    def dominant(self) -> str:
        """The drive with the highest urgency. Tie-break by name."""
        return max(DRIVE_NAMES, key=lambda n: (self.urgency(n), n))


async def load(runtime: CompanionRuntime, *, user_id: str) -> DriveState:
    """Read DriveState from the DB. Provisions defaults on first load.

    Per-user scoping is the load-bearing invariant: each user's drive
    levels are isolated. Calling without a user_id falls back to the
    legacy seed (user_id='') for backward compat.
    """
    user_id = user_id or ""
    backend = runtime.backend
    try:
        cur = await backend.conn.execute(
            "SELECT curiosity_level, competence_level, connection_level, rest_level, "
            "       curiosity_satiated_at, competence_satiated_at, "
            "       connection_satiated_at, rest_satiated_at, "
            "       last_decay_at "
            "FROM companion_drive_state "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("drives_load_failed", user_id=user_id, exc_info=True)
        return DriveState(
            user_id=user_id, companion_id=runtime.companion_id,
            levels=dict(DEFAULT_LEVELS),
        )

    if row is None:
        # Provision defaults
        try:
            await backend.conn.execute(
                "INSERT OR IGNORE INTO companion_drive_state "
                "(user_id, companion_id) VALUES (?, ?)",
                (user_id, runtime.companion_id),
            )
            await backend.conn.commit()
        except Exception:
            log.warning("drive_state_seed_failed", exc_info=True)
        return DriveState(
            user_id=user_id, companion_id=runtime.companion_id,
            levels=dict(DEFAULT_LEVELS),
        )

    return DriveState(
        user_id=user_id,
        companion_id=runtime.companion_id,
        levels={
            CURIOSITY: float(row[0] or DEFAULT_LEVELS[CURIOSITY]),
            COMPETENCE: float(row[1] or DEFAULT_LEVELS[COMPETENCE]),
            CONNECTION: float(row[2] or DEFAULT_LEVELS[CONNECTION]),
            REST: float(row[3] or DEFAULT_LEVELS[REST]),
        },
        satiated_at={
            CURIOSITY: row[4],
            COMPETENCE: row[5],
            CONNECTION: row[6],
            REST: row[7],
        },
        last_decay_at=row[8],
    )


def _half_life_hours() -> float:
    """Configurable half-life from settings."""
    try:
        from augmentum.config import settings
        return float(getattr(settings, "companion_drive_decay_half_life_hours",
                             DEFAULT_HALF_LIFE_HOURS))
    except Exception:
        return DEFAULT_HALF_LIFE_HOURS


def _clamp(v: float) -> float:
    return max(LEVEL_FLOOR, min(LEVEL_CEILING, v))


async def decay(runtime: CompanionRuntime, *, user_id: str) -> DriveState:
    """Apply exponential decay toward baseline based on elapsed time
    since last_decay_at. Returns the new state.

    Called by the tick loop on each iteration. Cheap — single
    SELECT + single UPDATE.
    """
    state = await load(runtime, user_id=user_id)
    if state.last_decay_at is None:
        # First load — nothing to decay yet
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
        return state  # less than a few seconds since last decay; skip

    half_life = max(_half_life_hours(), 0.1)
    # Decay fraction: each half-life halves the distance to baseline
    decay_fraction = 1.0 - math.exp(-elapsed_hours * math.log(2) / half_life)

    new_levels: dict[str, float] = {}
    for name in DRIVE_NAMES:
        current = state.levels.get(name, DEFAULT_LEVELS[name])
        baseline = DEFAULT_LEVELS[name]
        # Move toward baseline by decay_fraction
        new_levels[name] = _clamp(current + (baseline - current) * decay_fraction)

    try:
        await runtime.backend.conn.execute(
            "UPDATE companion_drive_state "
            "SET curiosity_level = ?, competence_level = ?, "
            "    connection_level = ?, rest_level = ?, "
            "    last_decay_at = datetime('now') "
            "WHERE user_id = ? AND companion_id = ?",
            (new_levels[CURIOSITY], new_levels[COMPETENCE],
             new_levels[CONNECTION], new_levels[REST],
             user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("drives_decay_persist_failed", exc_info=True)

    state.levels = new_levels
    return state


async def satiate(
    runtime: CompanionRuntime, *, user_id: str, drive: str,
    amount: float = SATIATION_AMOUNT,
) -> None:
    """Reduce a drive's level after a matching activity performs.

    Idempotent — repeat calls within the same tick just stack
    satiation. Bounded by LEVEL_FLOOR.
    """
    if drive not in DRIVE_NAMES:
        log.debug("drives_satiate_unknown", drive=drive)
        return
    state = await load(runtime, user_id=user_id)
    current = state.levels.get(drive, DEFAULT_LEVELS.get(drive, 0.5))
    new_level = _clamp(current - amount)

    column_level = f"{drive}_level"
    column_satiated = f"{drive}_satiated_at"
    try:
        await runtime.backend.conn.execute(
            f"UPDATE companion_drive_state "
            f"SET {column_level} = ?, {column_satiated} = datetime('now') "
            f"WHERE user_id = ? AND companion_id = ?",
            (new_level, user_id, runtime.companion_id),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("drives_satiate_persist_failed", drive=drive, exc_info=True)


__all__ = [
    "DriveState",
    "DRIVE_NAMES",
    "CURIOSITY", "COMPETENCE", "CONNECTION", "REST",
    "DEFAULT_LEVELS",
    "load", "decay", "satiate",
]
