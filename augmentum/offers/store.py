"""Offer suppression store — CRUD over the ``offer_suppressions`` table.

See migration 224 + spec 2026-06-02-offer-substrate-design.md.

The store has a single primary key ``(user_id, kind, target_id)``;
suppression is per-user, per-kind, per-target — deliberately NOT
per-thread, because "never suggest this again" is a standing preference,
not a per-chat one.

Only ``never`` is reachable from the UI. "Not now" dismisses a single chip
without writing here at all, so the only rows a user can create are
permanent ones they asked for by name.

The active offers themselves live in the ``notifications`` table on
channel ``system.offer``; no separate offer-row store is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


# Year-9999 sentinel for permanent suppression. Choosing year-9999
# (not, e.g., year-2999) so it's obvious in the row dump that this is
# a sentinel value and not a real future date. ISO-8601 UTC string
# so it round-trips through ``TEXT TIMESTAMP`` columns cleanly.
SUPPRESSION_NEVER: str = "9999-12-31T00:00:00+00:00"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class OfferSuppression:
    """One row from ``offer_suppressions``."""

    user_id: str
    kind: str
    target_id: str
    suppressed_until: str
    reason: str
    created_at: str

    @property
    def is_permanent(self) -> bool:
        return self.suppressed_until >= SUPPRESSION_NEVER


async def set_suppression(
    conn: "aiosqlite.Connection",
    *,
    user_id: str,
    kind: str,
    target_id: str,
    suppressed_until: str,
    reason: str = "snooze",
) -> None:
    """Upsert a suppression row.

    Calling with the same ``(user_id, kind, target_id)`` replaces any
    existing row — letting a snooze be promoted to Never, or vice
    versa, without bookkeeping at the call site.
    """

    if not user_id:
        raise ValueError("user_id is required (per-user isolation)")
    if not kind:
        raise ValueError("kind is required")
    if not target_id:
        raise ValueError("target_id is required")
    if not suppressed_until:
        raise ValueError("suppressed_until is required")

    await conn.execute(
        """INSERT INTO offer_suppressions
               (user_id, kind, target_id, suppressed_until, reason, created_at)
             VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, kind, target_id) DO UPDATE SET
               suppressed_until = excluded.suppressed_until,
               reason           = excluded.reason""",
        (user_id, kind, target_id, suppressed_until, reason, _now_iso()),
    )
    await conn.commit()


async def snooze(
    conn: "aiosqlite.Connection",
    *,
    user_id: str,
    kind: str,
    target_id: str,
    days: int = 30,
) -> str:
    """Snooze this offer for ``days`` days. Returns the suppression date.

    NOT wired to any chip action. The "Not now" button dismisses the one
    chip and writes nothing — a single decline on one device must not mute
    a capability everywhere for a month (see ``handlers/system_offer.py``
    and migration 326). This primitive stays because ``reason='snooze'``
    rows may exist from before that change and
    ``sweep_expired_suppressions`` must keep pruning them; wire it to a
    new UI only behind a control whose label states the real duration.
    """

    until_dt = datetime.now(UTC) + timedelta(days=int(days))
    until_iso = until_dt.isoformat()
    await set_suppression(
        conn,
        user_id=user_id,
        kind=kind,
        target_id=target_id,
        suppressed_until=until_iso,
        reason="snooze",
    )
    return until_iso


async def never(
    conn: "aiosqlite.Connection",
    *,
    user_id: str,
    kind: str,
    target_id: str,
) -> None:
    """Permanently suppress this offer (user can Undo from Settings)."""

    await set_suppression(
        conn,
        user_id=user_id,
        kind=kind,
        target_id=target_id,
        suppressed_until=SUPPRESSION_NEVER,
        reason="never",
    )


async def is_suppressed(
    conn: "aiosqlite.Connection",
    *,
    user_id: str,
    kind: str,
    target_id: str,
) -> bool:
    """True if there's an active suppression for this triple.

    Active = ``suppressed_until > NOW``. Expired snoozes are *not*
    treated as suppressed — they'll be pruned by a sweep but the
    dispatcher proceeds normally in the meantime.
    """

    if not user_id or not kind or not target_id:
        return False
    cur = await conn.execute(
        """SELECT suppressed_until FROM offer_suppressions
            WHERE user_id = ? AND kind = ? AND target_id = ?""",
        (user_id, kind, target_id),
    )
    row = await cur.fetchone()
    await cur.close()
    if row is None:
        return False
    return row[0] > _now_iso()


async def list_suppressions(
    conn: "aiosqlite.Connection", *, user_id: str,
) -> list[OfferSuppression]:
    """All current suppressions for the user (active + expired)."""

    if not user_id:
        return []
    cur = await conn.execute(
        """SELECT user_id, kind, target_id, suppressed_until, reason,
                  created_at
             FROM offer_suppressions
            WHERE user_id = ?
            ORDER BY created_at DESC""",
        (user_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [
        OfferSuppression(
            user_id=r[0],
            kind=r[1],
            target_id=r[2],
            suppressed_until=r[3],
            reason=r[4],
            created_at=r[5],
        )
        for r in rows
    ]


async def delete_suppression(
    conn: "aiosqlite.Connection",
    *,
    user_id: str,
    kind: str,
    target_id: str,
) -> bool:
    """Remove a suppression (the Undo path). Returns whether a row went."""

    if not user_id or not kind or not target_id:
        return False
    cur = await conn.execute(
        """DELETE FROM offer_suppressions
            WHERE user_id = ? AND kind = ? AND target_id = ?""",
        (user_id, kind, target_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def sweep_expired_suppressions(
    conn: "aiosqlite.Connection",
) -> int:
    """Prune snooze rows whose ``suppressed_until`` is in the past.

    Never rows (``reason='never'``) are preserved regardless of date.
    Intended to run on a periodic sweep (weekly is fine). Returns the
    number of rows removed.
    """

    cur = await conn.execute(
        """DELETE FROM offer_suppressions
            WHERE reason = 'snooze' AND suppressed_until <= ?""",
        (_now_iso(),),
    )
    await conn.commit()
    return cur.rowcount or 0
