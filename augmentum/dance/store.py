"""SQLite stores for the companion widget dance timeline.

Mirrors the ``AvatarStore`` pattern: every method takes ``user_id`` and
all queries scope by it. Unlike ``avatars`` there is no notion of
bundled rows here — both history and ratings are inherently per-user.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


# Hard server-side cap on how many history rows we keep per user. The
# widget reads the most recent 50 for the timeline; we keep a bit more
# headroom for future "what did I like this week" surfaces but stop
# short of unbounded growth. Pruning happens lazily on append (see
# ``DanceHistoryStore.append``).
_HISTORY_RETENTION = 200

# Cap on the per-id slot bonus from stacked 'longer' ratings, in
# seconds. Matches the legacy localStorage cap so behavior is
# unchanged from the user's perspective.
_SLOT_BONUS_CAP_SEC = 60


def _make_history_id() -> str:
    """Per-row uuid for dance_history. Time-prefixed so insertion order
    is preserved when sorted lexicographically (useful for debugging)."""
    ts = int(time.time() * 1000)
    h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]
    return f"dh_{ts}_{h}"


class DanceHistoryStore:
    """User-scoped ring buffer of dance playbacks."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def append(
        self,
        anim_id: str,
        label: str,
        played_at: int,
        duration_sec: float,
        mode: str | None = None,
        *,
        user_id: str = "",
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("dance_history append requires user_id")
        if not anim_id:
            raise ValueError("dance_history append requires anim_id")
        row_id = _make_history_id()
        await self._conn.execute(
            "INSERT INTO dance_history "
            "(id, user_id, anim_id, label, played_at, duration_sec, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row_id, user_id, anim_id, label, int(played_at),
             float(duration_sec), mode),
        )
        # Lazy retention: trim down to the cap after each insert. Cheap
        # because the index covers (user_id, played_at DESC).
        await self._conn.execute(
            "DELETE FROM dance_history WHERE user_id = ? AND id NOT IN ("
            "  SELECT id FROM dance_history WHERE user_id = ? "
            "  ORDER BY played_at DESC LIMIT ?"
            ")",
            (user_id, user_id, _HISTORY_RETENTION),
        )
        await self._conn.commit()
        return {
            "id": row_id,
            "anim_id": anim_id,
            "label": label,
            "played_at": int(played_at),
            "duration_sec": float(duration_sec),
            "mode": mode,
        }

    async def list_recent(
        self, limit: int = 50, *, user_id: str = "",
    ) -> list[dict[str, Any]]:
        if not user_id:
            return []
        cursor = await self._conn.execute(
            "SELECT id, anim_id, label, played_at, duration_sec, mode "
            "FROM dance_history WHERE user_id = ? "
            "ORDER BY played_at DESC LIMIT ?",
            (user_id, max(1, min(limit, _HISTORY_RETENTION))),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "anim_id": r[1],
                "label": r[2],
                "played_at": r[3],
                "duration_sec": r[4],
                "mode": r[5],
            }
            for r in rows
        ]

    async def clear(self, *, user_id: str = "") -> int:
        if not user_id:
            raise ValueError("dance_history clear requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM dance_history WHERE user_id = ?", (user_id,),
        )
        await self._conn.commit()
        return cursor.rowcount or 0


class DanceRatingsStore:
    """User-scoped curation ratings for animations.

    Ratings are upserted by (user_id, anim_id). ``kind`` is the
    like/dislike/broken value or NULL after a clear; ``slot_bonus_sec``
    accumulates from 'longer' ratings independently of ``kind``.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def list_all(self, *, user_id: str = "") -> dict[str, dict[str, Any]]:
        """Returns ratings keyed by anim_id — matches the JS shape
        ``{ animId: { kind, slotBonusSec, ts } }`` so the conductor's
        in-memory format is identical to what the API returns."""
        if not user_id:
            return {}
        cursor = await self._conn.execute(
            "SELECT anim_id, kind, slot_bonus_sec, updated_at "
            "FROM dance_ratings WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            entry: dict[str, Any] = {
                "slotBonusSec": r[2] or 0,
                "ts": r[3],
            }
            if r[1]:
                entry["kind"] = r[1]
            out[r[0]] = entry
        return out

    async def set_kind(
        self, anim_id: str, kind: str, *, user_id: str = "",
    ) -> dict[str, Any]:
        """Set like/dislike/broken. Preserves any accumulated slot bonus."""
        if not user_id:
            raise ValueError("dance_ratings set_kind requires user_id")
        if kind not in ("like", "dislike", "broken"):
            raise ValueError(f"dance_ratings invalid kind: {kind}")
        now = int(time.time() * 1000)
        await self._conn.execute(
            "INSERT INTO dance_ratings "
            "(user_id, anim_id, kind, slot_bonus_sec, updated_at) "
            "VALUES (?, ?, ?, 0, ?) "
            "ON CONFLICT(user_id, anim_id) DO UPDATE SET "
            "  kind = excluded.kind, updated_at = excluded.updated_at",
            (user_id, anim_id, kind, now),
        )
        await self._conn.commit()
        return await self._get(anim_id, user_id=user_id)

    async def add_slot_bonus(
        self, anim_id: str, increment_sec: int, *, user_id: str = "",
    ) -> dict[str, Any]:
        """Accumulate a 'longer' rating. Capped at _SLOT_BONUS_CAP_SEC.
        Preserves any existing kind."""
        if not user_id:
            raise ValueError("dance_ratings add_slot_bonus requires user_id")
        now = int(time.time() * 1000)
        cap = _SLOT_BONUS_CAP_SEC
        await self._conn.execute(
            "INSERT INTO dance_ratings "
            "(user_id, anim_id, kind, slot_bonus_sec, updated_at) "
            "VALUES (?, ?, NULL, MIN(?, ?), ?) "
            "ON CONFLICT(user_id, anim_id) DO UPDATE SET "
            "  slot_bonus_sec = MIN(?, dance_ratings.slot_bonus_sec + ?), "
            "  updated_at = excluded.updated_at",
            (user_id, anim_id, increment_sec, cap, now, cap, increment_sec),
        )
        await self._conn.commit()
        return await self._get(anim_id, user_id=user_id)

    async def clear(self, anim_id: str, *, user_id: str = "") -> None:
        """Remove the row entirely — un-likes, un-breaks, AND zeroes
        any slot bonus. Mirrors the JS 'clear' semantics."""
        if not user_id:
            raise ValueError("dance_ratings clear requires user_id")
        await self._conn.execute(
            "DELETE FROM dance_ratings WHERE user_id = ? AND anim_id = ?",
            (user_id, anim_id),
        )
        await self._conn.commit()

    async def _get(
        self, anim_id: str, *, user_id: str = "",
    ) -> dict[str, Any]:
        cursor = await self._conn.execute(
            "SELECT anim_id, kind, slot_bonus_sec, updated_at "
            "FROM dance_ratings WHERE user_id = ? AND anim_id = ?",
            (user_id, anim_id),
        )
        row = await cursor.fetchone()
        if not row:
            return {"anim_id": anim_id, "slotBonusSec": 0, "ts": 0}
        out: dict[str, Any] = {
            "anim_id": row[0],
            "slotBonusSec": row[2] or 0,
            "ts": row[3],
        }
        if row[1]:
            out["kind"] = row[1]
        return out
