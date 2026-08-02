"""Cast event audit log — server-side record of every dispatch.

Recorded when a surface is dispatched (``cast_surface`` succeeds).
Ended when the surface closes (replaced, user_stop, receiver
disconnect, surface ended). Powers two UIs:

  - "Currently showing on this TV" — query active events by trusted_id
  - Recent Activity — query latest N events per user, ordered desc

Multi-tenant scoped per the augmentum-dev rule. Cross-user reads
return empty.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


END_REASON_USER_STOP = "user_stop"           # explicit cast/send/close
END_REASON_REPLACED = "replaced"             # another surface took the slot
END_REASON_DISCONNECTED = "disconnected"     # receiver lost its WS
END_REASON_ENDED = "ended"                   # natural surface end (e.g. video finished)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _make_id() -> str:
    return f"cev_{secrets.token_hex(8)}"


@dataclass(slots=True)
class CastEvent:
    id: str
    user_id: str
    trusted_id: str
    registration_id: str
    surface_id: str
    surface_kind: str
    surface_url: str
    slot: str
    started_at: str
    ended_at: str
    end_reason: str

    @property
    def is_active(self) -> bool:
        return not self.ended_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trusted_id": self.trusted_id,
            "registration_id": self.registration_id,
            "surface_id": self.surface_id,
            "surface_kind": self.surface_kind,
            "surface_url": self.surface_url,
            "slot": self.slot,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "active": self.is_active,
        }


_SELECT_COLS = (
    "id, user_id, trusted_id, registration_id, surface_id, surface_kind, "
    "surface_url, slot, started_at, ended_at, end_reason"
)


def _row_to_event(row: Any) -> CastEvent:
    return CastEvent(
        id=row[0], user_id=row[1], trusted_id=row[2], registration_id=row[3],
        surface_id=row[4], surface_kind=row[5], surface_url=row[6], slot=row[7],
        started_at=row[8], ended_at=row[9], end_reason=row[10],
    )


class CastEventStore:
    def __init__(self, conn: "aiosqlite.Connection") -> None:
        self._conn = conn

    async def record_start(
        self,
        *,
        user_id: str,
        trusted_id: str = "",
        registration_id: str = "",
        surface_id: str,
        surface_kind: str,
        surface_url: str = "",
        slot: str = "main",
    ) -> str:
        """Append an event. Returns the new event id so the caller can
        reference it for ``mark_end``."""
        if not user_id or not surface_id:
            return ""
        new_id = _make_id()
        await self._conn.execute(
            "INSERT INTO receiver_cast_events "
            "(id, user_id, trusted_id, registration_id, surface_id, "
            " surface_kind, surface_url, slot, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id, user_id, trusted_id, registration_id,
             surface_id, surface_kind, surface_url, slot, _now_iso()),
        )
        await self._conn.commit()
        return new_id

    async def mark_end_by_surface(
        self,
        *,
        user_id: str,
        surface_id: str,
        reason: str = END_REASON_USER_STOP,
    ) -> bool:
        """Close the most recent active event for ``surface_id``.
        Idempotent — events that are already closed remain unchanged.
        """
        if not user_id or not surface_id:
            return False
        cursor = await self._conn.execute(
            "UPDATE receiver_cast_events SET ended_at = ?, end_reason = ? "
            "WHERE user_id = ? AND surface_id = ? AND ended_at = ''",
            (_now_iso(), reason, user_id, surface_id),
        )
        try:
            updated = cursor.rowcount > 0
        finally:
            await cursor.close()
        if updated:
            await self._conn.commit()
        return updated

    async def mark_end_by_registration(
        self,
        *,
        user_id: str,
        registration_id: str,
        reason: str = END_REASON_DISCONNECTED,
    ) -> int:
        """Close every active event tied to a runtime registration_id.
        Used when a WS disconnects — everything that was casting to
        that receiver ends 'disconnected'."""
        if not user_id or not registration_id:
            return 0
        cursor = await self._conn.execute(
            "UPDATE receiver_cast_events SET ended_at = ?, end_reason = ? "
            "WHERE user_id = ? AND registration_id = ? AND ended_at = ''",
            (_now_iso(), reason, user_id, registration_id),
        )
        try:
            count = cursor.rowcount
        finally:
            await cursor.close()
        if count > 0:
            await self._conn.commit()
        return count

    async def list_recent(
        self, *, user_id: str, limit: int = 25,
    ) -> list[CastEvent]:
        if not user_id:
            return []
        cursor = await self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM receiver_cast_events "
            "WHERE user_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (user_id, max(1, min(int(limit), 500))),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_event(r) for r in rows]

    async def list_active_for_trusted(
        self, trusted_id: str, *, user_id: str,
    ) -> list[CastEvent]:
        if not trusted_id or not user_id:
            return []
        cursor = await self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM receiver_cast_events "
            "WHERE user_id = ? AND trusted_id = ? AND ended_at = '' "
            "ORDER BY started_at DESC",
            (user_id, trusted_id),
        )
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_event(r) for r in rows]
