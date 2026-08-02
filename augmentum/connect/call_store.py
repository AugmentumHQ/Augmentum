"""Connect call store — read-side DAO for call_sessions + call_events.

Write paths are owned by ``call_routing.py`` and ``call_lifecycle.py``
(per the convention that each substrate's wiring layer is the
authoritative writer). This module is the READ-ONLY surface so HTTP
routes don't need to reach into the migration schema directly.

Returns dataclasses (CallRow, CallEventRow) shaped to match the
on-the-wire JSON the UI consumes. ``to_dict()`` is the canonical
serialisation — the HTTP routes can return rows directly.

Per CLAUDE.md: every method scopes by ``user_id``. There's no
cross-user read path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# ── Data shapes ────────────────────────────────────────────────────


@dataclass
class CallRow:
    call_id: str
    user_id: str
    initiator_did: str
    receiver_did: str
    modalities: str
    state: str
    end_reason: str
    initiated_at: str
    connected_at: str | None
    ended_at: str | None
    quality_rating: int | None
    quality_notes: str
    becca_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "user_id": self.user_id,
            "initiator_did": self.initiator_did,
            "receiver_did": self.receiver_did,
            "modalities": self.modalities,
            "state": self.state,
            "end_reason": self.end_reason,
            "initiated_at": self.initiated_at,
            "connected_at": self.connected_at,
            "ended_at": self.ended_at,
            "duration_seconds": self._duration_seconds(),
            "quality_rating": self.quality_rating,
            "quality_notes": self.quality_notes,
            "becca_present": self.becca_present,
            # UX hint — which side of the call this user is on. The
            # initiator's DID matches local_did_for(user_id); the
            # receiver's is the peer DID. Computed here so the UI
            # doesn't need to reconstruct it.
            "direction": (
                "outgoing"
                if self.initiator_did.startswith(self.user_id + "@")
                else "incoming"
            ),
            "peer_did": (
                self.receiver_did
                if self.initiator_did.startswith(self.user_id + "@")
                else self.initiator_did
            ),
        }

    def _duration_seconds(self) -> int | None:
        if not self.connected_at or not self.ended_at:
            return None
        try:
            from datetime import datetime
            start = datetime.fromisoformat(self.connected_at)
            end = datetime.fromisoformat(self.ended_at)
            delta = (end - start).total_seconds()
            if delta < 0:
                return None
            return int(delta)
        except (ValueError, TypeError):
            return None


@dataclass
class CallEventRow:
    event_id: int
    call_id: str
    user_id: str
    event_type: str
    event_data: dict[str, Any]
    occurred_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "occurred_at": self.occurred_at,
        }


# ── Reads ──────────────────────────────────────────────────────────


async def list_calls_for_user(
    conn: Any,
    *,
    user_id: str,
    limit: int = 100,
    before: str | None = None,
    state_filter: str | None = None,
) -> list[CallRow]:
    """Most-recent-first by ``initiated_at``.

    ``before`` is a pagination cursor (``initiated_at`` of the oldest
    row already loaded). ``state_filter`` narrows to one state
    (e.g. ``'missed'`` for the missed-calls inbox).
    """

    where = ["user_id = ?"]
    params: list[Any] = [user_id]
    if before:
        where.append("initiated_at < ?")
        params.append(before)
    if state_filter:
        where.append("state = ?")
        params.append(state_filter)

    cur = await conn.execute(
        f"""SELECT call_id, user_id, initiator_did, receiver_did,
                   modalities, state, end_reason,
                   initiated_at, connected_at, ended_at,
                   quality_rating, quality_notes, becca_present
              FROM call_sessions
             WHERE {' AND '.join(where)}
             ORDER BY initiated_at DESC, call_id DESC
             LIMIT ?""",
        (*params, max(1, min(limit, 500))),
    )
    return [_row_to_call(r) for r in await cur.fetchall()]


async def get_call(
    conn: Any, *, call_id: str, user_id: str,
) -> CallRow | None:
    cur = await conn.execute(
        """SELECT call_id, user_id, initiator_did, receiver_did,
                  modalities, state, end_reason,
                  initiated_at, connected_at, ended_at,
                  quality_rating, quality_notes, becca_present
             FROM call_sessions
            WHERE call_id = ? AND user_id = ?""",
        (call_id, user_id),
    )
    row = await cur.fetchone()
    return _row_to_call(row) if row else None


async def list_events_for_call(
    conn: Any, *, call_id: str, user_id: str, limit: int = 200,
) -> list[CallEventRow]:
    cur = await conn.execute(
        """SELECT event_id, call_id, user_id, event_type,
                  event_data, occurred_at
             FROM call_events
            WHERE call_id = ? AND user_id = ?
            ORDER BY occurred_at ASC, event_id ASC
            LIMIT ?""",
        (call_id, user_id, max(1, min(limit, 1000))),
    )
    return [_row_to_event(r) for r in await cur.fetchall()]


async def set_quality_rating(
    conn: Any, *, call_id: str, user_id: str,
    rating: int | None, notes: str = "",
) -> bool:
    """Stamp a post-call rating. None resets to unrated.

    Returns whether a row was updated.
    """

    if rating is not None and rating not in (-1, 0, 1):
        raise ValueError("rating must be -1, 0, 1, or None")
    cur = await conn.execute(
        """UPDATE call_sessions
              SET quality_rating = ?, quality_notes = ?
            WHERE call_id = ? AND user_id = ?""",
        (rating, notes or "", call_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


# ── Row converters ─────────────────────────────────────────────────


def _row_to_call(row: Any) -> CallRow:
    return CallRow(
        call_id=row[0],
        user_id=row[1],
        initiator_did=row[2],
        receiver_did=row[3],
        modalities=row[4] or "audio",
        state=row[5] or "invited",
        end_reason=row[6] or "",
        initiated_at=row[7] or "",
        connected_at=row[8],
        ended_at=row[9],
        quality_rating=row[10],
        quality_notes=row[11] or "",
        becca_present=bool(row[12]),
    )


def _row_to_event(row: Any) -> CallEventRow:
    try:
        data = json.loads(row[4]) if row[4] else {}
        if not isinstance(data, dict):
            data = {"value": data}
    except (json.JSONDecodeError, TypeError):
        data = {}
    return CallEventRow(
        event_id=int(row[0]),
        call_id=row[1],
        user_id=row[2],
        event_type=row[3],
        event_data=data,
        occurred_at=row[5] or "",
    )
