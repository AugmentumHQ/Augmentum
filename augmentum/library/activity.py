"""Library activity ledger — per-artifact event timeline.

Powers two things in the UI:

* Detail-pane timeline ("Opened 2d ago · Cast 5d ago · Edited last week").
* Sidebar Recent + Continue collections (server-derived from the ledger,
  not stored as explicit collections).

The store is intentionally append-only at the route layer. There's no
``edit`` or ``delete`` event method — pruning happens in a scheduled job
once we hit retention concerns (not today; the table is cheap).
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Literal

import aiosqlite

from augmentum.library.ids import is_publication_id
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_ID_PREFIX = "act_"
_ID_NONCE_BYTES = 8

Action = Literal["open", "cast", "edit", "pin", "unpin", "tag"]
Surface = Literal["desktop", "mobile", "tv", "cast"]


def _new_activity_id() -> str:
    return _ID_PREFIX + secrets.token_hex(_ID_NONCE_BYTES)


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


class ActivityStore:
    """Append + read for ``library_activity``."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def record(
        self,
        *,
        user_id: str,
        artifact_id: str,
        action: Action,
        surface: Surface = "desktop",
        payload: dict[str, Any] | None = None,
    ) -> str:
        if not user_id:
            raise ValueError("record requires user_id")
        if not artifact_id:
            raise ValueError("record requires artifact_id")

        # Ownership check across BOTH namespaces. Migration 309 dropped the
        # artifacts(id) FK so union ids (artifacts + pub_ publications) share
        # this table; this is now the only guard that a forged id can't
        # scribble onto another tenant's feed. Route by id prefix to the
        # right backing table.
        owner_table = (
            "library_publications" if is_publication_id(artifact_id) else "artifacts"
        )
        cursor = await self._conn.execute(
            f"SELECT 1 FROM {owner_table} WHERE id = ? AND user_id = ? LIMIT 1",
            (artifact_id, user_id),
        )
        if not await cursor.fetchone():
            raise PermissionError("item not found for this user")

        aid = _new_activity_id()
        await self._conn.execute(
            "INSERT INTO library_activity "
            "(id, user_id, artifact_id, action, surface, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (aid, user_id, artifact_id, action, surface,
             json.dumps(payload or {})),
        )
        await self._conn.commit()
        return aid

    async def list_for_artifact(
        self, artifact_id: str, *, user_id: str, limit: int = 50,
    ) -> list[dict[str, Any]]:
        # rowid DESC is the deterministic tiebreaker — datetime('now')
        # is 1-second resolution so multiple records in the same second
        # would otherwise sort in undefined order. rowid mirrors
        # insertion order, so newest-row-wins.
        cursor = await self._conn.execute(
            "SELECT * FROM library_activity "
            "WHERE artifact_id = ? AND user_id = ? "
            "ORDER BY occurred_at DESC, rowid DESC LIMIT ?",
            (artifact_id, user_id, int(limit)),
        )
        rows = await cursor.fetchall()
        return [_decode(_row_to_dict(cursor, r)) for r in rows]

    async def recent_artifact_ids(
        self,
        *,
        user_id: str,
        actions: tuple[Action, ...] = ("open", "cast", "edit"),
        limit: int = 12,
    ) -> list[str]:
        """Most-recent distinct artifact_ids the user touched.

        DISTINCT collapses repeated opens; the GROUP BY/MAX is the
        cheapest way to keep "most recent touch wins" semantics across
        actions without a window function (SQLite has them but the
        plan-explorer reads more clearly this way for the few-thousand
        rows we expect)."""
        if not user_id:
            return []
        placeholders = ",".join("?" * len(actions))
        cursor = await self._conn.execute(
            "SELECT artifact_id, MAX(occurred_at) AS last_seen, MAX(rowid) AS last_row "
            "FROM library_activity "
            f"WHERE user_id = ? AND action IN ({placeholders}) "
            "GROUP BY artifact_id "
            "ORDER BY last_seen DESC, last_row DESC LIMIT ?",
            (user_id, *actions, int(limit)),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def continue_artifact_ids(
        self, *, user_id: str, limit: int = 6,
    ) -> list[str]:
        """Artifacts the user *interacted with* but probably hasn't
        finished. v1 heuristic: artifacts opened or cast in the last
        30 days, sorted by most-recent touch, excluding anything tagged
        ``done``. Cheap server-side proxy for "continue where you left
        off" until per-item progress lands."""
        if not user_id:
            return []
        cursor = await self._conn.execute(
            "SELECT la.artifact_id, MAX(la.occurred_at) AS last_seen "
            "FROM library_activity la "
            "JOIN artifacts a ON a.id = la.artifact_id AND a.user_id = la.user_id "
            "WHERE la.user_id = ? "
            "  AND la.action IN ('open','cast') "
            "  AND la.occurred_at >= datetime('now', '-30 days') "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM json_each(a.tags) WHERE json_each.value = 'done'"
            "  ) "
            "GROUP BY la.artifact_id "
            "ORDER BY last_seen DESC, MAX(la.rowid) DESC LIMIT ?",
            (user_id, int(limit)),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    if "payload" in row and isinstance(row["payload"], str):
        try:
            row["payload"] = json.loads(row["payload"])
        except json.JSONDecodeError:
            row["payload"] = {}
    return row
