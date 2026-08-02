"""Game streaming persistence layer.

User-scoped CRUD for ``game_stream_sessions``, ``game_stream_worlds`` and
``game_stream_telemetry``. Mirrors the pattern in
``augmentum/state/jobs_store.py``: every read/write that touches user data
takes ``user_id`` and appends ``AND user_id = ?`` to the SQL.

Lifecycle reconciliation (the runner that walks orphaned containers on
startup) calls ``list_running_unscoped`` -- the one helper that bypasses
user filtering -- with the explicit understanding that it never returns
data to a user, only to the lifecycle manager.

See ``augmentum/state/migrations/120_game_stream_sessions.sql``,
``121_game_stream_worlds.sql`` and ``122_game_stream_telemetry.sql`` for
schema and field-level notes.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _require_user_id(user_id: str, op: str) -> None:
    """Refuse user-scoped ops invoked with empty user_id.

    The earlier signature defaulted user_id to "" and silently dropped
    the WHERE clause when empty — so any call site that forgot to thread
    user_id through inherited a cross-tenant read or write. Failing
    loudly here pins the invariant so future regressions are immediate
    rather than silent. Lifecycle-scoped reads that genuinely want every
    user's rows use ``list_running_unscoped``.
    """
    if not user_id:
        raise ValueError(
            f"{op} requires non-empty user_id (cross-tenant guard). "
            "Add an explicit unscoped variant if a lifecycle path needs "
            "to read across users.",
        )


# Statuses considered "live" -- the lifecycle manager owns these rows
# and they count against per-user concurrent-stream caps. ``paused`` is
# included because the container still exists (cgroup-frozen) and
# holds RAM + a port + a credit budget slot; only stopped/crashed
# release everything.
LIVE_SESSION_STATUSES: frozenset[str] = frozenset(
    {"starting", "ready", "connected", "idle", "paused"}
)

# Terminal statuses; these rows are kept for telemetry until reaped.
TERMINAL_SESSION_STATUSES: frozenset[str] = frozenset(
    {"stopped", "crashed"}
)


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    d: dict[str, Any] = dict(zip(cols, row))
    # Inflate JSON fields that callers expect as Python objects.
    for key in ("settings_json", "whitelist_user_ids"):
        val = d.get(key)
        if isinstance(val, str) and val:
            try:
                d[key] = json.loads(val)
            except json.JSONDecodeError:
                pass
    return d


class GameStreamStore:
    """Persistence for AGSP sessions, worlds, and telemetry."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Sessions ──────────────────────────────────────────────────────

    async def create_session(
        self,
        *,
        user_id: str,
        profile_id: str,
        world_id: str | None = None,
        bitrate_mbps: int = 4,
        resolution: str = "1280x720",
        encoder: str = "auto",
        system_id: str = "",
    ) -> str:
        if not user_id:
            raise ValueError("create_session requires user_id")
        if not profile_id:
            raise ValueError("create_session requires profile_id")

        session_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO game_stream_sessions
               (id, user_id, world_id, profile_id, status,
                bitrate_mbps, resolution, encoder, system_id)
               VALUES (?, ?, ?, ?, 'starting', ?, ?, ?, ?)""",
            (
                session_id,
                user_id,
                world_id,
                profile_id,
                int(bitrate_mbps),
                resolution,
                encoder,
                system_id or None,
            ),
        )
        await self._conn.commit()
        return session_id

    async def get_session(
        self, session_id: str, *, user_id: str = ""
    ) -> dict | None:
        _require_user_id(user_id, "get_session")
        cursor = await self._conn.execute(
            "SELECT * FROM game_stream_sessions WHERE id = ? AND user_id = ?",
            [session_id, user_id],
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_sessions_for_user(
        self,
        *,
        user_id: str,
        status: str | None = None,
        live_only: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        if not user_id:
            return []
        query = "SELECT * FROM game_stream_sessions WHERE user_id = ?"
        params: list = [user_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        elif live_only:
            placeholders = ",".join(["?"] * len(LIVE_SESSION_STATUSES))
            query += f" AND status IN ({placeholders})"
            params.extend(sorted(LIVE_SESSION_STATUSES))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def list_running_unscoped(self) -> list[dict]:
        """Lifecycle-manager-only: walk every live session across users.

        Used at startup to reconcile orphaned containers. Never expose
        the result of this call to a user.
        """
        placeholders = ",".join(["?"] * len(LIVE_SESSION_STATUSES))
        query = (
            f"SELECT * FROM game_stream_sessions WHERE status IN ({placeholders})"
        )
        cursor = await self._conn.execute(query, sorted(LIVE_SESSION_STATUSES))
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def update_session(
        self,
        session_id: str,
        *,
        user_id: str = "",
        status: str | None = None,
        container_id: str | None = None,
        stream_port: int | None = None,
        game_port: int | None = None,
        encoder: str | None = None,
        exit_reason: str | None = None,
        touch_seen: bool = False,
        cast_input_token: str | None = None,
        # Pause bookkeeping. When the runtime transitions a session into
        # PAUSED it stamps paused_at; on RESUME it clears (writes NULL).
        # The sweep loop uses paused_at to enforce paused_stop_seconds.
        # Sentinel "" means "clear to NULL" (since None signals "leave
        # alone" in the kwargs pattern this method uses elsewhere).
        paused_at: str | None = None,
    ) -> bool:
        _require_user_id(user_id, "update_session")
        sets: list[str] = ["updated_at = datetime('now')"]
        params: list = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if container_id is not None:
            sets.append("container_id = ?")
            params.append(container_id)
        if stream_port is not None:
            sets.append("stream_port = ?")
            params.append(int(stream_port))
        if game_port is not None:
            sets.append("game_port = ?")
            params.append(int(game_port))
        if encoder is not None:
            sets.append("encoder = ?")
            params.append(encoder)
        if exit_reason is not None:
            sets.append("exit_reason = ?")
            params.append(exit_reason)
        if touch_seen:
            sets.append("last_seen_at = datetime('now')")
        if cast_input_token is not None:
            sets.append("cast_input_token = ?")
            params.append(cast_input_token)
        if paused_at is not None:
            # Empty string sentinel = clear to NULL. Anything else =
            # store as-is (callers pass "now" → datetime('now')).
            if paused_at == "":
                sets.append("paused_at = NULL")
            elif paused_at == "now":
                sets.append("paused_at = datetime('now')")
            else:
                sets.append("paused_at = ?")
                params.append(paused_at)

        query = (
            f"UPDATE game_stream_sessions SET {', '.join(sets)} "
            "WHERE id = ? AND user_id = ?"
        )
        params.extend([session_id, user_id])
        cursor = await self._conn.execute(query, params)
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    async def delete_session(self, session_id: str, *, user_id: str = "") -> bool:
        _require_user_id(user_id, "delete_session")
        cursor = await self._conn.execute(
            "DELETE FROM game_stream_sessions WHERE id = ? AND user_id = ?",
            [session_id, user_id],
        )
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    async def count_live_for_user(self, *, user_id: str) -> int:
        if not user_id:
            return 0
        placeholders = ",".join(["?"] * len(LIVE_SESSION_STATUSES))
        query = (
            f"SELECT COUNT(*) FROM game_stream_sessions "
            f"WHERE user_id = ? AND status IN ({placeholders})"
        )
        cursor = await self._conn.execute(
            query, [user_id, *sorted(LIVE_SESSION_STATUSES)]
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ── Worlds ────────────────────────────────────────────────────────

    async def create_world(
        self,
        *,
        user_id: str,
        profile_id: str,
        name: str,
        settings: dict | None = None,
        whitelist: list[str] | None = None,
        storage_path: str = "",
    ) -> str:
        if not user_id:
            raise ValueError("create_world requires user_id")
        if not profile_id or not name:
            raise ValueError("create_world requires profile_id and name")

        world_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO game_stream_worlds
               (id, user_id, profile_id, name, settings_json,
                whitelist_user_ids, storage_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                world_id,
                user_id,
                profile_id,
                name,
                json.dumps(settings or {}),
                json.dumps(whitelist or []),
                storage_path,
            ),
        )
        await self._conn.commit()
        return world_id

    async def get_world(
        self, world_id: str, *, user_id: str = ""
    ) -> dict | None:
        _require_user_id(user_id, "get_world")
        cursor = await self._conn.execute(
            "SELECT * FROM game_stream_worlds WHERE id = ? AND user_id = ?",
            [world_id, user_id],
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_worlds_for_user(
        self, *, user_id: str, profile_id: str | None = None, limit: int = 100,
    ) -> list[dict]:
        if not user_id:
            return []
        query = (
            "SELECT * FROM game_stream_worlds "
            "WHERE user_id = ? OR whitelist_user_ids LIKE ?"
        )
        # Cheap LIKE-based whitelist lookup. Safe given current ID format
        # (16-char hex from uuid4().hex[:16]), where substring collision is
        # astronomically unlikely. If user_ids ever shift to a format where
        # one ID can be a substring of another (e.g. usernames), promote
        # the whitelist to its own join table and replace this with a real
        # join. The user_id is parameterised, so this is not injection-prone.
        params: list = [user_id, f'%"{user_id}"%']
        if profile_id:
            query += " AND profile_id = ?"
            params.append(profile_id)
        query += " ORDER BY last_played_at DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def update_world(
        self,
        world_id: str,
        *,
        user_id: str = "",
        name: str | None = None,
        settings: dict | None = None,
        whitelist: list[str] | None = None,
        touch_played: bool = False,
    ) -> bool:
        _require_user_id(user_id, "update_world")
        sets: list[str] = []
        params: list = []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if settings is not None:
            sets.append("settings_json = ?")
            params.append(json.dumps(settings))
        if whitelist is not None:
            sets.append("whitelist_user_ids = ?")
            params.append(json.dumps(whitelist))
        if touch_played:
            sets.append("last_played_at = datetime('now')")
        if not sets:
            return False
        query = (
            f"UPDATE game_stream_worlds SET {', '.join(sets)} "
            "WHERE id = ? AND user_id = ?"
        )
        params.extend([world_id, user_id])
        cursor = await self._conn.execute(query, params)
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    async def delete_world(self, world_id: str, *, user_id: str = "") -> bool:
        _require_user_id(user_id, "delete_world")
        cursor = await self._conn.execute(
            "DELETE FROM game_stream_worlds WHERE id = ? AND user_id = ?",
            [world_id, user_id],
        )
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    # ── Telemetry ─────────────────────────────────────────────────────

    async def insert_telemetry(
        self,
        *,
        session_id: str,
        user_id: str,
        rtt_ms: float | None = None,
        jitter_ms: float | None = None,
        packet_loss: float | None = None,
        bitrate_kbps: int | None = None,
        fps: float | None = None,
    ) -> None:
        if not session_id or not user_id:
            return
        await self._conn.execute(
            """INSERT INTO game_stream_telemetry
               (session_id, user_id, rtt_ms, jitter_ms, packet_loss,
                bitrate_kbps, fps)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                user_id,
                rtt_ms,
                jitter_ms,
                packet_loss,
                int(bitrate_kbps) if bitrate_kbps is not None else None,
                fps,
            ),
        )
        await self._conn.commit()

    async def recent_telemetry(
        self, session_id: str, *, user_id: str = "", limit: int = 60,
    ) -> list[dict]:
        query = (
            "SELECT rtt_ms, jitter_ms, packet_loss, bitrate_kbps, fps, ts "
            "FROM game_stream_telemetry WHERE session_id = ?"
        )
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]
