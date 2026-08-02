"""Cast couch co-op guest profile store (Phase 2).

Named guest identities under a host's account — see migration 229 and
``docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md``.

Guests aren't Augmentum users. They have no account, no password, no
session. A guest profile is the host's *record* of a recurring
playmate, stored alongside the host's other user-scoped data so:

  * The host's account-delete cascade tears down their guest list
    automatically (FK ``ON DELETE CASCADE``).
  * Per-host UNIQUE on display_name lets "alice" exist independently
    at two different hosts without collision.
  * Phase 4 can join ``game_saves.guest_profile_id`` to surface
    per-player progress without breaking the host's own saves.

Style: mirrors the per-user-scoping pattern of the other stores
(:mod:`augmentum.state.game_stream_store` etc.) — every method that
returns rows takes ``host_user_id`` and appends ``AND host_user_id =
?`` to the SQL.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Limit for the identify-endpoint's existing-profile list (privacy:
# don't leak the full guest roster to anyone who scans a QR).
MAX_PROFILES_PER_LIST = 8


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row, strict=False))


class GuestStore:
    """Persistence for cast couch co-op guest profiles."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Create / lookup ────────────────────────────────────────────────

    async def create_profile(
        self,
        *,
        host_user_id: str,
        display_name: str,
        color: str = "",
    ) -> dict[str, Any]:
        """Mint a new guest profile.

        Raises :class:`aiosqlite.IntegrityError` on UNIQUE collision
        (display_name already taken at this host). Caller is expected
        to catch that and surface the existing profile via
        :meth:`get_by_name`.
        """
        if not host_user_id:
            raise ValueError("create_profile requires host_user_id")
        clean_name = display_name.strip()
        if not clean_name:
            raise ValueError("display_name cannot be empty")

        profile_id = f"gp_{secrets.token_hex(6)}"
        now = int(time.time())
        await self._conn.execute(
            """INSERT INTO guest_profiles
               (id, host_user_id, display_name, color,
                created_at, last_seen_at, play_count)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (profile_id, host_user_id, clean_name, color, now, now),
        )
        await self._conn.commit()
        log.info(
            "guest_profile_created",
            profile_id=profile_id, host_user_id=host_user_id,
            display_name=clean_name,
        )
        return {
            "id": profile_id, "host_user_id": host_user_id,
            "display_name": clean_name, "color": color,
            "created_at": now, "last_seen_at": now, "play_count": 0,
        }

    async def get(
        self, profile_id: str, *, host_user_id: str,
    ) -> dict[str, Any] | None:
        """Read a profile by id, scoped to its owning host.

        ``host_user_id`` is required — passing an empty string raises
        ``ValueError`` so a route handler that forgets to thread the
        scope fails loudly instead of silently leaking another host's
        guest profile.
        """
        if not host_user_id:
            raise ValueError("guest_store.get requires host_user_id")
        cursor = await self._conn.execute(
            "SELECT * FROM guest_profiles WHERE id = ? AND host_user_id = ?",
            (profile_id, host_user_id),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def get_by_name(
        self, *, host_user_id: str, display_name: str,
    ) -> dict[str, Any] | None:
        clean = display_name.strip()
        if not clean or not host_user_id:
            return None
        cursor = await self._conn.execute(
            """SELECT * FROM guest_profiles
               WHERE host_user_id = ? AND display_name = ?""",
            (host_user_id, clean),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_for_host(
        self, *, host_user_id: str, limit: int = MAX_PROFILES_PER_LIST,
    ) -> list[dict[str, Any]]:
        """Most-recently-seen first. Used to populate the identify
        endpoint's existing-profile picker so guests don't have to
        re-type their name.

        Capped by ``limit`` (default 8) for privacy — anyone holding
        an invite token can call identify, so the public roster
        leak surface is bounded.
        """
        if not host_user_id:
            return []
        cursor = await self._conn.execute(
            """SELECT * FROM guest_profiles
               WHERE host_user_id = ?
               ORDER BY last_seen_at DESC
               LIMIT ?""",
            (host_user_id, max(1, min(int(limit), MAX_PROFILES_PER_LIST))),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    # ── Mutate ─────────────────────────────────────────────────────────

    async def touch_last_seen(
        self, profile_id: str, *, host_user_id: str,
        increment_play_count: bool = False,
    ) -> None:
        """Bump ``last_seen_at`` to now. Optionally bumps ``play_count``
        — call with ``increment_play_count=True`` exactly once per join,
        not per frame.

        ``host_user_id`` is required — see :meth:`get` for rationale.
        """
        if not host_user_id:
            raise ValueError("guest_store.touch_last_seen requires host_user_id")
        now = int(time.time())
        if increment_play_count:
            sql = (
                "UPDATE guest_profiles SET last_seen_at = ?, "
                "play_count = play_count + 1 "
                "WHERE id = ? AND host_user_id = ?"
            )
        else:
            sql = (
                "UPDATE guest_profiles SET last_seen_at = ? "
                "WHERE id = ? AND host_user_id = ?"
            )
        await self._conn.execute(sql, (now, profile_id, host_user_id))
        await self._conn.commit()

    async def rename(
        self, profile_id: str, *, host_user_id: str, new_name: str,
    ) -> bool:
        """Host-side rename (Phase 2: optional, no UI yet — useful for
        the Phase 2-stretch "Manage guests" screen later).

        Returns False on UNIQUE collision (a different profile under
        this host already has the new name).
        """
        clean = new_name.strip()
        if not clean or not host_user_id:
            return False
        try:
            await self._conn.execute(
                """UPDATE guest_profiles SET display_name = ?
                   WHERE id = ? AND host_user_id = ?""",
                (clean, profile_id, host_user_id),
            )
            await self._conn.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def delete(
        self, profile_id: str, *, host_user_id: str,
    ) -> bool:
        """Host-side delete. Cascades to guest_devices via the FK
        constraint (Phase 3) and detaches game_saves via SET NULL
        (Phase 4).
        """
        if not host_user_id:
            return False
        cursor = await self._conn.execute(
            "DELETE FROM guest_profiles WHERE id = ? AND host_user_id = ?",
            (profile_id, host_user_id),
        )
        await self._conn.commit()
        deleted = (cursor.rowcount or 0) > 0
        if deleted:
            log.info(
                "guest_profile_deleted",
                profile_id=profile_id, host_user_id=host_user_id,
            )
        return deleted
