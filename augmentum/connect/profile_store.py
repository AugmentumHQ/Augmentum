"""Connect profile store — bio / status presentation layer (Phase 1).

A small CRUD layer over ``connect_profiles`` (migration 281). The profile is
the soft, editable presentation of a user on the Connect surface — distinct
from the canonical ``users.display_name``. See the design doc
``docs/superpowers/specs/2026-06-20-connect-comms-platform-design.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import aiosqlite

_FIELDS = ("bio", "status_message", "status_emoji", "avatar_ref")

# Soft length caps so a profile can't bloat the directory payload. Enforced at
# write time (truncation, not rejection — a too-long bio just gets clipped).
_CAPS = {"bio": 500, "status_message": 140, "status_emoji": 16, "avatar_ref": 256}


def _empty_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id, "bio": "", "status_message": "",
        "status_emoji": "", "avatar_ref": "", "updated_at": "",
    }


async def get_profile(conn: aiosqlite.Connection, *, user_id: str) -> dict[str, Any]:
    """Return the user's profile, or a zero-value profile if none exists yet."""
    cur = await conn.execute(
        "SELECT user_id, bio, status_message, status_emoji, avatar_ref, updated_at "
        "FROM connect_profiles WHERE user_id = ?",
        (user_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return _empty_profile(user_id)
    return {
        "user_id": row[0], "bio": row[1], "status_message": row[2],
        "status_emoji": row[3], "avatar_ref": row[4], "updated_at": row[5],
    }


async def upsert_profile(
    conn: aiosqlite.Connection, *, user_id: str, **fields: Any,
) -> dict[str, Any]:
    """Create/patch the caller's profile. Only the fields passed are written.

    Unknown keys are ignored; ``None`` values are skipped (leave the existing
    value); strings are stripped and truncated to their soft cap. Returns the
    full profile after the write.
    """
    updates: dict[str, str] = {}
    for key in _FIELDS:
        if key in fields and fields[key] is not None:
            val = str(fields[key]).strip()[: _CAPS[key]]
            updates[key] = val
    if not updates:
        return await get_profile(conn, user_id=user_id)

    cols = ", ".join(updates)
    placeholders = ", ".join("?" for _ in updates)
    set_clause = ", ".join(f"{k} = excluded.{k}" for k in updates)
    await conn.execute(
        f"""INSERT INTO connect_profiles (user_id, {cols}, updated_at)
            VALUES (?, {placeholders}, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                {set_clause}, updated_at = datetime('now')""",
        (user_id, *updates.values()),
    )
    await conn.commit()
    return await get_profile(conn, user_id=user_id)


async def get_profiles_for(
    conn: aiosqlite.Connection, user_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Bulk-fetch profiles for a set of user_ids (directory enrichment).

    Returns a ``{user_id: profile_dict}`` map containing only users who have a
    profile row — callers fall back to defaults for the rest.
    """
    if not user_ids:
        return {}
    placeholders = ", ".join("?" for _ in user_ids)
    cur = await conn.execute(
        "SELECT user_id, bio, status_message, status_emoji, avatar_ref, updated_at "
        f"FROM connect_profiles WHERE user_id IN ({placeholders})",
        tuple(user_ids),
    )
    rows = await cur.fetchall()
    await cur.close()
    return {
        r[0]: {
            "user_id": r[0], "bio": r[1], "status_message": r[2],
            "status_emoji": r[3], "avatar_ref": r[4], "updated_at": r[5],
        }
        for r in rows
    }
