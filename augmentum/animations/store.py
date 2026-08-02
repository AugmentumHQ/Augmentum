"""SQLite store for user-uploaded animations (VRMA / BVH).

Mirrors the ``AvatarStore`` shape. Every method takes ``user_id`` and
scopes all queries by it. No notion of bundled rows here — bundled
entries live in code (``ui/scripts/anim-atlas.js``) and are merged
with these rows at runtime by the widget.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


def _make_animation_id() -> str:
    """Namespaced id so an atlas collision with a future bundled id is
    impossible AND so render code can distinguish uploads at a glance
    (e.g. to surface a delete control on uploads only)."""
    ts = int(time.time())
    h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:10]
    return f"user:{ts}_{h}"


# Defaults applied at upload time. Match the most common ATLAS shape so
# new uploads behave reasonably out of the box; user can edit via PUT.
_DEFAULT_EMOTION = {"warmth": 0.6, "energy": 0.6, "openness": 0.6, "focus": 0.6}
_DEFAULT_ROLES = ["dance"]
_DEFAULT_MODES = ["chat-call", "narrative"]
_DEFAULT_COST = 0.5
_DEFAULT_COOLDOWN = 300.0

_ALLOWED_TYPES = ("vrma", "bvh")


def _row_to_dict(row: aiosqlite.Row | tuple, cols: list[str]) -> dict[str, Any]:
    d = dict(zip(cols, row))
    for jcol in ("roles", "emotion", "modes"):
        try:
            d[jcol] = json.loads(d.get(jcol) or "null")
        except Exception:
            d[jcol] = None
    return d


class UserAnimationStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create(
        self,
        animation_type: str,
        source_path: str,
        label: str,
        roles: list[str] | None = None,
        emotion: dict[str, float] | None = None,
        modes: list[str] | None = None,
        cost: float = _DEFAULT_COST,
        duration_sec: float = 0.0,
        cooldown_sec: float = _DEFAULT_COOLDOWN,
        framing: str | None = None,
        trim_start: float | None = None,
        trim_end: float | None = None,
        speed: float | None = None,
        loop_flag: bool = False,
        explicit_only: bool = False,
        notes: str | None = None,
        thumbnail_path: str | None = None,
        *,
        user_id: str = "",
        animation_id: str | None = None,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("user_animations create requires user_id")
        if animation_type not in _ALLOWED_TYPES:
            raise ValueError(
                f"user_animations create unsupported type: {animation_type}"
            )
        if not label.strip():
            raise ValueError("user_animations create requires label")
        anim_id = animation_id or _make_animation_id()
        now = time.time()
        await self._conn.execute(
            "INSERT INTO user_animations "
            "(id, user_id, type, source_path, label, roles, emotion, modes, "
            " cost, duration_sec, cooldown_sec, framing, trim_start, "
            " trim_end, speed, loop_flag, explicit_only, notes, "
            " thumbnail_path, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            " ?, ?, ?)",
            (
                anim_id, user_id, animation_type, source_path, label,
                json.dumps(roles if roles is not None else _DEFAULT_ROLES),
                json.dumps(emotion or _DEFAULT_EMOTION),
                json.dumps(modes if modes is not None else _DEFAULT_MODES),
                float(cost), float(duration_sec), float(cooldown_sec),
                framing, trim_start, trim_end, speed,
                int(bool(loop_flag)), int(bool(explicit_only)),
                notes, thumbnail_path, now, now,
            ),
        )
        await self._conn.commit()
        result = await self.get(anim_id, user_id=user_id)
        if result is None:
            raise RuntimeError("user_animations create: insert disappeared")
        return result

    async def get(
        self, animation_id: str, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        # Isolation floor. This read historically dropped the WHERE clause
        # when user_id was falsy, returning ANY user's row by id. There are
        # no bundled/shared rows here (see module docstring), so an empty
        # user_id is always a caller bug — every sibling (create/update/
        # delete) already raises, so match them rather than leak.
        if not user_id:
            raise ValueError("user_animations get requires user_id")
        cursor = await self._conn.execute(
            "SELECT * FROM user_animations WHERE id = ? AND user_id = ?",
            (animation_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_dict(row, cols)

    async def list_for_user(
        self, *, user_id: str = "",
    ) -> list[dict[str, Any]]:
        if not user_id:
            return []
        cursor = await self._conn.execute(
            "SELECT * FROM user_animations WHERE user_id = ? "
            "ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_dict(r, cols) for r in rows]

    async def update(
        self,
        animation_id: str,
        updates: dict[str, Any],
        *,
        user_id: str = "",
    ) -> dict[str, Any] | None:
        """Partial update. Only known columns are applied; everything
        else is silently ignored so callers can pass through whatever
        the upload form sends.
        """
        if not user_id:
            raise ValueError("user_animations update requires user_id")
        # Whitelist of editable columns. id / user_id / source_path /
        # created_at are not user-editable.
        editable_text = {"label", "framing", "notes"}
        editable_json = {"roles", "emotion", "modes"}
        editable_real = {
            "cost", "duration_sec", "cooldown_sec",
            "trim_start", "trim_end", "speed",
        }
        editable_bool = {"loop_flag", "explicit_only"}
        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            if key in editable_text:
                set_parts.append(f"{key} = ?")
                params.append(None if value is None else str(value))
            elif key in editable_json:
                set_parts.append(f"{key} = ?")
                params.append(json.dumps(value))
            elif key in editable_real:
                set_parts.append(f"{key} = ?")
                params.append(None if value is None else float(value))
            elif key in editable_bool:
                set_parts.append(f"{key} = ?")
                params.append(int(bool(value)))
        if not set_parts:
            return await self.get(animation_id, user_id=user_id)
        set_parts.append("updated_at = ?")
        params.append(time.time())
        params.extend([animation_id, user_id])
        await self._conn.execute(
            f"UPDATE user_animations SET {', '.join(set_parts)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get(animation_id, user_id=user_id)

    async def delete(
        self, animation_id: str, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        """Returns the row before deletion so callers can clean up the
        on-disk file (route handler does this)."""
        if not user_id:
            raise ValueError("user_animations delete requires user_id")
        existing = await self.get(animation_id, user_id=user_id)
        if not existing:
            return None
        await self._conn.execute(
            "DELETE FROM user_animations WHERE id = ? AND user_id = ?",
            (animation_id, user_id),
        )
        await self._conn.commit()
        return existing

    # ── Bundled-atlas overrides (user_atlas_overrides, migration 256) ──
    #
    # Per-user customization of the BUNDLED anim-atlas entries that ship
    # in code. disabled removes an entry from the selection pool; patch
    # is a JSON object of atlas-field overrides merged client-side.

    # Atlas fields a patch may override. Anything else is dropped so a
    # malicious/buggy client can't smuggle arbitrary keys (e.g. source)
    # into entries the conductor will play.
    _PATCH_FIELDS = frozenset({
        "roles", "emotion", "modes", "cost", "duration", "cooldown",
        "loop", "explicitOnly", "notes", "speed", "framing",
        "trimStart", "trimEnd", "trimStartFrac",
    })

    def _sanitize_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in patch.items() if k in self._PATCH_FIELDS}

    async def list_overrides(
        self, *, user_id: str = "",
    ) -> list[dict[str, Any]]:
        if not user_id:
            return []
        cursor = await self._conn.execute(
            "SELECT atlas_id, disabled, patch, updated_at "
            "FROM user_atlas_overrides WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        out: list[dict[str, Any]] = []
        for atlas_id, disabled, patch, updated_at in rows:
            try:
                parsed = json.loads(patch) if patch else {}
            except Exception:
                log.warning(
                    "atlas_override_patch_unparseable",
                    atlas_id=atlas_id, user_id=user_id,
                )
                parsed = {}
            out.append({
                "atlas_id": atlas_id,
                "disabled": bool(disabled),
                "patch": parsed,
                "updated_at": updated_at,
            })
        return out

    async def set_override(
        self,
        atlas_id: str,
        *,
        disabled: bool | None = None,
        patch: dict[str, Any] | None = None,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Upsert one override. ``disabled=None`` / ``patch=None`` keep
        the existing value for that half; pass explicit values to set.
        A patch replaces the stored patch wholesale (the client always
        sends the full edited field set, so merge semantics live there).
        """
        if not user_id:
            raise ValueError("atlas_overrides set requires user_id")
        if not atlas_id or not atlas_id.strip():
            raise ValueError("atlas_overrides set requires atlas_id")
        atlas_id = atlas_id.strip()
        existing = await self._get_override(atlas_id, user_id=user_id)
        new_disabled = (
            existing["disabled"] if disabled is None and existing else
            bool(disabled)
        )
        if patch is None:
            new_patch = existing["patch"] if existing else {}
        else:
            new_patch = self._sanitize_patch(patch)
        await self._conn.execute(
            "INSERT INTO user_atlas_overrides "
            "(user_id, atlas_id, disabled, patch, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(user_id, atlas_id) DO UPDATE SET "
            "disabled = excluded.disabled, patch = excluded.patch, "
            "updated_at = excluded.updated_at",
            (user_id, atlas_id, int(new_disabled), json.dumps(new_patch)),
        )
        await self._conn.commit()
        return {
            "atlas_id": atlas_id,
            "disabled": new_disabled,
            "patch": new_patch,
        }

    async def _get_override(
        self, atlas_id: str, *, user_id: str,
    ) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            "SELECT disabled, patch FROM user_atlas_overrides "
            "WHERE user_id = ? AND atlas_id = ?",
            (user_id, atlas_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        try:
            parsed = json.loads(row[1]) if row[1] else {}
        except Exception:
            parsed = {}
        return {"disabled": bool(row[0]), "patch": parsed}

    async def clear_override(
        self, atlas_id: str, *, user_id: str = "",
    ) -> bool:
        """Drop the override row — bundled entry returns to defaults.
        Returns True if a row was deleted."""
        if not user_id:
            raise ValueError("atlas_overrides clear requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM user_atlas_overrides "
            "WHERE user_id = ? AND atlas_id = ?",
            (user_id, atlas_id),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)
