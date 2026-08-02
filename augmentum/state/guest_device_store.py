"""Cast couch co-op guest device fingerprint store (Phase 3).

Links browser-on-phone identities to guest profiles so a returning
guest auto-resumes as themselves without retyping their name. See
migration 230 and
``docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md``.

Fingerprint shape:
  * localStorage UUID — minted by the guest's browser on first visit
    and survives until they clear browser data. The strong signal.
  * UA hash — sha1ish of UA string + viewport. Defends against the
    "cleared localStorage but same device" case; requiring it to
    match keeps the welcome-back path tight.

The store does NOT touch IP addresses, canvas/audio fingerprints,
or any other invasive signal. Privacy thesis: guest can self-revoke
the link from the guest-side "Forget this device" control.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row, strict=False))


class GuestDeviceStore:
    """Persistence for guest device fingerprints."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Link / lookup ─────────────────────────────────────────────────

    async def match(
        self,
        *,
        host_user_id: str,
        device_uuid: str,
        ua_hash: str = "",
    ) -> dict[str, Any] | None:
        """Resolve a device fingerprint to the owning guest profile.

        Returns the joined guest_profiles row when (host_user_id,
        device_uuid) matches AND (when ``ua_hash`` is provided) the
        stored ua_hash matches. Mismatch on ua_hash returns None so
        the welcome-back path falls back to the name picker — the
        device may have legitimately changed user agents, but the
        guest can re-link via the explicit name pick.
        """
        if not host_user_id or not device_uuid:
            return None
        cursor = await self._conn.execute(
            """SELECT gd.*, gp.display_name AS display_name,
                      gp.color AS color
               FROM guest_devices gd
               JOIN guest_profiles gp ON gp.id = gd.guest_profile_id
               WHERE gd.host_user_id = ? AND gd.device_uuid = ?""",
            (host_user_id, device_uuid),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        data = _row_to_dict(cursor, row)
        if ua_hash and data.get("ua_hash") and data["ua_hash"] != ua_hash:
            # UA changed — treat as mismatch. Guest can re-link via
            # the name picker, which calls link_device with the new
            # UA hash and updates the row.
            return None
        # Best-effort touch — successful identify means the device
        # is in active use.
        try:
            await self._conn.execute(
                "UPDATE guest_devices SET last_seen_at = ? WHERE id = ?",
                (int(time.time()), data["id"]),
            )
            await self._conn.commit()
        except Exception as exc:
            log.warning(
                "guest_device_touch_failed",
                device_id=data["id"], error=str(exc)[:160],
            )
        return {
            "id": data["guest_profile_id"],
            "display_name": data["display_name"],
            "color": data.get("color", ""),
        }

    async def link_device(
        self,
        *,
        guest_profile_id: str,
        host_user_id: str,
        device_uuid: str,
        ua_hash: str = "",
        label: str = "",
    ) -> dict[str, Any]:
        """Create or update the link between a device and a profile.

        UPSERT semantics: if (host_user_id, device_uuid) exists,
        the row is rebound to the new ``guest_profile_id`` + the
        ua_hash is refreshed. This is the "not me?" re-link path —
        guest picks a different profile, the device row moves with
        them.

        Returns the canonical row dict.
        """
        if not (guest_profile_id and host_user_id and device_uuid):
            raise ValueError(
                "link_device requires guest_profile_id, host_user_id, device_uuid",
            )
        now = int(time.time())

        # Check existing.
        cursor = await self._conn.execute(
            """SELECT * FROM guest_devices
               WHERE host_user_id = ? AND device_uuid = ?""",
            (host_user_id, device_uuid),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            existing_d = _row_to_dict(cursor, existing)
            await self._conn.execute(
                """UPDATE guest_devices
                   SET guest_profile_id = ?, ua_hash = ?, label = ?,
                       last_seen_at = ?
                   WHERE id = ?""",
                (
                    guest_profile_id, ua_hash, label or existing_d["label"],
                    now, existing_d["id"],
                ),
            )
            await self._conn.commit()
            existing_d.update({
                "guest_profile_id": guest_profile_id,
                "ua_hash": ua_hash,
                "label": label or existing_d["label"],
                "last_seen_at": now,
            })
            log.info(
                "guest_device_relinked",
                device_id=existing_d["id"],
                guest_profile_id=guest_profile_id,
                host_user_id=host_user_id,
            )
            return existing_d

        device_id = f"gd_{secrets.token_hex(6)}"
        await self._conn.execute(
            """INSERT INTO guest_devices
               (id, guest_profile_id, host_user_id, device_uuid,
                ua_hash, label, first_seen_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                device_id, guest_profile_id, host_user_id, device_uuid,
                ua_hash, label, now, now,
            ),
        )
        await self._conn.commit()
        log.info(
            "guest_device_linked",
            device_id=device_id,
            guest_profile_id=guest_profile_id,
            host_user_id=host_user_id,
        )
        return {
            "id": device_id, "guest_profile_id": guest_profile_id,
            "host_user_id": host_user_id, "device_uuid": device_uuid,
            "ua_hash": ua_hash, "label": label,
            "first_seen_at": now, "last_seen_at": now,
        }

    async def forget(
        self,
        *,
        host_user_id: str,
        device_uuid: str,
    ) -> bool:
        """Drop the device row. Profile itself is preserved."""
        if not (host_user_id and device_uuid):
            return False
        cursor = await self._conn.execute(
            """DELETE FROM guest_devices
               WHERE host_user_id = ? AND device_uuid = ?""",
            (host_user_id, device_uuid),
        )
        await self._conn.commit()
        gone = (cursor.rowcount or 0) > 0
        if gone:
            log.info(
                "guest_device_forgotten",
                host_user_id=host_user_id,
                device_uuid_prefix=device_uuid[:8],
            )
        return gone

    async def list_for_profile(
        self,
        *,
        guest_profile_id: str,
        host_user_id: str,
    ) -> list[dict[str, Any]]:
        """List every device linked to a profile.

        Useful for the future host-side "Manage guests" screen and
        for tests. host_user_id scope is mandatory.
        """
        if not (guest_profile_id and host_user_id):
            return []
        cursor = await self._conn.execute(
            """SELECT * FROM guest_devices
               WHERE guest_profile_id = ? AND host_user_id = ?
               ORDER BY last_seen_at DESC""",
            (guest_profile_id, host_user_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]
