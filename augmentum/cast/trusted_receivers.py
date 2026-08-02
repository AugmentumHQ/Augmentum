"""Trusted-receiver store — persistent identity for paired TVs.

Bridges the runtime ReceiverRegistry (in-memory, ephemeral) and the
durable user-management surface ("my TVs"). Every receiver that
supplies a stable ``device_id`` on its ``ready`` event gets a row
here; reconnects rebind to the same row so the user-visible label,
created-at, last-seen, and revocation state survive reboots.

Browser-tab receivers (no device_id) intentionally bypass this — they
were always ephemeral and treating them as durable would clutter the
user's "my TVs" view with one row per browser session.

Multi-tenant invariants per the augmentum-dev rule:
  - Every method takes ``*, user_id: str = ""`` and appends
    ``AND user_id = ?`` to all queries
  - Cross-user reads return None / empty list
  - revoke is user-scoped — only the owner can disable a receiver
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _make_id() -> str:
    return f"tr_{secrets.token_hex(8)}"


# Canonical MAC = lowercase colon-separated octets. Accept any of the
# usual wire formats the receiver app (or a user pasting from a router
# admin page) might produce: AA-BB-CC-DD-EE-FF, AABBCCDDEEFF,
# AABB.CCDD.EEFF, mixed case, surrounding whitespace.
_MAC_HEX_RE = re.compile(r"^[0-9a-f]{12}$")


def normalise_mac(raw: str) -> str:
    """Return canonical ``aa:bb:cc:dd:ee:ff`` or '' when invalid.

    Tolerates the three common write-styles seen in router UIs and
    pasted serial-debug output. Empty input passes through as ''.
    """
    if not raw:
        return ""
    stripped = "".join(ch for ch in str(raw).lower() if ch in "0123456789abcdef")
    if not _MAC_HEX_RE.fullmatch(stripped):
        return ""
    return ":".join(stripped[i:i + 2] for i in (0, 2, 4, 6, 8, 10))


# IPv4 = dotted quad of 0..255. Returns canonical form ('' = invalid).
_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


def normalise_ipv4(raw: str) -> str:
    if not raw:
        return ""
    m = _IPV4_RE.fullmatch(str(raw).strip())
    if not m:
        return ""
    octets = [int(o) for o in m.groups()]
    if any(o > 255 for o in octets):
        return ""
    return ".".join(str(o) for o in octets)


@dataclass(slots=True)
class TrustedReceiver:
    id: str
    user_id: str
    label: str
    platform: str
    device_id: str
    info: dict[str, Any]
    created_at: str
    last_seen_at: str
    last_cast_at: str
    revoked_at: str
    prefs: dict[str, Any]
    mac_address: str = ""
    last_local_ip: str = ""
    wol_broadcast_override: str = ""

    @property
    def is_revoked(self) -> bool:
        return bool(self.revoked_at)

    def to_dict(self, *, include_user_id: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label or self.id,
            "platform": self.platform,
            "device_id": self.device_id,
            "info": dict(self.info or {}),
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "last_cast_at": self.last_cast_at,
            "revoked": self.is_revoked,
            "revoked_at": self.revoked_at,
            "mac_address": self.mac_address,
            "last_local_ip": self.last_local_ip,
            "wol_broadcast_override": self.wol_broadcast_override,
            "wol_ready": bool(self.mac_address),
        }
        if include_user_id:
            out["user_id"] = self.user_id
        return out


def _row_to_record(row: Any) -> TrustedReceiver:
    info_raw = row[5] if not isinstance(row, dict) else row["info"]
    try:
        info = json.loads(info_raw) if info_raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        info = {}
    # prefs_json column (added in migration 188). Tolerate older rows
    # that predate the migration — they read as the empty bag and the
    # client merges defaults on top.
    prefs_raw = row[10] if not isinstance(row, dict) else row.get("prefs_json", "")
    try:
        prefs = json.loads(prefs_raw) if prefs_raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        prefs = {}
    if not isinstance(prefs, dict):
        prefs = {}
    return TrustedReceiver(
        id=row[0] if not isinstance(row, dict) else row["id"],
        user_id=row[1] if not isinstance(row, dict) else row["user_id"],
        label=row[2] if not isinstance(row, dict) else row["label"],
        platform=row[3] if not isinstance(row, dict) else row["platform"],
        device_id=row[4] if not isinstance(row, dict) else row["device_id"],
        info=info,
        created_at=row[6] if not isinstance(row, dict) else row["created_at"],
        last_seen_at=row[7] if not isinstance(row, dict) else row["last_seen_at"],
        last_cast_at=row[8] if not isinstance(row, dict) else row["last_cast_at"],
        revoked_at=row[9] if not isinstance(row, dict) else row["revoked_at"],
        prefs=prefs,
        mac_address=(row[11] if not isinstance(row, dict)
                     else row.get("mac_address", "")) or "",
        last_local_ip=(row[12] if not isinstance(row, dict)
                       else row.get("last_local_ip", "")) or "",
        wol_broadcast_override=(row[13] if not isinstance(row, dict)
                                else row.get("wol_broadcast_override", "")) or "",
    )


_SELECT_COLS = (
    "id, user_id, label, platform, device_id, info, "
    "created_at, last_seen_at, last_cast_at, revoked_at, prefs_json, "
    "mac_address, last_local_ip, wol_broadcast_override"
)


class TrustedReceiverStore:
    """User-scoped CRUD over the trusted_receivers table."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def list_for_user(
        self, *, user_id: str, include_revoked: bool = False,
    ) -> list[TrustedReceiver]:
        if not user_id:
            return []
        query = f"SELECT {_SELECT_COLS} FROM trusted_receivers WHERE user_id = ?"
        if not include_revoked:
            query += " AND revoked_at = ''"
        query += " ORDER BY created_at DESC"
        cursor = await self._conn.execute(query, (user_id,))
        try:
            rows = await cursor.fetchall()
        finally:
            await cursor.close()
        return [_row_to_record(r) for r in rows]

    async def get(self, trusted_id: str, *, user_id: str) -> TrustedReceiver | None:
        if not trusted_id or not user_id:
            return None
        cursor = await self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM trusted_receivers "
            "WHERE id = ? AND user_id = ?",
            (trusted_id, user_id),
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        return _row_to_record(row) if row else None

    async def get_by_device(
        self, device_id: str, *, user_id: str,
    ) -> TrustedReceiver | None:
        """Look up by stable per-device id. Returns None when the device
        has never connected for this user, OR when it's revoked.
        """
        if not device_id or not user_id:
            return None
        cursor = await self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM trusted_receivers "
            "WHERE user_id = ? AND device_id = ?",
            (user_id, device_id),
        )
        try:
            row = await cursor.fetchone()
        finally:
            await cursor.close()
        return _row_to_record(row) if row else None

    async def upsert_on_connect(
        self,
        *,
        user_id: str,
        device_id: str,
        platform: str,
        info: dict[str, Any] | None = None,
        label: str = "",
    ) -> TrustedReceiver | None:
        """Called when a receiver connects with a stable device_id.
        Creates a new row if first-seen; otherwise refreshes
        last_seen_at + platform/info. Returns None when the device
        has been revoked (caller MUST reject the connection)."""
        if not user_id or not device_id:
            return None

        existing = await self.get_by_device(device_id, user_id=user_id)
        info_json = json.dumps(info or {}, separators=(",", ":"))
        now = _now_iso()

        # Receiver-supplied WoL hints, if present. The Android TV
        # bundle can read MAC + IP via WifiInfo and ships them on the
        # ready event; browser receivers can't and these stay empty
        # until the user types a MAC into the cast-control UI.
        ready_mac = normalise_mac(str((info or {}).get("mac_address") or ""))
        ready_ip = normalise_ipv4(str((info or {}).get("local_ip") or ""))

        if existing is None:
            new_id = _make_id()
            # Default label: derived from info / platform — user can
            # rename later via the UI.
            chosen_label = label or (info or {}).get("label") or platform or "TV"
            await self._conn.execute(
                "INSERT INTO trusted_receivers "
                "(id, user_id, label, platform, device_id, info, "
                " last_seen_at, mac_address, last_local_ip) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id, user_id, chosen_label, platform, device_id,
                 info_json, now, ready_mac, ready_ip),
            )
            await self._conn.commit()
            log.info(
                "trusted_receiver_created",
                trusted_id=new_id, user_id=user_id, device_id=device_id,
                platform=platform, label=chosen_label,
                has_mac=bool(ready_mac), has_ip=bool(ready_ip),
            )
            return await self.get(new_id, user_id=user_id)

        if existing.is_revoked:
            log.warning(
                "trusted_receiver_revoked_connect_attempt",
                trusted_id=existing.id, user_id=user_id, device_id=device_id,
            )
            return None

        # Refresh metadata that may have changed (platform upgrade,
        # new capabilities, etc.) but preserve the user-chosen label.
        # MAC + IP are auto-filled but never overwritten with empty —
        # a manually-entered MAC survives even when the receiver
        # downgrades to a non-reporting version.
        await self._conn.execute(
            "UPDATE trusted_receivers SET platform = ?, info = ?, last_seen_at = ?, "
            "mac_address = CASE WHEN ? = '' THEN mac_address ELSE ? END, "
            "last_local_ip = CASE WHEN ? = '' THEN last_local_ip ELSE ? END "
            "WHERE id = ? AND user_id = ?",
            (platform, info_json, now,
             ready_mac, ready_mac,
             ready_ip, ready_ip,
             existing.id, user_id),
        )
        await self._conn.commit()
        return await self.get(existing.id, user_id=user_id)

    async def update_label(
        self, trusted_id: str, *, user_id: str, label: str,
    ) -> bool:
        if not trusted_id or not user_id:
            return False
        cursor = await self._conn.execute(
            "UPDATE trusted_receivers SET label = ? "
            "WHERE id = ? AND user_id = ? AND revoked_at = ''",
            (label, trusted_id, user_id),
        )
        try:
            updated = cursor.rowcount > 0
        finally:
            await cursor.close()
        if updated:
            await self._conn.commit()
        return updated

    async def revoke(self, trusted_id: str, *, user_id: str) -> bool:
        """Mark a receiver as revoked. Subsequent connects from the
        same device_id are rejected. Returns True when the revocation
        was applied (False = unknown id or already revoked)."""
        if not trusted_id or not user_id:
            return False
        now = _now_iso()
        cursor = await self._conn.execute(
            "UPDATE trusted_receivers SET revoked_at = ? "
            "WHERE id = ? AND user_id = ? AND revoked_at = ''",
            (now, trusted_id, user_id),
        )
        try:
            updated = cursor.rowcount > 0
        finally:
            await cursor.close()
        if updated:
            await self._conn.commit()
            log.info(
                "trusted_receiver_revoked",
                trusted_id=trusted_id, user_id=user_id,
            )
        return updated

    async def restore(self, trusted_id: str, *, user_id: str) -> bool:
        """Reverse a revoke — the device can connect again under its
        existing trust entry. Returns True when restored (False = id
        unknown for this user, or row not currently revoked).

        Recovery path for accidental revokes — the cast-stage UI's
        two-step pattern occasionally races with the live-data poll
        re-render, so a user who 'first-clicked' before the page
        refreshed can see their next click commit immediately. This
        endpoint lets them recover without DB surgery.
        """
        if not trusted_id or not user_id:
            return False
        cursor = await self._conn.execute(
            "UPDATE trusted_receivers SET revoked_at = '' "
            "WHERE id = ? AND user_id = ? AND revoked_at != ''",
            (trusted_id, user_id),
        )
        try:
            updated = cursor.rowcount > 0
        finally:
            await cursor.close()
        if updated:
            await self._conn.commit()
            log.info(
                "trusted_receiver_restored",
                trusted_id=trusted_id, user_id=user_id,
            )
        return updated

    async def set_wol_address(
        self,
        trusted_id: str,
        *,
        user_id: str,
        mac_address: str | None = None,
        wol_broadcast_override: str | None = None,
    ) -> bool:
        """Update WoL fields. ``None`` = leave alone; ``''`` = clear.

        Returns True when at least one column was changed. The caller
        is expected to have already canonicalised inputs via
        ``normalise_mac`` / ``normalise_ipv4`` — bad values land here
        only if the user typed garbage and we want the 422 to fire at
        the route layer, not silently overwrite with broken data.
        """
        if not trusted_id or not user_id:
            return False

        sets: list[str] = []
        args: list[Any] = []
        if mac_address is not None:
            sets.append("mac_address = ?")
            args.append(mac_address)
        if wol_broadcast_override is not None:
            sets.append("wol_broadcast_override = ?")
            args.append(wol_broadcast_override)
        if not sets:
            return False
        args.extend([trusted_id, user_id])
        cursor = await self._conn.execute(
            f"UPDATE trusted_receivers SET {', '.join(sets)} "
            "WHERE id = ? AND user_id = ? AND revoked_at = ''",
            tuple(args),
        )
        try:
            updated = cursor.rowcount > 0
        finally:
            await cursor.close()
        if updated:
            await self._conn.commit()
        return updated

    async def mark_cast(self, trusted_id: str, *, user_id: str) -> None:
        """Bump last_cast_at — called when a surface is dispatched."""
        if not trusted_id or not user_id:
            return
        await self._conn.execute(
            "UPDATE trusted_receivers SET last_cast_at = ? "
            "WHERE id = ? AND user_id = ?",
            (_now_iso(), trusted_id, user_id),
        )
        await self._conn.commit()

    async def get_prefs(
        self, trusted_id: str, *, user_id: str,
    ) -> dict[str, Any] | None:
        """Read the raw prefs bag for a receiver. Returns the stored
        dict (possibly empty) when the row exists, or None when the
        trust id is unknown for this user. The route layer is
        responsible for merging defaults — we return the stored shape
        verbatim so the caller can tell apart 'never written' from
        'explicitly wrote False for everything'.
        """
        record = await self.get(trusted_id, user_id=user_id)
        if record is None:
            return None
        return dict(record.prefs or {})

    async def set_prefs(
        self,
        trusted_id: str,
        *,
        user_id: str,
        prefs: dict[str, Any],
    ) -> bool:
        """Replace the prefs bag wholesale. Callers MUST pass a dict
        already cleaned by ``receiver_prefs.coerce_prefs`` — we don't
        re-validate here so the store stays a thin CRUD layer.
        Returns True on success, False when the row doesn't exist for
        this user or is revoked.
        """
        if not trusted_id or not user_id:
            return False
        prefs_json = json.dumps(prefs or {}, separators=(",", ":"))
        cursor = await self._conn.execute(
            "UPDATE trusted_receivers SET prefs_json = ? "
            "WHERE id = ? AND user_id = ? AND revoked_at = ''",
            (prefs_json, trusted_id, user_id),
        )
        try:
            updated = cursor.rowcount > 0
        finally:
            await cursor.close()
        if updated:
            await self._conn.commit()
        return updated
