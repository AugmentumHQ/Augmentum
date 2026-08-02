"""SQLite persistence for the device substrate.

Three stores share this file because they share the same access patterns
and read each other's rows in places (registry merges discovery results
against saved devices; play history references device_id).

User scoping is enforced on every read. The auth/pairing JSON blobs are
encrypted at rest using the same Fernet helper that protects media-server
access tokens. Drivers never see ciphertext.
"""

from __future__ import annotations

import json
import secrets
from typing import TYPE_CHECKING, Any

from augmentum.devices.device import Device
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def _dump(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return default


def _load(text: str, default: Any) -> Any:
    if not text:
        return default
    try:
        loaded = json.loads(text)
        if loaded is None:
            return default
        return loaded
    except (json.JSONDecodeError, TypeError):
        return default


def _encrypt_blob(plaintext: dict[str, Any]) -> str:
    if not plaintext:
        return ""
    encoded = json.dumps(plaintext)
    return encrypt_api_key(encoded) or ""


def _decrypt_blob(ciphertext: str) -> dict[str, Any]:
    if not ciphertext:
        return {}
    decoded = decrypt_api_key(ciphertext) or ""
    if not decoded:
        return {}
    try:
        loaded = json.loads(decoded)
        if isinstance(loaded, dict):
            return loaded
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ----------------------------------------------------------------------------
# saved_devices
# ----------------------------------------------------------------------------


class DeviceStore:
    _SELECT_COLS = (
        "id, user_id, driver, native_id, label, capabilities, address, "
        "auth, status, last_seen_at, metadata, config, bindings, "
        "created_at, updated_at"
    )

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    def _row_to_device(self, row) -> Device:
        (
            id_, user_id, driver, native_id, label, capabilities_json,
            address_json, auth_blob, status, last_seen_at, metadata_json,
            config_json, bindings_json, created_at, updated_at,
        ) = row
        return Device(
            id=id_,
            user_id=user_id,
            driver=driver,
            native_id=native_id,
            label=label,
            capabilities=_load(capabilities_json, []),
            address=_load(address_json, {}),
            auth=_decrypt_blob(auth_blob),
            status=status,
            last_seen_at=last_seen_at or "",
            metadata=_load(metadata_json, {}),
            config=_load(config_json, {}),
            bindings=_load(bindings_json, []),
            created_at=created_at or "",
            updated_at=updated_at or "",
        )

    async def upsert(self, device: Device, *, user_id: str) -> Device:
        if not user_id or device.user_id != user_id:
            raise ValueError("device upsert requires matching user_id")
        await self._conn.execute(
            "INSERT INTO saved_devices "
            "(id, user_id, driver, native_id, label, capabilities, address, "
            " auth, status, last_seen_at, metadata, config, bindings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, driver, native_id) DO UPDATE SET "
            "  label = excluded.label, "
            "  capabilities = excluded.capabilities, "
            "  address = excluded.address, "
            "  auth = excluded.auth, "
            "  status = excluded.status, "
            "  last_seen_at = excluded.last_seen_at, "
            "  metadata = excluded.metadata, "
            "  config = excluded.config, "
            "  bindings = excluded.bindings, "
            "  updated_at = datetime('now')",
            (
                device.id,
                user_id,
                device.driver,
                device.native_id,
                device.label,
                _dump(device.capabilities, "[]"),
                _dump(device.address, "{}"),
                _encrypt_blob(device.auth or {}),
                device.status,
                device.last_seen_at,
                _dump(device.metadata, "{}"),
                _dump(device.config, "{}"),
                _dump(device.bindings, "[]"),
            ),
        )
        await self._conn.commit()
        log.info(
            "device_upsert",
            id=device.id,
            user_id=user_id,
            driver=device.driver,
            label=device.label,
        )
        result = await self.get(device.id, user_id=user_id)
        assert result is not None
        return result

    async def get(self, device_id: str, *, user_id: str) -> Device | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM saved_devices "
            "WHERE id = ? AND user_id = ?",
            (device_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_device(row) if row else None

    async def find_by_native(
        self,
        *,
        user_id: str,
        driver: str,
        native_id: str,
    ) -> Device | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM saved_devices "
            "WHERE user_id = ? AND driver = ? AND native_id = ?",
            (user_id, driver, native_id),
        )
        row = await cursor.fetchone()
        return self._row_to_device(row) if row else None

    async def list_for_user(self, *, user_id: str) -> list[Device]:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM saved_devices "
            "WHERE user_id = ? ORDER BY label COLLATE NOCASE ASC",
            (user_id,),
        )
        return [self._row_to_device(r) for r in await cursor.fetchall()]

    async def update_fields(
        self,
        device_id: str,
        *,
        user_id: str,
        label: str | None = None,
        status: str | None = None,
        last_seen_at: str | None = None,
        capabilities: list[str] | None = None,
        address: dict[str, Any] | None = None,
        auth: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        bindings: list[dict[str, Any]] | None = None,
    ) -> Device | None:
        fields: list[str] = []
        params: list[object] = []
        if label is not None:
            fields.append("label = ?")
            params.append(label)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if last_seen_at is not None:
            fields.append("last_seen_at = ?")
            params.append(last_seen_at)
        if capabilities is not None:
            fields.append("capabilities = ?")
            params.append(_dump(capabilities, "[]"))
        if address is not None:
            fields.append("address = ?")
            params.append(_dump(address, "{}"))
        if auth is not None:
            fields.append("auth = ?")
            params.append(_encrypt_blob(auth))
        if metadata is not None:
            fields.append("metadata = ?")
            params.append(_dump(metadata, "{}"))
        if config is not None:
            fields.append("config = ?")
            params.append(_dump(config, "{}"))
        if bindings is not None:
            fields.append("bindings = ?")
            params.append(_dump(bindings, "[]"))
        if not fields:
            return await self.get(device_id, user_id=user_id)

        fields.append("updated_at = datetime('now')")
        params.extend([device_id, user_id])
        await self._conn.execute(
            f"UPDATE saved_devices SET {', '.join(fields)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get(device_id, user_id=user_id)

    async def delete(self, device_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM saved_devices WHERE id = ? AND user_id = ?",
            (device_id, user_id),
        )
        await self._conn.commit()
        ok = cursor.rowcount > 0
        if ok:
            log.info("device_deleted", id=device_id, user_id=user_id)
        return ok


# ----------------------------------------------------------------------------
# device_pairings
# ----------------------------------------------------------------------------


class DevicePairingStore:
    _SELECT_COLS = (
        "id, device_id, user_id, state, pairing_data, expires_at, "
        "created_at, updated_at"
    )

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    def _row_to_dict(self, row) -> dict[str, Any]:
        (
            pair_id, device_id, user_id, state, pairing_blob,
            expires_at, created_at, updated_at,
        ) = row
        return {
            "id": pair_id,
            "device_id": device_id,
            "user_id": user_id,
            "state": state,
            "pairing_data": _decrypt_blob(pairing_blob),
            "expires_at": expires_at or "",
            "created_at": created_at or "",
            "updated_at": updated_at or "",
        }

    async def create(
        self,
        *,
        device_id: str,
        user_id: str,
        state: str = "pending",
        pairing_data: dict[str, Any] | None = None,
        expires_at: str = "",
    ) -> dict[str, Any]:
        pair_id = f"pair_{secrets.token_hex(8)}"
        await self._conn.execute(
            "INSERT INTO device_pairings "
            "(id, device_id, user_id, state, pairing_data, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                pair_id,
                device_id,
                user_id,
                state,
                _encrypt_blob(pairing_data or {}),
                expires_at,
            ),
        )
        await self._conn.commit()
        result = await self.get(pair_id, user_id=user_id)
        assert result is not None
        return result

    async def get(self, pair_id: str, *, user_id: str) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM device_pairings "
            "WHERE id = ? AND user_id = ?",
            (pair_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def get_active_for_device(
        self,
        device_id: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM device_pairings "
            "WHERE device_id = ? AND user_id = ? AND state = 'active' "
            "ORDER BY updated_at DESC LIMIT 1",
            (device_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def update(
        self,
        pair_id: str,
        *,
        user_id: str,
        state: str | None = None,
        pairing_data: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any] | None:
        fields: list[str] = []
        params: list[object] = []
        if state is not None:
            fields.append("state = ?")
            params.append(state)
        if pairing_data is not None:
            fields.append("pairing_data = ?")
            params.append(_encrypt_blob(pairing_data))
        if expires_at is not None:
            fields.append("expires_at = ?")
            params.append(expires_at)
        if not fields:
            return await self.get(pair_id, user_id=user_id)

        fields.append("updated_at = datetime('now')")
        params.extend([pair_id, user_id])
        await self._conn.execute(
            f"UPDATE device_pairings SET {', '.join(fields)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get(pair_id, user_id=user_id)

    async def delete(self, pair_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM device_pairings WHERE id = ? AND user_id = ?",
            (pair_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0


# ----------------------------------------------------------------------------
# device_play_history (MRU + favorites for smart-match)
# ----------------------------------------------------------------------------


class DevicePlayHistoryStore:
    """Tracks every cast event for MRU/favorites-driven smart-match.

    The smart-match heuristic for "play the lofi music on the living room TV":
      1. Favorites matching the query (is_favorite=1, content_kind+label match)
      2. Most-recently-played matching the query
      3. Most-played matching the query
      4. Arbitrary first match

    All four pivots query this table.
    """

    _SELECT_COLS = (
        "id, user_id, device_id, capability_id, action, file_id, "
        "content_key, content_label, content_kind, is_favorite, success, "
        "extra, created_at"
    )

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    def _row_to_dict(self, row) -> dict[str, Any]:
        (
            id_, user_id, device_id, capability_id, action, file_id,
            content_key, content_label, content_kind, is_favorite,
            success, extra_json, created_at,
        ) = row
        return {
            "id": id_,
            "user_id": user_id,
            "device_id": device_id,
            "capability_id": capability_id,
            "action": action,
            "file_id": file_id or "",
            "content_key": content_key or "",
            "content_label": content_label or "",
            "content_kind": content_kind or "",
            "is_favorite": bool(is_favorite),
            "success": bool(success),
            "extra": _load(extra_json, {}),
            "created_at": created_at or "",
        }

    async def log(
        self,
        *,
        user_id: str,
        device_id: str,
        capability_id: str,
        action: str = "play",
        file_id: str = "",
        content_key: str = "",
        content_label: str = "",
        content_kind: str = "",
        success: bool = True,
        is_favorite: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry_id = f"dph_{secrets.token_hex(8)}"
        await self._conn.execute(
            "INSERT INTO device_play_history "
            "(id, user_id, device_id, capability_id, action, file_id, "
            " content_key, content_label, content_kind, is_favorite, "
            " success, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry_id,
                user_id,
                device_id,
                capability_id,
                action,
                file_id,
                content_key,
                content_label,
                content_kind,
                1 if is_favorite else 0,
                1 if success else 0,
                _dump(extra, "{}"),
            ),
        )
        await self._conn.commit()
        result = await self._get(entry_id, user_id=user_id)
        assert result is not None
        return result

    async def _get(self, entry_id: str, *, user_id: str) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM device_play_history "
            "WHERE id = ? AND user_id = ?",
            (entry_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_dict(row) if row else None

    async def recent_for_kind(
        self,
        *,
        user_id: str,
        content_kind: str = "",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        if content_kind:
            cursor = await self._conn.execute(
                f"SELECT {self._SELECT_COLS} FROM device_play_history "
                "WHERE user_id = ? AND content_kind = ? AND success = 1 "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, content_kind, max(1, int(limit))),
            )
        else:
            cursor = await self._conn.execute(
                f"SELECT {self._SELECT_COLS} FROM device_play_history "
                "WHERE user_id = ? AND success = 1 "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, max(1, int(limit))),
            )
        return [self._row_to_dict(r) for r in await cursor.fetchall()]

    async def favorites_for_kind(
        self,
        *,
        user_id: str,
        content_kind: str = "",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        if content_kind:
            cursor = await self._conn.execute(
                f"SELECT {self._SELECT_COLS} FROM device_play_history "
                "WHERE user_id = ? AND content_kind = ? AND is_favorite = 1 "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, content_kind, max(1, int(limit))),
            )
        else:
            cursor = await self._conn.execute(
                f"SELECT {self._SELECT_COLS} FROM device_play_history "
                "WHERE user_id = ? AND is_favorite = 1 "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, max(1, int(limit))),
            )
        return [self._row_to_dict(r) for r in await cursor.fetchall()]

    async def play_count(
        self,
        *,
        user_id: str,
        content_key: str,
    ) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM device_play_history "
            "WHERE user_id = ? AND content_key = ? AND success = 1",
            (user_id, content_key),
        )
        row = await cursor.fetchone()
        return int(row[0] if row else 0)

    async def set_favorite(
        self,
        *,
        user_id: str,
        content_key: str,
        is_favorite: bool,
    ) -> int:
        """Mark all rows for a content_key as favorite (or not).

        Returns the number of rows affected.
        """
        if not content_key:
            return 0
        cursor = await self._conn.execute(
            "UPDATE device_play_history SET is_favorite = ? "
            "WHERE user_id = ? AND content_key = ?",
            (1 if is_favorite else 0, user_id, content_key),
        )
        await self._conn.commit()
        return cursor.rowcount or 0
