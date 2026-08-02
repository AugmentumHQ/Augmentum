"""SaveStore -- index over save blobs in the BlobStore.

Per-(user, title, kind, slot) UNIQUE. PUT semantics: writing a slot
that already exists releases the old blob (refcount-1) and replaces
the row with the new sha. Reads return the (record, bytes) tuple via
``get_with_bytes``; lists return records only (clients fetch bytes
separately when they decide to load).

The store enforces per-slot size caps but does NOT enforce per-user
total quotas -- that lives at the route layer where it can be tied to
user-facing settings.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


SAVE_KINDS: frozenset[str] = frozenset({"sram", "state", "screenshot"})

# Default cap per slot (50 MB). Settings layer can override via
# ``emulator_save_max_per_slot_mb``.
DEFAULT_MAX_PER_SLOT_BYTES = 50 * 1024 * 1024


class SaveServiceError(Exception):
    """Base for save-service errors. Carriers map to 400/404/413."""


class SaveTooLargeError(SaveServiceError):
    """Raised when a save blob exceeds the configured per-slot cap."""


# String enum-ish for kind. Keeping it as a string (not Enum) so SQL
# and JSON paths stay copy-friendly without import indirection.
class SaveKind:
    SRAM = "sram"
    STATE = "state"
    SCREENSHOT = "screenshot"


@dataclass(frozen=True)
class SaveRecord:
    id: str
    user_id: str
    artifact_id: str
    core_id: str
    kind: str
    slot: int
    sha256: str
    size_bytes: int
    label: str
    created_at: str
    updated_at: str
    # Couch co-op (Phase 4): when this save belongs to a named guest
    # under the host's account. Empty/None = host's own save (the
    # default for every row pre-Phase-4).
    guest_profile_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "artifact_id": self.artifact_id,
            "core_id": self.core_id,
            "kind": self.kind,
            "slot": self.slot,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "label": self.label,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "guest_profile_id": self.guest_profile_id,
        }


class SaveStore:
    """CRUD over ``game_saves`` rows + delegates blob I/O.

    ``blob_store`` is an ``augmentum.vfs.blobs.BlobStore`` instance.
    Tests can pass a fake (anything with the same write/release/get
    interface) without dragging the VFS layer in.
    """

    def __init__(self, conn: aiosqlite.Connection, blob_store: Any) -> None:
        self._conn = conn
        self._blobs = blob_store

    # ── Reads ────────────────────────────────────────────────────────

    async def list_for_title(
        self,
        *,
        user_id: str,
        artifact_id: str,
        kind: str | None = None,
        guest_profile_id: str | None = None,
    ) -> list[SaveRecord]:
        """List save rows for a title.

        ``guest_profile_id`` filters:
          * ``None`` (default): every row regardless of guest ownership.
            Used by the host's "manage saves" view to see everything.
          * ``""``: only the host's saves (NULL guest_profile_id).
            Used at runtime when the host is playing solo.
          * ``"gp_xxx"``: only that specific guest's saves.
        """
        if not user_id or not artifact_id:
            return []
        query = (
            "SELECT * FROM game_saves "
            "WHERE user_id = ? AND artifact_id = ?"
        )
        params: list[Any] = [user_id, artifact_id]
        if kind:
            if kind not in SAVE_KINDS:
                return []
            query += " AND kind = ?"
            params.append(kind)
        if guest_profile_id is not None:
            if guest_profile_id:
                query += " AND guest_profile_id = ?"
                params.append(guest_profile_id)
            else:
                query += " AND guest_profile_id IS NULL"
        query += " ORDER BY kind, slot"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_record(dict(zip(cols, r))) for r in rows]

    async def get(
        self,
        *,
        user_id: str,
        artifact_id: str,
        kind: str,
        slot: int,
        guest_profile_id: str = "",
    ) -> SaveRecord | None:
        """Read one save row. ``guest_profile_id="" `` matches the
        host's saves (NULL); pass an explicit ``gp_*`` id to read
        a guest's slot. Uniqueness is (user, artifact, kind, slot,
        guest_profile_id) per the Phase-4 index.
        """
        if guest_profile_id:
            sql = (
                "SELECT * FROM game_saves "
                "WHERE user_id = ? AND artifact_id = ? AND kind = ? "
                "AND slot = ? AND guest_profile_id = ?"
            )
            params: tuple = (
                user_id, artifact_id, kind, int(slot), guest_profile_id,
            )
        else:
            sql = (
                "SELECT * FROM game_saves "
                "WHERE user_id = ? AND artifact_id = ? AND kind = ? "
                "AND slot = ? AND guest_profile_id IS NULL"
            )
            params = (user_id, artifact_id, kind, int(slot))
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_record(dict(zip(cols, row)))

    async def get_with_bytes(
        self,
        *,
        user_id: str,
        artifact_id: str,
        kind: str,
        slot: int,
        guest_profile_id: str = "",
    ) -> tuple[SaveRecord, bytes] | None:
        record = await self.get(
            user_id=user_id, artifact_id=artifact_id,
            kind=kind, slot=slot, guest_profile_id=guest_profile_id,
        )
        if record is None:
            return None
        # BlobStore returns metadata; we read bytes off disk.
        meta = await self._blobs.get(record.sha256)
        if meta is None:
            return None
        try:
            with open(meta["real_path"], "rb") as f:
                data = f.read()
        except OSError as exc:
            log.warning(
                "save_blob_read_failed",
                sha=record.sha256, error=str(exc),
            )
            return None
        return record, data

    async def total_bytes_for_user(self, *, user_id: str) -> int:
        if not user_id:
            return 0
        cursor = await self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) "
            "FROM game_saves WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ── Writes ───────────────────────────────────────────────────────

    async def put(
        self,
        *,
        user_id: str,
        artifact_id: str,
        kind: str,
        slot: int,
        data: bytes,
        core_id: str = "",
        label: str = "",
        max_per_slot_bytes: int | None = None,
        guest_profile_id: str = "",
    ) -> SaveRecord:
        if not user_id or not artifact_id:
            raise SaveServiceError("user_id and artifact_id required")
        if kind not in SAVE_KINDS:
            raise SaveServiceError(
                f"unknown save kind {kind!r} (known: {sorted(SAVE_KINDS)})"
            )
        if not isinstance(data, (bytes, bytearray)):
            raise SaveServiceError("data must be bytes")
        if not data:
            raise SaveServiceError("save data is empty")
        cap = max_per_slot_bytes or DEFAULT_MAX_PER_SLOT_BYTES
        if len(data) > cap:
            raise SaveTooLargeError(
                f"save is {len(data)} bytes; cap is {cap}",
            )

        # Write the blob (dedup-safe; same bytes round-trip cheaply).
        mime = "application/octet-stream"
        if kind == SaveKind.SCREENSHOT:
            mime = "image/png"
        blob_meta = await self._blobs.write(bytes(data), mime_type=mime)
        new_sha = blob_meta["sha256"]

        # Look up the existing row so we can release the old blob if
        # the slot is being overwritten with a different sha. Scoped
        # to the guest if one's provided so alice's slot 0 doesn't
        # blow away the host's slot 0.
        existing = await self.get(
            user_id=user_id, artifact_id=artifact_id,
            kind=kind, slot=slot,
            guest_profile_id=guest_profile_id,
        )

        if existing is None:
            row_id = uuid.uuid4().hex[:16]
            await self._conn.execute(
                """INSERT INTO game_saves
                   (id, user_id, artifact_id, core_id, kind, slot,
                    sha256, size_bytes, label, guest_profile_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row_id, user_id, artifact_id, core_id, kind,
                    int(slot), new_sha, len(data), label,
                    guest_profile_id or None,
                ),
            )
            await self._conn.commit()
        else:
            # Update — scope the WHERE clause by guest_profile_id so
            # we update only the matching row (the per-guest UNIQUE
            # index guarantees uniqueness, but the WHERE has to
            # match on the same key for correctness).
            if guest_profile_id:
                update_sql = (
                    "UPDATE game_saves SET "
                    "sha256 = ?, size_bytes = ?, core_id = ?, "
                    "label = ?, updated_at = datetime('now') "
                    "WHERE user_id = ? AND artifact_id = ? "
                    "AND kind = ? AND slot = ? AND guest_profile_id = ?"
                )
                update_params: tuple = (
                    new_sha, len(data), core_id, label,
                    user_id, artifact_id, kind, int(slot),
                    guest_profile_id,
                )
            else:
                update_sql = (
                    "UPDATE game_saves SET "
                    "sha256 = ?, size_bytes = ?, core_id = ?, "
                    "label = ?, updated_at = datetime('now') "
                    "WHERE user_id = ? AND artifact_id = ? "
                    "AND kind = ? AND slot = ? AND guest_profile_id IS NULL"
                )
                update_params = (
                    new_sha, len(data), core_id, label,
                    user_id, artifact_id, kind, int(slot),
                )
            await self._conn.execute(update_sql, update_params)
            await self._conn.commit()
            if existing.sha256 == new_sha:
                # Same bytes re-uploaded. BlobStore.write dedup-bumped
                # the refcount, but conceptually we still have exactly
                # one reference (one row points at the blob). Release
                # once to keep the count honest.
                await self._blobs.release(new_sha)
            elif existing.sha256:
                # Old slot used different bytes -- release that blob.
                await self._blobs.release(existing.sha256)

        record = await self.get(
            user_id=user_id, artifact_id=artifact_id,
            kind=kind, slot=slot,
            guest_profile_id=guest_profile_id,
        )
        # get() must succeed since we just wrote; assert for clarity
        assert record is not None
        return record

    async def delete(
        self,
        *,
        user_id: str,
        artifact_id: str,
        kind: str,
        slot: int,
        guest_profile_id: str = "",
    ) -> bool:
        existing = await self.get(
            user_id=user_id, artifact_id=artifact_id,
            kind=kind, slot=slot,
            guest_profile_id=guest_profile_id,
        )
        if existing is None:
            return False
        if guest_profile_id:
            await self._conn.execute(
                "DELETE FROM game_saves "
                "WHERE user_id = ? AND artifact_id = ? "
                "AND kind = ? AND slot = ? AND guest_profile_id = ?",
                (
                    user_id, artifact_id, kind, int(slot),
                    guest_profile_id,
                ),
            )
        else:
            await self._conn.execute(
                "DELETE FROM game_saves "
                "WHERE user_id = ? AND artifact_id = ? "
                "AND kind = ? AND slot = ? AND guest_profile_id IS NULL",
                (user_id, artifact_id, kind, int(slot)),
            )
        await self._conn.commit()
        if existing.sha256:
            await self._blobs.release(existing.sha256)
        return True

    async def delete_all_for_title(
        self, *, user_id: str, artifact_id: str,
    ) -> int:
        """Cascade-delete every save for a title. Used when the user
        deletes the title artifact. Releases all blobs."""
        records = await self.list_for_title(
            user_id=user_id, artifact_id=artifact_id,
        )
        for rec in records:
            if rec.sha256:
                await self._blobs.release(rec.sha256)
        cursor = await self._conn.execute(
            "DELETE FROM game_saves "
            "WHERE user_id = ? AND artifact_id = ?",
            (user_id, artifact_id),
        )
        await self._conn.commit()
        return cursor.rowcount or 0


def _row_to_record(row: dict) -> SaveRecord:
    return SaveRecord(
        id=str(row.get("id", "")),
        user_id=str(row.get("user_id", "")),
        artifact_id=str(row.get("artifact_id", "")),
        core_id=str(row.get("core_id", "")),
        kind=str(row.get("kind", "")),
        slot=int(row.get("slot", 0)),
        sha256=str(row.get("sha256", "")),
        size_bytes=int(row.get("size_bytes", 0)),
        label=str(row.get("label", "")),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
        # NULL on pre-Phase-4 rows; cast to "" so callers don't have
        # to special-case None vs empty.
        guest_profile_id=str(row.get("guest_profile_id") or ""),
    )
