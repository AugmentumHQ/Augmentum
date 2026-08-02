"""BiosStore -- index over user-installed BIOS blobs.

Mirrors the SaveStore pattern: bytes live in the BlobStore (refcount-
tracked, content-addressed, dedup'd across users), this table records
which user installed which canonical BIOS for which system.

Public API:
    install(user_id, system_id, canonical_filename, data, ...)
    get(user_id, system_id, canonical_filename)
    list_for_user(user_id, system_id=None)
    list_status(user_id, system_id) -> per-file present/missing
    delete(user_id, system_id, canonical_filename)
    delete_all_for_user(user_id) -- account-deletion cascade

The store is hash-honest: install() expects pre-classified bytes
(the bulk-import classifier did the BIOS recognition upstream) and
just records them. It does NOT re-validate the bytes against the
catalog; the caller is responsible for that. This keeps the store
mechanical and lets the classifier own the policy.

Store-first (migration 324): a file the user drops on a system row is
stored whether or not we can identify it, exactly as RetroArch's
System folder and EmuDeck's ``emulation/bios`` accept anything copied
into them. Identification produces ``verify_status`` -- a label the UI
renders -- rather than an admission decision. That means the store now
holds files with no catalog slot, so :meth:`list_status` reports
catalog slots AND extras; anything else would make a stored file
invisible in the very panel that manages it.
"""

from __future__ import annotations

import hashlib
import os
import uuid
import zlib
from dataclasses import dataclass
from typing import Any

import aiosqlite

from augmentum.titles.bios_catalog import (
    BiosFile,
    all_for_system,
    required_for_system,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class BiosServiceError(Exception):
    """Base for BIOS-store errors."""


# How much a given match provenance actually proves. A cryptographic
# digest identifies the bytes; a filename+size agreement identifies the
# slot but not the contents; anything else is the user's assertion,
# which we honour and label honestly.
_VERIFY_BY_MATCH: dict[str, str] = {
    "sha1": "verified",
    "md5": "verified",
    "crc32": "verified",
    "name_size": "named",
    "manual": "named",
    "user_asserted": "unverified",
}


@dataclass(frozen=True)
class BiosRecord:
    id: str
    user_id: str
    system_id: str
    canonical_filename: str
    blob_sha256: str
    original_filename: str
    sha1: str
    size_bytes: int
    matched_by: str          # 'sha1'|'md5'|'crc32'|'name_size'|'manual'|'user_asserted'
    installed_at: str
    verify_status: str = "unverified"  # 'verified' | 'named' | 'unverified'
    md5: str = ""
    crc32: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "system_id": self.system_id,
            "canonical_filename": self.canonical_filename,
            "blob_sha256": self.blob_sha256,
            "original_filename": self.original_filename,
            "sha1": self.sha1,
            "md5": self.md5,
            "crc32": self.crc32,
            "size_bytes": self.size_bytes,
            "matched_by": self.matched_by,
            "verify_status": self.verify_status,
            "installed_at": self.installed_at,
        }


@dataclass(frozen=True)
class BiosStatusEntry:
    """One row in the per-system BIOS status panel."""
    system_id: str
    canonical_filename: str
    description: str
    optional: bool
    present: bool
    matched_by: str          # '' when not present
    installed_filename: str  # '' when not present (the original name)
    verify_status: str = ""  # '' when not present
    size_bytes: int = 0
    # True for a stored file with no catalog slot -- an "extra". The
    # vault lists these under the system so a store-first install is
    # never invisible in the panel that manages it.
    is_extra: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "canonical_filename": self.canonical_filename,
            "description": self.description,
            "optional": self.optional,
            "present": self.present,
            "matched_by": self.matched_by,
            "installed_filename": self.installed_filename,
            "verify_status": self.verify_status,
            "size_bytes": self.size_bytes,
            "is_extra": self.is_extra,
        }


class BiosStore:
    def __init__(self, conn: aiosqlite.Connection, blob_store: Any) -> None:
        self._conn = conn
        self._blobs = blob_store

    # ── Reads ────────────────────────────────────────────────────────

    async def get(
        self,
        *,
        user_id: str,
        system_id: str,
        canonical_filename: str,
    ) -> BiosRecord | None:
        if not user_id:
            return None
        cursor = await self._conn.execute(
            "SELECT * FROM user_bios_files "
            "WHERE user_id = ? AND system_id = ? AND canonical_filename = ?",
            (user_id, system_id, canonical_filename),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_record(dict(zip(cols, row, strict=False)))

    async def get_with_bytes(
        self,
        *,
        user_id: str,
        system_id: str,
        canonical_filename: str,
    ) -> tuple[BiosRecord, bytes] | None:
        record = await self.get(
            user_id=user_id, system_id=system_id,
            canonical_filename=canonical_filename,
        )
        if record is None:
            return None
        meta = await self._blobs.get(record.blob_sha256)
        if meta is None:
            return None
        try:
            with open(meta["real_path"], "rb") as f:
                data = f.read()
        except OSError as exc:
            log.warning(
                "bios_blob_read_failed",
                sha=record.blob_sha256, error=str(exc),
            )
            return None
        return record, data

    async def get_blob_path(
        self,
        *,
        user_id: str,
        system_id: str,
        canonical_filename: str,
    ) -> tuple[BiosRecord, str] | None:
        """Like :meth:`get_with_bytes` but returns the on-disk path
        instead of the bytes, so the serve route can stream the file
        (Range-aware, no full-file RAM load -- matters for the big
        modern firmware blobs the catalog lists, e.g. DSi NAND ~240 MB).
        Returns ``(record, real_path)`` or None when not installed /
        the backing blob is missing.
        """
        record = await self.get(
            user_id=user_id, system_id=system_id,
            canonical_filename=canonical_filename,
        )
        if record is None:
            return None
        meta = await self._blobs.get(record.blob_sha256)
        if meta is None or not meta.get("real_path"):
            return None
        real_path = str(meta["real_path"])
        if not os.path.isfile(real_path):
            log.warning(
                "bios_blob_file_missing",
                sha=record.blob_sha256, path=real_path,
            )
            return None
        return record, real_path

    async def list_for_user(
        self, *, user_id: str, system_id: str | None = None,
    ) -> list[BiosRecord]:
        if not user_id:
            return []
        if system_id:
            cursor = await self._conn.execute(
                "SELECT * FROM user_bios_files "
                "WHERE user_id = ? AND system_id = ? "
                "ORDER BY system_id, canonical_filename",
                (user_id, system_id),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM user_bios_files WHERE user_id = ? "
                "ORDER BY system_id, canonical_filename",
                (user_id,),
            )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_record(dict(zip(cols, r, strict=False))) for r in rows]

    async def list_status(
        self, *, user_id: str, system_id: str,
    ) -> list[BiosStatusEntry]:
        """Per-file present/missing for the BIOS panel.

        Combines the catalog (what should be there) with the store
        (what the user actually installed) to produce one row per
        catalogued file, THEN appends every stored file that has no
        catalog slot as an 'extra'.

        The extras pass is load-bearing, not cosmetic. Under
        store-first (migration 324) a user can install a firmware dump
        we cannot identify -- a regional PS2 revision, a redump that
        differs by a byte, a console-specific NAND. Iterating the
        catalog alone would store that file, charge a blob refcount
        for it, serve it to the emulator, and render nothing in the
        vault: no delete button, no way to tell it apart from a failed
        upload. The user would reasonably conclude the upload silently
        failed -- which is the exact bug store-first exists to kill.
        """
        catalog = all_for_system(system_id)
        installed = {
            r.canonical_filename: r
            for r in await self.list_for_user(
                user_id=user_id, system_id=system_id,
            )
        }
        out: list[BiosStatusEntry] = []
        for spec in catalog:
            rec = installed.pop(spec.filename, None)
            out.append(BiosStatusEntry(
                system_id=spec.system_id,
                canonical_filename=spec.filename,
                description=spec.description,
                optional=spec.optional,
                present=rec is not None,
                matched_by=rec.matched_by if rec else "",
                installed_filename=rec.original_filename if rec else "",
                verify_status=rec.verify_status if rec else "",
                size_bytes=rec.size_bytes if rec else spec.size_bytes,
            ))

        # Whatever is left in `installed` occupies no catalog slot.
        for rec in sorted(installed.values(), key=lambda r: r.canonical_filename):
            out.append(BiosStatusEntry(
                system_id=system_id,
                canonical_filename=rec.canonical_filename,
                description="Stored by you — not in the known-BIOS database",
                optional=True,      # never gates launch
                present=True,
                matched_by=rec.matched_by,
                installed_filename=rec.original_filename,
                verify_status=rec.verify_status,
                size_bytes=rec.size_bytes,
                is_extra=True,
            ))
        return out

    async def missing_required(
        self, *, user_id: str, system_id: str,
    ) -> list[BiosFile]:
        """Required-but-not-installed BIOS files. Used by the launch
        path to refuse to boot with an actionable error."""
        installed = {
            r.canonical_filename
            for r in await self.list_for_user(
                user_id=user_id, system_id=system_id,
            )
        }
        return [
            f for f in required_for_system(system_id)
            if f.filename not in installed
        ]

    # ── Writes ───────────────────────────────────────────────────────

    async def install(
        self,
        *,
        user_id: str,
        system_id: str,
        canonical_filename: str,
        data: bytes,
        original_filename: str = "",
        sha1: str = "",
        matched_by: str = "sha1",
        verify_status: str = "",
        md5: str = "",
        crc32: str = "",
    ) -> BiosRecord:
        """Idempotent install. Re-installing the same canonical slot
        replaces the existing row + releases the old blob (only when
        the bytes differ; identical bytes round-trip cheaply since
        BlobStore dedup'd them on write).

        ``canonical_filename`` no longer has to name a catalog slot.
        Under store-first it is simply the name the file will be served
        under, which for an unidentified dump is the name the user
        uploaded. ``verify_status`` records how much we actually know:
        derived from ``matched_by`` when the caller doesn't say.
        """
        if not user_id:
            raise BiosServiceError("user_id required")
        if not system_id:
            raise BiosServiceError("system_id required")
        if not canonical_filename:
            raise BiosServiceError("canonical_filename required")
        if not isinstance(data, bytes | bytearray):
            raise BiosServiceError("data must be bytes")
        if not data:
            raise BiosServiceError("BIOS data is empty")

        # Compute the digests the caller didn't supply (defensive; the
        # classifier supplies all three, the manual-override path may
        # supply none). We record all three even when only one made
        # the match, so a later hash-database refresh can upgrade an
        # 'unverified' row in place without asking for a re-upload.
        if not sha1:
            sha1 = hashlib.sha1(data).hexdigest()
        if not md5:
            md5 = hashlib.md5(data).hexdigest()
        if not crc32:
            crc32 = format(zlib.crc32(bytes(data)) & 0xFFFFFFFF, "08x")
        if not verify_status:
            verify_status = _VERIFY_BY_MATCH.get(matched_by, "unverified")

        blob_meta = await self._blobs.write(
            bytes(data), mime_type="application/octet-stream",
        )
        new_sha = blob_meta["sha256"]

        existing = await self.get(
            user_id=user_id, system_id=system_id,
            canonical_filename=canonical_filename,
        )

        if existing is None:
            row_id = uuid.uuid4().hex[:16]
            await self._conn.execute(
                """INSERT INTO user_bios_files
                   (id, user_id, system_id, canonical_filename,
                    blob_sha256, original_filename, sha1, size_bytes,
                    matched_by, verify_status, md5, crc32)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row_id, user_id, system_id, canonical_filename,
                    new_sha, original_filename, sha1, len(data),
                    matched_by, verify_status, md5, crc32,
                ),
            )
            await self._conn.commit()
        else:
            await self._conn.execute(
                """UPDATE user_bios_files SET
                     blob_sha256 = ?, original_filename = ?, sha1 = ?,
                     size_bytes = ?, matched_by = ?, verify_status = ?,
                     md5 = ?, crc32 = ?,
                     installed_at = datetime('now')
                   WHERE user_id = ? AND system_id = ?
                     AND canonical_filename = ?""",
                (
                    new_sha, original_filename, sha1, len(data),
                    matched_by, verify_status, md5, crc32,
                    user_id, system_id, canonical_filename,
                ),
            )
            await self._conn.commit()
            if existing.blob_sha256 == new_sha:
                # Same bytes re-installed. BlobStore.write bumped the
                # refcount; we still own one logical reference. Release
                # once to keep the count honest.
                await self._blobs.release(new_sha)
            elif existing.blob_sha256:
                await self._blobs.release(existing.blob_sha256)

        record = await self.get(
            user_id=user_id, system_id=system_id,
            canonical_filename=canonical_filename,
        )
        assert record is not None
        log.info(
            "bios_installed",
            user_id=user_id, system=system_id,
            canonical=canonical_filename, sha1=sha1,
            matched_by=matched_by,
        )
        return record

    async def delete(
        self,
        *,
        user_id: str,
        system_id: str,
        canonical_filename: str,
    ) -> bool:
        existing = await self.get(
            user_id=user_id, system_id=system_id,
            canonical_filename=canonical_filename,
        )
        if existing is None:
            return False
        await self._conn.execute(
            "DELETE FROM user_bios_files "
            "WHERE user_id = ? AND system_id = ? AND canonical_filename = ?",
            (user_id, system_id, canonical_filename),
        )
        await self._conn.commit()
        if existing.blob_sha256:
            await self._blobs.release(existing.blob_sha256)
        return True

    async def delete_all_for_user(self, *, user_id: str) -> int:
        """Account-deletion cascade. Releases every BIOS blob the
        user owned."""
        records = await self.list_for_user(user_id=user_id)
        for rec in records:
            if rec.blob_sha256:
                await self._blobs.release(rec.blob_sha256)
        cursor = await self._conn.execute(
            "DELETE FROM user_bios_files WHERE user_id = ?", (user_id,),
        )
        await self._conn.commit()
        return cursor.rowcount or 0


def _row_to_record(row: dict) -> BiosRecord:
    return BiosRecord(
        id=str(row.get("id", "")),
        user_id=str(row.get("user_id", "")),
        system_id=str(row.get("system_id", "")),
        canonical_filename=str(row.get("canonical_filename", "")),
        blob_sha256=str(row.get("blob_sha256", "")),
        original_filename=str(row.get("original_filename", "")),
        sha1=str(row.get("sha1", "")),
        size_bytes=int(row.get("size_bytes", 0)),
        matched_by=str(row.get("matched_by", "")),
        installed_at=str(row.get("installed_at", "")),
        verify_status=str(row.get("verify_status", "") or "unverified"),
        md5=str(row.get("md5", "") or ""),
        crc32=str(row.get("crc32", "") or ""),
    )
