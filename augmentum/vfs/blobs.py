"""Content-addressed blob store.

Bytes are keyed by their SHA-256. Two logical files with identical content
share one on-disk blob — dedup is automatic. Reference counting drives
cleanup: every logical reference to a blob bumps refcount on insert and
decrements on delete; when refcount hits zero the physical file and the
blobs row both go away.

Storage layout:
    <data_dir>/blobs/<sha[:2]>/<sha[2:4]>/<sha>

Two 2-char hex shard levels cap any one directory at ~15 entries even at
1M blobs (16^4 = 65,536 shards). The hex digest is filename-safe, has no
dot/space issues, and doesn't need escaping on any OS. Total path length
stays well under the Windows 260-char limit at typical data_dir depths.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class BlobStore:
    """Async SQLite-backed content-addressed blob store."""

    def __init__(self, conn: aiosqlite.Connection, base_dir: str | None = None) -> None:
        self._conn = conn
        self._base = Path(base_dir) if base_dir else (Path(settings.data_dir) / "blobs")
        self._base.mkdir(parents=True, exist_ok=True)

    # --- Paths ---------------------------------------------------------

    def path_for(self, sha: str) -> Path:
        """Canonical filesystem path for a blob's bytes."""
        return self._base / sha[:2] / sha[2:4] / sha

    # --- Metadata ------------------------------------------------------

    async def get(self, sha: str) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT sha256, size_bytes, mime_type, real_path, refcount "
            "FROM blobs WHERE sha256 = ?",
            (sha,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "sha256": row[0],
            "size_bytes": row[1],
            "mime_type": row[2] or "",
            "real_path": row[3],
            "refcount": row[4],
        }

    async def exists(self, sha: str) -> bool:
        return (await self.get(sha)) is not None

    # --- Write ---------------------------------------------------------

    async def write(self, data: bytes, *, mime_type: str = "") -> dict:
        """Insert bytes or bump refcount for an existing blob. Returns
        the current blob row (post-increment).

        If the on-disk file is missing for a known sha (manual cleanup,
        corruption), re-creates it from the provided bytes — the refcount
        accounting stays correct because the sha matches.
        """
        sha = hashlib.sha256(data).hexdigest()
        existing = await self.get(sha)

        if existing:
            dest = Path(existing["real_path"])
            if not dest.exists():
                # Heal a missing file. Refcount is still valid — bytes are
                # identical by sha, so callers will see consistent state.
                log.warning("blob_missing_file_rewrote", sha=sha)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            await self._conn.execute(
                "UPDATE blobs SET refcount = refcount + 1 WHERE sha256 = ?",
                (sha,),
            )
            await self._conn.commit()
            existing["refcount"] += 1
            log.info("blob_dedup_hit", sha=sha, refcount=existing["refcount"])
            return existing

        dest = self.path_for(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-insert so a failed disk write doesn't leave an orphan row.
        try:
            dest.write_bytes(data)
        except OSError as err:
            log.warning("blob_write_failed", sha=sha, err=str(err))
            raise RuntimeError(f"blob write failed: {err}") from err

        await self._conn.execute(
            "INSERT INTO blobs (sha256, size_bytes, mime_type, real_path, refcount) "
            "VALUES (?, ?, ?, ?, 1)",
            (sha, len(data), mime_type, str(dest)),
        )
        await self._conn.commit()
        log.info("blob_created", sha=sha, size=len(data))
        return {
            "sha256": sha,
            "size_bytes": len(data),
            "mime_type": mime_type,
            "real_path": str(dest),
            "refcount": 1,
        }

    # --- Release + cleanup --------------------------------------------

    async def release(self, sha: str) -> bool:
        """Decrement refcount. When it hits zero, remove the file + row.

        Returns True if we decremented (a reference existed). Returns
        False if the sha was unknown or already at zero.
        """
        cursor = await self._conn.execute(
            "UPDATE blobs SET refcount = refcount - 1 "
            "WHERE sha256 = ? AND refcount > 0",
            (sha,),
        )
        await self._conn.commit()
        if cursor.rowcount == 0:
            return False

        blob = await self.get(sha)
        if blob and blob["refcount"] <= 0:
            fp = blob["real_path"]
            if fp:
                try:
                    if os.path.exists(fp):
                        os.remove(fp)
                        # Opportunistically prune empty shard directories.
                        parent = Path(fp).parent
                        try:
                            parent.rmdir()
                            parent.parent.rmdir()
                        except OSError:
                            pass  # not empty — fine
                except OSError as err:
                    log.warning("blob_file_remove_failed", sha=sha, err=str(err))
            await self._conn.execute("DELETE FROM blobs WHERE sha256 = ?", (sha,))
            await self._conn.commit()
            log.info("blob_purged", sha=sha)
        return True

    # --- Orphan sweep --------------------------------------------------

    async def sweep_orphans(self, *, limit: int = 1000) -> int:
        """Remove blob rows + files for refcount<=0 entries.

        Reaches zero through `release()` ordinarily, but bulk file_index
        deletes (e.g. `purge_all_old_trash` running without per-source
        adapter dispatch) can leave the row at 0 with no further callers.
        This is the safety net that catches that case.

        Returns the number of blobs purged.
        """
        cursor = await self._conn.execute(
            "SELECT sha256, real_path FROM blobs WHERE refcount <= 0 LIMIT ?",
            (int(limit),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return 0
        purged = 0
        for sha, real_path in rows:
            if real_path:
                try:
                    if os.path.exists(real_path):
                        os.remove(real_path)
                        parent = Path(real_path).parent
                        try:
                            parent.rmdir()
                            parent.parent.rmdir()
                        except OSError:
                            pass
                except OSError as err:
                    log.warning("orphan_blob_remove_failed", sha=sha, err=str(err))
                    continue
            await self._conn.execute("DELETE FROM blobs WHERE sha256 = ?", (sha,))
            purged += 1
        if purged:
            await self._conn.commit()
            log.info("orphan_blobs_swept", count=purged)
        return purged

    # --- Stats (for /api/files/stats growth) --------------------------

    async def totals(self) -> dict:
        cursor = await self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), "
            "COALESCE(SUM(refcount), 0) FROM blobs",
        )
        row = await cursor.fetchone()
        return {
            "blob_count": row[0],
            "stored_bytes": row[1],
            "reference_count": row[2],
        }
