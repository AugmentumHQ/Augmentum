"""Uploads adapter — user-uploaded files routed through the blob store.

Each upload gets its own `uploads` row (filename, timestamp, metadata)
pointing at a shared blob by sha. Identical bytes across multiple uploads
share one blob — so uploading the same PDF five times costs one copy on
disk. Delete path decrements the blob refcount and purges the physical
file when the last reference drops.
"""

from __future__ import annotations

import json
import secrets
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.vfs import register_file, unregister_file
from augmentum.vfs.blobs import BlobStore
from augmentum.vfs.validation import get_user_storage_used

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class UploadsAdapter:
    name = "uploads"

    def __init__(self, conn: aiosqlite.Connection, blobs: BlobStore) -> None:
        self._conn = conn
        self._blobs = blobs

    # --- Save ----------------------------------------------------------

    async def save(
        self,
        data: bytes,
        filename: str,
        *,
        mime_type: str = "",
        mime_sniffed: str = "",
        user_id: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Store an uploaded file. Returns upload metadata + dedup flag.

        `mime_type` is the client-claimed Content-Type; `mime_sniffed` is
        the server-detected MIME from magic bytes. The sniffed value is
        what we surface to downstream consumers (preview, search filters)
        because the client-claimed value is user-controlled.
        """
        if not user_id:
            raise ValueError("uploads require a user_id")
        if not data:
            raise ValueError("uploads must be non-empty")

        # Use the sniffed MIME for the blob row when present — that's the
        # truth about what the bytes actually are.  file_index also gets
        # the sniffed value so search/preview behave correctly even when a
        # malicious client lied about Content-Type.
        effective_mime = mime_sniffed or mime_type
        blob = await self._blobs.write(data, mime_type=effective_mime)
        upload_id = f"ul_{secrets.token_hex(8)}"
        meta_json = json.dumps({
            **(metadata or {}),
            "blob_sha": blob["sha256"],
        })
        await self._conn.execute(
            "INSERT INTO uploads (id, user_id, filename, blob_sha, "
            "size_bytes, mime_type, mime_sniffed, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (upload_id, user_id, filename, blob["sha256"],
             blob["size_bytes"], mime_type, mime_sniffed, meta_json),
        )
        await self._conn.commit()

        # Index in file_index — kind/source color tinting, search, chips all
        # flow from this row. `source_metadata.blob_sha` lets the detail panel
        # surface dedup info if we ever want to.
        await register_file(
            user_id=user_id, source=self.name, source_id=upload_id,
            name=filename, mime_type=effective_mime,
            size_bytes=blob["size_bytes"],
            real_path=blob["real_path"],
            source_metadata={
                "blob_sha": blob["sha256"],
                "mime_claimed": mime_type,
                "mime_sniffed": mime_sniffed,
                **(metadata or {}),
            },
        )

        deduped = blob["refcount"] > 1
        log.info(
            "upload_saved", id=upload_id, user_id=user_id,
            filename=filename, size=blob["size_bytes"], deduped=deduped,
            mime_claimed=mime_type, mime_sniffed=mime_sniffed,
        )
        return {
            "id": upload_id,
            "filename": filename,
            "blob_sha": blob["sha256"],
            "size_bytes": blob["size_bytes"],
            "mime_type": effective_mime,
            "mime_claimed": mime_type,
            "mime_sniffed": mime_sniffed,
            "deduped": deduped,
        }

    async def storage_used(self, user_id: str) -> int:
        """Return total bytes the user has occupied via uploads."""
        return await get_user_storage_used(self._conn, user_id)

    # --- Adapter Protocol ---------------------------------------------

    async def resolve(self, source_id: str, *, user_id: str) -> str | None:
        cursor = await self._conn.execute(
            "SELECT blob_sha FROM uploads WHERE id = ? AND user_id = ?",
            (source_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        blob = await self._blobs.get(row[0])
        if not blob or not blob.get("real_path"):
            return None
        return blob["real_path"]

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT id FROM uploads WHERE user_id = ?", (user_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "SELECT blob_sha FROM uploads WHERE id = ? AND user_id = ?",
            (source_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        blob_sha = row[0]
        await self._conn.execute(
            "DELETE FROM uploads WHERE id = ? AND user_id = ?",
            (source_id, user_id),
        )
        await self._conn.commit()
        # Decrement blob refcount — frees the physical file when this was
        # the last reference. Other uploads / sources pointing at the same
        # sha keep it alive.
        await self._blobs.release(blob_sha)
        await unregister_file(self.name, source_id, user_id=user_id)
        log.info("upload_deleted", id=source_id, user_id=user_id, blob_sha=blob_sha)
        return True
