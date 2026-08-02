"""One-time population of file index from existing data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

    from augmentum.vfs.index import FileIndexService

log = get_logger(__name__)


# Source → (backing table, id column) — stays in sync with populate_from_existing
_SOURCE_TABLES: dict[str, tuple[str, str]] = {
    "artifacts": ("artifacts", "id"),
    "images": ("image_generations", "image_id"),
    "documents": ("documents", "id"),
    "chat_images": ("chat_images", "id"),
}


async def repair_real_paths(db: aiosqlite.Connection) -> tuple[int, int]:
    """Fix historical file_index rows with broken real_path or size_bytes=0.

    The original populate pass stored relative artifact paths and never picked
    up image sizes, which is why downloads 404 and cards show 0 B. This pass:
      - Resolves relative artifact paths to absolute via the artifact base dir
      - Pulls image real_paths from image_generations.file_path when missing
      - Refreshes size_bytes from disk when 0 and the file is readable
    One UPDATE per row only when something actually changes. Safe to run on
    every boot.
    """
    import os

    try:
        cursor = await db.execute(
            "SELECT id, source, source_id, real_path, size_bytes FROM file_index "
            "WHERE is_trashed = 0",
        )
        rows = await cursor.fetchall()
    except Exception:
        log.warning("repair_real_paths_select_failed", exc_info=True)
        return 0, 0

    artifact_base = _artifact_base_dir()
    path_fixed = 0
    size_fixed = 0

    # Pre-load source lookups in bulk (one query per source)
    artifact_paths: dict[str, str] = {}
    image_paths: dict[str, str] = {}
    try:
        cur = await db.execute("SELECT id, path FROM artifacts")
        artifact_paths = {r[0]: r[1] or "" for r in await cur.fetchall()}
    except Exception as exc:
        log.debug("vfs_populate_artifacts_load_failed", error=str(exc))
    try:
        cur = await db.execute("SELECT image_id, file_path FROM image_generations")
        image_paths = {r[0]: r[1] or "" for r in await cur.fetchall()}
    except Exception as exc:
        log.debug("vfs_populate_image_generations_load_failed", error=str(exc))

    for fid, source, source_id, real_path, size_bytes in rows:
        new_path = real_path or ""
        # Try to fix a missing / broken real_path from the source row.
        if not new_path or not os.path.exists(new_path):
            if source == "artifacts" and source_id in artifact_paths:
                rel = artifact_paths[source_id]
                candidate = os.path.join(artifact_base, rel) if rel else ""
                if candidate and os.path.exists(candidate):
                    new_path = candidate
            elif source == "images" and source_id in image_paths:
                candidate = image_paths[source_id]
                if candidate and os.path.exists(candidate):
                    new_path = candidate

        path_changed = new_path != (real_path or "")
        new_size = size_bytes or 0
        if (not new_size) and new_path and os.path.exists(new_path):
            new_size = _size_of(new_path)

        if path_changed or new_size != (size_bytes or 0):
            await db.execute(
                "UPDATE file_index SET real_path = ?, size_bytes = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (new_path or None, new_size, fid),
            )
            if path_changed:
                path_fixed += 1
            if new_size != (size_bytes or 0):
                size_fixed += 1

    if path_fixed or size_fixed:
        await db.commit()
        log.info("file_index_repaired", paths=path_fixed, sizes=size_fixed)
    return path_fixed, size_fixed


async def backfill_kind(db: aiosqlite.Connection) -> int:
    """Classify any file_index rows left with an empty kind.

    The SQL backfill in migration 085 handles mime-typed rows. This pass fills
    in the long tail: empty / octet-stream / unknown mimes where the filename
    extension is the only signal. One UPDATE per (kind, id-set) batch.
    """
    from collections import defaultdict

    from augmentum.vfs.classify import derive_kind

    try:
        cursor = await db.execute(
            "SELECT id, mime_type, name FROM file_index WHERE kind = '' OR kind IS NULL",
        )
        rows = await cursor.fetchall()
    except Exception:
        # Column may not exist yet if migration hasn't run
        log.warning("backfill_kind_select_failed", exc_info=True)
        return 0

    if not rows:
        return 0

    by_kind: dict[str, list[str]] = defaultdict(list)
    for fid, mime, name in rows:
        by_kind[derive_kind(mime, name)].append(fid)

    total = 0
    for kind, ids in by_kind.items():
        if not ids:
            continue
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" for _ in batch)
            cursor = await db.execute(
                f"UPDATE file_index SET kind = ? WHERE id IN ({placeholders})",
                [kind, *batch],
            )
            total += cursor.rowcount or 0
    await db.commit()
    if total:
        log.info("file_index_kind_backfilled", count=total)
    return total


async def reconcile_private_images(db: aiosqlite.Connection) -> int:
    """Drop file_index rows for images that are flagged private upstream.

    Back-compat sweep for users who marked images private *before* the
    set_private → file_index cascade landed. Idempotent — re-running
    never removes anything new once the cascade is in place, because
    new private toggles unregister immediately. Batched single DELETE so
    a large backlog doesn't produce per-row overhead.

    Thumb cache is cleaned in the same pass — a row that was never
    supposed to surface in Files shouldn't leave orphaned previews in
    ``data/thumbs/images/`` either.
    """
    try:
        cursor = await db.execute(
            "SELECT fi.source_id FROM file_index fi "
            "JOIN image_generations ig ON ig.image_id = fi.source_id "
            "WHERE fi.source = 'images' AND ig.is_private = 1",
        )
        ids = [row[0] for row in await cursor.fetchall() if row[0]]
    except Exception:
        log.warning("reconcile_private_images_select_failed", exc_info=True)
        return 0

    if not ids:
        return 0

    try:
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" for _ in batch)
            await db.execute(
                f"DELETE FROM file_index WHERE source = 'images' "
                f"AND source_id IN ({placeholders})",
                batch,
            )
        await db.commit()
    except Exception:
        log.warning("reconcile_private_images_delete_failed", exc_info=True)
        return 0

    # Best-effort thumbnail purge — safe no-op if the service isn't
    # installed (e.g. test harness or stripped-down config).
    try:
        from augmentum.vfs import purge_thumbnails
        for image_id in ids:
            purge_thumbnails("images", image_id)
    except Exception:
        log.warning("reconcile_private_images_thumb_purge_failed", exc_info=True)

    log.info("file_index_private_images_reconciled", purged=len(ids))
    return len(ids)


async def reconcile_orphaned_images(
    index: FileIndexService, db: aiosqlite.Connection,
) -> int:
    """Register non-private ``image_generations`` rows that never made
    it into ``file_index``.

    The inverse of :func:`reconcile_private_images`. Catches images
    that exist in ``image_generations`` but have no matching
    ``file_index`` row — either because ``save_generation`` was called
    before ``register_file`` was wired in, because the realtime
    register failed silently, or because the per-user one-time
    populate ran while those rows hadn't been generated yet (the
    populate marker prevents it from re-running). Result: the gallery
    + Files panel + global search all miss those images forever
    until something repopulates the index.

    Idempotent — re-running finds nothing once every public image
    has its index row. Background images are included because they're
    still legitimate library content; private images are excluded
    because :func:`reconcile_private_images` would just delete them
    on the next pass anyway.
    """
    try:
        cursor = await db.execute(
            "SELECT ig.image_id, ig.user_id, ig.file_path, ig.prompt, "
            "ig.model, ig.seed, ig.width, ig.height, ig.job_type "
            "FROM image_generations ig "
            "LEFT JOIN file_index fi "
            "  ON fi.source = 'images' "
            "  AND fi.source_id = ig.image_id "
            "  AND fi.user_id = ig.user_id "
            "WHERE COALESCE(ig.is_private, 0) = 0 AND fi.id IS NULL",
        )
        rows = await cursor.fetchall()
    except Exception:
        log.warning("reconcile_orphaned_images_select_failed", exc_info=True)
        return 0

    if not rows:
        return 0

    registered = 0
    for image_id, user_id, file_path, prompt, model, seed, width, height, job_type in rows:
        try:
            await index.register(
                user_id=user_id, source="images", source_id=image_id,
                name=f"{image_id}.png", mime_type="image/png",
                size_bytes=_size_of(file_path or ""),
                real_path=file_path or "",
                description=(prompt or "")[:500],
                source_metadata={
                    "prompt": prompt or "", "model": model or "",
                    "seed": seed, "width": width, "height": height,
                    "job_type": job_type or "txt2img",
                },
            )
            registered += 1
        except Exception:
            log.warning(
                "reconcile_orphaned_images_register_failed",
                image_id=image_id, user_id=user_id, exc_info=True,
            )

    if registered:
        log.info("file_index_orphaned_images_backfilled", registered=registered)
    return registered


async def reconcile_stranded(index: FileIndexService, db: aiosqlite.Connection) -> int:
    """Remove file_index rows whose backing source row no longer exists.

    Safety net for rows that were stranded before source delete paths learned
    to cascade into file_index. Scans each source in turn and issues a single
    bulk delete for any ids missing from the source table.
    """
    total = 0
    for source, (table, id_col) in _SOURCE_TABLES.items():
        try:
            cursor = await db.execute(
                "SELECT source_id FROM file_index WHERE source = ?", (source,),
            )
            indexed = {row[0] for row in await cursor.fetchall() if row[0]}
            if not indexed:
                continue
            cursor = await db.execute(f"SELECT {id_col} FROM {table}")
            alive = {row[0] for row in await cursor.fetchall() if row[0]}
            stranded = indexed - alive
            if not stranded:
                continue
            # Use the index's parameterized delete (keeps hooks / fts in sync)
            placeholders = ",".join("?" for _ in stranded)
            await db.execute(
                f"DELETE FROM file_index WHERE source = ? AND source_id IN ({placeholders})",
                [source, *stranded],
            )
            total += len(stranded)
        except Exception:
            log.warning("reconcile_source_failed", source=source, exc_info=True)
    if total:
        await db.commit()
        log.info("file_index_reconciled", purged=total)
    return total


def _artifact_base_dir() -> str:
    """Resolve the artifact base dir the same way ArtifactStore does."""
    from pathlib import Path

    from augmentum.config import settings

    configured = getattr(settings, "agentic_artifact_dir", "data/artifacts")
    p = Path(configured)
    if not p.is_absolute():
        p = Path(settings.data_dir) / p
    return str(p)


def _size_of(path: str) -> int:
    """Best-effort filesystem size lookup. Returns 0 on any failure."""
    import os

    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


async def populate_from_existing(index: FileIndexService, db: aiosqlite.Connection, user_id: str) -> int:
    """Scan all existing subsystem data and register in file index. Returns count."""
    import os

    total = 0
    artifact_base = _artifact_base_dir()

    # Artifacts — `path` column is relative to artifact_base. Resolve to
    # absolute so downloads can read it directly.
    cursor = await db.execute(
        "SELECT id, filename, display_name, format, size_bytes, path FROM artifacts WHERE user_id = ?",
        (user_id,),
    )
    for row in await cursor.fetchall():
        rel_path = row[5] or ""
        real_path = os.path.join(artifact_base, rel_path) if rel_path else ""
        size_bytes = row[4] or _size_of(real_path)
        await index.register(
            user_id=user_id, source="artifacts", source_id=row[0],
            name=row[1], size_bytes=size_bytes, real_path=real_path,
            description=row[2] or row[1],
            source_metadata={"format": row[3] or ""},
        )
        total += 1

    # Images — image_generations doesn't carry size, so pick it up from disk.
    # Private images are excluded so the gallery's "private" flag actually
    # hides them from the Files panel after a restart.
    cursor = await db.execute(
        "SELECT image_id, file_path, prompt, model, seed, width, height "
        "FROM image_generations WHERE user_id = ? AND is_private = 0",
        (user_id,),
    )
    for row in await cursor.fetchall():
        real_path = row[1] or ""
        await index.register(
            user_id=user_id, source="images", source_id=row[0],
            name=f"{row[0]}.png", mime_type="image/png",
            size_bytes=_size_of(real_path),
            real_path=real_path, description=(row[2] or "")[:500],
            source_metadata={"prompt": row[2] or "", "model": row[3] or "",
                             "seed": row[4], "width": row[5], "height": row[6]},
        )
        total += 1

    # Documents
    cursor = await db.execute(
        "SELECT id, filename, mime_type, file_size, chunk_count FROM documents WHERE user_id = ?",
        (user_id,),
    )
    for row in await cursor.fetchall():
        await index.register(
            user_id=user_id, source="documents", source_id=row[0],
            name=row[1], mime_type=row[2] or "", size_bytes=row[3] or 0,
            description=f"{row[1]} ({row[4] or 0} chunks)",
            source_metadata={"chunk_count": row[4] or 0},
        )
        total += 1

    # Chat images
    cursor = await db.execute(
        "SELECT id, mime_type, length(data), session_id FROM chat_images WHERE user_id = ?",
        (user_id,),
    )
    for row in await cursor.fetchall():
        ext = (row[1] or "image/png").split("/")[-1]
        await index.register(
            user_id=user_id, source="chat_images", source_id=row[0],
            name=f"chat_{row[0][:8]}.{ext}", mime_type=row[1] or "image/png",
            size_bytes=row[2] or 0,
            source_metadata={"session_id": row[3] or ""},
        )
        total += 1

    log.info("file_index_populated", user_id=user_id, total=total)
    return total
