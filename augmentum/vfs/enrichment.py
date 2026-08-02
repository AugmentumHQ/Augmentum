"""Background enrichment — thumbnails, descriptions, and embeddings for indexed files."""

from __future__ import annotations

import asyncio
import base64
import io
import struct
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.vfs.index import FileIndexService

log = get_logger(__name__)


# How long to skip a file after a failed enrichment pass. Prevents the
# 600-errors-per-30-min retry storm seen 2026-05-09 where 5 EPUB files
# kept hitting database-lock contention on their UPDATE. After the
# interval the file becomes eligible again; if it still fails we wait
# another interval. A successful pass naturally clears the file from
# the pending set on its own (embedding/thumbnail predicates stop
# matching).
_ENRICHMENT_RETRY_BACKOFF = "-1 hour"


async def enrich_pending(index: FileIndexService, db) -> int:
    """Enrich files that lack thumbnails or embeddings. Returns count processed."""
    if not settings.files_enrichment_enabled:
        return 0

    # Three branches, each with its own index path — split into a
    # UNION-of-IDs subquery so the optimizer doesn't fall back to a
    # full table scan because of the OR structure.
    #
    #   B1: embedding IS NULL                              → partial index
    #   B2: epub AND thumbnail IS NULL                     → partial index
    #   B3: artifact-sourced epub needing cover_url sync   → idx_file_index_mime
    #
    # The trailing per-branch AND on last_enrichment_attempt skips
    # files attempted in the last hour so a repeatedly-failing
    # enrichment can't poison the loop (``enrich_file_atomic`` always
    # bumps the stamp via ``stamp_attempt=True`` so this filter is
    # honest).
    #
    # Pre-fix this SELECT cost 110-160ms every cycle on a 63k-row
    # table because the OR structure prevented index use. With the
    # partial index from migration 134 it drops to a sub-millisecond
    # seek (the partial index typically holds 0-10 rows).
    cursor = await db.execute(
        "SELECT id, user_id, name, description, tags, mime_type, real_path, source, "
        "       source_id, thumbnail, "
        "       embedding IS NULL AS needs_embed "
        "FROM file_index "
        "WHERE id IN ( "
        "    SELECT id FROM file_index "
        "    WHERE embedding IS NULL "
        "      AND (last_enrichment_attempt IS NULL "
        "        OR last_enrichment_attempt < datetime('now', ?)) "
        "    UNION "
        "    SELECT id FROM file_index "
        "    WHERE mime_type = 'application/epub+zip' AND thumbnail IS NULL "
        "      AND (last_enrichment_attempt IS NULL "
        "        OR last_enrichment_attempt < datetime('now', ?)) "
        "    UNION "
        "    SELECT id FROM file_index "
        "    WHERE mime_type = 'application/epub+zip' AND source = 'artifacts' "
        "      AND thumbnail IS NOT NULL "
        "      AND (SELECT json_extract(COALESCE(metadata, '{}'), '$.cover_url') "
        "           FROM artifacts WHERE id = file_index.source_id "
        "                            AND user_id = file_index.user_id) IS NULL "
        "      AND (last_enrichment_attempt IS NULL "
        "        OR last_enrichment_attempt < datetime('now', ?)) "
        "    UNION "
        # B4 (Piece 3 backfill): image rows that don't yet have a vision
        # caption. Detected by mime prefix + empty description. The
        # caption job is enqueued (not run inline) so vision inference
        # doesn't block the enrichment loop — the SmolVLM sibling
        # processes captions on its own at background priority.
        "    SELECT id FROM file_index "
        "    WHERE mime_type LIKE 'image/%' "
        "      AND (description IS NULL OR description = '') "
        "      AND is_trashed = 0 "
        "      AND (last_enrichment_attempt IS NULL "
        "        OR last_enrichment_attempt < datetime('now', ?)) "
        ") "
        "LIMIT 32",
        (_ENRICHMENT_RETRY_BACKOFF,) * 4,
    )
    rows = await cursor.fetchall()
    if not rows:
        return 0

    count = 0
    for row in rows:
        (file_id, user_id, name, desc, tags, mime, real_path, source,
         source_id, existing_thumb, needs_embed) = row

        # Collect every change for this file into one dict. We commit
        # all of them — plus the attempt stamp — in a single transaction
        # via ``enrich_file_atomic``. Previously each enrichment path
        # (description / EPUB cover / embedding / image thumb / stamp)
        # had its own commit, so an EPUB pass could fire 4-5 commits
        # per file. With 32 files per batch that was ~150 writer-lock
        # acquisitions per pass.
        changes: dict = {}
        cover_url_for_artifact: tuple[str, str] | None = None

        try:
            # If we have a real file on disk and no description yet, try to
            # pull the actual contents through the extractors. The description
            # column then carries real signal for both search hits and the
            # embedding step below — instead of only matching against filename.
            if (not desc) and real_path:
                extracted = await _extract_description(real_path, mime or "", name or "")
                if extracted:
                    desc = extracted
                    changes["description"] = extracted

            # EPUB: pull structured metadata (author/publisher/language/date)
            # into source_metadata and generate a cover thumbnail. The text
            # extractor above already seeded `description`; this pass adds
            # the fields plain text extraction can't surface.
            if mime == "application/epub+zip" and real_path:
                cover_url = existing_thumb
                if cover_url is None:
                    extracted_cover, extras = await _extract_epub_meta(real_path, file_id)
                    cover_url = extracted_cover
                    if extracted_cover:
                        changes["thumbnail"] = extracted_cover
                    if extras:
                        changes["source_metadata_merge"] = extras
                # Sync the cover into artifacts.metadata.cover_url so the
                # library UI can render it. Different table, so this stays
                # a separate commit; it only fires for artifact-sourced
                # EPUBs that have a cover.
                if cover_url and source == "artifacts" and source_id:
                    cover_url_for_artifact = (source_id, cover_url)

            if needs_embed:
                text = f"{name} {desc or ''} {tags or ''}".strip()
                if text:
                    embedding = await _generate_embedding(text)
                    if embedding:
                        changes["embedding"] = embedding
                        count += 1

            # Image thumbnail — only if this file isn't already getting
            # one from the EPUB path above.
            if (mime and mime.startswith("image/") and real_path
                    and "thumbnail" not in changes
                    and not await _has_thumbnail(db, file_id)):
                thumb = await _generate_thumbnail(real_path)
                if thumb:
                    changes["thumbnail"] = thumb

            # B4 (Piece 3 backfill): image rows lacking a vision caption.
            # The enqueue is best-effort — the row's enrichment continues
            # whether or not the caption queue is reachable. Caption
            # results land asynchronously via the SmolVLM sibling; the
            # next enrichment pass will see the now-populated description
            # and naturally skip re-enqueue. file_caption handler is
            # itself idempotent (skips when description is set).
            jobs_store = getattr(index, "_jobs_store", None)
            if (jobs_store is not None
                    and mime and mime.startswith("image/")
                    and not (desc or "").strip()
                    and real_path):
                # Without real_path the caption handler short-circuits on
                # "file bytes not on disk" — enqueueing creates a churn
                # loop because last_enrichment_attempt is bumped but
                # description stays NULL, so the next pass re-enqueues.
                try:
                    await jobs_store.create(
                        user_id=user_id, job_type="file_caption",
                        payload={"file_id": file_id, "user_id": user_id},
                        priority=2, max_attempts=2,
                    )
                except Exception:
                    log.warning(
                        "enrichment_caption_enqueue_failed", file_id=file_id,
                    )
        except Exception:
            log.warning("enrichment_failed", file_id=file_id, exc_info=True)

        # ONE commit per file: every collected field plus the attempt
        # stamp, in a single ``BEGIN IMMEDIATE`` / ``COMMIT``. The stamp
        # always goes in so the next pass's hour-backoff filter applies
        # whether or not anything was enrichable.
        try:
            await index.enrich_file_atomic(
                file_id, user_id=user_id, stamp_attempt=True, **changes,
            )
        except Exception:
            # Atomic commit failed (DB lock, etc.). Fall back to a
            # standalone stamp so a repeatedly-failing UPDATE can't
            # poison the loop — this is what fixed the 600-errors-per-
            # 30-min retry storm on 2026-05-09.
            log.debug("enrichment_atomic_commit_failed",
                      file_id=file_id, exc_info=True)
            await _stamp_enrichment_attempt(db, file_id, user_id)

        if cover_url_for_artifact:
            sid, curl = cover_url_for_artifact
            await _sync_cover_to_artifact(db, sid, user_id, curl)

        # Yield to the event loop between files. Without this, a 32-file
        # batch monopolizes the connection's writer thread for several
        # seconds; every other coroutine (auth, polling, chat sync) sits
        # in queue. ``sleep(0)`` is the minimal yield — costs nothing but
        # lets pending callbacks fire between files.
        await asyncio.sleep(0)

    return count


async def _stamp_enrichment_attempt(db, file_id: str, user_id: str) -> None:
    """Mark this file as recently attempted. Best-effort: failures here
    are silent because they don't matter — if the stamp fails the worst
    case is the file gets retried on the next pass instead of in an
    hour, which is exactly the previous behaviour we're fixing."""
    try:
        await db.execute(
            "UPDATE file_index SET last_enrichment_attempt = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (file_id, user_id),
        )
        await db.commit()
    except Exception:
        log.debug("enrichment_attempt_stamp_failed", file_id=file_id, exc_info=True)


async def _extract_epub_meta(
    real_path: str,
    file_id: str,
) -> tuple[str | None, dict | None]:
    """Pull EPUB cover + structured metadata. Pure extraction, no writes.

    Returns ``(cover_thumbnail_or_none, extras_dict_or_none)``. The
    caller folds these into the per-file enrichment dict so all field
    writes commit in a single transaction.
    """
    from augmentum.vfs.epub_extractor import extract as epub_extract

    def _do_extract():
        return epub_extract(real_path, max_thumb_px=settings.files_max_thumbnail_px)

    try:
        meta = await asyncio.to_thread(_do_extract)
    except Exception:
        log.debug("epub_extract_thread_failed", file_id=file_id, exc_info=True)
        return None, None
    if meta is None:
        return None, None
    extras = meta.as_source_metadata() or None
    cover = meta.cover_thumbnail or None
    return cover, extras


async def _sync_cover_to_artifact(
    db, source_id: str, user_id: str, cover_url: str,
) -> None:
    """Set artifacts.metadata.cover_url if not already present.

    Idempotent: the WHERE clause only matches rows missing cover_url, so
    subsequent passes no-op. This keeps the enrichment backfill pass
    cheap once every EPUB has been synced.
    """
    try:
        await db.execute(
            "UPDATE artifacts "
            "SET metadata = json_set(COALESCE(metadata, '{}'), '$.cover_url', ?) "
            "WHERE id = ? AND user_id = ? "
            "  AND json_extract(COALESCE(metadata, '{}'), '$.cover_url') IS NULL",
            (cover_url, source_id, user_id),
        )
        await db.commit()
    except Exception:
        log.warning("cover_sync_failed", source_id=source_id, exc_info=True)


async def _extract_description(real_path: str, mime: str, filename: str) -> str:
    """Pull text content from disk via the vfs extractor dispatch and
    truncate to the configured description budget.

    Runs the file read + parse off the event loop so a slow PDF doesn't
    block the whole enrichment cycle. Empty string on any failure —
    callers treat this as "nothing extracted, move on".
    """
    from augmentum.vfs.extractors import extract, supported_for

    if not supported_for(mime, filename):
        return ""

    def _read_and_extract() -> str:
        try:
            with open(real_path, "rb") as fp:
                data = fp.read()
        except OSError:
            return ""
        return extract(data, mime, filename=filename)

    try:
        text = await asyncio.to_thread(_read_and_extract)
    except Exception:
        log.debug("extract_thread_failed", real_path=real_path, exc_info=True)
        return ""

    if not text:
        return ""
    cap = max(0, int(settings.files_description_max_chars))
    return text[:cap] if cap else text


async def _generate_embedding(text: str) -> bytes | None:
    """Generate embedding vector as bytes.

    The actual embedding inference is CPU/GPU-bound and synchronous;
    push it through asyncio.to_thread so a 32-file enrichment batch
    doesn't stall the event loop for 3-10 seconds (each call is
    100-300ms on CPU; 32 of them serially blocks the loop and starves
    every request handler). This was the root cause of the
    event_loop_stall warnings observed alongside the enrichment-loop
    DB lock storm on 2026-05-09.
    """
    try:
        from augmentum.memory.embeddings import EmbeddingService
        vec = await asyncio.to_thread(EmbeddingService.embed_one, text)
        # Pack as float32 array
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception:
        return None


async def _generate_thumbnail(real_path: str) -> str | None:
    """Generate base64 thumbnail for an image file.

    PIL decode + JPEG re-encode + base64 are all CPU-bound; a 4K image
    can spend 200-500ms here. Run off the event loop for the same
    reason as ``_generate_embedding`` above.
    """
    def _do_thumbnail() -> str | None:
        try:
            from PIL import Image
            max_px = settings.files_max_thumbnail_px
            img = Image.open(real_path)
            img.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None
    return await asyncio.to_thread(_do_thumbnail)


async def _has_thumbnail(db, file_id: str) -> bool:
    cursor = await db.execute(
        "SELECT thumbnail FROM file_index WHERE id = ? AND thumbnail IS NOT NULL",
        (file_id,),
    )
    return (await cursor.fetchone()) is not None


async def enrichment_loop(index: FileIndexService, db):
    """Background loop that enriches files periodically."""
    while True:
        try:
            count = await enrich_pending(index, db)
            if count:
                log.info("enrichment_batch", count=count)
        except Exception:
            log.warning("enrichment_loop_error", exc_info=True)
        await asyncio.sleep(30)  # Check every 30 seconds
