"""``gutenberg_fetch`` job handler.

Downloads the Project Gutenberg plaintext that corresponds to a pinned
LibriVox audiobook, strips boilerplate, writes the body to disk under
``{data_dir}/gutenberg/{file_id}.txt``, and updates the file_index row's
``source_metadata`` with pointers the UI can use for read-along.

Registered once on server startup via the factory below — the factory
closes over ``app`` so ``app.state`` lookups happen at job-run time (not
at registration time, when ``http_client``/``file_index`` may still be
None).

Payload shape (set by :func:`augmentum.proxy.media_routes.pin_item`):

    {"file_id": "fi_...", "url_text_source": "https://www.gutenberg.org/..."}

Idempotent: a re-run on an already-fetched row short-circuits with a
``skipped`` result instead of re-downloading. This matters because the
job queue re-queues in-flight rows on restart — we don't want to burn
Gutenberg bandwidth on a book we already have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from augmentum.config import settings
from augmentum.jobs.context import JobContext, JobRetryable
from augmentum.media.gutenberg import (
    GutenbergError,
    fetch_plaintext,
    resolve_ebook_id,
    strip_boilerplate,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _gutenberg_dir() -> Path:
    """Return (and create) the directory where we store fetched texts."""
    path = Path(settings.data_dir) / "gutenberg"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_gutenberg_fetch_handler(app):
    """Build a handler bound to ``app.state`` services.

    The returned coroutine is what :mod:`augmentum.jobs.runner` dispatches
    when a ``gutenberg_fetch`` row is claimed.
    """

    async def handler(ctx: JobContext) -> dict[str, Any]:
        http_client = getattr(app.state, "http_client", None)
        idx = getattr(app.state, "file_index", None)
        if http_client is None or idx is None:
            # Shouldn't happen in a healthy server — surface loudly so
            # the runner can mark the job failed instead of spinning
            # silently.
            raise RuntimeError("gutenberg_fetch: http_client or file_index not initialized")

        file_id = str(ctx.payload.get("file_id") or "")
        url = str(ctx.payload.get("url_text_source") or "")
        if not file_id or not url:
            raise RuntimeError(
                f"gutenberg_fetch: malformed payload (file_id={file_id!r}, url={url!r})",
            )

        await ctx.update_progress(0.05, stage="resolving")
        entry = await idx.get(file_id, user_id=ctx.user_id)
        if entry is None:
            # Row was unpinned between enqueue and dispatch. Not a bug,
            # not worth retrying — mark complete with a skipped note.
            log.info(
                "gutenberg_fetch_entry_gone",
                job_id=ctx.job_id, file_id=file_id, user_id=ctx.user_id,
            )
            return {"skipped": "entry_missing"}

        existing_meta = dict(entry.source_metadata or {})
        if existing_meta.get("gutenberg_status") == "fetched":
            # Idempotency shortcut — a restart-requeued job or a manual
            # re-enqueue lands here and does nothing.
            await ctx.update_progress(1.0, stage="already_fetched")
            return {"skipped": "already_fetched"}

        try:
            ebook_id = resolve_ebook_id(url)
        except GutenbergError as exc:
            # Bad/non-Gutenberg URL on the pin row. Permanent — don't
            # retry. Mark metadata so the UI can show an "unavailable"
            # hint instead of silent absence.
            await _mark_unavailable(idx, file_id, ctx.user_id, existing_meta, str(exc))
            return {"skipped": "not_gutenberg", "reason": str(exc)}

        await ctx.update_progress(0.15, stage="downloading")
        await ctx.check_cancel()
        try:
            raw = await fetch_plaintext(http_client, ebook_id)
        except GutenbergError as exc:
            # Every candidate URL failed. Treat as transient — Gutenberg
            # intermittently serves 403s during load spikes and their
            # cache sometimes lags behind new books. Raise JobRetryable
            # so the runner reverts to pending and the next attempt can
            # try the fallback URLs again.
            raise JobRetryable(f"gutenberg fetch failed: {exc}") from exc
        await ctx.check_cancel()

        await ctx.update_progress(0.70, stage="cleaning")
        # Boilerplate stripping is pure Python string ops; safe on the
        # event loop for typical book sizes (~500 KB). Offload anyway
        # for very large omnibus volumes that can hit multi-MB, where
        # the regex scan starts to show up in profiles.
        cleaned = await ctx.run_in_thread(strip_boilerplate, raw)
        if not cleaned.strip():
            raise RuntimeError(
                f"gutenberg fetch for {ebook_id} produced empty text after stripping",
            )

        await ctx.update_progress(0.85, stage="storing")
        out_path = _gutenberg_dir() / f"{file_id}.txt"
        await ctx.run_in_thread(_atomic_write, out_path, cleaned)

        word_count = sum(1 for _ in cleaned.split())
        byte_size = len(cleaned.encode("utf-8"))
        updated_meta = dict(existing_meta)
        updated_meta["gutenberg_status"] = "fetched"
        updated_meta["gutenberg_ebook_id"] = ebook_id
        updated_meta["gutenberg_path"] = str(out_path)
        updated_meta["gutenberg_word_count"] = word_count
        updated_meta["gutenberg_byte_size"] = byte_size
        # Clear any previous error so a successful retry wipes a stale
        # failure state left over from an earlier attempt.
        updated_meta.pop("gutenberg_error", None)

        ok = await idx.update_source_metadata(
            file_id, updated_meta, user_id=ctx.user_id,
        )
        if not ok:
            # The row vanished mid-fetch (unpinned). Clean up the blob
            # so we don't leak it, then report skip.
            await ctx.run_in_thread(_safe_unlink, out_path)
            return {"skipped": "entry_missing_after_fetch"}

        await ctx.update_progress(1.0, stage="done")
        log.info(
            "gutenberg_fetched",
            file_id=file_id, user_id=ctx.user_id,
            ebook_id=ebook_id, word_count=word_count, byte_size=byte_size,
        )
        return {
            "ebook_id": ebook_id,
            "word_count": word_count,
            "byte_size": byte_size,
            "path": str(out_path),
        }

    return handler


async def _mark_unavailable(
    idx,
    file_id: str,
    user_id: str,
    existing_meta: dict,
    reason: str,
) -> None:
    """Flag a file_index row as having no fetchable Gutenberg source.

    Used for permanent failures (bad URL, non-Gutenberg URL) so the UI
    can render a differentiated state — "no read-along available for
    this recording" vs "fetching…".
    """
    meta = dict(existing_meta)
    meta["gutenberg_status"] = "unavailable"
    meta["gutenberg_error"] = reason[:500]
    try:
        await idx.update_source_metadata(file_id, meta, user_id=user_id)
    except Exception:
        log.warning("gutenberg_mark_unavailable_failed", exc_info=True)


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a tmp-then-rename, so a crash
    mid-write can't leave a truncated file that would then read back as
    a valid but corrupt book body."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        log.warning("gutenberg_safe_unlink_failed", path=str(path), exc_info=True)
