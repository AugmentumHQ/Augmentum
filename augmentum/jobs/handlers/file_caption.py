"""``file_caption`` job handler.

Generates a short vision description for a newly-registered file_index
row via the always-on :class:`VisionRouter`. The caption is written
back to ``file_index.description`` so both the FTS5 index (via the
existing triggers) and the vec0 mirror (via the embedder pipeline)
pick it up automatically.

Payload shape (set by :meth:`FileIndexService.register`):

    {"file_id": "fi_...", "user_id": "..."}

Workload hint: ``BACKGROUND``. The router never routes background
captioning through the primary model — only through the SmolVLM
sibling — so the primary's KV cache stays clean while the user is
mid-conversation. This is the load-bearing design choice for the
always-on captioner.

Idempotent: rows that already have a non-empty description short-circuit
with ``status=skipped, reason="description already set"`` instead of
re-captioning. The auto-enqueue path only fires on initial INSERTs but
the job queue may re-run on container restart — idempotency keeps that
safe.

Provider availability: when no vision provider is ready (vision_provider
disabled, model files missing, sibling still starting), the handler
returns ``status=skipped, reason="no vision provider"`` without raising.
The ingest pipeline isn't blocked on vision availability; captions
appear as the substrate comes online and the backfill (Piece 4) sweeps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Mime-type prefixes the captioner can usefully describe. Skipping
# everything else short-circuits cheaply at job-claim time instead of
# burning a llama-server roundtrip on a PDF or .zip we couldn't
# describe usefully anyway.
_CAPTIONABLE_PREFIXES = ("image/",)

# Caption text caps for storage sanity. SmolVLM at max_tokens=80
# typically returns ~50-60 tokens / ~300 chars; the hard cap is a
# safety net against runaway output.
_CAPTION_HARD_CHAR_CAP = 1000


def make_file_caption_handler(app):
    """Bind the handler to ``app.state`` services.

    The returned coroutine is what :mod:`augmentum.jobs.runner` dispatches
    when a ``file_caption`` row is claimed.
    """

    async def handler(ctx: JobContext) -> dict[str, Any]:
        payload = ctx.payload or {}
        file_id = str(payload.get("file_id") or "")
        user_id = str(payload.get("user_id") or "")
        if not file_id or not user_id:
            return {"status": "skipped", "reason": "missing file_id or user_id"}

        idx = getattr(app.state, "file_index", None)
        router = getattr(app.state, "vision_router", None)
        if idx is None:
            log.info("file_caption_skipped_no_index", file_id=file_id)
            return {"status": "skipped", "reason": "file_index unavailable"}
        if router is None or not router.has_any_provider:
            log.info("file_caption_skipped_no_provider", file_id=file_id)
            return {"status": "skipped", "reason": "no vision provider"}
        if not await router.is_available():
            log.info("file_caption_skipped_provider_not_ready", file_id=file_id)
            return {"status": "skipped", "reason": "vision provider not ready"}

        entry = await idx.get(file_id, user_id=user_id)
        if entry is None:
            return {"status": "skipped", "reason": "row not found"}

        mime = (entry.mime_type or "").lower()
        if not mime.startswith(_CAPTIONABLE_PREFIXES):
            return {
                "status": "skipped",
                "reason": f"non-captionable mime: {mime}",
            }

        if (entry.description or "").strip():
            return {"status": "skipped", "reason": "description already set"}

        real_path = entry.real_path or ""
        if not real_path or not Path(real_path).is_file():
            return {"status": "skipped", "reason": "file bytes not on disk"}

        try:
            image_bytes = await ctx.run_in_thread(
                lambda: Path(real_path).read_bytes(),
            )
        except Exception as exc:
            log.warning(
                "file_caption_read_failed",
                file_id=file_id, error=str(exc)[:200],
            )
            return {"status": "failed", "reason": "read failed"}

        # Background workload hint — router routes through SmolVLM
        # sibling regardless of primary VL availability, keeping the
        # primary's KV cache clean for the active user conversation.
        from augmentum.vision.router import Workload
        caption = await router.caption(
            image_bytes,
            prompt="Describe this image in one short sentence.",
            workload=Workload.BACKGROUND,
            max_tokens=80,
            timeout_s=60.0,
        )
        if not caption:
            return {"status": "skipped", "reason": "empty caption"}

        if len(caption) > _CAPTION_HARD_CHAR_CAP:
            caption = caption[:_CAPTION_HARD_CHAR_CAP].rstrip() + "…"

        ok = await idx.update_enrichment(
            file_id, user_id=user_id, description=caption,
        )
        # Clear the (likely-stale) embedding so enrich_pending
        # regenerates it from the now-captioned description on its
        # next pass. Without this, image rows captioned after their
        # first enrichment keep an embedding computed from filename
        # alone — search_by_embedding hits would be weak. See
        # FileIndexService.clear_embedding for the rationale.
        if ok:
            try:
                await idx.clear_embedding(file_id, user_id=user_id)
            except Exception:
                log.warning(
                    "file_caption_clear_embedding_failed",
                    file_id=file_id, exc_info=True,
                )
        log.info(
            "file_caption_written",
            file_id=file_id,
            chars=len(caption),
            mime=mime,
            success=ok,
        )
        return {
            "status": "ok" if ok else "noop",
            "caption_chars": len(caption),
        }

    return handler
