"""``wake_word_corpus_download`` job handler — fetch LibriSpeech dev-clean.

Real-audio negatives are the structural fix for the false-positive +
self-feedback failures of synthetic-only wake-word training. This handler
hands off to :func:`negatives_corpus.ensure_downloaded`, which is
idempotent — a re-run on an already-installed corpus short-circuits with
``skipped: already_installed`` instead of re-downloading.

Payload is empty (``{}``). The corpus is global, not per-user, so there's
nothing to parameterize. The job is enqueued with a user_id only so the
existing jobs UI can show the operator their own corpus-install progress.

Result shape::

    {
        "installed": true,
        "path": "/data/wake_word_corpora/librispeech-dev-clean",
        "num_files": 2703,
        "total_bytes": 337926624,
        "skipped": "already_installed"   # only if it was a no-op
    }

Idempotent on retry: re-running mid-extract re-verifies the existing
tarball via MD5 and resumes from there (or wipes and re-downloads if
the tarball was corrupted).
"""

from __future__ import annotations

from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def make_wake_word_corpus_download_handler(app):
    """Build the ``wake_word_corpus_download`` handler bound to ``app``."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        # Lazy-import — corpus module pulls torchaudio, which is heavy and
        # not every server startup loads it (the runner just dispatches).
        from augmentum.voice.wake_word import negatives_corpus

        if negatives_corpus.is_installed():
            summary = negatives_corpus.installed_summary()
            log.info(
                "wake_word_corpus_already_installed",
                job_id=ctx.job_id, path=summary.get("path"),
                num_files=summary.get("num_files"),
            )
            await ctx.update_progress(1.0, stage="already installed")
            summary["skipped"] = "already_installed"
            return summary

        log.info("wake_word_corpus_download_starting", job_id=ctx.job_id)

        async def _progress(frac: float, stage: str) -> None:
            await ctx.update_progress(frac, stage=stage)

        async def _cancel() -> None:
            await ctx.check_cancel()

        try:
            summary = await negatives_corpus.ensure_downloaded(
                progress_cb=_progress,
                cancel_cb=_cancel,
            )
        finally:
            # Drop the in-process catalog cache so subsequent training jobs
            # in the same server lifetime see the new manifest immediately.
            negatives_corpus.invalidate_cache()

        log.info(
            "wake_word_corpus_download_complete",
            job_id=ctx.job_id,
            num_files=summary.get("num_files"),
            total_bytes=summary.get("total_bytes"),
        )
        return summary

    return handler
