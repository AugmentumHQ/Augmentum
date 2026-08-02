"""``image_pull`` job handler.

Wraps the image-model download path (``pull_from_huggingface`` /
``pull_from_civitai``) in the background-job queue so an in-flight pull
survives a server restart:

  1. POST ``/api/image/models/pull`` creates a job (instead of a fire-and-
     forget ``asyncio.create_task``). ``task_id == job_id``.
  2. JobRunner dispatches the job; the handler calls the same
     ``_run_pull_task`` body the legacy flow uses, threaded with a
     :class:`JobContext` so progress is mirrored to the persistent DB.
  3. On restart, ``JobsStore.requeue_crashed`` flips ``running`` →
     ``pending`` and the runner re-dispatches. The handler enters with the
     same payload, ``pull_from_huggingface`` resumes from HF's
     ``.cache/huggingface/download/*.incomplete`` partial (see
     ``ModelManager._has_resumable_partial``), and ``_pull_tasks`` is
     rebuilt for the UI poll.

Idempotency: the handler is idempotent because the underlying HF pull is.
A re-entry after restart will either find the model already installed
(``status: exists``) or pick up the partial and finish it.
"""

from __future__ import annotations

from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def make_image_pull_handler(app):
    """Build the image-pull handler bound to ``app.state`` services."""

    async def handler(ctx: JobContext) -> dict[str, Any] | None:
        # Lazy imports to avoid the circular dep with image_routes.
        from augmentum.image.model_manager import ModelManager
        from augmentum.proxy.image_routes import _pull_tasks, _run_pull_task

        payload = ctx.payload or {}
        source = str(payload.get("source") or "").strip()
        if not source:
            raise RuntimeError("image_pull: payload missing 'source'")

        model_mgr = getattr(app.state, "image_model_manager", None)
        persistence = getattr(app.state, "image_persistence", None)
        if not model_mgr:
            from augmentum.config import settings as _settings
            model_dir = _settings.image_model_dir or f"{_settings.data_dir}/image_models"
            model_mgr = ModelManager(model_dir)

        # Pre-populate _pull_tasks for the UI poll. After a restart this is
        # empty for our task_id; populating it here is what makes the
        # `reconnectCatalogDownloads` flow show a live progress bar again.
        task_id = ctx.job_id
        if task_id not in _pull_tasks:
            _pull_tasks[task_id] = {
                "status": "running",
                "source": source,
                "progress": {},
                "last_event": {},
                "result": None,
                "error": None,
            }

        await _run_pull_task(
            task_id,
            model_mgr,
            persistence,
            source,
            str(payload.get("name") or ""),
            payload.get("allow_patterns") or None,
            variant=str(payload.get("variant") or ""),
            asset_type=str(payload.get("asset_type") or ""),
            trigger_words=payload.get("trigger_words") or None,
            base_model=str(payload.get("base_model") or ""),
            ctx=ctx,
        )

        # _run_pull_task writes its outcome to the shared dict. Translate
        # back into the JobRunner's return contract: dict → mark_completed
        # with that result; raise → mark_failed.
        final = _pull_tasks.get(task_id, {})
        status = final.get("status", "running")
        if status == "error":
            err = final.get("error") or "image_pull failed"
            raise RuntimeError(err)
        # Refresh resource inventory + disk cache — a new image model
        # just landed on disk.
        try:
            from augmentum.resource.ledger import invalidate as _invalidate_resource
            _invalidate_resource(app.state, "image", disk=True)
        except Exception as exc:
            log.debug("image_pull_resource_invalidate_failed", error=str(exc))
        return final.get("result") or {}

    return handler
