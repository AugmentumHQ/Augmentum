"""Async generation queue with single worker and job tracking."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from augmentum.image.schemas import JobStatus, JobType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget background tasks."""
    if not task.cancelled() and task.exception():
        # exc_info=exc preserves the traceback; str(exc) alone strips it and
        # turns tensor off-by-ones into unfindable needles (see 2026-04-23
        # "index 26 is out of bounds for dimension 0 with size 26" with no
        # stack frame to point at).
        exc = task.exception()
        log.error("background_task_failed", error=str(exc), exc_info=exc)


@dataclass
class GenerationJob:
    """A pending or completed image generation job."""

    job_id: str = ""
    job_type: JobType = JobType.TXT2IMG
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    preset: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = ""
    scheduler: str = ""
    loras: list[dict] = field(default_factory=list)
    session_id: str = ""
    # img2img / inpaint fields
    source_image: str = ""  # base64 or file path
    mask_image: str = ""    # base64 or file path (inpaint only)
    strength: float = 0.75
    mask_blur: int = 4
    inpaint_mode: str = "default"  # "default" | "improve" | "modify"
    source_image_id: str = ""  # original image_id for provenance
    condense_model: str = ""   # LLM model for prompt condensation
    enhance_prompt: bool = True   # Whether to run LLM prompt enhancement
    condense_prompt: bool = True  # Whether to condense prompt if over token limit
    inpaint_full_res: bool = False
    inpaint_padding: int = 32
    # Per-generation quality optimizations
    guidance_rescale: float = 0.0
    hires_fix: bool = False
    hires_scale: float = 1.5
    hires_denoise: float = 0.5
    clip_skip: int | None = None       # Skip last N CLIP layers (SD1.5/SDXL)
    ip_adapter_image: str | list[str] = ""  # single or multiple ref images
    ip_adapter_scale: float = 0.55   # IP-Adapter influence strength
    user_id: str = ""  # Owner for multi-tenant persistence
    # Provenance, not silos: 'companion' when she initiated the
    # generation (tool call / architect dispatch), '' = user. Flows to
    # image_generations.origin so the gallery can filter on it.
    origin: str = ""
    priority: int = 0  # Higher = more important (dequeued first)
    # Source category for progress-indicator routing. ``user`` covers
    # explicit clicks (Illustrate moment, Generate scene image, Image
    # Studio panel); ``auto_bg`` is the per-turn narrative background
    # autogenerator. Polling endpoints filter by this so the in-message
    # loader and the corner-of-scene badge don't BOTH light up for the
    # same job. Default ``user`` because explicit clicks are the
    # majority path; the auto-bg path overrides this on its own jobs.
    category: str = "user"
    status: JobStatus = JobStatus.QUEUED
    stage: str = ""  # Human-readable progress stage (e.g. "Loading model", "Generating")
    # Determinate step progress for the diffusion loop. ``steps_total``
    # is set to ``job.steps`` when the diffusion call begins;
    # ``steps_done`` increments via the diffusers ``callback_on_step_end``
    # hook so UIs can render a real progress bar instead of an
    # indeterminate spinner. Both stay at 0 outside the diffusion phase
    # (model load, prompt distillation, VAE decode, save) — UIs use
    # ``stage`` text in those phases and the bar only during diffusion.
    steps_total: int = 0
    steps_done: int = 0
    started_at: float = 0.0  # monotonic timestamp when job reached RUNNING — drives elapsed display
    result: dict | None = None
    error: str = ""
    finished_at: float = 0.0  # monotonic timestamp when job reached COMPLETED/FAILED
    future: asyncio.Future | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = uuid.uuid4().hex[:16]


class GenerationQueue:
    """Async queue for image generation with a single worker.

    GPU-bound work doesn't benefit from parallelism, so we process
    jobs sequentially with a single worker.
    """

    _STALE_TIMEOUT_S: float = 300  # 5 minutes
    _FINISHED_RETAIN_S: float = 60  # keep finished jobs for status polling

    def __init__(self, max_size: int = 10) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, GenerationJob]] = (
            asyncio.PriorityQueue(maxsize=max_size)
        )
        self._jobs: dict[str, GenerationJob] = {}
        self._worker_task: asyncio.Task | None = None
        self._running = False
        self._generate_fn = None  # Set by start()
        self._max_size = max_size
        self._submit_counter: int = 0
        # Track the currently running generation task so it can be cancelled
        self._current_task: asyncio.Task | None = None
        self._current_job_id: str | None = None

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running

    def get_job(self, job_id: str) -> GenerationJob | None:
        return self._jobs.get(job_id)

    def start(self, generate_fn) -> None:
        """Start the queue worker.

        Args:
            generate_fn: Async callable(GenerationJob) -> dict that performs
                the actual image generation and returns a result dict.
        """
        if self._running:
            return
        self._generate_fn = generate_fn
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        self._worker_task.add_done_callback(self._on_worker_done)
        log.info("generation_queue_started", max_size=self._max_size)

    def _on_worker_done(self, task: asyncio.Task) -> None:
        """Auto-restart the worker if it crashes unexpectedly."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("queue_worker_crashed", error=str(exc))
            if self._running:
                log.info("queue_worker_restarting")
                self._worker_task = asyncio.create_task(self._worker())
                self._worker_task.add_done_callback(self._on_worker_done)

    async def stop(self) -> None:
        """Stop the queue worker."""
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        if self._worker_task:
            self._worker_task.cancel()
            import contextlib
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        log.info("generation_queue_stopped")

    def active_count_for_user(self, user_id: str) -> int:
        """Count this user's not-yet-finished jobs (QUEUED or RUNNING).

        Finished jobs linger in ``_jobs`` for ~60s before cleanup, so we
        count only in-flight ones — the figure that matters for fairness.
        """
        if not user_id:
            return 0
        return sum(
            1 for j in self._jobs.values()
            if j.user_id == user_id
            and j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        )

    async def submit(self, job: GenerationJob) -> GenerationJob:
        """Submit a job to the queue. Returns the job with a future attached."""
        # Per-user fairness cap — stop one caller from monopolising the
        # shared GPU queue on a multi-tenant box (the global ``full()`` check
        # below treats all users as one pool). Default-on; 0 disables.
        from augmentum.config import settings as _cfg
        _per_user = int(getattr(_cfg, "image_max_inflight_per_user", 0) or 0)
        if (
            _per_user > 0
            and job.user_id
            and self.active_count_for_user(job.user_id) >= _per_user
        ):
            raise RuntimeError(
                f"Too many image jobs in progress (limit {_per_user} per user) "
                "— try again shortly"
            )
        if self._queue.full():
            raise RuntimeError(f"Generation queue is full ({self._max_size} jobs)")

        loop = asyncio.get_running_loop()
        job.future = loop.create_future()
        job._queued_at = time.monotonic()  # type: ignore[attr-defined]
        self._jobs[job.job_id] = job
        self._submit_counter += 1
        await self._queue.put((-job.priority, self._submit_counter, job))

        log.info("job_submitted", job_id=job.job_id, priority=job.priority, queue_size=self._queue.qsize())
        return job

    async def wait_for_result(self, job: GenerationJob, timeout: float = 600.0) -> dict:
        """Wait for a job to complete and return its result.

        If the timeout expires OR the caller is cancelled (Phase 9.5),
        the running generation task is cancelled so the worker is
        freed for the next job. Without the caller-cancel handling
        below, a browser disconnecting mid-render would leak the
        generation: the route handler's task gets CancelledError,
        wait_for_result re-raises, but the underlying job keeps
        running on the GPU until natural completion — wasted compute
        + queue slot blocked from the next user.
        """
        if not job.future:
            raise RuntimeError("Job has no future — was it submitted?")
        try:
            return await asyncio.wait_for(job.future, timeout=timeout)
        except TimeoutError:
            self.cancel_job(job.job_id)
            raise
        except asyncio.CancelledError:
            # Browser cancel / route teardown / parent task cancel.
            # Kill the underlying job before re-raising so the
            # queue + GPU are freed up.
            self.cancel_job(job.job_id)
            raise

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Returns True if cancellation was initiated, False if the job was
        already finished or not found.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return False

        if job.status == JobStatus.RUNNING and self._current_job_id == job_id:
            # Cancel the in-progress generation task
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                log.info("job_cancelled_running", job_id=job_id)
                return True

        if job.status == JobStatus.QUEUED:
            # Mark as failed so the worker skips it when dequeued
            job.status = JobStatus.FAILED
            job.error = "Cancelled"
            job.finished_at = time.monotonic()
            if job.future and not job.future.done():
                job.future.set_exception(
                    TimeoutError("Job cancelled")
                )
            log.info("job_cancelled_queued", job_id=job_id)
            return True

        return False

    def get_position(self, job_id: str) -> int:
        """Get the position of a job in the queue (0 = currently processing)."""
        job = self._jobs.get(job_id)
        if not job:
            return -1
        if job.status == JobStatus.RUNNING:
            return 0
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return -1
        # Count queued jobs ahead of this one
        pos = 1
        for jid, j in self._jobs.items():
            if jid == job_id:
                break
            if j.status == JobStatus.QUEUED:
                pos += 1
        return pos

    def _cleanup_stale_jobs(self) -> None:
        """Auto-cancel jobs that have been QUEUED for too long."""
        now = time.monotonic()
        for job in list(self._jobs.values()):
            if job.status == JobStatus.QUEUED and hasattr(job, "_queued_at"):
                if now - job._queued_at > self._STALE_TIMEOUT_S:  # type: ignore[attr-defined]
                    job.status = JobStatus.FAILED
                    job.error = "Job expired (queued too long)"
                    job.finished_at = now
                    if job.future and not job.future.done():
                        job.future.set_exception(TimeoutError(job.error))
                    log.warning("stale_job_cancelled", job_id=job.job_id)

    def _cleanup_finished_jobs(self) -> None:
        """Remove finished jobs older than the retention window."""
        now = time.monotonic()
        to_remove = [
            jid
            for jid, j in self._jobs.items()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            and j.finished_at > 0
            and now - j.finished_at > self._FINISHED_RETAIN_S
        ]
        for jid in to_remove:
            del self._jobs[jid]
        if to_remove:
            log.info("finished_jobs_cleaned", count=len(to_remove))

    async def _worker(self) -> None:
        """Main worker loop: process jobs sequentially."""
        while self._running:
            self._cleanup_stale_jobs()
            self._cleanup_finished_jobs()
            try:
                _, _, job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Skip jobs that were cancelled while queued
            if job.status == JobStatus.FAILED:
                self._queue.task_done()
                continue

            job.status = JobStatus.RUNNING
            job.started_at = time.monotonic()
            self._current_job_id = job.job_id
            log.info("job_processing", job_id=job.job_id, prompt_chars=len(job.prompt))
            log.debug("job_processing_prompt", job_id=job.job_id, prompt=job.prompt[:50])

            try:
                # Run generation in a separate task so it can be cancelled
                # independently of the worker loop
                self._current_task = asyncio.create_task(self._generate_fn(job))
                self._current_task.add_done_callback(_task_done_callback)
                result = await self._current_task
                job.status = JobStatus.COMPLETED
                job.result = result
                job.finished_at = time.monotonic()
                if job.future and not job.future.done():
                    job.future.set_result(result)
                log.info("job_completed", job_id=job.job_id)
            except asyncio.CancelledError:
                job.status = JobStatus.FAILED
                job.error = "Cancelled"
                job.finished_at = time.monotonic()
                if job.future and not job.future.done():
                    job.future.set_exception(
                        TimeoutError("Generation cancelled")
                    )
                log.warning("job_cancelled", job_id=job.job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = time.monotonic()
                if job.future and not job.future.done():
                    job.future.set_exception(exc)
                log.error("job_failed", job_id=job.job_id, error=str(exc), exc_info=exc)
            finally:
                self._current_task = None
                self._current_job_id = None
                self._queue.task_done()
