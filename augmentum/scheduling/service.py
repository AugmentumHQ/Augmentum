"""SchedulerService — the app-level dispatcher for standing tasks.

Before this service, standing tasks only fired through the companion
runtime's ``tick_scheduler`` verb, which (a) required
``companion_runtime_enabled`` and (b) dispatched ONLY for the bound
owner — every other user's schedules sat in the table and never ran.
This service closes both gaps:

  * **Companion OFF** — the service builds a *headless* dispatch
    context (an unstarted :class:`CompanionRuntime`: ``__init__`` is
    I/O-free and provides exactly what the engine and task kinds reach
    for — ``backend``, ``companion_id``, ``memory``, ``_app_state``)
    and dispatches every user's due tasks itself.
  * **Companion ON** — the companion's tick verb keeps dispatching the
    OWNER's tasks (presence-gated, verb-ledger-cited, unchanged); this
    service picks up every OTHER user, using the real runtime as the
    context so runners get full wiring.

Presence gate: user-created schedules are the user's own explicit ask,
not companion initiative, so this dispatcher does NOT apply the
presence/silent autonomy gate (``respect_presence_gate=False``). The
companion tick path keeps its historical gating.

Cadence mirrors the tick verb: a 60s scan, ONE due task per user per
scan, 3-minute wallclock ceiling per run (the standing-task kinds'
internal budgets all sit below that; see tick_scheduler.py for the
2026-06-08 incident that set the number).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)

_SCAN_INTERVAL_S = 60
_RUN_WALLCLOCK_S = 180  # per-run ceiling, matches tick_scheduler's envelope


class SchedulerService:
    """60s scan loop dispatching due standing tasks across all users."""

    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        app_state: Any,
        companion_runtime: CompanionRuntime | None = None,
        companion_id: str = "becca",
    ) -> None:
        self._backend = backend
        self._app_state = app_state
        self._companion_runtime = companion_runtime
        self._companion_id = companion_id
        self._ctx: CompanionRuntime | None = companion_runtime
        self._task: asyncio.Task | None = None
        self._stopping = False

    # ── context ─────────────────────────────────────────────────────

    def _ensure_ctx(self) -> CompanionRuntime:
        """Dispatch context for the engine + task-kind runners.

        The real runtime when the companion is on; otherwise a headless
        (never ``start()``-ed) CompanionRuntime — no behavior loops, no
        tick ladder, no persona work. Construction is I/O-free.
        """
        if self._ctx is None:
            from augmentum.companion_runtime.runtime import CompanionRuntime
            self._ctx = CompanionRuntime(
                self._backend,
                companion_id=self._companion_id,
                app_state=self._app_state,
            )
            # Wire the memory facade so journal delivery works headless —
            # runtime.start() would normally do this. Best-effort: without
            # it, task results still notify (hub path); only the drawer
            # note is skipped (and logged) by _surface_result.
            try:
                store = getattr(self._app_state, "memory_store", None)
                core = getattr(self._app_state, "core_profile_manager", None)
                if store is not None and core is not None:
                    self._ctx.memory.attach(store, core)
            except Exception:
                log.warning("scheduler_headless_memory_attach_failed",
                            exc_info=True)
            log.info("scheduler_headless_ctx_created")
        return self._ctx

    @property
    def ctx(self) -> CompanionRuntime:
        """Context for creation surfaces (tools/routes) to CRUD tasks
        against when ``app.state.companion_runtime`` is absent."""
        return self._ensure_ctx()

    # ── lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(
            self._loop(), name="scheduler-service",
        )
        log.info(
            "scheduler_service_started",
            companion_dispatcher=self._companion_runtime is not None,
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        log.info("scheduler_service_stopped")

    # ── scan loop ───────────────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("scheduler_scan_failed", exc_info=True)
            await asyncio.sleep(_SCAN_INTERVAL_S)

    async def _due_user_ids(self) -> list[str]:
        cur = await self._backend.conn.execute(
            """SELECT DISTINCT user_id FROM companion_standing_tasks
               WHERE companion_id = ? AND enabled = 1
                 AND (next_run_at IS NULL OR next_run_at <= datetime('now'))
               ORDER BY user_id""",
            (self._companion_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [r[0] for r in rows if r[0]]

    async def _scan_once(self) -> None:
        from augmentum.config import settings
        if not getattr(settings, "companion_standing_tasks_enabled", True):
            return

        users = await self._due_user_ids()
        if not users:
            return

        # When the companion dispatcher is live, the owner's tasks are
        # its lane (presence-gated, cited on the verb ledger) — skip
        # them here so a due task can't double-fire across dispatchers.
        skip_owner = ""
        if self._companion_runtime is not None:
            skip_owner = getattr(
                self._companion_runtime, "owner_user_id", "",
            ) or ""

        ctx = self._ensure_ctx()
        from augmentum.companion_runtime import standing_tasks
        for uid in users:
            if uid == skip_owner:
                continue
            started = time.monotonic()
            try:
                ran = await asyncio.wait_for(
                    standing_tasks.step(
                        ctx, user_id=uid, respect_presence_gate=False,
                    ),
                    timeout=_RUN_WALLCLOCK_S,
                )
            except TimeoutError:
                # step() shields its persist on cancellation, so the
                # row's schedule still advanced (budget-timeout path).
                log.warning(
                    "scheduler_run_wallclock_exceeded",
                    user_id=uid,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
                continue
            except Exception:
                log.warning(
                    "scheduler_user_step_failed", user_id=uid, exc_info=True,
                )
                continue
            if ran is not None:
                log.debug("scheduler_task_ran", user_id=uid, task_id=ran)


__all__ = ["SchedulerService"]
