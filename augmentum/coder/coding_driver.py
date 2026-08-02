"""The Coding Driver — dispatch a coding task, whichever engine runs it.

One record (``coding_runs``, migration 319), two engines with genuinely
different handling:

- **InternalCoderDriver** — Augmentum DRIVES the loop: capture the workspace
  HEAD as the diff anchor, then enqueue a ``coder_background_run`` job (the
  existing headless mission stack). Full control, works for every user with
  zero external setup.
- **HarnessCoderDriver** — Augmentum DELEGATES + OBSERVES: assign a task to a
  live checked-in external agent via the agent bridge; it runs the loop on the
  user's machine and self-reports. (Assignment lands here in a later phase;
  the unified list already surfaces external agents alongside internal runs.)

Both are observed through the same record so the Agents window can show one
list, live status, and — via ``git_review_diff`` anchored on ``base_commit`` —
exactly what each run changed. Everything is user-scoped.
"""

from __future__ import annotations

import uuid
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

DRIVERS = ("internal", "harness")
_ACTIVE = ("queued", "working")


def _conn(app_state: Any):
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


# ── record CRUD ────────────────────────────────────────────────────────

async def create_run(
    app_state: Any, *, user_id: str, driver: str, workspace_id: str, task: str,
    model: str = "", base_commit: str = "", engine_ref: str = "",
    origin_surface: str = "coder", status: str = "queued",
) -> str | None:
    conn = _conn(app_state)
    if conn is None or not user_id:
        return None
    if driver not in DRIVERS:
        driver = "internal"
    run_id = f"cr_{uuid.uuid4().hex[:12]}"
    await conn.execute(
        "INSERT INTO coding_runs (id, user_id, driver, engine_ref, workspace_id, "
        "task, model, base_commit, origin_surface, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [run_id, user_id, driver, engine_ref, workspace_id, task, model,
         base_commit, origin_surface, status],
    )
    await conn.commit()
    return run_id


async def update_run(
    app_state: Any, *, user_id: str, run_id: str, status: str = "",
    summary: str = "", engine_ref: str = "", broker_run_id: str = "",
    review_turn_id: str = "",
) -> None:
    conn = _conn(app_state)
    if conn is None or not user_id or not run_id:
        return
    await conn.execute(
        "UPDATE coding_runs SET "
        "status = COALESCE(NULLIF(?, ''), status), "
        "summary = COALESCE(NULLIF(?, ''), summary), "
        "engine_ref = COALESCE(NULLIF(?, ''), engine_ref), "
        "run_id = COALESCE(NULLIF(?, ''), run_id), "
        "review_turn_id = COALESCE(NULLIF(?, ''), review_turn_id), "
        "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
        [status, summary, engine_ref, broker_run_id, review_turn_id, run_id, user_id],
    )
    await conn.commit()


async def get_run(app_state: Any, *, user_id: str, run_id: str) -> dict | None:
    conn = _conn(app_state)
    if conn is None or not user_id or not run_id:
        return None
    cur = await conn.execute(
        "SELECT id, driver, engine_ref, workspace_id, task, model, base_commit, "
        "run_id, status, summary, origin_surface, created_at, updated_at, "
        "review_turn_id "
        "FROM coding_runs WHERE id = ? AND user_id = ?",
        [run_id, user_id],
    )
    r = await cur.fetchone()
    if r is None:
        return None
    return _row_to_dict(r)


async def list_internal_runs(
    app_state: Any, *, user_id: str, workspace_id: str = "", limit: int = 50,
) -> list[dict]:
    conn = _conn(app_state)
    if conn is None or not user_id:
        return []
    where = "user_id = ?"
    params: list[Any] = [user_id]
    if workspace_id:
        where += " AND workspace_id = ?"
        params.append(workspace_id)
    cur = await conn.execute(
        "SELECT id, driver, engine_ref, workspace_id, task, model, base_commit, "
        "run_id, status, summary, origin_surface, created_at, updated_at, "
        "review_turn_id "
        f"FROM coding_runs WHERE {where} ORDER BY updated_at DESC LIMIT ?",
        [*params, int(limit)],
    )
    rows = [_row_to_dict(r) for r in await cur.fetchall()]
    # Live status enrichment from the jobs store (the job is authoritative).
    await _enrich_status(app_state, user_id, rows)
    return rows


def _row_to_dict(r) -> dict:
    return {
        "id": r[0], "driver": r[1], "engine_ref": r[2], "workspace_id": r[3],
        "task": r[4], "model": r[5], "base_commit": r[6], "run_id": r[7],
        "status": r[8], "summary": r[9], "origin_surface": r[10],
        "created_at": r[11], "updated_at": r[12], "review_turn_id": r[13],
    }


# Map jobs-store status → coding-run status vocabulary.
_JOB_STATUS = {
    "pending": "queued", "queued": "queued", "running": "working",
    "in_progress": "working", "succeeded": "done", "completed": "done",
    "done": "done", "failed": "failed", "error": "failed",
    "cancelled": "cancelled", "canceled": "cancelled",
}


async def _enrich_status(app_state: Any, user_id: str, rows: list[dict]) -> None:
    """For internal runs still marked active, read the live job status so the
    Agents window reflects reality without a background sync loop."""
    jobs_store = getattr(app_state, "jobs_store", None)
    if jobs_store is None:
        return
    for row in rows:
        if row["driver"] != "internal" or not row["engine_ref"]:
            continue
        if row["status"] not in _ACTIVE:
            continue
        try:
            job = await jobs_store.get(row["engine_ref"])
        except Exception:
            job = None
        if not job:
            continue
        raw = str((job.get("status") if isinstance(job, dict)
                   else getattr(job, "status", "")) or "").lower()
        mapped = _JOB_STATUS.get(raw)
        # On completion the job result carries the broker run_id + the
        # review_turn_id — the two handles the Agents window needs to reuse
        # the live stream and the real review panel. Persist them once.
        result = job.get("result") if isinstance(job, dict) else None
        review_turn_id = broker_run = ""
        if isinstance(result, dict):
            review_turn_id = str(result.get("review_turn_id") or "")
            broker_run = str(result.get("run_id") or "")
        if mapped and mapped != row["status"]:
            row["status"] = mapped
            await update_run(app_state, user_id=user_id, run_id=row["id"],
                             status=mapped, review_turn_id=review_turn_id,
                             broker_run_id=broker_run)
        elif review_turn_id or broker_run:
            await update_run(app_state, user_id=user_id, run_id=row["id"],
                             review_turn_id=review_turn_id, broker_run_id=broker_run)
        if review_turn_id and not row.get("review_turn_id"):
            row["review_turn_id"] = review_turn_id
        if broker_run and not row.get("run_id"):
            row["run_id"] = broker_run


# ── internal driver ────────────────────────────────────────────────────

class InternalCoderDriver:
    """Augmentum drives the loop via the coder_background_run job."""

    def __init__(self, app_state) -> None:
        self._app = app_state

    async def dispatch(
        self, *, user_id: str, workspace_id: str, task: str, model: str,
        coder_strategy: str = "", origin_surface: str = "coder",
    ) -> dict:
        if not (user_id and workspace_id and task and model):
            return {"ok": False, "error": "user_id, workspace_id, task, model required"}
        jobs_store = getattr(self._app, "jobs_store", None)
        job_runner = getattr(self._app, "job_runner", None)
        if jobs_store is None or job_runner is None:
            return {"ok": False, "error": "background jobs unavailable"}

        # Diff anchor: HEAD at dispatch. Best-effort — an uninitialized repo
        # just yields an empty base (diff shows nothing until first commit).
        base_commit = ""
        cm = getattr(self._app, "container_manager", None)
        if cm is not None:
            try:
                base_commit = await cm.git_head_short(workspace_id)
            except Exception:
                base_commit = ""

        run_id = await create_run(
            self._app, user_id=user_id, driver="internal",
            workspace_id=workspace_id, task=task, model=model,
            base_commit=base_commit, origin_surface=origin_surface,
        )
        if run_id is None:
            return {"ok": False, "error": "could not create coding run"}

        job_id = await jobs_store.create(
            user_id=user_id,
            job_type="coder_background_run",
            payload={
                "workspace_id": workspace_id,
                "prompt": task,
                "model": model,
                "coder_strategy": coder_strategy,
            },
            priority=5,
            max_attempts=2,
        )
        await update_run(self._app, user_id=user_id, run_id=run_id,
                         engine_ref=job_id, status="queued")
        job_runner.wake()
        log.info("coding_run_dispatched", user_id=user_id, run_id=run_id,
                 workspace_id=workspace_id, job_id=job_id, model=model)
        return {"ok": True, "run_id": run_id, "job_id": job_id,
                "base_commit": base_commit}

    async def interrupt(self, *, user_id: str, run_id: str) -> dict:
        run = await get_run(self._app, user_id=user_id, run_id=run_id)
        if run is None:
            return {"ok": False, "error": "run not found"}
        jobs_store = getattr(self._app, "jobs_store", None)
        if jobs_store is not None and run["engine_ref"]:
            try:
                await jobs_store.request_cancel(run["engine_ref"], user_id=user_id)
            except Exception:
                log.warning("coding_run_cancel_failed", run_id=run_id, exc_info=True)
        await update_run(self._app, user_id=user_id, run_id=run_id, status="cancelled")
        return {"ok": True, "run_id": run_id}


class HarnessCoderDriver:
    """Augmentum delegates + observes: assign a task to a live external agent
    (Claude Code / pi) via the bridge. The agent runs the loop on the user's
    machine and picks the assignment up at its next check-in."""

    def __init__(self, app_state) -> None:
        self._app = app_state

    async def dispatch(
        self, *, user_id: str, agent_session_id: str, task: str,
        origin_surface: str = "coder", harness: str = "", project: str = "",
    ) -> dict:
        if not (user_id and agent_session_id and task):
            return {"ok": False, "error": "user_id, agent_session_id, task required"}
        from augmentum.proxy import agent_bridge
        # Create the status row FIRST so the assignment can carry its id — that
        # link lets check-in advance the row (queued → working → done) as the
        # agent on the user's machine picks the task up and reports back.
        run_id = await create_run(
            self._app, user_id=user_id, driver="harness",
            workspace_id="", task=task, engine_ref=agent_session_id,
            origin_surface=origin_surface, status="queued",
        )
        assigned = await agent_bridge.create_assignment(
            self._app, user_id=user_id, agent_session_id=agent_session_id,
            task=task, harness=harness, project=project, linked_run_id=run_id or "",
        )
        if not assigned:
            # Roll the orphaned status row back to a terminal state so it doesn't
            # linger as a phantom 'queued' run.
            if run_id:
                await update_run(self._app, user_id=user_id, run_id=run_id,
                                 status="failed", summary="agent not found")
            return {"ok": False, "error": "no such agent (or it isn't yours)"}
        log.info("coding_run_assigned", user_id=user_id, run_id=run_id,
                 agent_session_id=agent_session_id)
        return {"ok": True, "run_id": run_id,
                "request_id": assigned.get("request_id", ""),
                "agent_session_id": agent_session_id}
