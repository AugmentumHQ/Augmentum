"""REST routes for Bug Finder runs.

Three endpoints:

* ``POST /api/bug-finder/runs`` — start a new run against an existing
  coder workspace (enqueues a background job).
* ``GET  /api/bug-finder/runs?workspace_id=...`` — list the caller's
  runs, optionally scoped to one workspace.
* ``GET  /api/bug-finder/runs/{run_id}`` — full run report.

These routes assume the workspace already exists in coder. There is no
clone/intake path here — that lives in coder mode.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from augmentum.bug_finder.capability import (
    PRIMARY_MODEL_BELOW_MINIMUM,
    capability_floor_label,
    is_capable,
)
from augmentum.bug_finder.store import BugFinderRunStore
from augmentum.bug_finder.stream import (
    BugFinderStreamHub,
    drop_hub,
    get_or_create_hub,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["bug-finder"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BugFinderRunCreate(BaseModel):
    """Body for POST /api/bug-finder/runs."""

    workspace_id: str
    primary_model: str
    """The user's currently-selected model — used for planner, detector,
    fixer. Comes from the same source the rest of coder uses."""

    verifier_model: str = ""
    """Optional per-workspace verifier override. Empty string means
    single-model self-verification (default — avoids model-swap thrash
    on local hardware)."""

    focus_paths: list[str] = Field(default_factory=list)

    detector_models: list[str] = Field(default_factory=list)
    """Optional ensemble of detector model IDs.

    When 2+ models are provided, the detector loop round-robins through
    them across `detector_runs_per_chunk` invocations. Findings flagged
    by 2+ vendor families gain a `families_to_confirm >= 2` signal in
    the report — Anthropic's research identifies correlated within-
    family errors as the dominant FP source.

    Empty list = single-model behavior (uses primary_model for every
    detector run). The verifier and fixer roles are unaffected.
    """

    threat_model: str = ""
    """Optional user-supplied threat model (free-form markdown).

    When provided, prepended to detector + verifier system prompts so
    both subagents work from the same authoritative trust-boundary
    definition. Anthropic's bug-finder research names mismatched
    threat models as the #1 cause of "valid finding but rejected by
    team" (40% FP rate where PoCs proved exploitability but the team
    dismissed them because they didn't fit the project's threat model).

    Recommended sections: Assets, Trust boundaries, Attacker
    capabilities, In scope, Out of scope.
    """

    # Optional tunables — orchestrator defaults apply when absent.
    detector_runs_per_chunk: int | None = None
    detector_concurrency: int | None = None
    max_chunks: int | None = None
    max_fix_attempts_per_finding: int | None = None
    overall_wallclock_seconds: float | None = None

    force_below_minimum: bool = False
    """Opt-in escape hatch for the primary-model capability gate.

    Bug-finder prompts target capable instruction-followers (see
    ``augmentum/bug_finder/capability.py``). Below-floor models tend to
    produce malformed JSON the parsers reject silently, yielding zero-
    finding runs that look successful. This flag bypasses the gate and
    attaches a note to the run report so a zero result can be
    interpreted in context.
    """

    user_goal: Any = None  # dict | str | None — Union not friendly in Pydantic v2
    """Optional structured north-star for this run.

    Two accepted shapes:

    * Full dict — ``{"mode": "named-bug", "description": "...",
      "repro_hint": "...", "scope_paths": [...], "severity_floor":
      "...", "time_budget_minutes": int}``. This is the shape an
      orchestrator agent (coder, companion, MCP client) fills in
      programmatically.
    * Bare string — convenience for direct UI use; treated as the
      ``description`` field in ``explore`` mode.

    Empty / null → default explore-mode goal. The goal renders into
    every subagent's system prompt as the alignment north-star and
    drives downstream ranking once the senior-engineer prioritizer
    lands.
    """

    enable_comprehension: bool | None = None
    """Opt-out: skip the structural comprehension stage on first
    contact. Defaults to True at the orchestrator. Skip for scripted
    one-shot runs that don't benefit from the persisted map."""

    enable_fuzz_leg: bool | None = None
    """Opt-out: skip the atheris fuzz leg. Defaults to True at the
    orchestrator for chunks classified as fuzzable. Disable for runs
    where the fuzz install cost isn't worth it."""

    enable_pen_test_leg: bool | None = None
    """Opt-IN (default False): run the dynamic pen-test leg on
    confirmed findings. Boots the workspace's app via the
    ``pen_test_boot_command`` hint and runs active HTTP probes.
    """

    pen_test_boot_command: str | None = None
    """Hint passed to the pen_tester's ``boot_under_test`` tool.
    Empty / null = let the subagent discover the command itself."""

    pen_test_boot_port: int | None = None
    """Port hint for the under-test app. 0 / null = subagent picks."""

    pen_test_healthcheck_path: str | None = None
    """Healthcheck path the pen_tester uses to verify boot. Default
    ``/`` works for most apps."""

    detector_model: str | None = None
    """Override the detector role's model id (e.g.
    ``'Qwen3.6-35B-A3B-IQ4_XS'``). Null/empty = use
    ``primary_model`` (current behavior). Lets a caller route the
    detector to a local reasoning model while keeping the verifier on
    a cloud model — the standard Mythos-replication shape."""

    detector_temperature: float | None = None
    """Detector sampling temperature. Defaults to 0.0 (the determinism
    floor). Raise to >0 only for variance / thinking-mode experiments.
    The temperature lockdown regression test carves out the detector
    site specifically for this knob."""

    detector_enable_thinking: bool | None = None
    """Per-request ``enable_thinking`` chat-template kwarg for the
    detector. None = let the model's template default apply. True =
    explicitly enable chain-of-thought on Qwen 3.x / GLM-4.x /
    EXAONE 4.x / Nemotron 3 Nano. False = explicitly disable."""

    detector_preserve_thinking: bool | None = None
    """When True, ``<think>`` traces are kept across multi-turn
    detector history (Qwen 3.6 ``preserve_thinking`` chat-template
    kwarg). Other model templates ignore this kwarg."""

    run_mode: str | None = None
    """Chunk-selection strategy. ``'planner'`` (default at orchestrator)
    uses the LLM planner. ``'static_chunk'`` walks the workspace via
    AST and emits one chunk per qualifying function — bypasses the
    planner's token-budget cliff at whole-project scope. Unknown
    values fall through to the planner path."""


class BugFinderRunCreated(BaseModel):
    run_id: str
    job_id: str
    status: str = "pending"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request) -> BugFinderRunStore | None:
    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    return BugFinderRunStore(conn) if conn is not None else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/api/bug-finder/runs", response_model=BugFinderRunCreated)
async def create_run(request: Request, body: BugFinderRunCreate) -> BugFinderRunCreated:
    """Enqueue a new bug-finder run.

    Returns ``{run_id, job_id, status}``. The client polls the standard
    jobs API for progress and the bug-finder runs API for the final
    report once the job completes.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")

    if not body.workspace_id.strip():
        raise HTTPException(status_code=400, detail="workspace_id is required")
    if not body.primary_model.strip():
        raise HTTPException(status_code=400, detail="primary_model is required")
    if not body.force_below_minimum and not is_capable(body.primary_model):
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": PRIMARY_MODEL_BELOW_MINIMUM,
                "message": (
                    f"primary_model '{body.primary_model}' is below the bug-"
                    "finder capability floor. The detector/verifier/fixer "
                    "prompts target capable instruction-followers; below-"
                    "floor models tend to produce malformed JSON the "
                    "parsers silently reject, yielding zero-finding runs "
                    "that look successful. Pick a capable model, or pass "
                    "`force_below_minimum: true` to override."
                ),
                "recommended_floor": capability_floor_label(),
            },
        )

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        raise HTTPException(
            status_code=503, detail="background job queue unavailable",
        )

    run_id = f"bfr_{uuid.uuid4().hex[:12]}"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "workspace_id": body.workspace_id.strip(),
        "primary_model": body.primary_model.strip(),
        "verifier_model": body.verifier_model.strip(),
        "focus_paths": list(body.focus_paths),
        "threat_model": (body.threat_model or "").strip(),
        "detector_models": [m.strip() for m in body.detector_models if m.strip()],
    }
    for opt in (
        "detector_runs_per_chunk",
        "detector_concurrency",
        "max_chunks",
        "max_fix_attempts_per_finding",
        "overall_wallclock_seconds",
    ):
        value = getattr(body, opt)
        if value is not None:
            payload[opt] = value
    if body.force_below_minimum:
        payload["force_below_minimum"] = True
    if body.user_goal is not None:
        payload["user_goal"] = body.user_goal
    if body.enable_comprehension is not None:
        payload["enable_comprehension"] = body.enable_comprehension
    if body.enable_fuzz_leg is not None:
        payload["enable_fuzz_leg"] = body.enable_fuzz_leg
    if body.enable_pen_test_leg is not None:
        payload["enable_pen_test_leg"] = body.enable_pen_test_leg
    if body.pen_test_boot_command is not None:
        payload["pen_test_boot_command"] = body.pen_test_boot_command
    if body.pen_test_boot_port is not None:
        payload["pen_test_boot_port"] = body.pen_test_boot_port
    if body.pen_test_healthcheck_path is not None:
        payload["pen_test_healthcheck_path"] = body.pen_test_healthcheck_path
    if body.detector_model and body.detector_model.strip():
        payload["detector_model"] = body.detector_model.strip()
    if body.detector_temperature is not None:
        payload["detector_temperature"] = body.detector_temperature
    if body.detector_enable_thinking is not None:
        payload["detector_enable_thinking"] = body.detector_enable_thinking
    if body.detector_preserve_thinking is not None:
        payload["detector_preserve_thinking"] = body.detector_preserve_thinking
    if body.run_mode and body.run_mode.strip():
        payload["run_mode"] = body.run_mode.strip()

    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="bug_finder_run",
        payload=payload,
        priority=5,
        # Bug-finder runs are token-expensive, so we don't want to retry
        # them on routine handler errors. BUT a single retry is the
        # difference between "useful subagent" and "always broken on
        # restart" — under autoheal, the augmentum container can cycle
        # mid-run, killing the orchestrator deep inside a long detector
        # loop. With max_attempts=1 that run is permanently lost; with
        # 2, the job_runner re-queues once on restart and the user pays
        # for one wasted partial attempt instead of losing all progress.
        # The handler itself doesn't re-raise on routine errors, so the
        # extra attempt only activates on the restart-crashed path.
        max_attempts=2,
    )
    job_runner.wake()
    log.info(
        "bug_finder_run_created",
        user_id=user_id, run_id=run_id, job_id=job_id,
        workspace_id=body.workspace_id,
        same_model=(not body.verifier_model.strip()),
    )

    return BugFinderRunCreated(run_id=run_id, job_id=job_id, status="pending")


@router.get("/api/bug-finder/runs")
async def list_runs(request: Request, limit: int = 50) -> dict[str, Any]:
    """List the caller's recent runs (newest first), without the heavy report blob."""
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="state store unavailable")
    rows = await store.list_runs(user_id=user_id, limit=max(1, min(int(limit), 200)))
    return {"runs": rows}


@router.get("/api/bug-finder/runs/{run_id}")
async def get_run(request: Request, run_id: str) -> dict[str, Any]:
    """Full run report (including the embedded BugFinderRunReport JSON).

    Read-time reconciliation: when the row says ``running`` but the
    underlying background_job is already terminal (failed / cancelled /
    completed without a report write), we flip the row to a terminal
    error before returning. Closes the "API polls forever" symptom
    that happens when a bug_finder job is killed between
    ``start_run`` and ``complete_run`` — e.g. autoheal-driven restart
    cascade exhausts ``max_attempts=1``, jobs_store correctly marks
    the job ``failed``, but the run row never gets a sync write.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="state store unavailable")
    row = await store.get_run(run_id, user_id=user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")

    if (row.get("stop_reason") or "running") == "running":
        reconciled = await _reconcile_orphan_run(
            request, store, run_id=run_id, user_id=user_id, row=row,
        )
        if reconciled is not None:
            return reconciled
    return row


async def _reconcile_orphan_run(
    request: Request,
    store: BugFinderRunStore,
    *,
    run_id: str,
    user_id: str,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    """Check whether a ``running`` row's underlying job is dead. If so,
    mark the row terminal and return the reconciled row. Returns None
    when no reconcile happened (job still live or unknown)."""
    job_id = (row.get("job_id") or "").strip()
    if not job_id:
        return None
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if jobs_store is None:
        return None
    try:
        job = await jobs_store.get(job_id, user_id=user_id)
    except Exception:  # noqa: BLE001 — reconcile is best-effort
        return None
    if not job:
        return None
    job_status = str(job.get("status") or "").lower()
    if job_status not in {"failed", "cancelled", "completed"}:
        return None
    if job_status == "completed":
        # Handler completed but never wrote the report — unlikely but
        # treat as error so the caller sees a meaningful terminal.
        stop_reason = "error"
        stop_detail = (
            "background job completed but no run report was written"
        )
    elif job_status == "cancelled":
        stop_reason = "cancelled"
        stop_detail = str(job.get("error") or "").strip() or "cancelled"
    else:  # failed
        stop_reason = "error"
        stop_detail = (
            str(job.get("error") or "").strip()
            or "background job failed without writing a report"
        )
    updated = await store.mark_orphaned(
        run_id,
        user_id=user_id,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
    )
    if not updated:
        return None
    log.info(
        "bug_finder_run_reconciled_orphan",
        user_id=user_id, run_id=run_id, job_id=job_id,
        stop_reason=stop_reason, stop_detail=stop_detail[:120],
    )
    return await store.get_run(run_id, user_id=user_id)


@router.get("/api/bug-finder/runs/{run_id}/events")
async def stream_run_events(request: Request, run_id: str) -> StreamingResponse:
    """SSE stream of stage transitions + final result for a run.

    Client connects after creating a run; we replay buffered events
    so the stream renders correctly even if the client took a few
    seconds to attach. ``event: done`` is the terminal signal — the
    server closes the stream after emitting it.

    Events emitted (all wrapped in SSE ``event:`` + ``data:`` lines):

    * ``stage``: ``{stage, progress, ...}`` — plan/detect/verify/fix
      transitions, plus a ``workspace_ready`` boot event with the
      detected language + test command from the baseline.
    * ``done``: terminal payload — stop_reason, finding counts, token
      totals. Subscribers should disconnect after this event.

    Auth scope: same as the polling endpoints — user must own the run.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="state store unavailable")

    # Ownership check via the persisted row. Even if the hub is alive,
    # we refuse to stream to a non-owner.
    row = await store.get_run(run_id, user_id=user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")

    registry = _streams_registry(request)
    hub = get_or_create_hub(registry, run_id)
    queue = hub.subscribe()

    async def _generator():
        # Heartbeat cadence keeps proxies / load balancers from
        # closing an idle stream during long stages (detect can take
        # 5+ minutes on big repos).
        heartbeat_interval = 15.0
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=heartbeat_interval,
                    )
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield event.to_sse()
                if event.terminal:
                    break
        finally:
            hub.unsubscribe(queue)
            if hub.closed and hub.subscriber_count == 0:
                drop_hub(registry, run_id)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _generator(), media_type="text/event-stream", headers=headers,
    )


def _streams_registry(request: Request) -> dict[str, BugFinderStreamHub]:
    """Lazy-init the per-app stream registry. Stored on ``app.state``
    so the job handler can push into the same dict the SSE route
    drains from."""
    registry = getattr(request.app.state, "bug_finder_streams", None)
    if registry is None:
        registry = {}
        request.app.state.bug_finder_streams = registry
    return registry


@router.delete("/api/bug-finder/runs/{run_id}")
async def cancel_run(request: Request, run_id: str) -> dict[str, Any]:
    """Cancel an in-flight bug-finder run.

    Resolves the run's underlying job_id and flips the cancel flag via
    the jobs store. The orchestrator's ``JobCancelled`` handling
    produces a clean ``BugFinderRunReport(stop_reason="cancelled")``,
    which the handler writes back to the row — so the persisted record
    stays consistent without us touching ``bug_finder_runs`` directly.

    Returns ``{cancelled: bool, status: str}`` where ``cancelled`` is
    False when the run is already terminal (complete / error /
    previously cancelled). 404 when the run is unknown or not owned.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="state store unavailable")
    row = await store.get_run(run_id, user_id=user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")

    job_id = (row.get("job_id") or "").strip()
    current_status = (row.get("stop_reason") or "running").strip()
    if current_status in ("complete", "cancelled", "error", "wallclock"):
        return {"cancelled": False, "status": current_status}

    if not job_id:
        # Defensive: row exists but lost its job mapping. Mark the row
        # terminal so the UI stops polling.
        log.warning(
            "bug_finder_cancel_orphan_run",
            user_id=user_id, run_id=run_id,
        )
        return {"cancelled": False, "status": current_status}

    jobs_store = getattr(request.app.state, "jobs_store", None)
    if jobs_store is None:
        raise HTTPException(
            status_code=503, detail="background job queue unavailable",
        )

    cancelled = await jobs_store.request_cancel(job_id, user_id=user_id)
    log.info(
        "bug_finder_run_cancel_requested",
        user_id=user_id, run_id=run_id, job_id=job_id,
        signalled=cancelled,
    )
    return {
        "cancelled": cancelled,
        "status": "cancelled" if cancelled else current_status,
    }


@router.delete("/api/bug-finder/knowledge/{workspace_id}")
async def forget_workspace_knowledge(
    request: Request, workspace_id: str,
) -> dict[str, Any]:
    """Delete the cached comprehender knowledge map for a workspace.

    Forces the next bug_finder run to re-comprehend the codebase from
    scratch. Two valid use cases:

    * **Variance / regression measurement** — between consecutive runs
      on the same target so each run starts with the same blank
      knowledge state. Without this, the comprehender's first-contact
      output biases every subsequent run's planner.
    * **Stale-map recovery** — when the workspace's code has changed
      substantially since the last run and the cached brief no longer
      matches reality (e.g. a refactor renamed half the modules).

    Idempotent: returns ``{forgotten: bool}`` where ``forgotten`` is
    False when no row existed in the first place. Always returns 200
    when the call shape is valid — there's no error case the caller
    needs to distinguish."""
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")

    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        raise HTTPException(status_code=503, detail="state store unavailable")

    from augmentum.bug_finder.knowledge_store import KnowledgeStore
    knowledge_store = KnowledgeStore(conn)
    before = await knowledge_store.get(
        user_id=user_id, workspace_id=workspace_id,
    )
    await knowledge_store.forget(user_id=user_id, workspace_id=workspace_id)
    log.info(
        "bug_finder_knowledge_forgotten",
        user_id=user_id, workspace_id=workspace_id,
        had_brief=bool(before.brief if before else False),
    )
    return {"forgotten": bool(before and before.is_populated)}
