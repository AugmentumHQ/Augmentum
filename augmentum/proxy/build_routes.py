"""FastAPI routes for first-class Build Mode runs."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from augmentum.builds.runtime import (
    _utc_now_iso,
    apply_project_progress,
    build_status_from_run,
    build_status_snapshot,
    load_persisted_build_run,
    progress_payload_from_state,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/builds", tags=["builds"])

# Cap on the in-memory tool-step trail surfaced in the build snapshot. A build
# is bounded to ~40 iterations, so this is a safety ceiling, not a normal limit.
_MAX_BUILD_STEPS = 400


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request):
    return getattr(request.app.state, "build_run_store", None)


def _get_active(build_id: str, *, user_id: str):
    from augmentum.modes.passthrough.handler import ACTIVE_BUILDS

    build = ACTIVE_BUILDS.get(build_id)
    if not build:
        return None
    owner = build.get("user_id") or ""
    if owner and (not user_id or owner != user_id):
        return None
    return build


def _make_build_event_sink(build_state: dict):
    """Bridge facade events into an ACTIVE_BUILDS entry.

    The facade emits ``stage`` (lifecycle) and ``build_progress`` (per
    loop-iteration tool activity) events. We merge them into the in-memory
    build_state and pulse ``_change_event`` so the existing SSE stream
    (``build_status_snapshot``) renders the build accumulating live — no new
    streaming machinery needed.
    """

    def sink(kind: str, payload: dict, terminal: bool = False) -> None:
        try:
            if payload.get("name"):
                build_state["name"] = payload["name"]
            if payload.get("workspace_id"):
                build_state["workspace_id"] = payload["workspace_id"]

            if kind == "build_progress":
                iteration = int(payload.get("iteration") or 0)
                phase = payload.get("phase") or ""
                tool = payload.get("tool") or ""
                preview = (payload.get("preview") or "").strip()
                label = (f"{tool}: {preview}".strip(": ") or tool or phase)[:120]
                build_state["currentFile"] = label
                build_state["totalTokens"] = int(payload.get("tokens_in") or 0) + int(payload.get("tokens_out") or 0)
                build_state["llmCalls"] = iteration
                # Ordered tool trail — the observable record the library monitor
                # renders live and the builder_eval harness judges. Only
                # tool_call phases are real "steps the agent took"; responding/
                # tool_result/done phases just update the live label above.
                if phase == "tool_call" and tool:
                    steps = build_state.setdefault("steps", [])
                    steps.append({
                        "i": len(steps),
                        "iteration": iteration,
                        "tool": tool,
                        "preview": preview[:200],
                        "ts": _utc_now_iso(),
                    })
                    if len(steps) > _MAX_BUILD_STEPS:
                        del steps[: len(steps) - _MAX_BUILD_STEPS]
                apply_project_progress(build_state, {
                    "pass": "build", "status": "running",
                    "iteration": iteration, "detail": label,
                })
            elif kind == "stage":
                stage = payload.get("stage", "")
                # Behavior contract rides several stages (planning/verifying/
                # fixing/complete) — keep the latest snapshot live.
                if isinstance(payload.get("behaviors"), list):
                    build_state["behaviors"] = payload["behaviors"]
                if stage in ("complete", "error"):
                    status = (payload.get("status") or "").lower()
                    # Budget/stuck stops PAUSE (non-fatal, resumable) rather than
                    # fail — preserve that so the UI shows a continue/stop gate.
                    if stage == "error":
                        build_state["status"] = "error"
                    elif status == "paused":
                        build_state["status"] = "paused"
                    elif status in ("completed", "complete", ""):
                        build_state["status"] = "complete"
                    else:
                        build_state["status"] = "error"
                    build_state["awaiting_continue"] = build_state["status"] == "paused"
                    if payload.get("stop_reason"):
                        build_state["stop_reason"] = payload["stop_reason"]
                    build_state["completed_at"] = _utc_now_iso()
                    if payload.get("error"):
                        build_state["error"] = payload["error"]
                    aid = payload.get("artifact_id", "")
                    if aid:
                        build_state["artifact_id"] = aid
                    # Inline quality verdict from the facade's floor judge.
                    verdict = payload.get("verdict") or {}
                    quality_status = payload.get("qualityStatus") or "clean"
                    warnings = payload.get("warnings") or []
                    blocking = payload.get("blockingErrors") or []
                    if verdict:
                        build_state["verdict"] = verdict
                    build_state["qualityStatus"] = quality_status
                    build_state["quality_status"] = quality_status
                    if warnings:
                        build_state["warnings"] = warnings
                    if blocking:
                        build_state["blockingErrors"] = blocking
                        build_state["blocking_errors"] = blocking
                    build_state["project"] = {
                        "name": build_state.get("name", ""),
                        "artifactId": aid,
                        "artifact_id": aid,
                        "workspaceId": build_state.get("workspace_id", ""),
                        "status": build_state["status"],
                        "qualityStatus": quality_status,
                        "quality_status": quality_status,
                        "warnings": warnings,
                        "blockingErrors": blocking,
                        "behaviors": build_state.get("behaviors") or [],
                    }
                else:
                    label = {
                        "planning_checks": "deriving acceptance checks",
                        "verifying": "verifying behaviors in a browser",
                        "fixing": "fixing failing behaviors",
                    }.get(stage, stage)
                    apply_project_progress(build_state, {
                        "pass": "build", "status": "running", "detail": label,
                    })
        except Exception:  # noqa: BLE001 — a sink error must never break the build
            log.debug("build_event_sink_failed", kind=kind, exc_info=True)
        finally:
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()

    return sink


@router.post("")
async def create_build(request: Request) -> JSONResponse:
    """Kick off an autonomous build (the coder-harness build-test-fix loop).

    Creates a workspace, runs the agent under the Frontend App Builder Power,
    streams progress through the existing build monitor, and publishes the
    result to the library. Returns immediately with the build_id; follow
    ``GET /api/builds/{id}/stream`` for live progress.
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    objective = str(body.get("description") or body.get("objective") or "").strip()
    if not objective:
        raise HTTPException(status_code=400, detail="A build description is required")
    model = str(body.get("model") or "").strip()
    profile_id = str(body.get("profile") or body.get("profile_id") or "static").strip() or "static"
    session_id = str(body.get("session_id") or "")

    store = _get_store(request)
    cm = getattr(request.app.state, "container_manager", None)
    artifact_store = getattr(request.app.state, "artifact_store", None)
    pr = getattr(request.app.state, "provider_registry", None)
    power_registry = getattr(request.app.state, "power_registry", None)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    if cm is None:
        raise HTTPException(status_code=503, detail="Container manager not available")
    if artifact_store is None:
        raise HTTPException(status_code=503, detail="Artifact storage not available")
    if pr is None:
        raise HTTPException(status_code=503, detail="Model provider registry not available")

    backend, clean_model = await pr.resolve_backend_with_fabric(model, user_id=uid)
    if backend is None:
        raise HTTPException(status_code=503, detail="No model backend available — select a model first")
    resolved_model = clean_model or model

    from augmentum.builds.facade import run_build
    from augmentum.modes.passthrough.handler import ACTIVE_BUILDS

    build_id = f"build_{uuid.uuid4().hex[:16]}"
    build_state: dict = {
        "id": build_id,
        "kind": "application",
        "user_id": uid,
        "session_id": session_id,
        "task_id": "",
        "started_at": asyncio.get_event_loop().time(),
        "started_at_iso": _utc_now_iso(),
        "model": resolved_model,
        "name": "",
        "status": "running",
        "passes": [],
        "error": None,
        "project": None,
        "workspace_id": "",
        "_checkpoint": {},
        "_change_event": asyncio.Event(),
    }
    ACTIVE_BUILDS[build_id] = build_state
    sink = _make_build_event_sink(build_state)

    async def _runner() -> None:
        try:
            await run_build(
                objective=objective,
                user_id=uid,
                backend=backend,
                model=resolved_model,
                container_manager=cm,
                artifact_store=artifact_store,
                build_run_store=store,
                power_registry=power_registry,
                profile_id=profile_id,
                session_id=session_id,
                build_id=build_id,
                event_sink=sink,
            )
        except asyncio.CancelledError:
            build_state["status"] = "cancelled"
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()
            raise
        except Exception as exc:  # noqa: BLE001 — surface as a failed build
            log.warning("build_run.task_failed", build_id=build_id, error=str(exc), exc_info=True)
            build_state["status"] = "error"
            build_state["error"] = str(exc)
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()

    build_state["_task"] = asyncio.create_task(_runner())
    log.info("build_run.started", build_id=build_id, profile=profile_id, model=resolved_model)
    return JSONResponse(
        {"build_id": build_id, "status": "running", "model": resolved_model, "name": ""},
        status_code=201,
    )


@router.post("/from-artifact")
async def build_from_artifact(request: Request) -> JSONResponse:
    """Rebuild / continue an existing application artifact in a coder workspace.

    This is the "Rebuild" action of the artifact workspace (ui/workspace.js),
    migrated off the retired quickjs pipeline (`/api/artifacts/iterate`) onto
    the SAME coder builder as build mode + the Library button. We seed a fresh
    workspace from the artifact's current files, then run ``run_build`` with the
    user's change request as ``instructions`` (resume framing) so the agent
    modifies the existing app and browser-verifies it, rather than regenerating
    blind. The workspace persists for "Open in Code" continuation, and a new
    artifact is published. Body: ``{artifact_id, instructions|description,
    model?, session_id?, name?}``.
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    artifact_id = str(body.get("artifact_id") or "").strip()
    instructions = str(body.get("instructions") or body.get("description") or "").strip()
    if not artifact_id:
        raise HTTPException(status_code=400, detail="artifact_id is required")
    if not instructions:
        raise HTTPException(status_code=400, detail="A change description is required")
    model = str(body.get("model") or "").strip()
    session_id = str(body.get("session_id") or "")
    name_hint = str(body.get("name") or "").strip()

    store = _get_store(request)
    cm = getattr(request.app.state, "container_manager", None)
    artifact_store = getattr(request.app.state, "artifact_store", None)
    pr = getattr(request.app.state, "provider_registry", None)
    power_registry = getattr(request.app.state, "power_registry", None)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    if cm is None:
        raise HTTPException(status_code=503, detail="Container manager not available")
    if artifact_store is None:
        raise HTTPException(status_code=503, detail="Artifact storage not available")
    if pr is None:
        raise HTTPException(status_code=503, detail="Model provider registry not available")

    backend, clean_model = await pr.resolve_backend_with_fabric(model, user_id=uid)
    if backend is None:
        raise HTTPException(status_code=503, detail="No model backend available — select a model first")
    resolved_model = clean_model or model

    artifact = await artifact_store.get(artifact_id, user_id=uid)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        src_data = json.loads(artifact.get("source_json") or "") if isinstance(artifact.get("source_json"), str) else (artifact.get("source_json") or {})
    except json.JSONDecodeError:
        src_data = {}
    ws_name = name_hint or (src_data or {}).get("name") or artifact.get("title") or "App"
    # For an incremental edit the OBJECTIVE is the CHANGE itself. This scopes the
    # derived behavior contract — and the agent's "verify every behavior" framing
    # — to the requested change, instead of re-deriving and re-verifying the
    # ENTIRE original app. Passing the whole-app objective here is what turned a
    # one-line "make the play button blue" into a full-game re-audit that burned
    # ~2.4M tokens and then discarded the (correct) change.
    objective = instructions

    # Reuse the artifact's existing coder workspace if one is still alive (from
    # a prior coder build/rebuild) instead of spinning up a fresh 2-3GB
    # container on every rebuild. Only seed a new workspace from the artifact
    # files when there's no live one to continue on.
    prior_ws = str(
        (artifact.get("metadata") or {}).get("workspace_id")
        or (src_data or {}).get("workspace_id")
        or (src_data or {}).get("workspaceId")
        or "",
    )
    workspace_id = ""
    if prior_ws and await _ensure_workspace_alive(cm, prior_ws):
        workspace_id = prior_ws
        log.info("build_from_artifact.reuse_workspace", workspace_id=workspace_id, artifact_id=artifact_id)
    if not workspace_id:
        workspace_id = await _workspace_from_artifact(
            request, uid=uid, artifact_id=artifact_id, name=ws_name,
        )
    if not workspace_id:
        raise HTTPException(
            status_code=409,
            detail="Couldn't recreate a workspace from this artifact (no usable files).",
        )

    from augmentum.builds.budgets import build_budget
    from augmentum.builds.facade import run_build
    from augmentum.modes.passthrough.handler import ACTIVE_BUILDS

    # Tier-scaled EDIT checkpoint budget: a light change checks in sooner than a
    # full build, and weaker/local models get more room. Reaching it PAUSES and
    # asks the user to continue or stop — it never force-fails — so this is a
    # check-in cadence, not a wall.
    edit_budget = build_budget(resolved_model, "edit")

    build_id = f"build_{uuid.uuid4().hex[:16]}"
    # resume=True skips run_build's own row creation, so we create it here.
    await store.create(
        user_id=uid, build_id=build_id, session_id=session_id, kind="application",
        status="running", name=ws_name,
        request={
            "objective": objective, "model": resolved_model,
            "from_artifact": artifact_id, "instructions": instructions,
        },
        profile_id="static", target="inline",
    )
    build_state: dict = {
        "id": build_id,
        "kind": "application",
        "user_id": uid,
        "session_id": session_id,
        "task_id": "",
        "started_at": asyncio.get_event_loop().time(),
        "started_at_iso": _utc_now_iso(),
        "model": resolved_model,
        "name": ws_name,
        "status": "running",
        "passes": [],
        "error": None,
        "project": None,
        "workspace_id": workspace_id,
        "_checkpoint": {},
        "_change_event": asyncio.Event(),
    }
    ACTIVE_BUILDS[build_id] = build_state
    sink = _make_build_event_sink(build_state)

    async def _runner() -> None:
        try:
            await run_build(
                objective=objective, user_id=uid, backend=backend, model=resolved_model,
                container_manager=cm, artifact_store=artifact_store, build_run_store=store,
                power_registry=power_registry, session_id=session_id, build_id=build_id,
                event_sink=sink, reuse_workspace_id=workspace_id, resume=True,
                instructions=instructions, prior_steps=[], prior_stop_reason="completed",
                budget=edit_budget,
            )
        except asyncio.CancelledError:
            build_state["status"] = "cancelled"
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()
            raise
        except Exception as exc:  # noqa: BLE001 — surface as a failed build
            log.warning("build_from_artifact.failed", build_id=build_id, error=str(exc), exc_info=True)
            build_state["status"] = "error"
            build_state["error"] = str(exc)
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()

    build_state["_task"] = asyncio.create_task(_runner())
    log.info("build_from_artifact.started", build_id=build_id, artifact_id=artifact_id,
             workspace_id=workspace_id, model=resolved_model)
    return JSONResponse(
        {"build_id": build_id, "status": "running", "workspace_id": workspace_id,
         "model": resolved_model, "name": ws_name},
        status_code=201,
    )


@router.get("")
async def list_build_runs(
    request: Request,
    session_id: str = "",
    limit: int = 50,
) -> JSONResponse:
    store = _get_store(request)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    uid = _user_id(request)
    runs = await store.list_for_session(session_id, user_id=uid, limit=limit)
    return JSONResponse({"runs": runs})


@router.get("/{build_id}")
async def get_build_run(build_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    active = _get_active(build_id, user_id=uid)
    if active:
        return JSONResponse({"run": build_status_snapshot(active, build_id)})
    store = _get_store(request)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    run = await load_persisted_build_run(store, build_id=build_id, user_id=uid)
    if not run:
        raise HTTPException(status_code=404, detail="Build run not found")
    return JSONResponse({"run": run, "status": build_status_from_run(run)})


@router.get("/{build_id}/stream")
async def stream_build_run(build_id: str, request: Request):
    uid = _user_id(request)
    store = _get_store(request)

    async def event_stream():
        last_serialised = ""
        heartbeat = 25.0
        deadline = asyncio.get_event_loop().time() + 1800
        while True:
            if await request.is_disconnected():
                return
            if asyncio.get_event_loop().time() > deadline:
                yield "event: end\ndata: {\"reason\":\"max-duration\"}\n\n"
                return

            active = _get_active(build_id, user_id=uid)
            if active:
                snap = build_status_snapshot(active, build_id)
            elif store:
                run = await load_persisted_build_run(store, build_id=build_id, user_id=uid)
                snap = build_status_from_run(run)
            else:
                snap = {"active": False}

            payload = json.dumps(snap, default=str)
            if payload != last_serialised:
                yield f"data: {payload}\n\n"
                last_serialised = payload

            if snap.get("status") in ("complete", "error", "cancelled", "paused"):
                # "paused" is idle-awaiting-user — close the stream and let the
                # UI render the continue/stop gate; a resume re-opens a stream.
                yield "event: end\ndata: {\"reason\":\"terminal\"}\n\n"
                return

            ev = active.get("_change_event") if active else None
            if ev is not None:
                try:
                    await asyncio.wait_for(ev.wait(), timeout=heartbeat)
                except TimeoutError:
                    pass
                ev.clear()
            else:
                await asyncio.sleep(heartbeat)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/{build_id}/cancel")
async def cancel_build_run(build_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    active = _get_active(build_id, user_id=uid)
    if not active:
        return JSONResponse({"cancelled": False, "reason": "build not running"}, status_code=404)
    active["status"] = "cancelled"
    active["_cancel"] = True
    # Cancel the running build task (the autonomous loop) if one is attached.
    task = active.get("_task")
    if task is not None and not task.done():
        task.cancel()
    ev = active.get("_change_event")
    if ev is not None:
        ev.set()
    store = _get_store(request)
    if store:
        await store.update(
            build_id,
            user_id=uid,
            status="canceled",
            progress=progress_payload_from_state(active),
        )
    log.info("build_run.cancelled", build_id=build_id)
    return JSONResponse({"cancelled": True, "build_id": build_id})


async def _ensure_workspace_alive(cm, workspace_id: str) -> bool:
    """True if the workspace exists and is now running. Starts a stopped/paused
    one; returns False if it's gone from Docker entirely (caller rebuilds)."""
    if not cm or not workspace_id:
        return False
    try:
        workspaces = await cm.list_workspaces()
    except Exception:  # noqa: BLE001
        log.warning("build_resume.list_workspaces_failed", exc_info=True)
        return False
    ws = next((w for w in workspaces if getattr(w, "id", "") == workspace_id), None)
    if ws is None:
        return False
    if getattr(ws, "status", "") == "running":
        return True
    try:
        await cm.start(workspace_id)
        return True
    except Exception:  # noqa: BLE001 — container gone / unstartable → rebuild path
        log.warning("build_resume.workspace_start_failed", workspace_id=workspace_id, exc_info=True)
        return False


async def _workspace_from_artifact(request: Request, *, uid: str, artifact_id: str, name: str) -> str:
    """Recreate a workspace from a published application artifact. Returns the
    new workspace id, or "" if the artifact is missing/unusable."""
    artifact_store = getattr(request.app.state, "artifact_store", None)
    cm = getattr(request.app.state, "container_manager", None)
    if not artifact_store or cm is None or not artifact_id:
        return ""
    artifact = await artifact_store.get(artifact_id, user_id=uid)
    if not artifact:
        return ""
    source = artifact.get("source_json") or ""
    try:
        source_data = json.loads(source) if isinstance(source, str) else source
    except json.JSONDecodeError:
        return ""
    if not isinstance(source_data, dict) or source_data.get("type") != "application":
        return ""
    payload: list[tuple[str, bytes]] = []
    for f in source_data.get("files") or []:
        path = str(f.get("path") or "").replace("\\", "/").lstrip("/")
        if not path or path.startswith("../") or "/../" in f"/{path}/":
            continue
        payload.append((path, str(f.get("content") or "").encode("utf-8")))
    if not payload:
        return ""
    info = await cm.create_workspace(
        name=name or source_data.get("name") or "Built App",
        publish_ports=True, tooling_profile="browser", user_id=uid,
    )
    await cm.file_upload(info.id, "/workspace", payload)
    await cm.git_checkpoint(info.id, f"Restored for resume of {artifact_id}")
    return info.id


@router.post("/{build_id}/resume")
async def resume_build_run(build_id: str, request: Request) -> JSONResponse:
    """Continue a stopped or finished build on its existing workspace.

    Workspace-as-checkpoint: the prior workspace (or, if it's gone, one rebuilt
    from the published artifact) is handed back to the autonomous loop with a
    continuation prompt — so a budget-exhausted build finishes its work and a
    completed build can be re-prompted ("also add dark mode"), instead of
    restarting from scratch. Body: ``{instructions?: str, model?: str}``.
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")

    store = _get_store(request)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    run = await store.get(build_id, user_id=uid)
    if not run:
        raise HTTPException(status_code=404, detail="Build run not found")

    # Reject resuming a build that's still actively running. A PAUSED build is
    # in ACTIVE_BUILDS too but idle (awaiting the user's continue) — allow it.
    active = _get_active(build_id, user_id=uid)
    if active is not None and (active.get("status") or "").lower() == "running":
        raise HTTPException(status_code=409, detail="Build is still running")

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    instructions = str(body.get("instructions") or "").strip()
    model_override = str(body.get("model") or "").strip()

    req = run.get("request") or {}
    result = run.get("result") or {}
    objective = str(req.get("objective") or run.get("name") or "").strip()
    if not objective:
        raise HTTPException(status_code=400, detail="Build has no original objective to resume")
    model = model_override or str(req.get("model") or "")
    profile_id = run.get("profile_id") or "static"
    prior_stop_reason = str(result.get("stop_reason") or run.get("status") or "")
    prior_steps = (run.get("progress") or {}).get("steps") or []

    cm = getattr(request.app.state, "container_manager", None)
    artifact_store = getattr(request.app.state, "artifact_store", None)
    pr = getattr(request.app.state, "provider_registry", None)
    power_registry = getattr(request.app.state, "power_registry", None)
    if cm is None or artifact_store is None or pr is None:
        raise HTTPException(status_code=503, detail="Build services not available")

    # Resolve the backend FIRST (cheap fail before any workspace work, and so a
    # transiently-unavailable model returns a clean 503 instead of a 500 — the
    # resolver RAISES ModelUnavailableError rather than returning None).
    try:
        backend, clean_model = await pr.resolve_backend_with_fabric(model, user_id=uid)
    except Exception as exc:  # noqa: BLE001 — surface as a retryable 503
        log.warning("build_run.resume_model_unavailable", build_id=build_id, model=model, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is unavailable right now — try again in a moment.",
        ) from exc
    if backend is None:
        raise HTTPException(status_code=503, detail="No model backend available — select a model first")
    resolved_model = clean_model or model

    # Resolve the workspace: reuse the live one, else rebuild from the artifact.
    workspace_id = run.get("workspace_id") or result.get("workspace_id") or ""
    artifact_id = run.get("artifact_id") or result.get("artifact_id") or ""
    ready = await _ensure_workspace_alive(cm, workspace_id)
    if not ready:
        workspace_id = await _workspace_from_artifact(
            request, uid=uid, artifact_id=artifact_id, name=run.get("name") or objective,
        )
        if not workspace_id:
            raise HTTPException(
                status_code=409,
                detail="This build's workspace is gone and it has no artifact to rebuild from — start a new build instead.",
            )

    # Flip the row back to running + bump the resume counter. The status guard
    # inside begin_resume is the authority on "is this resumable".
    new_count = await store.begin_resume(build_id, user_id=uid)
    if not new_count:
        raise HTTPException(status_code=409, detail="Build is not in a resumable state")

    from augmentum.builds.facade import run_build
    from augmentum.modes.passthrough.handler import ACTIVE_BUILDS

    build_state: dict = {
        "id": build_id,
        "kind": "application",
        "user_id": uid,
        "session_id": run.get("session_id", ""),
        "task_id": "",
        "started_at": asyncio.get_event_loop().time(),
        "started_at_iso": _utc_now_iso(),
        "model": resolved_model,
        "name": run.get("name", ""),
        "status": "running",
        "passes": [],
        "error": None,
        "project": None,
        "workspace_id": workspace_id,
        # Carry the prior trail so the live monitor continues it visually.
        "steps": list(prior_steps),
        "resume_count": new_count,
        "_checkpoint": {},
        "_change_event": asyncio.Event(),
    }
    ACTIVE_BUILDS[build_id] = build_state
    sink = _make_build_event_sink(build_state)

    async def _runner() -> None:
        try:
            await run_build(
                objective=objective, user_id=uid, backend=backend, model=resolved_model,
                container_manager=cm, artifact_store=artifact_store, build_run_store=store,
                power_registry=power_registry, profile_id=profile_id,
                session_id=run.get("session_id", ""), build_id=build_id, event_sink=sink,
                reuse_workspace_id=workspace_id, resume=True, instructions=instructions,
                prior_steps=prior_steps, prior_stop_reason=prior_stop_reason, kind=profile_id,
            )
        except asyncio.CancelledError:
            build_state["status"] = "cancelled"
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("build_run.resume_failed", build_id=build_id, error=str(exc), exc_info=True)
            build_state["status"] = "error"
            build_state["error"] = str(exc)
            ev = build_state.get("_change_event")
            if ev is not None:
                ev.set()

    build_state["_task"] = asyncio.create_task(_runner())
    log.info("build_run.resumed", build_id=build_id, resume_count=new_count,
             workspace_id=workspace_id, has_instructions=bool(instructions))
    return JSONResponse(
        {"build_id": build_id, "status": "running", "model": resolved_model,
         "resume_count": new_count, "workspace_id": workspace_id},
        status_code=202,
    )


@router.post("/{build_id}/ack")
async def ack_build_run(build_id: str, request: Request) -> JSONResponse:
    """Mark a terminal build as dismissed so it stops resurfacing on
    subsequent /build-status polls (cross-device). No-op for already-acked
    or unknown builds — the client UX is fire-and-forget."""
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")
    store = _get_store(request)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    acked = await store.mark_acked(build_id, user_id=uid)
    return JSONResponse({"acked": acked, "build_id": build_id})


@router.post("/{build_id}/open-in-code")
async def open_build_in_code(build_id: str, request: Request) -> JSONResponse:
    """Create a coder workspace from an application build artifact."""
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")

    store = _get_store(request)
    if not store:
        raise HTTPException(status_code=503, detail="Build run store not available")
    run = await store.get(build_id, user_id=uid)
    if not run:
        raise HTTPException(status_code=404, detail="Build run not found")

    artifact_id = run.get("artifact_id") or (run.get("result") or {}).get("artifact_id") or ""
    if not artifact_id:
        project = (run.get("result") or {}).get("project") or {}
        artifact_id = project.get("artifactId") or project.get("artifact_id") or ""
    if not artifact_id:
        raise HTTPException(status_code=400, detail="Build has no saved artifact")

    artifact_store = getattr(request.app.state, "artifact_store", None)
    if not artifact_store:
        raise HTTPException(status_code=503, detail="Artifact storage not available")
    artifact = await artifact_store.get(artifact_id, user_id=uid)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    source = artifact.get("source_json") or ""
    try:
        source_data = json.loads(source) if isinstance(source, str) else source
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Artifact has invalid source_json") from exc
    if not isinstance(source_data, dict) or source_data.get("type") != "application":
        raise HTTPException(status_code=400, detail="Artifact is not an application")

    files = source_data.get("files") or []
    payload: list[tuple[str, bytes]] = []
    for f in files:
        path = str(f.get("path") or "").replace("\\", "/").lstrip("/")
        if not path or path.startswith("../") or "/../" in f"/{path}/":
            continue
        payload.append((path, str(f.get("content") or "").encode("utf-8")))
    if not payload:
        raise HTTPException(status_code=400, detail="Application artifact has no files")

    mgr = getattr(request.app.state, "container_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Container manager not available")

    name = run.get("name") or source_data.get("name") or artifact.get("display_name") or "Built App"
    info = await mgr.create_workspace(
        name=name,
        publish_ports=True,
        tooling_profile="browser",
        user_id=uid,
    )
    await mgr.file_upload(info.id, "/workspace", payload)
    await mgr.git_checkpoint(info.id, f"Imported build {build_id}")
    return JSONResponse({
        "workspace_id": info.id,
        "workspace": info.__dict__,
        "artifact_id": artifact_id,
        "build_id": build_id,
    }, status_code=201)


@router.post("/artifacts/{artifact_id}/open-in-code")
async def open_artifact_in_code(artifact_id: str, request: Request) -> JSONResponse:
    """Create a coder workspace from a saved application artifact."""
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthorized")

    artifact_store = getattr(request.app.state, "artifact_store", None)
    if not artifact_store:
        raise HTTPException(status_code=503, detail="Artifact storage not available")
    artifact = await artifact_store.get(artifact_id, user_id=uid)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    source = artifact.get("source_json") or ""
    try:
        source_data = json.loads(source) if isinstance(source, str) else source
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Artifact has invalid source_json") from exc
    if not isinstance(source_data, dict) or source_data.get("type") != "application":
        raise HTTPException(status_code=400, detail="Artifact is not an application")

    payload: list[tuple[str, bytes]] = []
    for f in source_data.get("files") or []:
        path = str(f.get("path") or "").replace("\\", "/").lstrip("/")
        if not path or path.startswith("../") or "/../" in f"/{path}/":
            continue
        payload.append((path, str(f.get("content") or "").encode("utf-8")))
    if not payload:
        raise HTTPException(status_code=400, detail="Application artifact has no files")

    mgr = getattr(request.app.state, "container_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Container manager not available")

    name = source_data.get("name") or artifact.get("display_name") or "Built App"
    info = await mgr.create_workspace(
        name=name,
        publish_ports=True,
        tooling_profile="browser",
        user_id=uid,
    )
    await mgr.file_upload(info.id, "/workspace", payload)
    await mgr.git_checkpoint(info.id, f"Imported artifact {artifact_id}")
    return JSONResponse({
        "workspace_id": info.id,
        "workspace": info.__dict__,
        "artifact_id": artifact_id,
    }, status_code=201)
