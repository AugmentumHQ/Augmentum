"""Coding Driver routes — the Agents window's backend.

One list across both engines (internal missions + external bridge agents),
dispatch of an internal run, the review diff (what a run changed), and
interrupt. User-scoped throughout; workspace ownership checked at the edge.
See augmentum/coder/coding_driver.py and docs/superpowers/specs/
2026-07-20-companion-coding-driver-design.md.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from augmentum.coder import coding_driver
from augmentum.coder.external.providers import public_providers
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["coding"])


def _uid(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


class DispatchRequest(BaseModel):
    driver: str = "internal"          # internal | harness
    workspace_id: str = ""            # internal
    task: str = ""
    model: str = ""                   # internal (never auto-picked)
    coder_strategy: str = ""
    agent_session_id: str = ""        # harness: which live agent to assign
    origin_surface: str = "coder"


@router.get("/api/coding/runs")
async def list_runs(request: Request) -> JSONResponse:
    """Unified list: internal coding runs + live external bridge agents,
    newest first. The two engines are handled differently underneath but
    surface as one stream the Agents window renders."""
    uid = _uid(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    workspace_id = str(request.query_params.get("workspace_id") or "").strip()
    state = request.app.state

    internal = await coding_driver.list_internal_runs(
        state, user_id=uid, workspace_id=workspace_id, limit=50,
    )

    # External driver: surface live bridge agents (observe-only for now).
    external: list[dict] = []
    try:
        from augmentum.proxy import agent_bridge
        agents = await agent_bridge.list_agents(state, user_id=uid)
        for a in agents:
            external.append({
                "id": a.get("agent_id", ""),
                "driver": "harness",
                "engine_ref": a.get("agent_id", ""),
                "workspace_id": "",
                "task": a.get("title", ""),
                "model": "",
                "base_commit": "",
                "run_id": "",
                "status": _map_agent_status(a.get("status", "")),
                "summary": a.get("summary", ""),
                "origin_surface": a.get("harness", "") or "harness",
                "pending_requests": a.get("pending_requests", 0),
                "created_at": a.get("created_at", ""),
                "updated_at": a.get("last_seen", ""),
            })
    except Exception:
        log.warning("coding_runs_external_list_failed", exc_info=True)

    runs = internal + external
    return JSONResponse({"runs": runs})


def _map_agent_status(s: str) -> str:
    return {"working": "working", "waiting": "working", "done": "done"}.get(
        str(s or "").lower(), "working")


def _norm_status(s: str) -> str:
    """Normalize every engine's status vocabulary to one set."""
    return {
        "running": "working", "in_progress": "working", "working": "working",
        "queued": "queued", "pending": "queued",
        "done": "done", "completed": "done", "succeeded": "done",
        "failed": "failed", "error": "failed",
        "cancelled": "cancelled", "canceled": "cancelled", "detached": "detached",
    }.get(str(s or "").lower(), str(s or "").lower() or "working")


def _conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    return getattr(getattr(sm, "backend", None), "conn", None) if sm else None


@router.get("/api/coder/agents/runs")
async def unified_agent_runs(request: Request) -> JSONResponse:
    """One faithful history across every engine, grouped by DERIVED locus.

    Internal (runs in an Augmentum workspace): local missions (coding_runs
    driver=internal) + Claude Code SDK-in-container (claude_runs). External
    (bare machine): bridge assignments (coding_runs driver=harness) + pushed
    pi terminal mirrors (pi_runs). Normalized to one row shape so the surface
    renders every agent the same way. Tool counts come from the *_run_events
    tables. Read-only; user-scoped.
    """
    uid = _uid(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"error": "state backend unavailable"}, status_code=503)
    workspace_id = str(request.query_params.get("workspace_id") or "").strip()
    internal: list[dict] = []
    external: list[dict] = []

    # 1) coding_runs — local missions (internal) + bridge assignments (external)
    try:
        runs = await coding_driver.list_internal_runs(
            request.app.state, user_id=uid, workspace_id=workspace_id, limit=50)
        for r in runs:
            row = {
                "id": r["id"], "agent": "augmentum",
                "locus": "external" if r["driver"] == "harness" else "internal",
                "goal": r["task"], "status": _norm_status(r["status"]),
                "model": r["model"], "turns": None, "tools": None,
                "cost_usd": None, "duration_ms": None, "result": r.get("summary", ""),
                "where": r["workspace_id"], "run_id": r.get("run_id", ""),
                "review_turn_id": r.get("review_turn_id", ""),
                "source": "coding_run", "updated_at": r["updated_at"],
            }
            (external if row["locus"] == "external" else internal).append(row)
    except Exception:
        log.warning("unified_runs_coding_failed", exc_info=True)

    # 2) claude_runs — Claude Code SDK in a workspace container (internal)
    try:
        where = "user_id = ?"
        params: list = [uid]
        if workspace_id:
            where += " AND workspace_id = ?"
            params.append(workspace_id)
        cur = await conn.execute(
            "SELECT id, workspace_id, task, status, outcome, cost_usd, num_turns, "
            "duration_ms, session_id, updated_at, "
            "(SELECT COUNT(*) FROM claude_run_events e WHERE e.run_id = r.id AND e.tool != ''), "
            "model "
            f"FROM claude_runs r WHERE {where} ORDER BY updated_at DESC LIMIT 40",
            params)
        for x in await cur.fetchall():
            internal.append({
                # Real model Claude ran (from stream init); "" until captured.
                "id": x[0], "agent": "claude", "locus": "internal",
                "goal": x[2], "status": _norm_status(x[3]), "model": x[11] or "",
                "turns": x[6], "tools": x[10], "cost_usd": x[5],
                "duration_ms": x[7], "result": x[4], "where": x[1],
                "run_id": x[0], "review_turn_id": "", "session_id": x[8],
                "source": "claude_run", "updated_at": x[9],
            })
    except Exception:
        log.warning("unified_runs_claude_failed", exc_info=True)

    # 3) pi_runs — pushed pi terminal mirrors on the user's own machine (external)
    try:
        cur = await conn.execute(
            "SELECT id, project, title, status, outcome, num_turns, model, updated_at, "
            "(SELECT COUNT(*) FROM pi_run_events e WHERE e.run_id = r.id AND e.tool != '') "
            "FROM pi_runs r WHERE user_id = ? ORDER BY updated_at DESC LIMIT 40",
            [uid])
        for x in await cur.fetchall():
            external.append({
                "id": x[0], "agent": "pi", "locus": "external",
                "goal": x[2] or x[1], "status": _norm_status(x[3]), "model": x[6],
                "turns": x[5], "tools": x[8], "cost_usd": None,
                "duration_ms": None, "result": x[4], "where": x[1],
                "run_id": x[0], "review_turn_id": "",
                "source": "pi_run", "updated_at": x[7],
            })
    except Exception:
        log.warning("unified_runs_pi_failed", exc_info=True)

    internal.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    external.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return JSONResponse({"internal": internal, "external": external})


@router.get("/api/coder/agents/providers")
async def agent_providers(request: Request) -> JSONResponse:
    """External-agent capabilities for the composer: which agents exist, how
    they dispatch, and (for model-targetable ones) their model catalog. The
    composer builds its Model control from this — provider-neutral, so adding
    Codex is a data edit in providers.py, not a UI/route change."""
    if not _uid(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({"providers": public_providers()})


@router.post("/api/coding/runs")
async def dispatch_run(body: DispatchRequest, request: Request) -> JSONResponse:
    """Dispatch a coding run. driver='internal' → Augmentum drives a
    background mission; driver='harness' → assign the task to a live external
    agent via the bridge (it picks it up at its next check-in)."""
    uid = _uid(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    task = (body.task or "").strip()
    if not task:
        return JSONResponse({"error": "task is required"}, status_code=400)
    driver_kind = (body.driver or "internal").strip().lower()

    if driver_kind == "harness":
        agent_id = (body.agent_session_id or "").strip()
        if not agent_id:
            return JSONResponse(
                {"error": "agent_session_id is required for the harness driver "
                          "(pick which live agent to assign)"}, status_code=400)
        driver = coding_driver.HarnessCoderDriver(request.app.state)
        result = await driver.dispatch(
            user_id=uid, agent_session_id=agent_id, task=task,
            origin_surface=(body.origin_surface or "coder").strip())
        if not result.get("ok"):
            return JSONResponse({"error": result.get("error") or "assign failed"},
                                status_code=400)
        return JSONResponse(result)

    # internal
    model = (body.model or "").strip()
    workspace_id = (body.workspace_id or "").strip()
    if not model:
        # Never auto-select a model on the user's behalf.
        return JSONResponse({"error": "model is required"}, status_code=400)
    if not workspace_id:
        return JSONResponse({"error": "workspace_id is required"}, status_code=400)

    # Edge ownership check (mirrors coder_routes._owns_workspace).
    try:
        from augmentum.proxy.coder_routes import _owns_workspace
        if not await _owns_workspace(request, workspace_id):
            return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception:
        pass

    driver = coding_driver.InternalCoderDriver(request.app.state)
    result = await driver.dispatch(
        user_id=uid, workspace_id=workspace_id, task=task, model=model,
        coder_strategy=(body.coder_strategy or "").strip(),
        origin_surface=(body.origin_surface or "coder").strip(),
    )
    if not result.get("ok"):
        return JSONResponse({"error": result.get("error") or "dispatch failed"},
                            status_code=400)
    return JSONResponse(result)


@router.get("/api/coding/runs/{run_id}/diff")
async def run_diff(run_id: str, request: Request) -> JSONResponse:
    """The review diff — what this run changed since it was dispatched."""
    uid = _uid(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    run = await coding_driver.get_run(request.app.state, user_id=uid, run_id=run_id)
    if run is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    if run["driver"] != "internal":
        return JSONResponse(
            {"error": "diff is only available for internal runs (external "
                      "agents run on their own machine)", "stat": "", "patch": ""},
            status_code=200)
    cm = getattr(request.app.state, "container_manager", None)
    if cm is None:
        return JSONResponse({"error": "container manager unavailable"}, status_code=503)
    try:
        diff = await cm.git_review_diff(run["workspace_id"], run["base_commit"])
    except Exception as exc:
        return JSONResponse({"error": f"diff failed: {exc}"}, status_code=500)
    return JSONResponse({
        "run_id": run_id, "workspace_id": run["workspace_id"],
        "task": run["task"], "status": run["status"], **diff,
    })


@router.post("/api/coding/runs/{run_id}/interrupt")
async def interrupt_run(run_id: str, request: Request) -> JSONResponse:
    uid = _uid(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    driver = coding_driver.InternalCoderDriver(request.app.state)
    result = await driver.interrupt(user_id=uid, run_id=run_id)
    if not result.get("ok"):
        return JSONResponse({"error": result.get("error") or "interrupt failed"},
                            status_code=400)
    return JSONResponse(result)
