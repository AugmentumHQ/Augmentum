"""HTTP endpoints for pushed pi (terminal agent) session mirrors.

The pi CLI runs on the user's host; with sync opted in (/sync on in pi),
its augmentum-missions extension pushes a normalized mirror of each
session here so terminal work is first-class in Augmentum — listed in the
web coder Agents panel next to Claude Code runs, transcript reopenable,
host session file recorded for `pi --session <file>` resume on the host.

Unlike external_coder_routes (which OWNS its runs via the RunManager and
spawns the CLI server-side), these routes are a passive sink: the host
owns the process; we persist what it pushes. The SSE stream is therefore
a replay + DB poll by seq (no bus, nothing server-owned to subscribe to).

Routes (prefix /api/coder/external/pi):
  POST /runs                    {run_id, project, session_file, title, model}
  POST /runs/{id}/events        {events: [{seq, kind, text, tool, path}, …]}
  POST /runs/{id}/finish        {status, outcome, error, files_changed, num_turns}
  GET  /runs?project=&limit=    run metadata, newest first
  GET  /runs/{id}?since_seq=    run + events (incremental via since_seq)
  GET  /runs/{id}/stream        SSE: replay persisted events, then poll by seq
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from augmentum.coder.external import pi_run_store
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/coder/external/pi", tags=["external-coder"])


def _conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None)


def _uid(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


class _RunBody(BaseModel):
    run_id: str = ""
    project: str = ""
    session_file: str = ""
    title: str = ""
    model: str = ""


class _EventsBody(BaseModel):
    events: list[dict] = Field(default_factory=list)


class _FinishBody(BaseModel):
    status: str = "detached"
    outcome: str = ""
    error: str = ""
    files_changed: list[str] = Field(default_factory=list)
    num_turns: int = 0


@router.post("/runs")
async def upsert_run(body: _RunBody, request: Request) -> JSONResponse:
    conn, uid = _conn(request), _uid(request)
    if conn is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    run_id = (body.run_id or "").strip()
    if not run_id:
        return JSONResponse({"error": "run_id required"}, status_code=400)
    try:
        await pi_run_store.upsert_run(
            conn, run_id=run_id, user_id=uid, project=body.project,
            session_file=body.session_file, title=body.title, model=body.model,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("pi_run_upsert_failed", run_id=run_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "run_id": run_id})


@router.post("/runs/{run_id}/events")
async def add_events(run_id: str, body: _EventsBody, request: Request) -> JSONResponse:
    conn, uid = _conn(request), _uid(request)
    if conn is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    if not body.events:
        return JSONResponse({"ok": True, "inserted": 0})
    try:
        inserted = await pi_run_store.add_events(
            conn, run_id=run_id, user_id=uid, events=body.events,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("pi_run_events_failed", run_id=run_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "inserted": inserted})


@router.post("/runs/{run_id}/finish")
async def finish_run(run_id: str, body: _FinishBody, request: Request) -> JSONResponse:
    conn, uid = _conn(request), _uid(request)
    if conn is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    try:
        await pi_run_store.finish_run(
            conn, run_id=run_id, user_id=uid, status=body.status,
            outcome=body.outcome, error=body.error,
            files_changed=body.files_changed, num_turns=body.num_turns,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("pi_run_finish_failed", run_id=run_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True})


@router.get("/runs")
async def list_runs(request: Request, project: str = "", limit: int = 50) -> JSONResponse:
    conn, uid = _conn(request), _uid(request)
    if conn is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    try:
        runs = await pi_run_store.list_runs(conn, user_id=uid, project=project, limit=limit)
    except Exception as exc:  # noqa: BLE001
        log.warning("pi_run_list_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"runs": runs})


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request, since_seq: int = -1) -> JSONResponse:
    conn, uid = _conn(request), _uid(request)
    if conn is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    run = await pi_run_store.get_run(conn, run_id=run_id, user_id=uid, since_seq=since_seq)
    if run is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(run)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_TERMINAL = ("done", "failed", "detached")


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    """Replay persisted events, then poll the DB by seq (1s) until the run
    reaches a terminal status. A pushed mirror has no server-owned task/bus,
    so polling IS the live view — v1 simplicity, same SSE frame shape as the
    claude_runs stream so the web panel reuses its renderer."""
    conn, uid = _conn(request), _uid(request)

    async def _gen():
        if conn is None or not uid:
            yield _sse({"kind": "failed", "text": "unavailable"})
            return
        last_seq = -1
        idle_polls = 0
        while True:
            run = await pi_run_store.get_run(conn, run_id=run_id, user_id=uid, since_seq=last_seq)
            if run is None:
                yield _sse({"kind": "failed", "text": "not found"})
                return
            for ev in run.get("events", []):
                last_seq = max(last_seq, int(ev.get("seq", -1)))
                yield _sse({"kind": ev.get("kind", "message"), "text": ev.get("text", ""),
                            "tool": ev.get("tool", ""), "path": ev.get("path", ""),
                            "seq": ev.get("seq")})
            status = run.get("status", "")
            if status in _TERMINAL:
                yield _sse({"kind": "done", "text": status})
                return
            idle_polls = idle_polls + 1 if not run.get("events") else 0
            # Keep intermediaries from idling out during long quiet stretches.
            if idle_polls and idle_polls % 15 == 0:
                yield ": ping\n\n"
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)
