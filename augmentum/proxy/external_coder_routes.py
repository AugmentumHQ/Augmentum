"""HTTP endpoints for connecting an external coding agent (Claude Code).

The "login through the webapp/Android" surface. The user runs
``claude setup-token`` (official browser OAuth → a 1-year ``sk-ant-oat01-…``
subscription token) once, pastes it here, and Augmentum stores it encrypted +
per-user (see ``coder/external/claude_token_store``). The driver loads it at
spawn time and hands it to the Claude-home container via env.

Token-paste login needs no image build (this route + the UI are enough). The
fuller "run setup-token from inside the webapp and capture the URL" handshake
lands once the Claude-home container exists (it's where the ``claude`` CLI is).

Routes (prefix /api/coder/external):
  POST   /claude/token   {token}      → store encrypted, return status
  GET    /claude/status               → {connected, kind, hint}
  DELETE /claude/token                → disconnect
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from augmentum.coder.containers import ExecAborted
from augmentum.coder.external import claude_token_store as cts
from augmentum.coder.external import run_store
from augmentum.coder.external.base import ExternalTask, to_engineering_record
from augmentum.coder.external.claude_cli import build_claude_argv
from augmentum.coder.external.providers import is_model_targetable
from augmentum.coder.external.run_manager import get_run_manager
from augmentum.coder.external.stream import ClaudeStreamCollector
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/coder/external", tags=["external-coder"])

# Redirect Claude Code's home (transcripts + config) under the persistent
# /workspace volume so its native session JSONL survives a container recreate —
# the prerequisite for `claude --resume <session_id>` working across runs.
_CLAUDE_CONFIG_DIR = "/workspace/.augmentum/claude-home"

# Hung-process backstop for a background agent run: no stdout at all for this
# long means the CLI is wedged, not thinking. Deliberately generous — the
# stream goes quiet for the whole duration of every tool call, so a slow test
# suite or docker build is a normal silence, not a stall. There is no
# wall-clock cap; see the run_command call in _execute_run.
_RUN_IDLE_TIMEOUT_S = 1800.0


def _conn(request: Request):
    """The aiosqlite connection from app state (None when DB is down)."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None)


class _TokenBody(BaseModel):
    token: str = ""


def _store_and_user(request: Request) -> tuple[object, str]:
    store = getattr(request.app.state, "settings_store", None)
    user = request.scope.get("user")
    uid = getattr(user, "id", "") if user else ""
    return store, uid


@router.post("/claude/token")
async def set_claude_token(body: _TokenBody, request: Request) -> JSONResponse:
    store, uid = _store_and_user(request)
    if store is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    token = (body.token or "").strip()
    if not token:
        return JSONResponse({"error": "token required"}, status_code=400)
    warn = None
    if not cts.looks_like_claude_credential(token):
        # Don't hard-reject (proxies/self-hosted differ), but flag it.
        warn = "That doesn't look like a Claude token (expected sk-ant-…); stored anyway."
    try:
        await cts.save_token(store, uid, token)
    except Exception as exc:  # noqa: BLE001
        log.warning("claude_token_save_failed", error=repr(exc))
        return JSONResponse({"error": "could not store token"}, status_code=500)
    log.info("claude_token_connected", user_id=uid)
    out = await cts.status(store, uid)
    if warn:
        out["warning"] = warn
    return JSONResponse(out)


@router.get("/claude/status")
async def claude_status(request: Request) -> JSONResponse:
    store, uid = _store_and_user(request)
    if store is None or not uid:
        return JSONResponse({"connected": False, "kind": "", "hint": ""})
    return JSONResponse(await cts.status(store, uid))


@router.delete("/claude/token")
async def clear_claude_token(request: Request) -> JSONResponse:
    store, uid = _store_and_user(request)
    if store is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    await cts.clear_token(store, uid)
    log.info("claude_token_disconnected", user_id=uid)
    return JSONResponse({"connected": False, "kind": "", "hint": ""})


class _RunBody(BaseModel):
    workspace_id: str = ""
    task: str = ""
    permission: str = "auto"  # confirm_mutations | auto | plan
    resume_run_id: str = ""   # continue a prior run via Claude's native --resume
    model: str = ""           # pinned model ("" = account default); → --model


async def _ensure_claude_cli(cm: object, workspace_id: str) -> str | None:
    """Guarantee the ``claude`` CLI is on PATH in the workspace before we exec it.

    Workspaces booted on a prebaked ``augmentum-workspace:*`` image already have
    the CLI (Dockerfile.workspace bakes it in) — the probe is a sub-second no-op
    there. Legacy / bootstrap workspaces on bare ``ubuntu:24.04`` predate that
    image and never got the CLI (``_FALLBACK_NPM_PACKAGES`` omits it), so without
    this the run dies with ``claude: command not found`` (exit 127) and the user
    sees a silent "ended without completing". We install it on demand (~15s,
    needs npm, which the fallback bootstrap does provide).

    Returns ``None`` on success, or a human-readable error string to surface.
    """
    probe_cmd = ["bash", "-lc", "command -v claude || true"]
    try:
        probe = await cm.run_command(workspace_id, probe_cmd, timeout=30.0)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return f"workspace not reachable: {exc!r}"
    if (probe or "").strip():
        return None  # already present

    log.info("claude_cli_install_start", workspace_id=workspace_id)
    install_cmd = [
        "bash", "-lc",
        "command -v npm >/dev/null 2>&1 || { echo NO_NPM; exit 1; }; "
        "npm install -g @anthropic-ai/claude-code 2>&1",
    ]
    try:
        out = await cm.run_command(  # type: ignore[attr-defined]
            workspace_id, install_cmd, timeout=300.0, idle_timeout=120.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("claude_cli_install_failed", workspace_id=workspace_id, error=repr(exc))
        return f"could not install the Claude CLI: {exc!r}"
    if "NO_NPM" in (out or ""):
        return ("this workspace has no Node/npm — recreate the workspace so it "
                "boots on the prebaked image with the Claude CLI included.")
    try:
        verify = await cm.run_command(workspace_id, probe_cmd, timeout=30.0)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return f"workspace not reachable after install: {exc!r}"
    if not (verify or "").strip():
        log.warning("claude_cli_install_unverified", workspace_id=workspace_id, tail=(out or "")[-300:])
        return "Claude CLI install did not produce a `claude` binary."
    log.info("claude_cli_installed", workspace_id=workspace_id)
    return None


async def _record_run_outcome(
    runtime: object, uid: str, et: ExternalTask, *,
    ok: bool, files: list[str], err: str,
    summary: str = "", session_ref: str = "",
) -> None:
    """Best-effort: record the run to the companion's engineering continuity
    ledger so Becca carries the work across sessions. Skips silently if the
    runtime isn't reachable. A memory write must never fail the run."""
    try:
        if runtime is None:
            return
        from augmentum.coder.external.base import ExternalRunResult
        from augmentum.companion_runtime import engineering_log
        rec = to_engineering_record(
            et,
            ExternalRunResult(
                ok=ok, files_changed=files, error=err,
                summary=summary, session_ref=session_ref,
            ),
            engine_label="Claude Code",
        )
        await engineering_log.record_engineering_outcome(
            runtime, user_id=uid,
            task=rec["task"], outcome=rec["outcome"],
            engine=rec["engine"], framing=rec["framing"],
            resume_ref=rec["resume_ref"],
        )
    except Exception:  # noqa: BLE001 — memory write must never fail the run
        log.debug("claude_run_engineering_log_failed", exc_info=True)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def _execute_run(
    run: object, *, conn: object, cm: object, runtime: object, uid: str,
    token: str, workspace_id: str, task: str, permission: str,
    resume_session_id: str, model: str = "",
) -> None:
    """The actual run, owned by the RunManager task (NOT a request). Persists
    each event + publishes it to the run's bus; finalizes on completion. This is
    what makes a run survive a viewer disconnecting."""
    run.publish({"kind": "status", "text": "Preparing workspace…"})
    install_err = await _ensure_claude_cli(cm, workspace_id)
    if install_err:
        log.warning("claude_run_cli_unavailable", user_id=uid, error=install_err)
        if conn is not None:
            with contextlib.suppress(Exception):
                await run_store.finish_run(
                    conn, run_id=run.run_id, user_id=uid, status="failed",
                    error=install_err, files_changed=[],
                )
        run.publish({"kind": "failed", "text": install_err})
        run.finish({"status": "failed", "ok": False, "error": install_err, "files_changed": []})
        return

    et = ExternalTask(prompt=task, workspace="/workspace", permission=permission, model=model)
    argv = build_claude_argv(et, resume_session_id=resume_session_id)

    seq = {"n": 0}

    async def _emit(item: dict) -> None:
        """Persist (with a seq) then publish one transcript event."""
        if conn is not None:
            seq["n"] += 1
            item = {**item, "seq": seq["n"]}
            with contextlib.suppress(Exception):
                await run_store.add_event(
                    conn, run_id=run.run_id, user_id=uid, seq=seq["n"],
                    kind=item.get("kind", ""), text=item.get("text", ""),
                    tool=item.get("tool", ""), path=item.get("path", ""),
                )
        run.publish(item)

    async def _on_session(sid: str) -> None:
        if conn is not None:
            with contextlib.suppress(Exception):
                await run_store.set_session_id(
                    conn, run_id=run.run_id, user_id=uid, session_id=sid
                )

    # The shared stream consumer (same one the self-edit driver uses).
    collector = ClaudeStreamCollector(emit=_emit, on_session_id=_on_session)

    abort_reason = ""
    try:
        await cm.run_command(
            workspace_id, argv,
            # No wall-clock cap. This is a BACKGROUND agent session: it runs
            # until it finishes, and any fixed budget is a guess about work we
            # haven't seen yet. The old 900s cap killed a healthy run at 161
            # tool calls mid-edit, then reported it as "claude ended without a
            # result" — the failure looked like Claude's, not ours. Liveness is
            # ``idle_timeout``'s job: it asks whether the process is still
            # doing anything, which is the question actually worth asking.
            timeout=None,
            # Generous, because silence here is normal: between a tool_use and
            # its result the CLI emits nothing on stdout, so a long build or
            # test suite is a legitimately quiet stretch. This is a hung-process
            # backstop, not a pace limit.
            idle_timeout=_RUN_IDLE_TIMEOUT_S,
            on_chunk=collector.on_chunk,
            environment={
                "CLAUDE_CODE_OAUTH_TOKEN": token,
                "CLAUDE_CONFIG_DIR": _CLAUDE_CONFIG_DIR,
            },
            # The ``claude`` CLI may live in the persistent-volume npm prefix
            # (installed on-demand by _ensure_claude_cli), which is only on
            # PATH under a login shell — run through one so exec resolves it.
            login_shell=True,
            # We stream via on_chunk and ignore the return value, so an
            # explanation appended to that string would be lost. Raise instead.
            strict=True,
        )
    except asyncio.CancelledError:
        # Stop pressed. Persist what we have (status set by the stop endpoint /
        # the manager's cancelled-finish); don't await more here.
        raise
    except ExecAborted as exc:
        # Augmentum terminated the process — say so plainly, and say it was us.
        abort_reason = str(exc.detail)
        log.warning(
            "claude_run_aborted", user_id=uid, run_id=run.run_id,
            kind=exc.kind, detail=exc.detail,
        )
        await _emit({"kind": "failed", "text": abort_reason})
    except Exception as exc:  # noqa: BLE001 — normalize to a failure event
        log.warning("claude_run_failed", user_id=uid, error=repr(exc), exc_info=True)
        abort_reason = f"{type(exc).__name__}: {exc}"
        await _emit({"kind": "failed", "text": abort_reason})
    finally:
        await collector.flush(abort_reason)

    status = collector.status
    if conn is not None:
        with contextlib.suppress(Exception):
            await run_store.finish_run(
                conn, run_id=run.run_id, user_id=uid, status=status,
                outcome=collector.outcome, error=collector.err,
                files_changed=collector.files, raw_jsonl=collector.raw_jsonl,
                session_id=collector.meta["session_id"],
                cost_usd=collector.meta["cost_usd"],
                num_turns=collector.meta["num_turns"],
                duration_ms=collector.meta["duration_ms"],
                # The model Claude actually ran (from stream init); fall back to
                # the pinned choice if the stream didn't report one.
                model=collector.meta.get("model") or model,
            )
    await _record_run_outcome(
        runtime, uid, et, ok=collector.ok, files=collector.files, err=collector.err,
        summary=collector.outcome, session_ref=workspace_id,
    )
    run.finish({
        "status": status, "ok": collector.ok, "error": collector.err,
        "files_changed": collector.files, "session_id": collector.meta["session_id"],
        "cost_usd": collector.meta["cost_usd"], "num_turns": collector.meta["num_turns"],
        "duration_ms": collector.meta["duration_ms"],
    })


def _done_from_row(row: dict) -> dict:
    """Terminal 'done' frame reconstructed from a persisted run row (for a viewer
    attaching to an already-finished run)."""
    return {
        "kind": "done", "ok": row.get("status") == "done",
        "run_id": row.get("id"), "error": row.get("error", ""),
        "files_changed": row.get("files_changed") or [],
        "session_id": row.get("session_id", ""), "cost_usd": row.get("cost_usd", 0),
        "num_turns": row.get("num_turns", 0), "duration_ms": row.get("duration_ms", 0),
    }


def _attach_response(conn: object, manager: object, run_id: str, uid: str) -> StreamingResponse:
    """SSE that reconstructs a run for any viewer: emit run/started, replay the
    persisted transcript, then tail the live bus (deduped by seq) if still
    running. Closing it only unsubscribes — it never cancels the run."""
    async def gen():
        row = await run_store.get_run(conn, run_id=run_id, user_id=uid)
        if row is None:
            yield _sse({"kind": "failed", "text": "run not found"})
            return
        yield _sse({"kind": "run", "run_id": run_id, "resumed_from": row.get("resumed_from", "")})
        yield _sse({"kind": "started", "text": (row.get("task") or "")[:120]})

        run = manager.get(run_id)
        # Subscribe BEFORE re-reading the DB so events landing in that window are
        # caught live and deduped by seq (rather than lost).
        q = run.subscribe() if (run is not None and not run.finished.is_set()) else None
        fresh = await run_store.get_run(conn, run_id=run_id, user_id=uid) or row
        last = 0
        for ev in (fresh.get("events") or []):
            last = ev.get("seq") or last
            yield _sse(ev)

        if q is None or run is None or run.finished.is_set():
            if q is not None and run is not None:
                run.unsubscribe(q)
            yield _sse(_done_from_row(fresh))
            return
        try:
            while True:
                item = await q.get()
                if item.get("kind") == "done":
                    yield _sse(item)
                    break
                s = item.get("seq") or 0
                if s and s <= last:
                    continue  # already replayed from the DB
                yield _sse(item)
        finally:
            run.unsubscribe(q)  # detach only — the run keeps going

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/claude/run/stream")
async def run_claude_stream(body: _RunBody, request: Request):
    """Start a run (server-owned, survives disconnect) and attach to its live
    stream. Re-attach later via ``GET /claude/run/{id}/stream``; stop via
    ``POST /claude/run/{id}/stop``.

    SSE frames carry a ``kind`` of ``run`` | ``status`` | ``started`` |
    ``message`` | ``thinking`` | ``file_change`` | ``command_exec`` |
    ``tool_call`` | ``mcp_call`` | ``completed`` | ``failed``, plus a terminal
    ``done`` with ``{ok, error, files_changed, session_id, cost_usd, num_turns,
    duration_ms}``.
    """
    store, uid = _store_and_user(request)
    if store is None or not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    task = (body.task or "").strip()
    if not task:
        return JSONResponse({"error": "task required"}, status_code=400)
    if not body.workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)
    token = await cts.load_token(store, uid)
    if not token:
        return JSONResponse(
            {"error": "Claude not connected — add your token in the Agents panel first."},
            status_code=400,
        )
    cm = getattr(request.app.state, "container_manager", None)
    if cm is None:
        return JSONResponse({"error": "coder containers unavailable"}, status_code=503)
    conn = _conn(request)

    # Resume? Resolve the prior run's Claude session id (native --resume).
    resume_session_id = ""
    resumed_from = ""
    if body.resume_run_id and conn is not None:
        prior = await run_store.session_for_run(conn, run_id=body.resume_run_id, user_id=uid)
        if not prior:
            return JSONResponse({"error": "run to resume not found"}, status_code=404)
        if not prior.get("session_id"):
            return JSONResponse(
                {"error": "that run has no Claude session id — can't resume it natively."},
                status_code=400,
            )
        resume_session_id = prior["session_id"]
        resumed_from = body.resume_run_id

    # Only forward a pinned model if this agent is model-targetable (Claude is;
    # the guard keeps the seam honest as other engines are added).
    pinned_model = (body.model or "").strip() if is_model_targetable("claude") else ""

    run_id = uuid.uuid4().hex
    if conn is not None:
        with contextlib.suppress(Exception):
            await run_store.create_run(
                conn, run_id=run_id, user_id=uid, workspace_id=body.workspace_id,
                task=task, permission=body.permission, resumed_from=resumed_from,
                model=pinned_model,
            )

    manager = get_run_manager(request.app.state)
    runtime = getattr(request.app.state, "companion_runtime", None)
    run = manager.create(
        run_id, workspace_id=body.workspace_id, user_id=uid,
        task=task, resumed_from=resumed_from,
    )
    manager.start(run, lambda r: _execute_run(
        r, conn=conn, cm=cm, runtime=runtime, uid=uid, token=token,
        workspace_id=body.workspace_id, task=task, permission=body.permission,
        resume_session_id=resume_session_id, model=pinned_model,
    ))
    return _attach_response(conn, manager, run_id, uid)


@router.get("/claude/run/{run_id}/stream")
async def attach_claude_run(run_id: str, request: Request):
    """Re-attach to a run's live stream (or replay a finished one)."""
    _store, uid = _store_and_user(request)
    conn = _conn(request)
    if not uid or conn is None:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    manager = get_run_manager(request.app.state)
    return _attach_response(conn, manager, run_id, uid)


@router.post("/claude/run/{run_id}/stop")
async def stop_claude_run(run_id: str, request: Request) -> JSONResponse:
    """Cancel a running run. The background task ends; subscribers get a final
    ``done`` (cancelled) frame; the DB row is marked cancelled."""
    _store, uid = _store_and_user(request)
    conn = _conn(request)
    if not uid:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    # Ownership check: only the run's owner can stop it.
    if conn is not None:
        owned = await run_store.session_for_run(conn, run_id=run_id, user_id=uid)
        if owned is None:
            return JSONResponse({"error": "not found"}, status_code=404)
    manager = get_run_manager(request.app.state)
    stopped = await manager.stop(run_id)
    if conn is not None:
        with contextlib.suppress(Exception):
            await run_store.mark_status(
                conn, run_id=run_id, user_id=uid, status="cancelled",
                error="stopped by user",
            )
    return JSONResponse({"stopped": stopped})


@router.get("/claude/runs")
async def list_claude_runs(request: Request) -> JSONResponse:
    """Per-workspace run history (metadata only, newest first)."""
    _store, uid = _store_and_user(request)
    conn = _conn(request)
    if not uid or conn is None:
        return JSONResponse({"runs": []})
    workspace_id = request.query_params.get("workspace_id", "")
    if not workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        limit = 50
    runs = await run_store.list_runs(
        conn, user_id=uid, workspace_id=workspace_id, limit=limit
    )
    # Reconcile orphans: a row still marked "running" that the manager has no
    # live task for was interrupted (server restart) — keep the UI honest
    # instead of spinning a phantom run forever.
    manager = get_run_manager(request.app.state)
    for run in runs:
        if run.get("status") == "running" and manager.get(run["id"]) is None:
            with contextlib.suppress(Exception):
                await run_store.mark_status(
                    conn, run_id=run["id"], user_id=uid, status="failed",
                    error="interrupted (server restarted)",
                )
            run["status"] = "failed"
            run["error"] = "interrupted (server restarted)"
    return JSONResponse({"runs": runs})


@router.get("/claude/runs/{run_id}")
async def get_claude_run(run_id: str, request: Request) -> JSONResponse:
    """One run with its normalized transcript. ``?raw=1`` adds the verbatim
    stream-json (full fidelity)."""
    _store, uid = _store_and_user(request)
    conn = _conn(request)
    if not uid or conn is None:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    include_raw = request.query_params.get("raw") in ("1", "true", "yes")
    run = await run_store.get_run(
        conn, run_id=run_id, user_id=uid, include_raw=include_raw
    )
    if run is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"run": run})
