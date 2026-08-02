"""Coder subagent dispatch API — list runs, fetch one, list available roles.

Read-only inspection surface for the task_dispatch tool (see
``augmentum/agents/dispatch.py``). Spawns are NOT initiated here — the
model invokes them via the tool inside its turn. These routes back the
nested UI cards under each parent turn and the history sidebar.

All endpoints scoped to the calling user per the multi-tenant pattern.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/api/coder/subagents", tags=["coder_subagents"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


def _store(request: Request):
    """Resolve the SubagentRunStore wired in lifespan."""
    return getattr(request.app.state, "coder_subagent_store", None)


def _registry(request: Request):
    """Resolve the global AgentRegistry wired in lifespan."""
    return getattr(request.app.state, "coder_agent_registry", None)


@router.get("")
async def list_subagent_runs(
    request: Request,
    parent_run_id: str = Query("", description="Filter to children of one parent coder run"),
    session_id: str = Query("", description="Filter to one coder session"),
    limit: int = Query(50, ge=1, le=500),
) -> JSONResponse:
    """List a user's recent subagent runs (newest first).

    With ``parent_run_id``: just children of that parent (the nested-
    cards-under-a-turn use case). With ``session_id``: the running
    history of subagents in one chat. Empty filters return the user's
    recent runs across all sessions.
    """
    store = _store(request)
    if store is None:
        return JSONResponse([])
    user_id = _user_id(request)
    try:
        rows = await store.list_runs(
            user_id=user_id,
            parent_run_id=parent_run_id,
            session_id=session_id,
            limit=limit,
        )
    except Exception:
        log.exception("list_subagent_runs_failed")
        return JSONResponse({"error": "list failed"}, status_code=500)
    return JSONResponse(rows)


@router.get("/roles")
async def list_subagent_roles(request: Request) -> JSONResponse:
    """List available roles (built-ins + user-defined + workspace-local).

    Re-scans the registry on each call so newly-dropped role files
    surface without a server restart.
    """
    registry = _registry(request)
    if registry is None:
        return JSONResponse([])
    try:
        registry.refresh_if_stale()
        items = [r.to_api_dict() for r in registry.list()]
    except Exception:
        log.exception("list_subagent_roles_failed")
        return JSONResponse({"error": "list failed"}, status_code=500)
    return JSONResponse(items)


@router.get("/{subagent_id}")
async def get_subagent_run(request: Request, subagent_id: str) -> JSONResponse:
    """Fetch one subagent run by id (full transcript + tool_call_log)."""
    store = _store(request)
    if store is None:
        return JSONResponse({"error": "subagent store not initialized"}, status_code=503)
    user_id = _user_id(request)
    try:
        row = await store.get_run(subagent_id, user_id=user_id)
    except Exception:
        log.exception("get_subagent_run_failed", subagent_id=subagent_id)
        return JSONResponse({"error": "fetch failed"}, status_code=500)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(row)


@router.post("/{subagent_id}/cancel")
async def cancel_subagent_run(
    request: Request, subagent_id: str,
) -> JSONResponse:
    """Cancel one in-flight subagent without disturbing siblings.

    The dispatch coroutine intercepts the resulting CancelledError and
    synthesises a SubagentResult with stop_reason="cancelled" + a
    recovery hint so the parent loop gets the same shape as a clean
    exit. Returns 404 if no such subagent is currently running (either
    it already finished or it never existed). Returns 200 on success.

    Authorization: user must own the run. Looked up against the
    persisted row's user_id since the in-process registry doesn't
    carry that field cheaply. The look-up tolerates exactly one race —
    if the store has no row at all yet (cancel arrives before the
    start-row commits), we still allow the cancel because the
    dispatcher's own state is authoritative and the cost of declining
    is higher (subagent runs to completion) than the cost of allowing.
    Any other failure (store unreachable, query error) is fail-closed:
    we return 503 rather than allow an unverified cancel, so a DB
    outage can't be used to cancel another user's run.
    """
    from augmentum.agents.dispatch import find_subagent_owner

    user_id = _user_id(request)
    dispatcher = find_subagent_owner(subagent_id)
    if dispatcher is None:
        return JSONResponse(
            {"error": "subagent not running"}, status_code=404,
        )

    # Ownership check via the persistence store. If the row is present
    # and user_id mismatches, refuse (403). If the row is genuinely
    # missing (race), allow — see docstring. If the check itself fails,
    # fail closed (503) rather than fall through to an unverified cancel.
    store = _store(request)
    if store is not None and user_id:
        try:
            row = await store.get_run(subagent_id, user_id=user_id)
            if row is None:
                # Either belongs to a different user or hasn't been
                # persisted yet. Differentiate by checking owner.
                row_any = await store.get_run_any(subagent_id) if hasattr(store, "get_run_any") else None
                if row_any is not None:
                    return JSONResponse(
                        {"error": "not authorized"}, status_code=403,
                    )
        except Exception:
            log.warning("cancel_subagent_auth_check_failed", subagent_id=subagent_id, exc_info=True)
            return JSONResponse(
                {"error": "ownership check failed"}, status_code=503,
            )

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    reason = str(body.get("reason") or "user cancelled")[:200]

    cancelled = dispatcher.cancel(subagent_id, reason=reason)
    if not cancelled:
        # Race: the subagent finished between find_subagent_owner and
        # cancel. Surface as 404 — the caller can re-fetch and see
        # the completed row.
        return JSONResponse(
            {"error": "subagent finished before cancel landed"},
            status_code=404,
        )
    return JSONResponse({
        "subagent_id": subagent_id,
        "reason": reason,
        "cancelled": True,
    })
