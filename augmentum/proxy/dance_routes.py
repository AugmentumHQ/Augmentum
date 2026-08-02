"""Dance timeline API routes — history append/list, ratings CRUD.

Server-authoritative replacement for the localStorage-only state in
``ui/scripts/becca-presence.js`` and ``ui/scripts/movement-conductor.js``.
"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.dance.loops_store import DanceLoopsStore
from augmentum.dance.store import DanceHistoryStore, DanceRatingsStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/dance", tags=["dance"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _conn(request: Request):
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        raise RuntimeError("Backend not initialized")
    return sm.backend.conn


def _history_store(request: Request) -> DanceHistoryStore:
    return DanceHistoryStore(_conn(request))


def _ratings_store(request: Request) -> DanceRatingsStore:
    return DanceRatingsStore(_conn(request))


def _loops_store(request: Request) -> DanceLoopsStore:
    return DanceLoopsStore(_conn(request))


# ── History ─────────────────────────────────────────────────────────


@router.get("/history")
async def get_history(request: Request, limit: int = 50):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _history_store(request)
    entries = await store.list_recent(limit=limit, user_id=uid)
    return JSONResponse({"entries": entries})


@router.post("/history")
async def append_history(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    anim_id = (body.get("anim_id") or body.get("animId") or "").strip()
    if not anim_id:
        return JSONResponse({"error": "anim_id required"}, status_code=400)
    label = str(body.get("label") or anim_id)
    played_at = int(body.get("played_at") or body.get("playedAt") or 0)
    if played_at <= 0:
        import time as _t
        played_at = int(_t.time() * 1000)
    duration_sec = float(body.get("duration_sec") or body.get("durationSec") or 0)
    mode = body.get("mode")
    store = _history_store(request)
    try:
        row = await store.append(
            anim_id=anim_id, label=label, played_at=played_at,
            duration_sec=duration_sec,
            mode=str(mode) if mode else None,
            user_id=uid,
        )
    except ValueError:
        return JSONResponse({"error": "Invalid history entry"}, status_code=400)
    return JSONResponse({"entry": row})


@router.delete("/history")
async def clear_history(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _history_store(request)
    cleared = await store.clear(user_id=uid)
    return JSONResponse({"cleared": cleared})


# ── Ratings ─────────────────────────────────────────────────────────


@router.get("/ratings")
async def get_ratings(request: Request):
    """Returns all ratings for the authenticated user, keyed by anim_id.

    Response shape mirrors the in-memory JS shape used by
    MovementConductor: ``{ animId: { kind?, slotBonusSec, ts } }``.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _ratings_store(request)
    ratings = await store.list_all(user_id=uid)
    return JSONResponse({"ratings": ratings})


@router.put("/ratings/{anim_id}")
async def put_rating(request: Request, anim_id: str):
    """Upsert a single rating. Body shape:

        {"kind": "like" | "dislike" | "broken" | "longer" | "clear",
         "increment_sec": 8}  # only used when kind == "longer"

    'longer' accumulates slot bonus (capped server-side).
    'clear' deletes the row entirely (matches legacy localStorage
    semantics).
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not anim_id.strip():
        return JSONResponse({"error": "anim_id required"}, status_code=400)
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    kind = (body.get("kind") or "").strip().lower()
    store = _ratings_store(request)
    try:
        if kind == "clear":
            await store.clear(anim_id, user_id=uid)
            return JSONResponse({"rating": None})
        if kind == "longer":
            inc = int(body.get("increment_sec")
                      or body.get("incrementSec") or 8)
            row = await store.add_slot_bonus(
                anim_id, inc, user_id=uid,
            )
            return JSONResponse({"rating": row})
        if kind in ("like", "dislike", "broken"):
            row = await store.set_kind(anim_id, kind, user_id=uid)
            return JSONResponse({"rating": row})
    except ValueError:
        return JSONResponse(
            {"error": "Invalid rating request"}, status_code=400,
        )
    return JSONResponse(
        {"error": "kind must be like|dislike|broken|longer|clear"},
        status_code=400,
    )


@router.delete("/ratings/{anim_id}")
async def delete_rating(request: Request, anim_id: str):
    """Convenience alias for PUT with kind=clear."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not anim_id.strip():
        return JSONResponse({"error": "anim_id required"}, status_code=400)
    store = _ratings_store(request)
    await store.clear(anim_id, user_id=uid)
    return JSONResponse({"cleared": True})


# ── Loops ──────────────────────────────────────────────────────────


def _safe_loop_id(loop_id: str) -> bool:
    if not loop_id:
        return False
    if "/" in loop_id or "\\" in loop_id or ".." in loop_id:
        return False
    return True


@router.get("/loops")
async def list_loops(request: Request):
    """List the user's curated loops plus a summary of which is active.

    Response:
      {"loops": [...], "active_id": "loop_..." | null}
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _loops_store(request)
    loops = await store.list_for_user(user_id=uid)
    active = next((l for l in loops if l.get("is_active")), None)
    return JSONResponse({
        "loops": loops,
        "active_id": active["id"] if active else None,
    })


@router.post("/loops")
async def create_loop(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be an object"}, status_code=400)
    name = str(body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    animation_ids = body.get("animation_ids") or []
    if not isinstance(animation_ids, list):
        return JSONResponse(
            {"error": "animation_ids must be a list"}, status_code=400,
        )
    notes = body.get("notes")
    store = _loops_store(request)
    try:
        row = await store.create(
            name=name,
            animation_ids=[str(x) for x in animation_ids],
            notes=str(notes) if notes else None,
            user_id=uid,
        )
    except ValueError:
        return JSONResponse({"error": "Invalid loop"}, status_code=400)
    return JSONResponse({"loop": row})


@router.put("/loops/active")
async def set_active_loop(request: Request):
    """Set or clear the active loop. Body: ``{"loop_id": "loop_..."}``
    activates that loop; ``{"loop_id": null}`` clears the active state
    (= unconstrained atlas)."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    loop_id = body.get("loop_id") if isinstance(body, dict) else None
    if loop_id is not None and not _safe_loop_id(str(loop_id)):
        return JSONResponse({"error": "Invalid loop_id"}, status_code=400)
    store = _loops_store(request)
    try:
        row = await store.set_active(
            None if loop_id is None else str(loop_id), user_id=uid,
        )
    except ValueError:
        return JSONResponse(
            {"error": "Invalid activation request"}, status_code=400,
        )
    return JSONResponse({"active": row})


@router.put("/loops/{loop_id}")
async def update_loop(request: Request, loop_id: str):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_loop_id(loop_id):
        return JSONResponse({"error": "Invalid loop_id"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be an object"}, status_code=400)
    store = _loops_store(request)
    try:
        row = await store.update(loop_id, body, user_id=uid)
    except ValueError:
        return JSONResponse({"error": "Invalid update"}, status_code=400)
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"loop": row})


@router.delete("/loops/{loop_id}")
async def delete_loop(request: Request, loop_id: str):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_loop_id(loop_id):
        return JSONResponse({"error": "Invalid loop_id"}, status_code=400)
    store = _loops_store(request)
    deleted = await store.delete(loop_id, user_id=uid)
    if not deleted:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"ok": True})
