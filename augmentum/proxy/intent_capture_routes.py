"""Stats + export endpoints for the opt-in intent-router training capture.

Lets you watch the dataset accrue (``/stats``) and pull it as JSONL for
fine-tuning / HuggingFace (``/export``). Both are strictly user-scoped — you
only ever see your own captures. Writing happens in voice_routes via
``augmentum/intent/capture_store.py`` when ``intent_capture_enabled`` is on.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from augmentum.config import settings
from augmentum.state.backends.sqlite import SQLiteBackend

router = APIRouter(prefix="/api/intent/capture", tags=["intent-capture"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return None
    return sm.backend.conn


@router.get("/stats")
async def capture_stats(request: Request) -> JSONResponse:
    """Counts so you can watch the set grow over the week."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    total = (
        await (await conn.execute(
            "SELECT COUNT(*) FROM intent_capture WHERE user_id = ?", (uid,)
        )).fetchone()
    )[0]

    by_goal = {
        r[0]: r[1]
        for r in await (await conn.execute(
            "SELECT effective_goal, COUNT(*) FROM intent_capture "
            "WHERE user_id = ? GROUP BY effective_goal ORDER BY 2 DESC",
            (uid,),
        )).fetchall()
    }
    by_surface = {
        r[0]: r[1]
        for r in await (await conn.execute(
            "SELECT surface, COUNT(*) FROM intent_capture "
            "WHERE user_id = ? GROUP BY surface",
            (uid,),
        )).fetchall()
    }
    span = await (await conn.execute(
        "SELECT MIN(captured_at), MAX(captured_at) FROM intent_capture WHERE user_id = ?",
        (uid,),
    )).fetchone()

    return JSONResponse({
        "enabled": bool(getattr(settings, "intent_capture_enabled", False)),
        "total": total,
        "by_goal": by_goal,
        "by_surface": by_surface,
        "first_captured_at": span[0],
        "last_captured_at": span[1],
        "export_url": "/api/intent/capture/export",
    })


@router.get("/recent")
async def capture_recent(request: Request) -> JSONResponse:
    """Most-recent captures so a correction UI can relabel a misroute.

    User-scoped, newest first. Pairs with POST ``/{id}/correct`` to drive the
    correction flywheel — surfacing the rows whose ``corrected_goal`` is still
    empty (the teacher's guess) and lettable the user fix them.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    try:
        limit = int(request.query_params.get("limit", "20"))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(100, limit))

    cursor = await conn.execute(
        """
        SELECT id, input_text, goal, effective_goal, corrected_goal,
               confidence, addressed, surface, captured_at
        FROM intent_capture
        WHERE user_id = ?
        ORDER BY captured_at DESC
        LIMIT ?
        """,
        (uid, limit),
    )
    rows = [dict(r) for r in await cursor.fetchall()]
    for d in rows:
        d["addressed"] = bool(d["addressed"])
    return JSONResponse({"captures": rows, "count": len(rows)})


@router.post("/{capture_id}/correct")
async def capture_correct(capture_id: str, request: Request) -> JSONResponse:
    """Set the user's corrected goal on a capture row — the correction flywheel.

    Closes the dead-column gap: ``corrected_goal`` shipped in migration 271 and
    is already consumed by the export's supervised label, but nothing wrote it.
    A user fixing a voice misroute now produces a gold training label.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed body → validation error below
        body = {}
    corrected = str((body or {}).get("corrected_goal", "")).strip().lower()

    from augmentum.intent.capture_store import VALID_GOALS, update_corrected_goal
    if corrected not in VALID_GOALS:
        return JSONResponse(
            {"error": f"corrected_goal must be one of {sorted(VALID_GOALS)}"},
            status_code=400,
        )

    ok = await update_corrected_goal(
        conn, user_id=uid, capture_id=capture_id, corrected_goal=corrected,
    )
    if not ok:
        return JSONResponse({"error": "Capture not found"}, status_code=404)
    return JSONResponse(
        {"ok": True, "capture_id": capture_id, "corrected_goal": corrected}
    )


@router.get("/export")
async def capture_export(request: Request) -> Response:
    """Export the user's captures as JSONL — one training row per line.

    Each line is the input the on-device model would see plus the teacher's
    verdict, ready for distillation / a HuggingFace dataset.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    cursor = await conn.execute(
        """
        SELECT input_text, last_assistant_response, last_dispatch_summary,
               active_surface, seconds_since_last_tts, media_active, explicit_capture,
               goal, effective_goal, coherent, addressed, confidence,
               teacher_model, parsed_from, reasoning, latency_ms,
               corrected_goal, surface, schema_version, captured_at
        FROM intent_capture
        WHERE user_id = ?
        ORDER BY captured_at ASC
        """,
        (uid,),
    )
    rows = await cursor.fetchall()

    lines = []
    for r in rows:
        d = dict(r)
        # Normalise SQLite ints back to bools for a clean dataset.
        for k in ("media_active", "explicit_capture", "coherent", "addressed"):
            d[k] = bool(d[k])
        # The supervised label is the user's correction when present, else the
        # teacher's effective goal.
        d["label"] = d["corrected_goal"] or d["effective_goal"]
        lines.append(json.dumps(d, ensure_ascii=False))

    body = "\n".join(lines) + ("\n" if lines else "")
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": 'attachment; filename="intent_capture.jsonl"',
            "X-Capture-Count": str(len(rows)),
        },
    )
