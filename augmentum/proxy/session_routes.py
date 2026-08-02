"""Session management API routes — export/list sessions."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


@router.get("/")
async def list_sessions(request: Request) -> JSONResponse:
    """List all known sessions."""
    state_manager = getattr(request.app.state, "state_manager", None)
    if not state_manager or not isinstance(state_manager.backend, SQLiteBackend):
        return JSONResponse({"sessions": []})

    uid = _user_id(request)
    conn = state_manager.backend.conn
    q = "SELECT id, mode, message_count, created_at, updated_at FROM sessions"
    params: list = []
    if uid:
        q += " WHERE user_id = ?"
        params.append(uid)
    q += " ORDER BY updated_at DESC"
    cursor = await conn.execute(q, params)
    rows = await cursor.fetchall()
    return JSONResponse({"sessions": [dict(r) for r in rows]})


@router.get("/{session_id}/export")
async def export_session(session_id: str, request: Request) -> Response:
    """Export a session and its narrative state as JSON."""
    state_manager = getattr(request.app.state, "state_manager", None)
    if not state_manager:
        return JSONResponse({"error": "State manager unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    result: dict = {"session_id": session_id}

    # Session metadata (scoped to the authenticated user)
    session = await state_manager.backend.get_session(session_id, user_id=uid)
    if session:
        result["session"] = dict(session)
    else:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    # Narrative state (if SQLite)
    if isinstance(state_manager.backend, SQLiteBackend):
        persistence = NarrativePersistence(state_manager.backend.conn)
        narrative_state = await persistence.load_session_state(session_id, user_id=uid)
        if narrative_state:
            result["narrative_state"] = {
                "message_count": narrative_state.message_count,
                "character_card_name": narrative_state.character_card_name,
                "entities": [
                    {"id": e.id, "name": e.name, "type": e.type.value, "state": e.state.value}
                    for e in narrative_state.entities
                ],
                "facts": [
                    {"id": f.id, "content": f.content, "confidence": f.confidence, "source": f.source}
                    for f in narrative_state.facts
                ],
                "contradictions": [
                    {"id": c.id, "description": c.description, "severity": c.severity}
                    for c in narrative_state.contradictions
                ],
                "memory_summary": narrative_state.memory_summary,
            }

    export_json = json.dumps(result, indent=2, default=str)
    return Response(
        content=export_json,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="session_{session_id}.json"'},
    )
