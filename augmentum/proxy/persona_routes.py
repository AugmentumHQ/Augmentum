"""User persona management API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.proxy import system_events
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/personas", tags=["personas"])


def _get_conn(request: Request):
    """Get SQLite connection or None."""
    state_mgr = getattr(request.app.state, "state_manager", None)
    if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
        return state_mgr.backend.conn
    return None


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


@router.get("/")
async def list_personas(request: Request) -> JSONResponse:
    """List the authenticated user's personas."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"personas": []})

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    cursor = await conn.execute(
        "SELECT id, name, appearance, description, is_default, created_at, updated_at, avatar "
        "FROM user_personas WHERE user_id = ? "
        "ORDER BY is_default DESC, updated_at DESC",
        (uid,),
    )
    rows = await cursor.fetchall()
    return JSONResponse({
        "personas": [
            {
                "id": r[0],
                "name": r[1],
                "appearance": r[2],
                "description": r[3],
                "is_default": bool(r[4]),
                "created_at": r[5],
                "updated_at": r[6],
                "avatar": r[7] if len(r) > 7 else "",
            }
            for r in rows
        ]
    })


@router.post("/")
async def create_persona(request: Request) -> JSONResponse:
    """Create a new persona for the authenticated user."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)

    persona_id = uuid.uuid4().hex[:16]
    appearance = body.get("appearance", "").strip()
    description = body.get("description", "").strip()
    avatar = body.get("avatar", "")
    is_default = bool(body.get("is_default", False))

    # If this is the default, clear the caller's existing default only —
    # not every user's default (that was the cross-tenant bug).
    if is_default:
        await conn.execute(
            "UPDATE user_personas SET is_default = 0 WHERE user_id = ?",
            (uid,),
        )

    await conn.execute(
        "INSERT INTO user_personas "
        "(id, name, appearance, description, avatar, is_default, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (persona_id, name, appearance, description, avatar, int(is_default), uid),
    )
    await conn.commit()
    system_events.publish("personas.changed", {"id": persona_id}, user_id=uid)

    log.info("persona_created", id=persona_id, name=name, user_id=uid)
    return JSONResponse({
        "id": persona_id,
        "name": name,
        "appearance": appearance,
        "description": description,
        "avatar": avatar,
        "is_default": is_default,
    }, status_code=201)


@router.get("/{persona_id}")
async def get_persona(persona_id: str, request: Request) -> JSONResponse:
    """Get one of the authenticated user's personas."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    cursor = await conn.execute(
        "SELECT id, name, appearance, description, is_default, created_at, updated_at, avatar "
        "FROM user_personas WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    row = await cursor.fetchone()
    if not row:
        return JSONResponse({"error": "Persona not found"}, status_code=404)

    return JSONResponse({
        "id": row[0],
        "name": row[1],
        "appearance": row[2],
        "description": row[3],
        "is_default": bool(row[4]),
        "created_at": row[5],
        "updated_at": row[6],
        "avatar": row[7] if len(row) > 7 else "",
    })


@router.put("/{persona_id}")
async def update_persona(persona_id: str, request: Request) -> JSONResponse:
    """Update one of the authenticated user's personas."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()

    # Check the persona exists and belongs to this user.
    cursor = await conn.execute(
        "SELECT id FROM user_personas WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    if not await cursor.fetchone():
        return JSONResponse({"error": "Persona not found"}, status_code=404)

    updates = ["updated_at = datetime('now')"]
    params: list = []

    if "name" in body:
        name = body["name"].strip()
        if not name:
            return JSONResponse({"error": "Name cannot be empty"}, status_code=400)
        updates.append("name = ?")
        params.append(name)
    if "appearance" in body:
        updates.append("appearance = ?")
        params.append(body["appearance"].strip())
    if "description" in body:
        updates.append("description = ?")
        params.append(body["description"].strip())
    if "avatar" in body:
        updates.append("avatar = ?")
        params.append(body["avatar"])

    params.extend([persona_id, uid])
    await conn.execute(
        f"UPDATE user_personas SET {', '.join(updates)} "
        "WHERE id = ? AND user_id = ?",
        params,
    )
    await conn.commit()
    system_events.publish("personas.changed", {"id": persona_id}, user_id=uid)

    log.info("persona_updated", id=persona_id, user_id=uid)

    # Return the updated persona
    cursor = await conn.execute(
        "SELECT id, name, appearance, description, is_default, avatar "
        "FROM user_personas WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    row = await cursor.fetchone()
    return JSONResponse({
        "id": row[0],
        "name": row[1],
        "appearance": row[2],
        "description": row[3],
        "is_default": bool(row[4]),
        "avatar": row[5] if len(row) > 5 else "",
    })


@router.delete("/{persona_id}")
async def delete_persona(persona_id: str, request: Request) -> JSONResponse:
    """Delete one of the authenticated user's personas."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    cursor = await conn.execute(
        "SELECT id FROM user_personas WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    if not await cursor.fetchone():
        return JSONResponse({"error": "Persona not found"}, status_code=404)

    await conn.execute(
        "DELETE FROM user_personas WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    await conn.commit()
    system_events.publish("personas.changed", {"id": persona_id, "deleted": True}, user_id=uid)

    log.info("persona_deleted", id=persona_id, user_id=uid)
    return JSONResponse({"deleted": persona_id})


@router.post("/{persona_id}/default")
async def set_default_persona(persona_id: str, request: Request) -> JSONResponse:
    """Set one of the authenticated user's personas as their default."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    cursor = await conn.execute(
        "SELECT id FROM user_personas WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    if not await cursor.fetchone():
        return JSONResponse({"error": "Persona not found"}, status_code=404)

    # Clear only the caller's existing default; the other tenants' defaults
    # stay where they are.
    await conn.execute(
        "UPDATE user_personas SET is_default = 0 WHERE user_id = ?",
        (uid,),
    )
    await conn.execute(
        "UPDATE user_personas SET is_default = 1, updated_at = datetime('now') "
        "WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    await conn.commit()
    system_events.publish("personas.changed", {"id": persona_id, "default": True}, user_id=uid)

    log.info("persona_set_default", id=persona_id, user_id=uid)
    return JSONResponse({"default": persona_id})


@router.post("/{persona_id}/undefault")
async def unset_default_persona(persona_id: str, request: Request) -> JSONResponse:
    """Remove default status from one of the authenticated user's personas."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    await conn.execute(
        "UPDATE user_personas SET is_default = 0, updated_at = datetime('now') "
        "WHERE id = ? AND user_id = ?",
        (persona_id, uid),
    )
    await conn.commit()
    system_events.publish("personas.changed", {"id": persona_id, "default": False}, user_id=uid)

    log.info("persona_unset_default", id=persona_id, user_id=uid)
    return JSONResponse({"undefaulted": persona_id})
