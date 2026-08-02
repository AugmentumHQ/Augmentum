"""Browse notes CRUD — persisted via NotesStore (individual SQLite rows)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.proxy.reputation import _update_reputation
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/browse", tags=["browse-notes"])


def _get_notes_store(request: Request):
    return getattr(request.app.state, "notes_store", None)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


@router.get("/notes")
async def list_notes(request: Request) -> JSONResponse:
    """List the authenticated user's notes (metadata only)."""
    store = _get_notes_store(request)
    if not store:
        return JSONResponse({"error": "Notes store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    stubs = await store.list_stubs(user_id=uid)
    return JSONResponse({"notes": stubs})


@router.get("/notes/{note_id}")
async def get_note(request: Request, note_id: str) -> JSONResponse:
    """Get one of the authenticated user's notes with full content."""
    store = _get_notes_store(request)
    if not store:
        return JSONResponse({"error": "Notes store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    note = await store.get(note_id, user_id=uid)
    if not note:
        return JSONResponse({"error": "Note not found"}, status_code=404)
    return JSONResponse(note)


@router.post("/notes")
async def create_note(request: Request) -> JSONResponse:
    """Create a new note for the authenticated user."""
    store = _get_notes_store(request)
    if not store:
        return JSONResponse({"error": "Notes store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    note = {
        "id": uuid.uuid4().hex[:12],
        "title": body.get("title", "Untitled"),
        "content": body.get("content", ""),
        "tags": body.get("tags", []),
        "source_url": body.get("source_url", ""),
        "source_title": body.get("source_title", ""),
        "format": body.get("format", "note"),
        "created_at": now,
        "updated_at": now,
    }

    await store.create(note, user_id=uid)

    # Boost domain reputation — user saved content as a note
    source_url = body.get("source_url", "")
    if source_url:
        await _update_reputation(request, source_url, success=True, user_action=True)

    return JSONResponse(note, status_code=201)


@router.put("/notes/{note_id}")
async def update_note(request: Request, note_id: str) -> JSONResponse:
    """Update one of the authenticated user's notes."""
    store = _get_notes_store(request)
    if not store:
        return JSONResponse({"error": "Notes store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    updated = await store.update(note_id, body, user_id=uid)
    if not updated:
        return JSONResponse({"error": "Note not found"}, status_code=404)
    return JSONResponse(updated)


@router.delete("/notes/{note_id}")
async def delete_note(request: Request, note_id: str) -> JSONResponse:
    """Delete one of the authenticated user's notes."""
    store = _get_notes_store(request)
    if not store:
        return JSONResponse({"error": "Notes store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    deleted = await store.delete(note_id, user_id=uid)
    if not deleted:
        return JSONResponse({"error": "Note not found"}, status_code=404)
    return JSONResponse({"ok": True})
