"""TTS pronunciation lexicon routes — the table under Voices.

CRUD for ``tts_lexicon_entries`` (migration 261). Per-voice term →
phonetics overrides, '' voice = every voice. Application happens on the
speech endpoints (audio_routes) via ``voice.lexicon_store.apply``;
previews reuse the existing POST /api/audio/voices/preview endpoint, so
this file is storage only.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from augmentum.utils.logging import get_logger
from augmentum.voice import lexicon_store

log = get_logger(__name__)

router = APIRouter()


def _conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


class _LexiconEntryBody(BaseModel):
    term: str
    phonetics: str = ""
    voice: str = ""


@router.get("/api/voice/lexicon")
async def lexicon_list(request: Request, voice: str | None = None) -> JSONResponse:
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False, "reason": "db_unavailable"}, status_code=503)
    user_id = _user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "reason": "auth"}, status_code=401)
    try:
        entries = await lexicon_store.list_entries(
            conn, user_id=user_id, voice=voice,
        )
    except Exception as exc:
        log.warning("lexicon_list_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    return JSONResponse({"ok": True, "entries": entries})


@router.post("/api/voice/lexicon")
async def lexicon_add(body: _LexiconEntryBody, request: Request) -> JSONResponse:
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False, "reason": "db_unavailable"}, status_code=503)
    user_id = _user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "reason": "auth"}, status_code=401)
    try:
        entry = await lexicon_store.add_entry(
            conn, user_id=user_id, voice=body.voice,
            term=body.term, phonetics=body.phonetics,
        )
    except Exception as exc:
        log.warning("lexicon_add_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    if entry is None:
        return JSONResponse(
            {"ok": False,
             "reason": "term required (max 80 chars; phonetics max 200)"},
            status_code=400,
        )
    return JSONResponse({"ok": True, "entry": entry})


@router.delete("/api/voice/lexicon/{entry_id}")
async def lexicon_remove(entry_id: int, request: Request) -> JSONResponse:
    conn = _conn(request)
    if conn is None:
        return JSONResponse({"ok": False, "reason": "db_unavailable"}, status_code=503)
    user_id = _user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "reason": "auth"}, status_code=401)
    try:
        removed = await lexicon_store.remove_entry(
            conn, entry_id=entry_id, user_id=user_id,
        )
    except Exception as exc:
        log.warning("lexicon_remove_failed", error=str(exc)[:200])
        return JSONResponse({"ok": False, "reason": "internal"}, status_code=500)
    if not removed:
        return JSONResponse({"ok": False, "reason": "not_found"}, status_code=404)
    return JSONResponse({"ok": True})
