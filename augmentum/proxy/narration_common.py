"""Shared logic for the EPUB-narration endpoints (files-mode + artifacts).

A "narration" is a synthesized TTS WAV paired with an EPUB — see
``augmentum/jobs/handlers/narration_synth.py``. Both surfaces (Studio
artifacts and file-index rows) expose ``POST .../narration`` (start, or
return the in-progress / finished one) and ``GET .../narration`` (status).
This module holds the bits they share so the route files stay thin.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from augmentum.state.narration_store import NarrationStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _backend(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    return getattr(sm, "backend", None) if sm else None


async def _builtin_synth_engine(conn, voice: str) -> tuple[str, str]:
    """Resolve which built-in TTS engine would serve ``voice``.

    Narration synthesis only supports the in-process engines (Kokoro /
    Pocket TTS) — they yield well-formed per-segment WAV, which
    stitches cleanly; external HTTP providers chunk a single stream
    arbitrarily.

    Returns ``(engine_id, bare_voice)`` where ``engine_id`` is one of
    ``kokoro-builtin`` / ``pockettts-builtin`` (or ``""`` when the voice
    can't be served by a built-in engine), and ``bare_voice`` is the
    voice name with any ``provider_id::`` prefix stripped — what the
    engine's ``stream_speech`` actually expects. Returning both
    prevents the leak where ``pockettts-builtin::eve`` reaches the
    engine as a literal voice name and silently falls back to default.
    """
    try:
        from augmentum.proxy.audio_routes import _BUILTIN_TTS_IDS, resolve_voice_provider
        provider, bare_voice = await resolve_voice_provider(conn, voice or "")
        pid = provider.get("id") if provider else ""
        if pid in _BUILTIN_TTS_IDS:
            return pid, bare_voice
        return "", bare_voice
    except Exception:  # noqa: BLE001 — treat resolution failure as "not available"
        return "", voice


def _narration_view(row: dict | None, job_row: dict | None) -> dict[str, Any]:
    if not row:
        return {"status": "none"}
    out: dict[str, Any] = {
        "status": row.get("status") or "none",
        "voice": row.get("voice") or "",
    }
    if row.get("status") == "done" and row.get("narration_artifact_id"):
        out["narration_artifact_id"] = row["narration_artifact_id"]
        out["download_url"] = f"/api/artifacts/{row['narration_artifact_id']}/download"
    if row.get("status") == "failed":
        out["error"] = row.get("error") or "Synthesis failed"
    if row.get("status") in ("pending", "running"):
        out["processed_chunks"] = row.get("processed_chunks") or 0
        out["total_chunks"] = row.get("total_chunks") or 0
        if job_row:
            out["progress"] = job_row.get("progress")
            out["stage"] = job_row.get("stage") or ""
    return out


async def narration_status(request: Request, epub_kind: str, epub_ref: str) -> dict[str, Any]:
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    nstore = NarrationStore(backend.conn)
    row = await nstore.get(epub_kind, epub_ref, user_id=uid)
    job_row = None
    if row and row.get("job_id") and row.get("status") in ("pending", "running"):
        jobs_store = getattr(request.app.state, "jobs_store", None)
        if jobs_store:
            job_row = await jobs_store.get(row["job_id"], user_id=uid)
            # If the job vanished/finished but the pairing row didn't update,
            # surface the job's terminal state so the UI doesn't hang.
            if job_row and job_row.get("status") in ("completed", "failed", "cancelled") and row.get("status") in ("pending", "running"):
                row = {**row, "status": "failed" if job_row["status"] != "completed" else "done"}
    return _narration_view(row, job_row)


async def narration_start(
    request: Request, epub_kind: str, epub_ref: str, *,
    title: str, voice: str = "", output_format: str = "mp3", force: bool = False,
) -> dict[str, Any]:
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    conn = backend.conn
    nstore = NarrationStore(conn)

    existing = await nstore.get(epub_kind, epub_ref, user_id=uid)

    eff_voice = voice or (existing.get("voice", "") if existing else "")
    engine_id, bare_voice = await _builtin_synth_engine(conn, eff_voice)
    if not engine_id:
        raise HTTPException(
            422,
            "Narration recording requires a built-in voice (Kokoro or Pocket TTS). "
            "Pick one of those as your TTS provider in Settings to use it.",
        )

    # A finished narration recorded with a different voice than the one
    # requested now is stale — re-record instead of replaying the cache,
    # so a voice change in Settings actually takes effect here.
    stale = bool(existing) and existing.get("status") == "done" and (
        (existing.get("voice") or "") != bare_voice
    )
    if existing and not force and not stale:
        if existing.get("status") == "done" and existing.get("narration_artifact_id"):
            return _narration_view(existing, None)
        if existing.get("status") in ("pending", "running"):
            return _narration_view(existing, None)

    jobs_store = getattr(request.app.state, "jobs_store", None)
    if jobs_store is None:
        raise HTTPException(503, "Background jobs are not available")

    fmt = (output_format or "mp3").lower()
    if fmt not in ("wav", "mp3"):
        fmt = "mp3"
    job_id = await jobs_store.create(
        user_id=uid,
        job_type="narration_synth",
        payload={
            "epub_kind": epub_kind, "epub_ref": epub_ref, "voice": bare_voice,
            "title": title, "engine_id": engine_id, "format": fmt,
        },
    )
    await nstore.begin(epub_kind, epub_ref, bare_voice, job_id, user_id=uid)
    return {"status": "running", "job_id": job_id, "voice": bare_voice}
