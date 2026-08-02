"""Comic-narration endpoints — the voiced motion-comic.

Pairs a comic chapter (file-index row of a Komga/Suwayomi chapter) with a
synthesized narration + a per-bubble timeline that drives pan-and-scan
playback (in the in-app reader and, via cast, on the TV — both consume the
same ``narration-player.js`` + this payload).

Mirror of the EPUB-narration surface (``narration_common.py``) with the
timeline added. Synthesis runs as the ``comic_narration_synth`` background job
(restart-survivable, per-page checkpoint).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from augmentum.ocr.reading_order import (
    default_reading_direction,
    normalize_reading_direction,
)
from augmentum.state.comic_narration_store import ComicNarrationStore
from augmentum.utils.logging import get_logger

router = APIRouter(prefix="/api/comic-narration", tags=["comic"])
log = get_logger(__name__)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _backend(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    return getattr(sm, "backend", None) if sm else None


async def _builtin_synth_engine(conn, voice: str) -> tuple[str, str]:
    """Resolve (engine_id, bare_voice) for a built-in TTS voice, like the EPUB
    narration path. Comic synth reuses the same per-segment-WAV engines."""
    try:
        from augmentum.proxy.audio_routes import _BUILTIN_TTS_IDS, resolve_voice_provider
        provider, bare_voice = await resolve_voice_provider(conn, voice or "")
        pid = provider.get("id") if provider else ""
        if pid in _BUILTIN_TTS_IDS:
            return pid, bare_voice
        return "", bare_voice
    except Exception:  # noqa: BLE001
        return "", voice


def _parse_pages(row: dict) -> list[dict]:
    try:
        return json.loads(row.get("pages") or "[]")
    except Exception:  # noqa: BLE001
        return []


def _parse_cast(row: dict) -> dict:
    """The row's Voice Cast, folding the legacy 2-bucket columns into the low
    registers so an older recording still reports which voices it used."""
    try:
        cast = json.loads(row.get("voice_cast") or "{}")
    except Exception:  # noqa: BLE001
        cast = {}
    if not isinstance(cast, dict):
        cast = {}
    if not cast.get("m_low") and row.get("voice_male"):
        cast["m_low"] = row["voice_male"]
    if not cast.get("f_low") and row.get("voice_female"):
        cast["f_low"] = row["voice_female"]
    return {k: v for k, v in cast.items() if v}


def _view(row: dict | None, job_row: dict | None, *, include_pages: bool = False) -> dict[str, Any]:
    """Streaming-aware status view.

    ``pages`` accumulates as synthesis runs, so the player can begin on page 1
    (``ready_pages >= 1``) while later pages are still rendering. Each page's
    audio is a standalone artifact; the player chains them and polls this view
    for more until ``status == 'done'``.
    """
    from augmentum.config import settings

    # The reader short-circuits to a ready narration without ever asking the
    # begin route, so the cache switch has to reach it here too — otherwise
    # turning the cache off would only affect chapters that happen to be
    # unnarrated, which is exactly the hunt it exists to end.
    cache_enabled = bool(getattr(settings, "comic_narration_cache_enabled", True))
    if not row:
        return {"status": "none", "cache_enabled": cache_enabled}
    pages = _parse_pages(row)
    out: dict[str, Any] = {
        "cache_enabled": cache_enabled,
        "status": row.get("status") or "none",
        "voice": row.get("voice") or "",
        "voice_male": row.get("voice_male") or "",
        "voice_female": row.get("voice_female") or "",
        "voice_cast": _parse_cast(row),
        "reading_direction": normalize_reading_direction(
            row.get("reading_direction"), fallback=default_reading_direction(),
        ),
        "comic_ref": row.get("comic_ref") or "",
        "ready_pages": len(pages),
        "processed_pages": row.get("processed_pages") or 0,
        "total_pages": row.get("total_pages") or 0,
        "page_url_template": f"/api/media/comic/page/{row.get('comic_ref', '')}?page={{page}}",
    }
    if include_pages:
        # A page with no artifact is a silent one (splash art, or an image the
        # source couldn't serve) — checkpointed so it isn't re-read, but it has
        # no audio. Emitting a URL for it would send every player at
        # ``/api/artifacts//download``; an empty string is the signal they all
        # already test for.
        out["pages"] = [
            {
                **p,
                "audio_url": (
                    f"/api/artifacts/{p['artifact_id']}/download"
                    if p.get("artifact_id") else ""
                ),
            }
            for p in pages
        ]
    if row.get("status") == "failed":
        out["error"] = row.get("error") or "Synthesis failed"
    if row.get("status") in ("pending", "running") and job_row:
        out["progress"] = job_row.get("progress")
        out["stage"] = job_row.get("stage") or ""
    return out


@router.get("/{comic_ref}")
async def get_comic_narration(comic_ref: str, request: Request):
    """Streaming status of this comic's narration (+ ready pages). The player
    polls this while playing to pick up newly-synthesized pages."""
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    store = ComicNarrationStore(backend.conn)
    row = await store.get("file", comic_ref, user_id=uid)
    job_row = None
    if row and row.get("job_id") and row.get("status") in ("pending", "running"):
        jobs_store = getattr(request.app.state, "jobs_store", None)
        if jobs_store:
            job_row = await jobs_store.get(row["job_id"], user_id=uid)
            if job_row and job_row.get("status") in ("completed", "failed", "cancelled") \
                    and row.get("status") in ("pending", "running"):
                row = {**row, "status": "failed" if job_row["status"] != "completed" else "done"}
    return JSONResponse(_view(row, job_row, include_pages=True))


@router.get("/{comic_ref}/script")
async def get_comic_narration_script(comic_ref: str, request: Request):
    """The chapter as a plain-text script — what the models actually read.

    Transcription quality is otherwise only auditable by LISTENING to it, which
    is serial, slow, and impossible to diff between two runs. The text already
    exists per page on the narration row; this just makes it readable, so a bad
    read is visible in seconds instead of discovered twenty minutes into the
    audio.

    Pages with no dialogue are kept as explicit markers rather than omitted —
    silent art and a page that failed to read look identical in a list of only
    the pages that produced text, and telling those apart is most of the point.
    """
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    store = ComicNarrationStore(backend.conn)
    row = await store.get("file", comic_ref, user_id=uid)
    if not row:
        raise HTTPException(404, "No narration for this comic yet")

    pages = _parse_pages(row)
    out: list[str] = [
        f"# {row.get('comic_ref') or comic_ref}",
        f"# reading direction: {normalize_reading_direction(row.get('reading_direction'), fallback=default_reading_direction())}"
        f" · voice: {row.get('voice') or '?'}"
        f" · status: {row.get('status') or '?'}"
        f" · pages transcribed: {len(pages)}/{row.get('total_pages') or '?'}",
        "",
    ]
    for p in sorted(pages, key=lambda r: r.get("page", 0)):
        lines = p.get("lines") or []
        out.append(f"--- page {int(p.get('page', 0)) + 1} ---")
        out.extend((ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip())
        if not lines:
            out.append("(no readable text)")
        out.append("")
    return PlainTextResponse("\n".join(out))


@router.post("/begin")
async def begin_comic_narration(request: Request):
    """Enqueue narration synthesis for a comic chapter.

    Body: ``{comic_ref, voice, reading_direction?, format?, force?}``.
    """
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    conn = backend.conn
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    comic_ref = str(body.get("comic_ref") or "").strip()
    if not comic_ref:
        raise HTTPException(400, "comic_ref is required")
    voice = str(body.get("voice") or "")
    # Voice Cast — up to five register buckets the reader casts each line into.
    # Accepts the modern ``voice_cast`` object (m_low/m_high/f_low/f_high/
    # narrator) and, for older clients, the 2-bucket voice_male/voice_female
    # pair which folds into the low registers. `voice` / the narrator slot is
    # the fallback for any bucket left unset.
    req_cast = body.get("voice_cast")
    req_cast = req_cast if isinstance(req_cast, dict) else {}
    legacy_male = str(body.get("voice_male") or "")
    legacy_female = str(body.get("voice_female") or "")
    # An omitted or unrecognized direction falls through to the install
    # default, not to left-to-right. The old code coerced BOTH cases to "ltr",
    # so a manga library re-answered this question on every surface that
    # forgot to send it.
    reading_direction = normalize_reading_direction(
        body.get("reading_direction"), fallback=default_reading_direction(),
    )
    fmt = str(body.get("format") or "mp3").lower()
    if fmt not in ("wav", "mp3"):
        fmt = "mp3"
    force = bool(body.get("force") or False)
    # 0-indexed page the reader is showing; synthesis starts there and wraps.
    try:
        start_page = max(0, int(body.get("start_page") or 0))
    except (TypeError, ValueError):
        start_page = 0

    from augmentum.config import settings
    if not bool(getattr(settings, "ocr_enabled", False)):
        raise HTTPException(
            422,
            "OCR is not enabled. Bring up the OCR sidecar (compose.ocr.yaml) and "
            "set ocr_enabled to narrate comics.",
        )

    store = ComicNarrationStore(conn)
    existing = await store.get("file", comic_ref, user_id=uid)

    eff_voice = voice or (existing.get("voice", "") if existing else "")
    engine_id, bare_voice = await _builtin_synth_engine(conn, eff_voice)
    if not engine_id:
        raise HTTPException(
            422,
            "Comic narration requires a built-in voice (Kokoro or Pocket TTS). "
            "Pick one of those as your TTS provider in Settings.",
        )

    # Per-speaker voices resolve to BARE voices on the SAME built-in engine as
    # the narrator — a single engine instance synthesizes the whole chapter, so
    # a voice that lives on a different provider can't be mixed in. One that
    # doesn't match falls back to the narrator voice rather than failing the
    # run, so a half-configured pick still narrates.
    async def _same_engine_voice(raw: str) -> str:
        raw = (raw or "").strip()
        if not raw:
            return ""
        eid, bare = await _builtin_synth_engine(conn, raw)
        if eid != engine_id:
            log.info(
                "comic_narration_voice_engine_mismatch",
                requested=raw, requested_engine=eid, narrator_engine=engine_id,
            )
            return ""
        return bare

    # Build the requested cast: modern object first, then legacy male/female
    # folded into the low registers, then whatever the existing row had (so an
    # unchanged Listen keeps its cast). Each slot is resolved to a bare voice on
    # the narrator's engine; a mismatch or empty slot falls back to the narrator.
    from augmentum.ocr.vlm_reader import VOICE_CAST_SLOTS

    try:
        existing_cast = json.loads(existing.get("voice_cast") or "{}") if existing else {}
    except Exception:  # noqa: BLE001
        existing_cast = {}
    if not isinstance(existing_cast, dict):
        existing_cast = {}

    def _slot_request(slot: str) -> str:
        if req_cast.get(slot):
            return str(req_cast[slot])
        if slot == "m_low" and legacy_male:
            return legacy_male
        if slot == "f_low" and legacy_female:
            return legacy_female
        if existing_cast.get(slot):
            return str(existing_cast[slot])
        # Back-compat with rows recorded on the 2-bucket schema (031).
        if slot == "m_low" and existing:
            return str(existing.get("voice_male") or "")
        if slot == "f_low" and existing:
            return str(existing.get("voice_female") or "")
        return ""

    bare_cast: dict[str, str] = {}
    for slot in VOICE_CAST_SLOTS:
        if slot == "narrator":
            continue  # the narrator slot IS `voice` / bare_voice
        bare = await _same_engine_voice(_slot_request(slot))
        if bare:
            bare_cast[slot] = bare
    cast_json = json.dumps(bare_cast, sort_keys=True)
    # Legacy columns kept in sync so a 2-bucket client / older row still reads.
    bare_male = bare_cast.get("m_low", "")
    bare_female = bare_cast.get("f_low", "")

    # The existing row's EFFECTIVE cast, folding its legacy columns into the low
    # registers — so a chapter recorded on the 2-bucket schema doesn't read as
    # stale (and re-record) on every Listen just because voice_cast was empty.
    existing_eff = dict(existing_cast)
    if existing:
        if not existing_eff.get("m_low") and existing.get("voice_male"):
            existing_eff["m_low"] = existing["voice_male"]
        if not existing_eff.get("f_low") and existing.get("voice_female"):
            existing_eff["f_low"] = existing["voice_female"]
    existing_cast_json = json.dumps(
        {k: v for k, v in existing_eff.items() if v}, sort_keys=True,
    )

    # A finished narration made with a different voice/engine than the one
    # requested now is stale — replaying it would ignore the user's current
    # voice choice. Re-record it instead of returning the cache. (In-flight
    # jobs are left alone; the user can force once they finish.)
    # Reading direction belongs in this test for the same reason voice does:
    # it's a user choice baked into the recording. A narration read left-to-
    # right is scrambled dialogue for a manga, and replaying it after the user
    # flips the reader to RTL would silently ignore the flip.
    # With the cache off, every finished narration is stale by definition —
    # each Listen re-reads the chapter with the current model and prompt.
    cache_on = bool(getattr(settings, "comic_narration_cache_enabled", True))
    stale = bool(existing) and existing.get("status") == "done" and (
        not cache_on
        or
        (existing.get("engine_id") or "") != engine_id
        or (existing.get("voice") or "") != bare_voice
        or existing_cast_json != cast_json
        or normalize_reading_direction(
            existing.get("reading_direction"), fallback=default_reading_direction(),
        ) != reading_direction
    )
    if stale:
        log.info(
            "comic_narration_stale_rerecord", comic_ref=comic_ref,
            old_engine=existing.get("engine_id"), new_engine=engine_id,
            old_voice=existing.get("voice"), new_voice=bare_voice,
            old_direction=existing.get("reading_direction"), new_direction=reading_direction,
        )
    if existing and not force and not stale:
        if existing.get("status") == "done":
            return JSONResponse(_view(existing, None, include_pages=True))
        if existing.get("status") in ("pending", "running"):
            return JSONResponse(_view(existing, None, include_pages=True))
    # Re-record: clean up the previous run's per-page audio artifacts before
    # the row is reset, so they don't orphan.
    if existing and (force or stale):
        await _delete_page_artifacts(request, uid, existing)

    # Vision preflight — only for a run we're about to actually synthesize
    # (a cached narration replays fine with no vision model loaded). Reading a
    # comic with the VLM engine is an EXPLICIT vision request, so a role that
    # can't see is reported here, in the same breath as the click, naming the
    # knob to turn. The alternative — finding out per page — spends the whole
    # chapter before reporting "no readable text in this comic", which is the
    # exact failure this check exists to prevent.
    if (getattr(settings, "ocr_engine", "docling") or "docling").lower() == "vlm":
        from augmentum.ocr.vlm_reader import VisionNotConfigured, resolve_reader

        try:
            await resolve_reader(request.app)
        except VisionNotConfigured as exc:
            log.warning("comic_narration_vision_unavailable", error=str(exc)[:200])
            raise HTTPException(422, str(exc)) from exc

    # Boxed reading needs the docling sidecar, which is started by hand. If
    # it's down the read still works — it just reads whole pages, which the
    # 224px vision tower sees as a thumbnail. That's a real quality drop, so
    # it's reported rather than absorbed: silently narrating a chapter at the
    # worse setting is how you end up re-recording it later.
    boxed_available = None
    if (getattr(settings, "ocr_engine", "docling") or "docling").lower() == "vlm" \
            and bool(getattr(settings, "ocr_vlm_use_boxes", True)):
        from augmentum.ocr import get_docling_client

        boxed_available = await get_docling_client(settings).health()
        if not boxed_available and not bool(body.get("accept_degraded")):
            log.info("comic_narration_boxed_unavailable", comic_ref=comic_ref)
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "boxed_reading_unavailable",
                    "message": (
                        "The OCR sidecar isn't running, so bubbles can't be cropped — "
                        "the model would read each page as a thumbnail instead. Start it "
                        "(compose.ocr.yaml) for the better transcription, or read whole "
                        "pages anyway."
                    ),
                },
            )

    jobs_store = getattr(request.app.state, "jobs_store", None)
    if jobs_store is None:
        raise HTTPException(503, "Background jobs are not available")

    # Title from the file-index entry name (best-effort).
    title = "Comic"
    idx = getattr(request.app.state, "file_index", None)
    if idx is not None:
        try:
            entry = await idx.get(comic_ref, user_id=uid)
            if entry and getattr(entry, "name", ""):
                title = entry.name
        except Exception:  # noqa: BLE001
            pass

    job_id = await jobs_store.create(
        user_id=uid,
        job_type="comic_narration_synth",
        payload={
            "comic_kind": "file", "comic_ref": comic_ref, "voice": bare_voice,
            "voice_cast": bare_cast,
            "voice_male": bare_male, "voice_female": bare_female,
            "title": title, "engine_id": engine_id, "format": fmt,
            "reading_direction": reading_direction, "start_page": start_page,
        },
    )
    await store.begin(
        "file", comic_ref, bare_voice, job_id,
        engine_id=engine_id, reading_direction=reading_direction,
        voice_male=bare_male, voice_female=bare_female, voice_cast=cast_json,
        user_id=uid,
    )
    return JSONResponse({
        "status": "running", "job_id": job_id, "voice": bare_voice,
        "voice_cast": bare_cast,
    })


@router.post("/{comic_ref}/cast")
async def cast_comic_narration(comic_ref: str, request: Request):
    """Return the streaming play payload (per-page audio + lines + page source)
    once at least one page is ready.

    Consumed by two surfaces, each driving its OWN renderer from the shared
    audio-clock engine (``ui/scripts/comic-reader/narration-clock.js``):
      - the in-app reader's narration bar, and
      - the cast-comic TV surface (``ui/cast-comic/cast-comic.js``), which
        read-aloud's the comic on a TV and can duck a music bed under it.
    Both build page images from ``/api/media/comic/page/{comic_ref}?page=N``
    and chain the per-page audio; the ``status``/poll fields let the clock
    stream pages as synthesis completes.

    409 when no page is ready yet (or the comic has no narration) — the cast
    surface treats that as 'stay image-only'."""
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    store = ComicNarrationStore(backend.conn)
    row = await store.get("file", comic_ref, user_id=uid)
    if not row or not _parse_pages(row):
        raise HTTPException(409, "Narration is not ready")
    return JSONResponse(_view(row, None, include_pages=True))


async def _delete_page_artifacts(request: Request, uid: str, row: dict) -> None:
    """Best-effort removal of a narration's per-page audio artifacts."""
    astore = getattr(request.app.state, "artifact_store", None)
    if astore is None:
        return
    for p in _parse_pages(row):
        aid = p.get("artifact_id")
        if not aid:
            continue
        try:
            await astore.delete(aid, user_id=uid)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            log.warning("comic_narration_artifact_cleanup_failed", artifact_id=aid)


@router.delete("/{comic_ref}")
async def delete_comic_narration(comic_ref: str, request: Request):
    """Drop a comic's narration (so it can be re-synthesized), cleaning up its
    per-page audio artifacts."""
    backend = _backend(request)
    if backend is None:
        raise HTTPException(503, "Database not available")
    uid = _user_id(request)
    store = ComicNarrationStore(backend.conn)
    row = await store.get("file", comic_ref, user_id=uid)
    if row:
        await _delete_page_artifacts(request, uid, row)
    deleted = await store.delete("file", comic_ref, user_id=uid)
    return JSONResponse({"deleted": deleted})
