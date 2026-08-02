"""Title routes -- the unified surface for the Augmentum Experience
Framework (AXF).

This is the new, framework-level API for everything playable. Existing
``/api/games/*`` endpoints continue to work unchanged during the
transition; they target a narrower legacy surface (js13k browse + pin).
The ``/api/titles/*`` surface is what the new Library / Marketplace UI
will consume going forward.

Master toggle: ``titles_enabled`` (defaults False; flipped on by setup
wizard or admin once the framework UI lands).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import uuid
import zipfile
import zlib
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

# Starlette's UploadFile is the *parent* of fastapi.UploadFile in the
# current FastAPI/Starlette pinned here. request.form() returns the
# parent class directly, so an isinstance check against the FastAPI
# subclass returns False and silently skips every file. Import the
# parent class under an alias so the runtime guard accepts both.
from starlette.datastructures import UploadFile as _StarletteUploadFile

from augmentum.config import settings
from augmentum.titles import (
    TITLE_KINDS,
    BiosServiceError,
    TitleNotFound,
    TitleNotPlayable,
    TitleService,
    TitleServiceError,
)
from augmentum.titles.file_classifier import Classification, classify
from augmentum.titles.rom_systems import detect_system, get_system
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/titles", tags=["titles"])


# ── helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _service(request: Request) -> TitleService | None:
    return getattr(request.app.state, "title_service", None)


def _gate(request: Request) -> JSONResponse | None:
    """Master toggle + dependency check.

    Returns a 503 response when titles are disabled or the service
    didn't initialise; otherwise None and the caller continues.
    """
    if not getattr(settings, "titles_enabled", False):
        return JSONResponse(
            {"error": "Titles framework is disabled"},
            status_code=503,
        )
    if _service(request) is None:
        return JSONResponse(
            {"error": "Titles framework is not available"},
            status_code=503,
        )
    return None


# ── Library ──────────────────────────────────────────────────────────


@router.get("/")
async def list_titles(
    request: Request,
    kind: str | None = None,
    pinned_only: bool = False,
    limit: int = 200,
) -> JSONResponse:
    """List the user's titles. Filters: kind, pinned-only."""
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if kind is not None and kind not in TITLE_KINDS:
        return JSONResponse(
            {"error": f"Unknown kind: {kind!r} (known: {sorted(TITLE_KINDS)})"},
            status_code=400,
        )

    svc = _service(request)
    titles = await svc.list_titles(
        user_id=uid,
        kind=kind,
        pinned_only=bool(pinned_only),
        limit=max(1, min(500, int(limit))),
    )
    return JSONResponse({"titles": [t.to_dict() for t in titles]})


@router.post("/")
async def import_title(request: Request) -> JSONResponse:
    """Import a title manifest via a registered Source.

    Body shape:
        {
          "source_id":  "internal",          # required
          "manifest":   {                    # required
            "kind":     "web_app",
            "title":    "...",
            "source_remote_id": "...",
            "runtime_preferred": "browser-iframe",
            "metadata": { ... }
          }
        }
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body: dict[str, Any] = await request.json()
    source_id = str(body.get("source_id", "")).strip()
    manifest_data = body.get("manifest")
    if not source_id or not isinstance(manifest_data, dict):
        return JSONResponse(
            {"error": "source_id and manifest are required"},
            status_code=400,
        )

    svc = _service(request)
    try:
        manifest, created = await svc.import_title(
            user_id=uid,
            source_id=source_id,
            manifest_data=manifest_data,
        )
    except TitleServiceError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    # 200 for "this already existed" / 201 for "newly created" so
    # bulk-import callers (ROM folder drop) can bucket "imported"
    # vs "duplicate" without a separate query. ``created`` is also
    # in the body for clients that don't inspect status codes.
    return JSONResponse(
        {"title": manifest.to_dict(), "created": bool(created)},
        status_code=201 if created else 200,
    )


@router.get("/{title_id}")
async def get_title(request: Request, title_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    svc = _service(request)
    try:
        manifest = await svc.get_title(title_id, user_id=uid)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    return JSONResponse({"title": manifest.to_dict()})


@router.patch("/{title_id}")
async def update_title(request: Request, title_id: str) -> JSONResponse:
    """Update mutable fields: pinned, metadata patch.

    Body shape:
        { "pinned": true|false, "metadata": { ... patch ... } }
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body: dict[str, Any] = await request.json()
    svc = _service(request)

    if "pinned" in body:
        ok = await svc.set_pinned(
            title_id, user_id=uid, pinned=bool(body["pinned"]),
        )
        if not ok:
            return JSONResponse({"error": "Title not found"}, status_code=404)

    patch = body.get("metadata")
    if isinstance(patch, dict):
        ok = await svc.update_metadata(title_id, user_id=uid, patch=patch)
        if not ok:
            return JSONResponse({"error": "Title not found"}, status_code=404)

    try:
        manifest = await svc.get_title(title_id, user_id=uid)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    return JSONResponse({"title": manifest.to_dict()})


@router.delete("/{title_id}")
async def delete_title(request: Request, title_id: str) -> JSONResponse:
    """Remove a title from the user's library.

    Drops the artifact row and its run history, deletes every save slot
    (releasing the save blobs they referenced), and -- for uploaded
    ROMs -- releases the ROM blob so the bytes are reclaimed once
    nothing else points at them. Also clears any legacy js13k save blob
    stashed in the user-settings KV. A second call 404s.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    svc = _service(request)
    try:
        manifest = await svc.delete_title(title_id, user_id=uid)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)

    # Cascade: per-title save slots + their blobs.
    saves_removed = 0
    save_store = getattr(request.app.state, "save_store", None)
    if save_store is not None:
        try:
            saves_removed = await save_store.delete_all_for_title(
                user_id=uid, artifact_id=title_id,
            )
        except Exception as exc:
            log.warning(
                "title_delete_saves_cleanup_failed",
                id=title_id, error=str(exc),
            )

    # Uploaded ROMs hold one refcount on their ROM blob; release it so
    # the bytes get GC'd by the blob janitor once nothing else refs it.
    rom_sha = str(manifest.raw_metadata.get("rom_sha256") or "").strip()
    if rom_sha:
        blobs = getattr(request.app.state, "blob_store", None)
        if blobs is not None:
            try:
                await blobs.release(rom_sha)
            except Exception as exc:
                log.warning(
                    "title_delete_rom_blob_release_failed",
                    id=title_id, sha=rom_sha, error=str(exc),
                )

    # Legacy js13k pins keep their save blob in the user_settings KV
    # (``game_save:{id}``), not the game_saves table -- mirror the
    # /api/games/pin DELETE cleanup so deleting through this surface
    # doesn't strand it. Harmless no-op for everything else.
    ss = getattr(request.app.state, "settings_store", None)
    if ss is not None:
        try:
            await ss.set_user(uid, f"game_save:{title_id}", None)
        except Exception as exc:
            log.warning(
                "title_delete_legacy_save_cleanup_failed",
                id=title_id, error=str(exc),
            )

    log.info(
        "title_deleted", id=title_id, user_id=uid, saves_removed=saves_removed,
    )
    return JSONResponse({"ok": True, "saves_removed": saves_removed})


# ── Launch / runs ────────────────────────────────────────────────────


@router.post("/{title_id}/launch")
async def launch_title(request: Request, title_id: str) -> JSONResponse:
    """Start a session for the given title.

    Body shape (all optional):
        {
          "prefer_runtime": "agsp-streamed",
          "ctx": { "resolution": "1920x1080", "bitrate_mbps": 6, ... }
        }
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    prefer_runtime = body.get("prefer_runtime")
    ctx = body.get("ctx") if isinstance(body.get("ctx"), dict) else {}

    svc = _service(request)
    try:
        result = await svc.launch(
            title_id,
            user_id=uid,
            ctx=ctx,
            prefer_runtime=str(prefer_runtime) if isinstance(prefer_runtime, str) else None,
        )
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    except TitleNotPlayable as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except TitleServiceError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(result, status_code=201)


@router.post("/{title_id}/runs/{run_id}/end")
async def end_run(
    request: Request, title_id: str, run_id: str,
) -> JSONResponse:
    """Mark a run as ended. Called by the client when the player surface
    closes (or by a server-side reaper for idle/crashed runs).

    Body shape (all optional):
        {
          "runtime_id":  "agsp-streamed",  # if present + session_id, runtime.stop is invoked
          "session_id":  "...",
          "exit_reason": "clean",
          "avg_fps":     58.2,
          "avg_rtt_ms":  42.0,
          "avg_bitrate_kbps": 4000,
          "crashes":     0,
          "metadata":    { ... }
        }
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    svc = _service(request)
    ok = await svc.end_run(
        run_id,
        user_id=uid,
        runtime_id=str(body.get("runtime_id", "")),
        session_id=str(body.get("session_id", "")),
        exit_reason=str(body.get("exit_reason", "clean")),
        avg_fps=_as_float(body.get("avg_fps")),
        avg_rtt_ms=_as_float(body.get("avg_rtt_ms")),
        avg_bitrate_kbps=_as_int(body.get("avg_bitrate_kbps")),
        crashes=int(body.get("crashes", 0) or 0),
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
    )
    if not ok:
        return JSONResponse({"error": "Run not found or already ended"}, status_code=404)
    return JSONResponse({"ok": True})


@router.get("/{title_id}/runs")
async def list_runs(
    request: Request, title_id: str, limit: int = 50,
) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    svc = _service(request)
    runs = await svc.list_runs(
        user_id=uid,
        title_id=title_id,
        limit=max(1, min(500, int(limit))),
    )
    return JSONResponse({"runs": runs})


# ── ROM upload ───────────────────────────────────────────────────────


_DEFAULT_ROM_MAX_BYTES = 5000 * 1024 * 1024  # 5000 MB -- fits Wii single-layer (4.7 GB) out of the box; dual-layer (8.5 GB) still needs AUGMENTUM_EMULATOR_ROM_MAX_MB bump
# Header window for magic-byte detection. 1 MiB covers everything we
# disambiguate today: NES/GC/Wii fit in the first 64 bytes; PSX/PS2
# need the ISO9660 Primary Volume Descriptor at 0x8000-0x87FF AND the
# SYSTEM.CNF "BOOT2" disambiguator. SYSTEM.CNF *content* lives at
# whatever LBA the directory record points to; on retail PS2 discs
# that's typically LBA 0x14-0x40 (40-128 KB), but pressed discs
# from later production runs sometimes push it past 256 KB. 1 MiB
# is the comfortable upper bound — cost is just slicing bytes
# already in memory, so the bump is free.
_HEADER_SAMPLE_BYTES = 1 * 1024 * 1024


@router.post("/upload-rom")
async def upload_rom(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    system_id: str | None = Form(None),
) -> JSONResponse:
    """Upload an emulator ROM. Two-step install:

    1. POST this endpoint with a multipart ``file`` field. We write
       the ROM to the blob store and detect the system from filename
       + header. Returns a ``manifest_data`` dict.
    2. POST that ``manifest_data`` to ``POST /api/titles/`` with
       ``source_id="internal-rom"`` to create the title.

    Why two-step: keeps the upload endpoint focused on bytes-handling
    (multipart, blob store, system detection) and the title creation
    on JSON shape (manifest schema, extra metadata).

    Body: multipart/form-data with:
        file       binary ROM bytes (required)
        title      override display name (optional; defaults to filename stem)
        system_id  override detected system (optional; force-pick when
                   detection is ambiguous, e.g. .iso for PSX vs Saturn)
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    blobs = getattr(request.app.state, "blob_store", None)
    if blobs is None:
        return JSONResponse(
            {"error": "Blob store unavailable"}, status_code=503,
        )

    # Read into memory. 2GB cap is generous for retro ROMs (PSP ISOs
    # at the upper end). The settings layer can lower this; keep the
    # hard ceiling so a hostile upload can't OOM the server.
    cap = _rom_max_bytes()
    data = await file.read(cap + 1)
    if len(data) > cap:
        return JSONResponse(
            {"error": f"ROM exceeds size cap ({cap} bytes)"},
            status_code=413,
        )
    if not data:
        return JSONResponse(
            {"error": "Empty file"}, status_code=400,
        )

    # System detection
    spec = None
    if system_id:
        spec = get_system(system_id)
        if spec is None:
            return JSONResponse(
                {"error": f"unknown system_id: {system_id!r}"},
                status_code=400,
            )
    else:
        spec = detect_system(file.filename or "", header=data[:_HEADER_SAMPLE_BYTES])
    if spec is None:
        return JSONResponse(
            {
                "error": (
                    "Could not detect system from filename or header. "
                    "Pass system_id explicitly."
                ),
            },
            status_code=400,
        )

    # Write the ROM blob (refcount-tracked, sha-addressed, dedup-safe).
    blob_meta = await blobs.write(data, mime_type="application/octet-stream")

    # Compose the manifest_data the InternalRomSource expects.
    display_name = (title or "").strip()
    if not display_name and file.filename:
        # Default: filename without extension, prettified.
        from pathlib import Path
        display_name = Path(file.filename).stem.replace("_", " ").strip()
    if not display_name:
        display_name = f"{spec.label} ROM"

    manifest_data = {
        "rom_sha256": blob_meta["sha256"],
        "rom_size_bytes": int(blob_meta["size_bytes"]),
        "system_id": spec.id,
        "title": display_name,
        "original_filename": file.filename or "",
    }

    log.info(
        "rom_uploaded",
        user_id=uid,
        system=spec.id,
        size_bytes=int(blob_meta["size_bytes"]),
        sha=blob_meta["sha256"],
    )
    return JSONResponse(
        {
            "manifest_data": manifest_data,
            "system": {
                "id": spec.id,
                "label": spec.label,
                "bios_required": spec.bios_required,
                "libretro_core": spec.libretro_core,
            },
            "next_step": "POST /api/titles/ with source_id='internal-rom' and the manifest_data above",
        },
        status_code=201,
    )


def _rom_max_bytes() -> int:
    mb = getattr(settings, "emulator_rom_max_mb", 0)
    if isinstance(mb, int) and mb > 0:
        return mb * 1024 * 1024
    return _DEFAULT_ROM_MAX_BYTES


# ── ROM bytes (served only for emulator titles owned by the user) ────


@router.api_route(
    "/{title_id}/rom",
    methods=["GET", "HEAD"],
    operation_id="get_rom_bytes",
)
async def get_rom_bytes(request: Request, title_id: str):
    """Serve the ROM bytes for an emulator title the user owns.

    HEAD is supported because EmulatorJS issues a HEAD probe before
    GET to read Content-Length for the download progress bar; if
    HEAD 405s the bootstrap crashes inside emulator.min.js with
    "Cannot read properties of undefined (reading 'content-length')".
    Starlette's FileResponse handles HEAD natively (skips the body
    write but still emits all headers) so this is just a route
    method-allow change.

    The browser-side EmulatorJS runtime fetches this URL and feeds
    the bytes into its core. Strict ownership check -- only the
    title's owner can read its ROM.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    svc = _service(request)
    try:
        manifest = await svc.get_title(title_id, user_id=uid)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)

    if manifest.kind != "emulator_rom":
        return JSONResponse(
            {"error": "Title is not an emulator ROM"}, status_code=400,
        )
    rom_sha = str(manifest.raw_metadata.get("rom_sha256", ""))
    if not rom_sha:
        return JSONResponse(
            {"error": "Title has no ROM blob"}, status_code=404,
        )

    blobs = getattr(request.app.state, "blob_store", None)
    if blobs is None:
        return JSONResponse(
            {"error": "Blob store unavailable"}, status_code=503,
        )
    blob = await blobs.get(rom_sha)
    if blob is None:
        return JSONResponse(
            {"error": "ROM blob missing"}, status_code=410,
        )

    from fastapi.responses import FileResponse
    return FileResponse(
        blob["real_path"],
        media_type="application/octet-stream",
        filename=str(manifest.raw_metadata.get("original_filename", "rom.bin")),
        headers={"X-Rom-SHA256": rom_sha},
    )


# ── Cover artwork lookup ───────────────────────────────────────────


@router.get("/{title_id}/cover-candidates")
async def get_cover_candidates(
    request: Request, title_id: str, limit: int = 12,
) -> JSONResponse:
    """Return candidate cover-art URLs for a ROM title.

    Pulls from three sources, ordered by quality:
      1. libretro-thumbnails Named_Boxarts (the gold standard --
         every No-Intro / Redump entry has one)
      2. libretro-thumbnails Named_Titles + Named_Snaps (title
         screens + in-game screenshots; better than nothing)
      3. SearXNG image search ("<title> <system> cover") -- last
         resort for renamed ROMs that don't match the exact
         libretro filename convention

    Each candidate carries ``{url, source, label}``. The frontend
    renders them in a picker; the user clicks one (or uploads
    their own) and we PATCH metadata.thumbnail_url to lock it in.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    svc = _service(request)
    try:
        manifest = await svc.get_title(title_id, user_id=uid)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    if manifest.kind != "emulator_rom":
        return JSONResponse(
            {"error": "Cover lookup only supported for emulator ROMs"},
            status_code=400,
        )

    meta = manifest.raw_metadata
    title = str(meta.get("title", manifest.title)).strip()
    system_id = str(meta.get("system_id", "")).strip()
    system_label = str(meta.get("system_label", "")).strip()
    original_filename = str(meta.get("original_filename", "")).strip()

    candidates: list[dict[str, Any]] = []

    # 1) libretro-thumbnails -- try the exact filename plus a few
    #    common normalisations. Each subfolder gets one URL per
    #    variant; the frontend's <img onerror> will silently drop
    #    URLs that 404.
    libretro_dir = _LIBRETRO_THUMB_DIRS.get(system_id)
    if libretro_dir:
        names = _libretro_filename_variants(title, original_filename)
        for kind, folder, label in (
            ("boxart", "Named_Boxarts", "Box art"),
            ("title", "Named_Titles", "Title screen"),
            ("snap",  "Named_Snaps",   "Screenshot"),
        ):
            for name in names:
                candidates.append({
                    "url": _libretro_thumb_url(libretro_dir, folder, name),
                    "source": f"libretro:{kind}",
                    "label": label,
                })

    # 2) SearXNG image search -- broader catch when libretro misses.
    #    Best-effort; failures don't bubble (we just return what we
    #    have from libretro).
    if int(limit) > len(candidates):
        try:
            search_results = await _searxng_image_search(
                f"{title} {system_label or system_id} cover",
                limit=int(limit) - len(candidates),
            )
            for r in search_results:
                candidates.append({
                    "url": r["url"],
                    "source": "searxng",
                    "label": (r.get("source") or "Web search"),
                })
        except Exception as exc:
            log.debug("cover_searxng_failed", error=str(exc))

    return JSONResponse({
        "candidates": candidates[: int(limit)],
        "current_url": str(meta.get("thumbnail_url", "")),
    })


@router.get("/_/cover-proxy")
async def proxy_cover_image(request: Request, url: str) -> Any:
    """Stream an arbitrary http(s) image through this same-origin
    endpoint so the cover picker can display SearXNG candidates from
    domains the parent CSP doesn't whitelist. Auth-gated so only
    logged-in users can use the proxy (and only for image content).
    Caps + content-type validation match cover-from-url so a hostile
    URL can't burn bandwidth or smuggle non-image content.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not url.lower().startswith(("http://", "https://")):
        return JSONResponse({"error": "url must be http(s)"}, status_code=400)

    global _HTTP_CLIENT_FOR_COVERS
    if _HTTP_CLIENT_FOR_COVERS is None:
        import httpx
        _HTTP_CLIENT_FOR_COVERS = httpx.AsyncClient(timeout=8.0)

    try:
        resp = await _HTTP_CLIENT_FOR_COVERS.get(url, follow_redirects=True)
    except Exception:
        return JSONResponse({"error": "fetch failed"}, status_code=502)
    if resp.status_code >= 400:
        return JSONResponse({"error": "upstream error"}, status_code=502)
    ctype = resp.headers.get("content-type", "").lower().split(";")[0].strip()
    if not ctype.startswith("image/"):
        return JSONResponse({"error": "not an image"}, status_code=400)
    if len(resp.content) > 4 * 1024 * 1024:
        return JSONResponse({"error": "too large"}, status_code=413)

    from fastapi.responses import Response
    return Response(
        content=resp.content,
        media_type=ctype,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{title_id}/cover-from-url")
async def set_cover_from_url(request: Request, title_id: str) -> JSONResponse:
    """Server-side image fetch + save for the cover picker.

    The picker shows candidates from libretro-thumbnails AND SearXNG
    image search. SearXNG results come from arbitrary hosts that the
    parent CSP's ``img-src`` doesn't (and shouldn't) allow. To
    sidestep CSP entirely we have the server fetch the bytes, base64
    them into a ``data:`` URL, and stash that on
    ``metadata.thumbnail_url``. From then on the saved cover renders
    same-origin and can't be revoked by the source going offline.

    Body: ``{"url": "https://..."}``
    Caps: 4 MB image fetch, 5s timeout, ``image/*`` content-type only.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    url = str(body.get("url", "")).strip()
    if not url.lower().startswith(("http://", "https://")):
        return JSONResponse({"error": "url must be http(s)://"}, status_code=400)

    global _HTTP_CLIENT_FOR_COVERS
    if _HTTP_CLIENT_FOR_COVERS is None:
        import httpx
        _HTTP_CLIENT_FOR_COVERS = httpx.AsyncClient(timeout=8.0)

    try:
        resp = await _HTTP_CLIENT_FOR_COVERS.get(url, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        return JSONResponse(
            {"error": f"image fetch failed: {exc}"}, status_code=502,
        )
    ctype = resp.headers.get("content-type", "").lower().split(";")[0].strip()
    if not ctype.startswith("image/"):
        return JSONResponse(
            {"error": f"not an image (got {ctype!r})"}, status_code=400,
        )
    raw = resp.content
    cap = 4 * 1024 * 1024  # 4 MB; covers are typically <500 KB
    if len(raw) > cap:
        return JSONResponse(
            {"error": f"image too large ({len(raw)} > {cap} bytes)"},
            status_code=413,
        )

    import base64 as _b64
    data_url = f"data:{ctype};base64,{_b64.b64encode(raw).decode('ascii')}"

    svc = _service(request)
    ok = await svc.update_metadata(
        title_id, user_id=uid, patch={"thumbnail_url": data_url},
    )
    if not ok:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    try:
        manifest = await svc.get_title(title_id, user_id=uid)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    return JSONResponse({"title": manifest.to_dict()})


# ── Bulk cover scrape ──────────────────────────────────────────────
#
# The per-title picker (``/cover-candidates`` + ``/cover-from-url``) is
# the manual path: it hands the browser a list of *unverified* URLs and
# lets ``<img onerror>`` discover which ones 404. That's fine for one
# title but useless for a freshly-imported folder of 47 ROMs, where the
# user would have to open 47 modals.
#
# This is the unattended path, and the difference that makes it possible
# is that the SERVER verifies. We HEAD each libretro candidate and only
# apply one that actually resolves, so a title either gets real box art
# or is reported as needing review -- never a broken image. Titles we
# can't resolve are left untouched for the user to pick manually, which
# is the "pause only on ambiguity" shape ES-DE uses, except the queue
# keeps moving instead of blocking on each miss.
#
# Jobs live in memory: a scrape is cheap to re-run and meaningless to
# resume across a restart (the applied covers are already persisted on
# the titles themselves).
_SCRAPE_JOBS: dict[str, dict[str, Any]] = {}
_SCRAPE_JOB_CAP = 32


async def _head_ok(url: str) -> bool:
    """True when ``url`` resolves to a real image.

    libretro serves 404s for names that don't match its No-Intro
    convention exactly, which is the entire reason this check exists --
    the candidate list is a set of *guesses* and only one of them is
    usually right. Some CDNs refuse HEAD, so fall back to a ranged GET
    rather than treating a 405 as a miss.
    """
    global _HTTP_CLIENT_FOR_COVERS
    if _HTTP_CLIENT_FOR_COVERS is None:
        import httpx
        _HTTP_CLIENT_FOR_COVERS = httpx.AsyncClient(timeout=8.0)
    try:
        resp = await _HTTP_CLIENT_FOR_COVERS.head(url, follow_redirects=True)
        if resp.status_code == 405:
            resp = await _HTTP_CLIENT_FOR_COVERS.get(
                url, follow_redirects=True, headers={"Range": "bytes=0-0"},
            )
        if resp.status_code >= 400:
            return False
        ctype = resp.headers.get("content-type", "").lower()
        # An empty content-type from a 200 is ambiguous; accept it.
        return not ctype or ctype.startswith("image/")
    except Exception:
        return False


async def _scrape_one(manifest: Any, svc: Any, uid: str) -> dict[str, Any]:
    """Resolve and apply box art for a single ROM manifest.

    Only Named_Boxarts is considered. Title screens and screenshots are
    offered in the manual picker as a better-than-nothing fallback, but
    auto-applying one would quietly fill a library with inconsistent art
    that looks scraped-and-done -- worse than an obvious gap the user can
    fix. A miss here is reported, not papered over.
    """
    meta = manifest.raw_metadata or {}
    entry = {
        "title_id": manifest.id,
        "title": manifest.title,
        "status": "skipped",
        "url": "",
    }
    system_id = str(meta.get("system_id", "")).strip()
    libretro_dir = _LIBRETRO_THUMB_DIRS.get(system_id)
    if not libretro_dir:
        entry["status"] = "no_system"
        return entry

    names = _libretro_filename_variants(
        str(meta.get("title", manifest.title)).strip(),
        str(meta.get("original_filename", "")).strip(),
    )
    for name in names:
        url = _libretro_thumb_url(libretro_dir, "Named_Boxarts", name)
        if await _head_ok(url):
            ok = await svc.update_metadata(
                manifest.id, user_id=uid, patch={"thumbnail_url": url},
            )
            entry["status"] = "applied" if ok else "failed"
            entry["url"] = url
            return entry
    entry["status"] = "needs_review"
    return entry


async def _run_scrape_job(job: dict[str, Any], svc: Any, uid: str) -> None:
    """Drive one scrape job to completion, updating ``job`` in place."""
    try:
        for manifest in job.pop("_queue", []):
            if job.get("state") == "cancelled":
                break
            job["current"] = manifest.title
            try:
                entry = await _scrape_one(manifest, svc, uid)
            except Exception as exc:
                log.warning(
                    "scrape_title_failed", title=manifest.id, error=str(exc),
                )
                entry = {
                    "title_id": manifest.id,
                    "title": manifest.title,
                    "status": "failed",
                    "url": "",
                }
            job["results"].append(entry)
            job["done"] = len(job["results"])
        if job.get("state") != "cancelled":
            job["state"] = "done"
    except Exception as exc:
        # Never leave a job wedged in "running" -- the UI polls until it
        # sees a terminal state and would spin forever.
        log.warning("scrape_job_failed", error=str(exc))
        job["state"] = "failed"
        job["error"] = str(exc)
    finally:
        job["current"] = ""
        job.pop("_queue", None)


@router.post("/_/scrape")
async def start_cover_scrape(request: Request) -> JSONResponse:
    """Kick off a bulk cover scrape over the user's ROM library.

    Body (all optional)::

        {"title_ids": ["..."],     # default: every emulator_rom
         "only_missing": true}     # default: true -- don't clobber
                                   # covers the user picked by hand

    Returns ``{job_id, total}``; poll ``GET /_/scrape/{job_id}``.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    svc = _service(request)

    try:
        body = await request.json()
    except Exception:
        body = {}
    only_missing = bool(body.get("only_missing", True))
    wanted = {str(t) for t in (body.get("title_ids") or [])}

    manifests = await svc.list_titles(
        user_id=uid, kind="emulator_rom", limit=5000,
    )
    queue = []
    for m in manifests:
        if wanted and m.id not in wanted:
            continue
        # A user-chosen cover is never overwritten by a guess.
        if only_missing and str((m.raw_metadata or {}).get("thumbnail_url", "")):
            continue
        queue.append(m)

    # Bound the registry so a user hammering the button can't grow it
    # without limit. Terminal jobs are the ones safe to drop.
    if len(_SCRAPE_JOBS) > _SCRAPE_JOB_CAP:
        for jid, j in list(_SCRAPE_JOBS.items()):
            if j.get("state") in ("done", "failed", "cancelled"):
                _SCRAPE_JOBS.pop(jid, None)

    job_id = uuid.uuid4().hex[:16]
    job: dict[str, Any] = {
        "job_id": job_id,
        "user_id": uid,
        "state": "running" if queue else "done",
        "total": len(queue),
        "done": 0,
        "current": "",
        "results": [],
        "_queue": queue,
    }
    _SCRAPE_JOBS[job_id] = job
    if queue:
        asyncio.create_task(_run_scrape_job(job, svc, uid))
    else:
        job.pop("_queue", None)
    return JSONResponse({"job_id": job_id, "total": job["total"]})


def _scrape_job_for(request: Request, job_id: str) -> dict[str, Any] | None:
    """Fetch a job, enforcing per-user ownership.

    Job ids are opaque but guessable-adjacent, and a scrape reveals the
    titles in someone's library -- so ownership is checked, not assumed.
    """
    job = _SCRAPE_JOBS.get(job_id)
    if job is None or job.get("user_id") != _user_id(request):
        return None
    return job


@router.get("/_/scrape/{job_id}")
async def get_cover_scrape(request: Request, job_id: str) -> JSONResponse:
    """Poll a scrape job's progress."""
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    job = _scrape_job_for(request, job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({
        k: v for k, v in job.items()
        if not k.startswith("_") and k != "user_id"
    })


@router.post("/_/scrape/{job_id}/cancel")
async def cancel_cover_scrape(request: Request, job_id: str) -> JSONResponse:
    """Stop a running scrape. Already-applied covers stay applied."""
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    job = _scrape_job_for(request, job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job.get("state") == "running":
        job["state"] = "cancelled"
    return JSONResponse({"state": job["state"], "done": job["done"]})


# libretro-thumbnails directory names per system, mirroring the
# frontend table in library-game-sources.js. Kept duplicated rather
# than imported because the front and back stacks are deliberately
# decoupled (UI can run against any backend).
_LIBRETRO_THUMB_DIRS: dict[str, str] = {
    "nes":          "Nintendo - Nintendo Entertainment System",
    "snes":         "Nintendo - Super Nintendo Entertainment System",
    "gb":           "Nintendo - Game Boy",
    "gbc":          "Nintendo - Game Boy Color",
    "gba":          "Nintendo - Game Boy Advance",
    "n64":          "Nintendo - Nintendo 64",
    "nds":          "Nintendo - Nintendo DS",
    "gamecube":     "Nintendo - GameCube",
    "wii":          "Nintendo - Wii",
    "genesis":      "Sega - Mega Drive - Genesis",
    "sms":          "Sega - Master System - Mark III",
    "gg":           "Sega - Game Gear",
    "saturn":       "Sega - Saturn",
    "psx":          "Sony - PlayStation",
    "ps2":          "Sony - PlayStation 2",
    "psp":          "Sony - PlayStation Portable",
    "atari2600":    "Atari - 2600",
    "lynx":         "Atari - Lynx",
    "pce":          "NEC - PC Engine - TurboGrafx 16",
    "colecovision": "Coleco - ColecoVision",
}


def _libretro_thumb_url(directory: str, folder: str, name: str) -> str:
    """Build a libretro-thumbnails URL for one candidate name.

    libretro stores every file with the characters that are illegal in
    a Windows filename replaced by ``_``, so "Kirby & The Amazing
    Mirror (USA)" is served as "Kirby _ The Amazing Mirror (USA).png".
    Without the substitution every title containing ``&`` -- a large
    slice of any real collection -- 404s. Percent-encode after, so
    spaces and punctuation survive the trip.
    """
    from urllib.parse import quote

    safe = re.sub(r'[&*/:`<>?|"\\]', "_", name)
    return (
        "https://thumbnails.libretro.com/"
        f"{quote(directory)}/{folder}/{quote(safe)}.png"
    )


def _libretro_filename_variants(title: str, original_filename: str) -> list[str]:
    """Generate likely libretro-thumbnail filename matches.

    libretro thumbnails are keyed by the No-Intro / Redump filename
    WITHOUT extension. User-uploaded ROMs frequently have stripped
    region tags, alternate punctuation, or pretty titles. This walks
    common normalisations so we hit on as many user variations as
    possible -- the frontend then drops 404s silently.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(candidate: str) -> None:
        s = candidate.strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    # Strip extension if a filename was given.
    #
    # ``original_filename`` is a RELATIVE PATH, not a basename -- the
    # folder picker reports webkitRelativePath, so bulk-imported ROMs
    # arrive as "gba/Metroid Fusion (USA).gba" or "roms/wii/...rvz".
    # Leaving the directory on produced ".../Named_Boxarts/gba/Metroid
    # Fusion (USA).png", which is a different path entirely and 404s for
    # every folder-imported ROM. Drop it before anything else.
    base = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
    if base:
        dot = base.rfind(".")
        if dot > 0:
            base = base[:dot]

    # Both the filename stem and the stored title are worth walking --
    # a scene dump has a mangled filename but the user may have renamed
    # the title, and vice versa.
    for raw in (base, title):
        for variant in _name_variants(raw):
            add(variant)
    return out


# Single-letter region tags used by GoodTools/scene dumps, mapped to the
# No-Intro spellings libretro actually files things under. A ROM named
# "Pokemon - Sapphire Version (U) (V1.1)" has to become "Pokemon -
# Sapphire Version (USA)" before it will resolve.
_REGION_EXPANSIONS: dict[str, str] = {
    "u": "USA",
    "e": "Europe",
    "j": "Japan",
    "usa": "USA",
    "eur": "Europe",
    "jap": "Japan",
    "jp": "Japan",
    "w": "World",
}


def _name_variants(raw: str) -> list[str]:
    """Normalisations of one name, most-likely-correct first.

    Real libraries are not No-Intro-clean. A live scrape over 56 GBA
    ROMs missed 23 of them, and every miss was a naming convention
    rather than a genuinely absent thumbnail: scene-release number
    prefixes ("1840 - Metal Slug Advance"), group tags ("(TRSI)"),
    GoodTools single-letter regions ("(U)" vs "(USA)"), bracket flags
    ("[!]"), and hand-appended system suffixes ("# GBA"). Each of those
    is mechanical to undo, so we undo them here rather than dumping the
    work on the user's manual picker.
    """
    s = str(raw or "").strip()
    if not s:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        c = re.sub(r"\s{2,}", " ", str(candidate or "")).strip()
        # A trailing "-" or "," is left behind by tag stripping and
        # never appears in a real libretro filename.
        c = c.strip(" -,")
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    add(s)

    # Detect the region BEFORE stripping parentheticals, so we can put
    # the No-Intro spelling back on the cleaned name.
    regions: list[str] = []
    for tag in re.findall(r"\(([^)]*)\)", s):
        key = tag.strip().lower()
        if key in _REGION_EXPANSIONS:
            regions.append(_REGION_EXPANSIONS[key])

    # Progressive cleanup, each step additive so we try the mildest
    # normalisation before the most aggressive one.
    work = s
    # Scene release number prefix: "1840 - Metal Slug Advance".
    work = re.sub(r"^\s*\d{3,5}\s*-\s*", "", work)
    add(work)
    # Hand-appended system suffix: "Final Fantasy 4 Advance # GBA".
    work = re.sub(r"\s*#.*$", "", work)
    add(work)
    # Bracket flags: "[!]", "[a1]", "[T+Eng]".
    work = re.sub(r"\s*\[[^\]]*\]\s*", " ", work)
    add(work)
    # Underscores for spaces (common in user-renamed dumps).
    add(work.replace("_", " "))

    # Fully bare: every parenthetical gone. libretro almost always
    # requires a region tag, so this rarely hits on its own -- but it's
    # the stem the region-tagged candidates are built from.
    bare = re.sub(r"\s*\([^)]*\)\s*", " ", work).strip()
    bare = re.sub(r"\s{2,}", " ", bare).strip(" -,")
    add(bare)
    add(bare.replace("_", " "))

    if bare:
        # Region-tagged reconstructions. The detected region goes first
        # (it's the dump's own claim), then the common fallbacks --
        # libretro's coverage is USA-heaviest, so that order is the one
        # most likely to resolve on the first HEAD.
        for region in [*regions, "USA", "Europe", "Japan"]:
            add(f"{bare} ({region})")

    return out


_HTTP_CLIENT_FOR_COVERS: Any = None


async def _searxng_image_search(query: str, *, limit: int) -> list[dict]:
    """Image-only SearXNG search. Filters to common image hosts +
    image extensions to avoid pulling random page URLs that won't
    render in <img>.
    """
    global _HTTP_CLIENT_FOR_COVERS
    if _HTTP_CLIENT_FOR_COVERS is None:
        import httpx
        _HTTP_CLIENT_FOR_COVERS = httpx.AsyncClient(timeout=10.0)

    base = getattr(settings, "searxng_base_url", "http://searxng:8080")
    try:
        resp = await _HTTP_CLIENT_FOR_COVERS.get(
            f"{base}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "images",
            },
            timeout=8.0,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.debug("searxng_image_search_failed", error=str(exc), query=query)
        return []

    out: list[dict] = []
    for item in (resp.json().get("results") or [])[: limit * 3]:
        # Image results in SearXNG come with ``img_src`` (direct
        # image URL). The ``url`` field is the source page; we want
        # the direct image so it renders inline.
        img = (item.get("img_src") or item.get("thumbnail_src")
               or item.get("url") or "").strip()
        if not img.lower().startswith(("http://", "https://")):
            continue
        out.append({
            "url": img,
            "source": item.get("engine") or "image",
            "title": item.get("title", ""),
        })
        if len(out) >= limit:
            break
    return out


# ── Bulk-import pre-flight ────────────────────────────────────────────


@router.get("/_/by-rom-sha")
async def find_title_by_rom_sha(
    request: Request, sha: str,
) -> JSONResponse:
    """Lookup an emulator-rom title by SHA256 of its ROM bytes.

    Returns ``{"title_id": "<id>"}`` (200) when one exists for this
    user, or 404 when not. The bulk import flow calls this BEFORE
    uploading bytes so a 1.5 GB GameCube ISO doesn't get re-sent
    just to discover it's already in the user's library.

    Pre-check is necessary because the upload-rom + import roundtrip
    streams the full file twice (network → blob store → import) even
    on duplicates; a 64-char hash compare here is ~5ms regardless of
    file size.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    sha_clean = sha.strip().lower()
    if len(sha_clean) != 64 or any(c not in "0123456789abcdef" for c in sha_clean):
        return JSONResponse({"error": "sha must be a 64-char hex sha256"}, status_code=400)

    conn = getattr(request.app.state, "db_pool_conn", None) \
        or getattr(request.app.state, "db", None)
    # Pull from the artifacts table directly -- same query
    # InternalRomSource._find_existing uses, but exposed as a
    # standalone read for the import preflight. Falls back to the
    # service if a direct conn isn't available.
    svc = _service(request)
    src = svc._sources.get("internal-rom") if svc else None
    if not src:
        return JSONResponse({"error": "internal-rom source unavailable"}, status_code=503)
    existing = await src._find_existing(sha_clean, user_id=uid)
    if not existing:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"title_id": existing})


# ── Discovery ────────────────────────────────────────────────────────


@router.get("/_/discover")
async def discover_titles(
    request: Request,
    source_id: str,
    sort: str | None = None,
    page: int = 1,
    q: str | None = None,
    limit: int = 50,
) -> JSONResponse:
    """Browse a Source's catalog. Returns DiscoveryItems.

    Items are decorated with an ``installed`` flag (true if the user
    already owns a matching title). Pass an item's
    ``source_remote_id`` back to ``POST /api/titles/`` with the same
    ``source_id`` to install it.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not source_id:
        return JSONResponse({"error": "source_id required"}, status_code=400)

    query: dict[str, Any] = {
        "page": max(1, int(page)),
        "limit": max(1, min(200, int(limit))),
    }
    if sort:
        query["sort"] = sort
    if q:
        query["q"] = q

    svc = _service(request)
    try:
        items = await svc.discover_titles(
            source_id=source_id, query=query, user_id=uid,
        )
    except TitleServiceError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({
        "source_id": source_id,
        "items": [item.to_dict() for item in items],
    })


# ── Registry introspection ───────────────────────────────────────────


@router.get("/_/runtimes")
async def list_runtimes(request: Request) -> JSONResponse:
    """Available runtimes -- the UI consults this to decide which
    'play with' options to surface for a given title."""
    if (gate := _gate(request)) is not None:
        return gate
    return JSONResponse({"runtimes": _service(request).list_runtimes()})


@router.get("/_/sources")
async def list_sources(request: Request) -> JSONResponse:
    """Available sources -- the import dropdown consumes this."""
    if (gate := _gate(request)) is not None:
        return gate
    return JSONResponse({"sources": _service(request).list_sources()})


@router.get("/_/systems")
async def list_emulator_systems(request: Request) -> JSONResponse:
    """The full ROM systems catalog (id, label, emulator, extensions).
    Used by the library's "Change system" picker so the user can
    correct an auto-classification mistake without re-uploading."""
    if (gate := _gate(request)) is not None:
        return gate
    from augmentum.titles.rom_systems import list_systems
    out = []
    for spec in list_systems():
        out.append({
            "id": spec.id,
            "label": spec.label,
            "emulator": spec.emulator,
            "extensions": list(spec.extensions),
            "bios_required": spec.bios_required,
        })
    return JSONResponse({"systems": out})


# ── Bulk import (drop-anything pipeline) ─────────────────────────────


# Per-file size cap for bulk import. Pre-classification we don't know
# if the file is a 4 KB GBA BIOS or a 4.7 GB PS2 ISO; we lean on the
# ROM cap as the upper bound. Junk files are usually KB-tier so we
# don't waste memory on them either way.
_BULK_PER_FILE_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
_BULK_MAX_FILES = 1024                              # safety net


@router.post("/bulk-import")
async def bulk_import(
    request: Request, system_id: str | None = None,
) -> JSONResponse:
    """Multi-file drop import: classify each file server-side and
    route to ROM/BIOS/junk handlers. The "drop a folder" UX endpoint.

    Body: ``multipart/form-data`` with one or more ``files`` fields.
    Each file is hashed, classified via ``file_classifier.classify``,
    then routed:

      * ROM      → existing InternalRomSource pipeline (creates title)
      * BIOS     → BiosStore.install (records in user_bios_files)
      * Archive  → .zip extracted in memory, members re-classified;
                   .7z/.rar/.tar surfaced as 'unknown' for now (the
                   user can extract manually)
      * Junk     → silently skipped with a count
      * Unknown  → returned in the summary with override candidates

    Returns a summary the UI renders as a post-import digest:
        {
          "imported":   [{filename, system, title_id}, ...],
          "bios":       [{filename, system, canonical_filename}, ...],
          "duplicates": [{filename, kind, existing_id}, ...],
          "junk":       [{filename, reason}, ...],
          "unknown":    [{filename, sha1, size, reason, candidates}, ...],
          "errors":     [{filename, error}, ...]
        }
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    blobs = getattr(request.app.state, "blob_store", None)
    bios_store = getattr(request.app.state, "bios_store", None)
    if blobs is None or bios_store is None:
        return JSONResponse(
            {"error": "Storage subsystems unavailable"}, status_code=503,
        )

    # Multipart parse. FastAPI's UploadFile is lazy so we walk form()
    # ourselves to read every 'files' field (the canonical multi-file
    # convention -- both 'files' and 'file' accepted for client
    # flexibility).
    #
    # Starlette's default Request.form() caps individual parts at
    # max_part_size=1 MB (added in starlette 0.45). A multi-GB ROM
    # blows past that and the file gets silently dropped — form()
    # returns empty and we'd 400 with "No files in request" even
    # though the body arrived intact. Pass an explicit cap matching
    # the per-file size limit so the parser keeps the upload.
    form = await request.form(
        max_files=_BULK_MAX_FILES,
        max_fields=64,
        max_part_size=_BULK_PER_FILE_MAX_BYTES,
    )
    uploads: list[UploadFile] = []
    raw_keys: list[str] = []
    for key in ("files", "file"):
        for item in form.getlist(key):
            # Check against the Starlette parent class — request.form()
            # returns starlette.datastructures.UploadFile directly, and
            # `isinstance(starlette_obj, fastapi.UploadFile)` is False
            # because fastapi.UploadFile is a strict subclass. Checking
            # against the parent class accepts both, future-proofing
            # against either side moving the implementation.
            if isinstance(item, _StarletteUploadFile):
                uploads.append(item)
            else:
                raw_keys.append(f"{key}={type(item).__name__}")
    if not uploads:
        log.warning(
            "bulk_import_no_files",
            form_keys=list(form.keys()),
            non_upload_items=raw_keys,
        )
        return JSONResponse(
            {"error": "No files in request (use multipart 'files' field)"},
            status_code=400,
        )
    if len(uploads) > _BULK_MAX_FILES:
        return JSONResponse(
            {"error": f"Too many files (max {_BULK_MAX_FILES} per request)"},
            status_code=413,
        )

    summary: dict[str, list[dict[str, Any]]] = {
        "imported": [], "bios": [], "duplicates": [],
        "junk": [], "unknown": [], "errors": [],
    }

    # ``?system_id=`` is set by the BIOS vault when the user targets a
    # specific system's row. It is an intent signal, not a filter: it
    # only decides where an UNIDENTIFIED file lands. A file we can
    # positively identify still goes where its hash says it belongs,
    # so dropping a Saturn BIOS on the PSX row files it under Saturn
    # rather than mis-filing it silently.
    bios_system_hint = (system_id or "").strip().lower()

    rom_cap = _rom_max_bytes()

    for upload in uploads:
        try:
            data = await upload.read(_BULK_PER_FILE_MAX_BYTES + 1)
            if len(data) > _BULK_PER_FILE_MAX_BYTES:
                summary["errors"].append({
                    "filename": upload.filename or "",
                    "error": f"file exceeds bulk-import cap "
                             f"({_BULK_PER_FILE_MAX_BYTES} bytes)",
                })
                continue
            await _classify_and_route(
                request=request,
                filename=upload.filename or "",
                data=data,
                summary=summary,
                rom_cap=rom_cap,
                blobs=blobs,
                bios_store=bios_store,
                user_id=uid,
                bios_system_hint=bios_system_hint,
            )
        except Exception as exc:                # noqa: BLE001 -- per-file isolation
            log.warning(
                "bulk_import_file_failed",
                filename=upload.filename, error=str(exc),
            )
            summary["errors"].append({
                "filename": upload.filename or "",
                "error": str(exc),
            })

    log.info(
        "bulk_import_complete",
        user_id=uid,
        imported=len(summary["imported"]),
        bios=len(summary["bios"]),
        duplicates=len(summary["duplicates"]),
        junk=len(summary["junk"]),
        unknown=len(summary["unknown"]),
        errors=len(summary["errors"]),
    )
    return JSONResponse({"summary": summary})


async def _classify_and_route(
    *,
    request: Request,
    filename: str,
    data: bytes,
    summary: dict[str, list[dict[str, Any]]],
    rom_cap: int,
    blobs: Any,
    bios_store: Any,
    user_id: str,
    bios_system_hint: str = "",
) -> None:
    """Classify one file's bytes and route to the right install path.

    Side effect: appends to the appropriate summary bucket. Recursive
    on .zip archives (members re-enter via this same function with
    the archive's filename prefixed for diagnostics).

    ``bios_system_hint`` is set when the user dropped this file on a
    specific system's row in the BIOS vault. That is an explicit
    statement of intent ("this is a BIOS for that system"), and it
    changes the routing in two ways, both matching how RetroArch and
    EmuDeck treat their BIOS folders:

      * an unidentified file is STORED rather than discarded, labelled
        'unverified' so the UI stays honest about what we know;
      * a file that merely looks like a ROM by extension is NOT
        installed as a game. ``.bin`` is a registered ROM extension,
        so without this a PSX BIOS dropped on the PSX row would
        silently become a broken library title.
    """
    if not data:
        summary["junk"].append({"filename": filename, "reason": "empty file"})
        return

    sha1 = hashlib.sha1(data).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    crc32 = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
    size = len(data)
    header = data[:_HEADER_SAMPLE_BYTES]

    verdict = classify(
        filename, sha1=sha1, sha256=sha256, md5=md5, crc32=crc32,
        size_bytes=size, header=header,
    )

    if verdict.kind == "junk":
        summary["junk"].append({
            "filename": filename, "reason": verdict.junk_reason,
        })
        return

    if verdict.kind == "bios":
        await _install_bios(
            request=request, filename=filename, data=data,
            sha1=sha1, md5=md5, crc32=crc32,
            verdict=verdict, summary=summary,
            bios_store=bios_store, user_id=user_id,
        )
        return

    if verdict.kind == "rom":
        # Dropped on a BIOS row: the user's intent beats the extension
        # heuristic. See the docstring -- '.bin' is a ROM extension,
        # and most BIOS dumps are '.bin'.
        if bios_system_hint:
            await _store_asserted_bios(
                filename=filename, data=data, sha1=sha1, md5=md5,
                crc32=crc32, system_id=bios_system_hint, summary=summary,
                bios_store=bios_store, user_id=user_id,
                reason=(
                    f"Stored as {bios_system_hint} BIOS on your say-so "
                    f"(it also looks like a {verdict.system.id} ROM by "
                    "extension, but you dropped it on the BIOS row)"
                    if verdict.system else ""
                ),
            )
            return
        if size > rom_cap:
            summary["errors"].append({
                "filename": filename,
                "error": (
                    f"ROM exceeds size cap ({size} > {rom_cap} bytes); "
                    "raise emulator_rom_max_mb"
                ),
            })
            return
        await _install_rom(
            request=request, filename=filename, data=data,
            verdict=verdict, summary=summary,
            blobs=blobs, user_id=user_id,
        )
        return

    if verdict.kind == "archive":
        if verdict.archive_format == "zip":
            await _process_zip_archive(
                request=request, filename=filename, data=data,
                summary=summary, rom_cap=rom_cap,
                blobs=blobs, bios_store=bios_store, user_id=user_id,
                bios_system_hint=bios_system_hint,
            )
            return
        # Non-zip archives surface as 'unknown' so the user knows we
        # saw the file but can't open it. Lifts the ceiling for
        # adding 7z support later without breaking semantics.
        summary["unknown"].append({
            "filename": filename, "sha1": sha1, "size": size,
            "reason": (
                f"{verdict.archive_format} archives aren't extracted yet "
                "-- extract manually and re-import"
            ),
            "candidates": [],
        })
        return

    # Unknown. If the user dropped this on a system row in the BIOS
    # vault, that is an explicit assertion and we store it -- the
    # RetroArch/EmuDeck model, where the BIOS folder accepts what you
    # put in it and verification is advisory. Without a hint we have
    # nothing to file it under, so it goes to the override picker.
    if bios_system_hint:
        await _store_asserted_bios(
            filename=filename, data=data, sha1=sha1, md5=md5, crc32=crc32,
            system_id=bios_system_hint, summary=summary,
            bios_store=bios_store, user_id=user_id,
            reason=verdict.reason,
        )
        return

    summary["unknown"].append({
        "filename": filename, "sha1": sha1, "size": size,
        "reason": verdict.reason,
        "candidates": _override_candidates(filename),
    })


async def _store_asserted_bios(
    *,
    filename: str,
    data: bytes,
    sha1: str,
    md5: str,
    crc32: str,
    system_id: str,
    summary: dict[str, list[dict[str, Any]]],
    bios_store: Any,
    user_id: str,
    reason: str = "",
) -> None:
    """Install a file we could not identify, on the user's assertion
    that it belongs to ``system_id``.

    Stored under the name the user uploaded (leaf only -- a zip member
    arrives as ``pack.zip!bios/scph1001.bin`` and must not become a
    slot called that). Labelled 'unverified' so the vault can show it
    truthfully rather than implying we checked it.
    """
    leaf = filename.rsplit("!", 1)[-1]
    leaf = leaf.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or filename
    try:
        record = await bios_store.install(
            user_id=user_id,
            system_id=system_id,
            canonical_filename=leaf,
            data=data,
            original_filename=leaf,
            sha1=sha1, md5=md5, crc32=crc32,
            matched_by="user_asserted",
            verify_status="unverified",
        )
    except Exception as exc:                    # noqa: BLE001
        log.warning(
            "bios_store_asserted_failed",
            filename=filename, system=system_id, error=str(exc),
        )
        summary["errors"].append({"filename": filename, "error": str(exc)})
        return

    summary["bios"].append({
        "filename": filename,
        "system": system_id,
        "canonical_filename": record.canonical_filename,
        "matched_by": "user_asserted",
        "verify_status": "unverified",
        "reason": reason or "Stored unverified — we don't recognise this file",
    })


async def _install_rom(
    *,
    request: Request,
    filename: str,
    data: bytes,
    verdict: Classification,
    summary: dict[str, list[dict[str, Any]]],
    blobs: Any,
    user_id: str,
) -> None:
    """Route a classified ROM through the InternalRomSource pipeline.
    Same shape as the single-file /upload-rom + POST /api/titles/
    roundtrip, but inlined so we don't pay the HTTP roundtrip per
    file in a 1000-file folder drop."""
    spec = verdict.system
    assert spec is not None  # classifier guarantees system on kind='rom'

    blob_meta = await blobs.write(data, mime_type="application/octet-stream")

    from pathlib import Path
    display_name = Path(filename).stem.replace("_", " ").strip() or spec.label

    manifest_data = {
        "rom_sha256": blob_meta["sha256"],
        "rom_size_bytes": int(blob_meta["size_bytes"]),
        "system_id": spec.id,
        "title": display_name,
        "original_filename": filename,
    }

    svc = _service(request)
    try:
        manifest, created = await svc.import_title(
            user_id=user_id,
            source_id="internal-rom",
            manifest_data=manifest_data,
        )
    except TitleServiceError as exc:
        summary["errors"].append({
            "filename": filename, "error": str(exc),
        })
        # The blob refcount was bumped by .write(); release it since
        # the import didn't create a row.
        await blobs.release(blob_meta["sha256"])
        return

    if created:
        summary["imported"].append({
            "filename": filename,
            "system": spec.id,
            "system_label": spec.label,
            "title_id": manifest.id,
        })
    else:
        summary["duplicates"].append({
            "filename": filename, "kind": "rom",
            "existing_id": manifest.id,
            "system": spec.id,
        })
        # Duplicate import means the existing title's blob already
        # owns the bytes; release the extra refcount our .write()
        # bumped.
        await blobs.release(blob_meta["sha256"])


async def _install_bios(
    *,
    request: Request,
    filename: str,
    data: bytes,
    sha1: str,
    verdict: Classification,
    summary: dict[str, list[dict[str, Any]]],
    bios_store: Any,
    user_id: str,
    md5: str = "",
    crc32: str = "",
) -> None:
    bios = verdict.bios_file
    assert bios is not None

    # Trust the classifier's own account of how it matched. Re-deriving
    # it here as "sha1 if the hashes are equal else name_size" was
    # wrong for every entry the catalog stores without a hash: those
    # have ``bios.sha1 is None``, which never equals the computed
    # digest, so a genuine MD5/CRC32 identification was recorded as the
    # weaker 'name_size'. verify_status is derived from this, so the
    # error was about to become user-visible.
    matched_by = verdict.matched_by or (
        "sha1" if bios.sha1 and bios.sha1 == sha1 else "name_size"
    )
    try:
        record = await bios_store.install(
            user_id=user_id,
            system_id=bios.system_id,
            canonical_filename=bios.filename,
            data=data,
            original_filename=filename.rsplit("!", 1)[-1],
            sha1=sha1, md5=md5, crc32=crc32,
            matched_by=matched_by,
        )
    except BiosServiceError as exc:
        summary["errors"].append({
            "filename": filename, "error": str(exc),
        })
        return
    summary["bios"].append({
        "filename": filename,
        "system": bios.system_id,
        "canonical_filename": record.canonical_filename,
        "matched_by": matched_by,
        "verify_status": record.verify_status,
        "description": bios.description,
    })


async def _process_zip_archive(
    *,
    request: Request,
    filename: str,
    data: bytes,
    summary: dict[str, list[dict[str, Any]]],
    rom_cap: int,
    blobs: Any,
    bios_store: Any,
    user_id: str,
    bios_system_hint: str = "",
) -> None:
    """Extract a .zip in memory and re-classify each member.

    Arcade exception: if the zip's members all look like arcade ROM
    parts (no BIOS hits, no recognisable extensions, multiple .bin/
    .rom files at the top level), we treat the zip itself as an
    arcade ROM (FBNeo expects the zip intact). Detection heuristic:
    >= 3 files inside, no member matches BIOS catalog, no member
    extension matches a non-arcade ROM system. Falls through to
    per-member extraction otherwise.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        from augmentum.utils.safe_archive import UnsafeArchiveError, ensure_zip_sane
        try:
            ensure_zip_sane(zf, source=f"titles_import:{filename}")
        except UnsafeArchiveError as exc:
            summary["errors"].append({"filename": filename, "error": str(exc)})
            return
        members = [m for m in zf.infolist() if not m.is_dir()]
    except zipfile.BadZipFile:
        summary["errors"].append({
            "filename": filename, "error": "corrupt zip file",
        })
        return

    if not members:
        summary["junk"].append({
            "filename": filename, "reason": "empty zip",
        })
        return

    # Arcade-zip heuristic: walk members WITHOUT extracting bytes
    # (cheap), see if the shape matches an arcade romset.
    if _looks_like_arcade_zip(members):
        # Hand the whole zip to InternalRomSource as an .arcade ROM.
        from augmentum.titles.rom_systems import get_system
        spec = get_system("arcade")
        if spec is not None:
            verdict = Classification(
                kind="rom", confidence="high",
                system=spec, reason="zip detected as arcade romset",
            )
            await _install_rom(
                request=request, filename=filename, data=data,
                verdict=verdict, summary=summary,
                blobs=blobs, user_id=user_id,
            )
            return

    # Regular per-member extraction.
    for member in members:
        try:
            member_data = zf.read(member)
        except Exception as exc:                # noqa: BLE001
            summary["errors"].append({
                "filename": f"{filename}!{member.filename}",
                "error": f"zip read failed: {exc}",
            })
            continue
        await _classify_and_route(
            request=request,
            filename=f"{filename}!{member.filename}",
            data=member_data,
            summary=summary, rom_cap=rom_cap,
            blobs=blobs, bios_store=bios_store, user_id=user_id,
            bios_system_hint=bios_system_hint,
        )


def _looks_like_arcade_zip(members: list) -> bool:
    """Heuristic: arcade ROMs are .zip with multiple .bin/.rom parts
    inside, no top-level files we'd recognise as ROMs/BIOS for other
    systems. Conservative: returns True only when the shape is
    unambiguous, so worst case we extract a real arcade zip and
    surface its members as 'unknown' (which the user can then
    re-import as opaque arcade.zip)."""
    if len(members) < 3:
        return False
    arcade_part_exts = (".bin", ".rom", ".u1", ".u2", ".u3", ".u4")
    arcade_parts = 0
    for m in members:
        name = m.filename.lower()
        # Skip directory traversal / nested paths (we only look at
        # top-level structure).
        if "/" in name.strip("/"):
            return False
        if name.endswith(arcade_part_exts):
            arcade_parts += 1
        else:
            # Any non-arcade-looking file kills the heuristic. We'd
            # rather extract and miss-classify (recoverable) than
            # collapse a legitimate BIOS pack into a single arcade
            # title (not recoverable without re-import).
            return False
    return arcade_parts >= 3


def _override_candidates(filename: str) -> list[dict[str, str]]:
    """Suggest 'mark as system X' candidates for an unknown file.
    Used by the UI's override picker so the user can rescue a file
    we couldn't classify automatically (renamed dump, weird region,
    obscure extension)."""
    from augmentum.titles.rom_systems import list_systems
    out: list[dict[str, str]] = []
    for spec in list_systems():
        out.append({
            "system_id": spec.id,
            "label": f"ROM: {spec.label}",
        })
    out.append({"system_id": "_bios", "label": "Mark as BIOS (pick system)"})
    out.append({"system_id": "_skip",  "label": "Skip this file"})
    return out


# ── helpers ──────────────────────────────────────────────────────────


def _as_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None
