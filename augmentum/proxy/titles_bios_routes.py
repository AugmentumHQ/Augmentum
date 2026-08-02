"""BIOS routes -- per-user BIOS file management surface.

Mounted at ``/api/titles/bios/*``. Three responsibilities:

  * ``GET  /api/titles/bios/status``       -- per-system checklist
  * ``GET  /api/titles/bios/{sys}/{name}`` -- serve BIOS bytes (auth-
                                              gated, used by the
                                              EmulatorBrowserRuntime
                                              to populate EJS_biosUrl)
  * ``DELETE /api/titles/bios/{sys}/{name}`` -- remove an installed
                                                 BIOS slot

Install lives in ``titles_routes.py::bulk_import`` (the drop-anything
flow) since BIOS install is one of several outcomes the classifier
routes to. Manual single-file BIOS install is rare enough that the
bulk endpoint covers it -- a UI "Install BIOS" button just submits
one file to ``/bulk-import`` and shows the same summary.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile as _StarletteUploadFile

from augmentum.config import settings
from augmentum.titles import bios_hashdb
from augmentum.titles.bios_catalog import all_for_system, systems_with_bios
from augmentum.titles.rom_systems import get_system
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/titles/bios", tags=["titles-bios"])

# Generous ceiling for a single manually-installed BIOS/firmware blob.
# Most are <4 MB; the catalog's heavyweight entries (DSi NAND ~240 MB,
# PS3 firmware PUP) are the reason this isn't tiny. A hostile upload
# still can't OOM the box: the read is bounded by this cap.
_BIOS_MAX_BYTES = 300 * 1024 * 1024


# ── helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _bios_store(request: Request):
    return getattr(request.app.state, "bios_store", None)


def _core_status_for(system_id: str) -> str:
    """Honest playability state for a BIOS-catalog system.

    ``bundled``            -- a libretro core ships in our EmulatorJS
                              build; playable in-browser right now.
    ``streaming_required`` -- needs the (not-yet-wired) AGSP streamed
                              emulator runtime; playable later.
    ``experimental``       -- core exists upstream but isn't bundled.
    ``unsupported``        -- the system is in the BIOS catalog only
                              (so we can recognise its dumps) but there
                              is no ROM-system entry / runtime for it.
    """
    spec = get_system(system_id)
    if spec is None:
        return "unsupported"
    return getattr(spec, "core_status", "bundled") or "bundled"


def _gate(request: Request) -> JSONResponse | None:
    if not getattr(settings, "titles_enabled", False):
        return JSONResponse(
            {"error": "Titles framework is disabled"}, status_code=503,
        )
    if _bios_store(request) is None:
        return JSONResponse(
            {"error": "BIOS service unavailable"}, status_code=503,
        )
    return None


def _is_safe_slot_name(name: str) -> bool:
    """Whether ``name`` may be used as a stored BIOS filename.

    Store-first means the user picks the name, so this is the boundary
    that keeps a chosen name from escaping the store. The name is
    echoed back as a path segment by the serve route, so anything that
    could traverse or split a path is refused. Everything else --
    spaces, unicode, odd extensions, unfamiliar names -- is allowed,
    because real BIOS sets are full of all four.
    """
    if not name or len(name) > 255:
        return False
    if name in (".", ".."):
        return False
    if any(c in name for c in ("/", "\\", "\x00")):
        return False
    # Control characters would corrupt headers on the serve path.
    return all(ord(c) >= 0x20 for c in name)


# ── Status ───────────────────────────────────────────────────────────


@router.get("/status")
async def bios_status(
    request: Request, system_id: str | None = None,
) -> JSONResponse:
    """Per-system BIOS checklist.

    Without a query: returns every system that has at least one
    catalogued BIOS file, with present/missing counts. The UI
    renders this as a checklist grouped by system.

    With ``?system_id=psx``: returns the full per-file status for
    that system (each catalogued file + present flag + matched-by
    mode).

    The response is the source of truth for the "BIOS panel" UI --
    the launch path consults the bios_store directly, not this
    endpoint.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = _bios_store(request)

    if system_id:
        if get_system(system_id) is None and system_id not in systems_with_bios():
            # System not in either catalog -- 404 the checklist
            # rather than silently returning an empty list.
            return JSONResponse(
                {"error": f"unknown system {system_id!r}"}, status_code=404,
            )
        entries = await store.list_status(user_id=uid, system_id=system_id)
        core_status = _core_status_for(system_id)
        return JSONResponse({
            "system_id": system_id,
            "entries": [e.to_dict() for e in entries],
            "all_required_present": not await store.missing_required(
                user_id=uid, system_id=system_id,
            ),
            "core_status": core_status,
            "playable": core_status == "bundled",
        })

    # No system filter: per-system summary roll-up.
    out: list[dict] = []
    for sid in systems_with_bios():
        entries = await store.list_status(user_id=uid, system_id=sid)
        required_total = sum(1 for e in entries if not e.optional)
        required_present = sum(
            1 for e in entries if not e.optional and e.present
        )
        optional_total = sum(1 for e in entries if e.optional)
        optional_present = sum(
            1 for e in entries if e.optional and e.present
        )
        spec = get_system(sid)
        core_status = _core_status_for(sid)
        out.append({
            "system_id": sid,
            "system_label": spec.label if spec else sid,
            "required_total": required_total,
            "required_present": required_present,
            "optional_total": optional_total,
            "optional_present": optional_present,
            # Totals across everything actually stored, catalogued or
            # not. The vault shows "3 stored · 2 verified" so a user
            # who installed an unrecognised dump can see it landed.
            "stored_total": sum(1 for e in entries if e.present),
            "verified_total": sum(
                1 for e in entries if e.present and e.verify_status == "verified"
            ),
            "extra_total": sum(1 for e in entries if e.is_extra),
            "ready": required_present == required_total,
            "core_status": core_status,
            "playable": core_status == "bundled",
        })
    return JSONResponse({"systems": out})


# ── Catalog introspection (for the UI to render the checklist) ──────
#
# IMPORTANT: this route MUST be registered before the parameterized
# /{system_id}/{filename} routes below. FastAPI resolves routes in
# declaration order, and `/_/catalog` matches the parameterized
# pattern (system_id="_", filename="catalog") if that route comes
# first — every catalog request would 404 as "BIOS not installed".


@router.get("/_/catalog")
async def get_catalog(request: Request) -> JSONResponse:
    """The full BIOS catalog (system / filename / size / optional /
    description) without any user state. The UI fetches this once at
    panel-open time and merges it with /status for "what's expected
    vs what's installed" rendering."""
    if (gate := _gate(request)) is not None:
        return gate
    out: list[dict] = []
    for sid in systems_with_bios():
        spec = get_system(sid)
        core_status = _core_status_for(sid)
        out.append({
            "system_id": sid,
            "system_label": spec.label if spec else sid,
            "core_status": core_status,
            "playable": core_status == "bundled",
            "files": [
                {
                    "canonical_filename": f.filename,
                    "size_bytes": f.size_bytes,
                    "optional": f.optional,
                    "description": f.description,
                    "has_known_hash": bool(f.sha1),
                }
                for f in all_for_system(sid)
            ],
            # Reference list from the libretro hash database: every
            # dump known to exist for this system, including revisions
            # we don't carry a slot for. Mirrors EmuDeck's per-system
            # cheat-sheet -- it tells the user what a valid file looks
            # like instead of leaving them guessing after a rejection.
            "known_files": [
                {
                    "filename": k.basename,
                    "size_bytes": k.size_bytes,
                    "md5": k.md5,
                    "sha1": k.sha1,
                }
                for k in bios_hashdb.known_for_system(sid)
            ],
        })
    return JSONResponse({
        "systems": out,
        "hash_db": bios_hashdb.stats(),
    })


# ── Serve BIOS bytes ─────────────────────────────────────────────────


@router.api_route(
    "/{system_id}/{filename}",
    methods=["GET", "HEAD"],
    operation_id="serve_bios",
)
async def serve_bios(
    request: Request, system_id: str, filename: str,
):
    """Return BIOS bytes for the runtime to fetch.

    Auth-gated: only the user that installed the BIOS can read it.
    Streamed from disk via FileResponse, so a 240 MB DSi NAND doesn't
    materialise in RAM and Range requests get a 206 for free. HEAD is
    supported for symmetry with the ROM endpoint (EmulatorJS HEAD-probes
    for content-length).

    The launch flow generates a URL of this shape and stuffs it into
    ``EJS_biosUrl``. The browser fetches over the same auth cookie the
    rest of the app uses.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = _bios_store(request)
    result = await store.get_blob_path(
        user_id=uid, system_id=system_id, canonical_filename=filename,
    )
    if result is None:
        return JSONResponse(
            {"error": "BIOS not installed"}, status_code=404,
        )
    record, real_path = result

    # BIOS bytes are private (per-user owned) -- never cross-user cached.
    headers = {
        "X-Bios-SHA1": record.sha1,
        "Cache-Control": "private, max-age=3600",
    }

    # HEAD: emit the headers without touching the file body.
    if request.method == "HEAD":
        return Response(
            status_code=200,
            headers={
                **headers,
                "Content-Length": str(record.size_bytes),
                "Content-Type": "application/octet-stream",
            },
        )

    return FileResponse(
        real_path,
        media_type="application/octet-stream",
        headers=headers,
    )


# ── Manual install (override the classifier) ─────────────────────────


@router.post("/{system_id}/{filename}")
async def install_bios(
    request: Request,
    system_id: str,
    filename: str,
) -> JSONResponse:
    """Force-install an uploaded file as a specific BIOS slot.

    The bulk-import classifier recognises BIOS dumps by SHA1, then by
    (canonical name, exact size), then by manufacturer-naming pattern.
    A dump that matches none of those (odd filename, no catalogued
    hash, e.g. a regionally-obscure PS2 BIOS named ``ps2_bios.bin``)
    comes back as ``unknown`` and there's otherwise no way to install
    it. This route is the escape hatch -- the same role the ROM
    system-badge picker plays for mis-detected ROMs.

    ``system_id`` must be a system we know about. ``filename`` (the URL
    path segment) does NOT have to be a catalogued slot: under
    store-first the user may install a firmware dump we have no entry
    for, exactly as they could drop one into RetroArch's System folder
    or EmuDeck's ``emulation/bios``. Requiring a pre-existing slot was
    the second door on the same bug the vault's import path had -- it
    refused precisely the obscure regional dumps this escape hatch
    exists to rescue.

    What we still refuse is a filename that isn't a filename: path
    separators, traversal, and control characters, since the name
    becomes a URL segment on the serve route.

    The bytes are stored as-is and recorded with ``matched_by='manual'``
    so the status panel can flag "you installed this by hand".
    Idempotent: re-installing the same slot replaces it.

    Body: ``multipart/form-data`` with a single ``file`` field. We walk
    ``request.form()`` ourselves (rather than the ``File(...)`` DI) so
    the per-part size cap matches ``_BIOS_MAX_BYTES`` -- Starlette's
    default 1 MB cap would silently reject a 4 MB PS2 BIOS, exactly the
    case this route exists for.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if system_id not in systems_with_bios() and get_system(system_id) is None:
        return JSONResponse(
            {"error": f"unknown system {system_id!r}"},
            status_code=404,
        )
    if not _is_safe_slot_name(filename):
        return JSONResponse(
            {
                "error": (
                    f"{filename!r} is not a usable BIOS filename "
                    "(no path separators, traversal, or control characters)"
                ),
            },
            status_code=400,
        )

    # Not a catalogued slot is fine -- it just means we can't claim to
    # have verified it. install() derives verify_status from
    # matched_by, and 'manual' maps to 'named', so the vault shows it
    # as user-installed rather than hash-verified.
    known_slot = filename in {f.filename for f in all_for_system(system_id)}

    form = await request.form(
        max_files=2, max_fields=8, max_part_size=_BIOS_MAX_BYTES,
    )
    upload = None
    for key in ("file", "files"):
        for item in form.getlist(key):
            if isinstance(item, _StarletteUploadFile):
                upload = item
                break
        if upload is not None:
            break
    if upload is None:
        return JSONResponse(
            {"error": "expected a multipart 'file' field"}, status_code=400,
        )
    data = await upload.read(_BIOS_MAX_BYTES + 1)
    if len(data) > _BIOS_MAX_BYTES:
        return JSONResponse(
            {"error": f"file exceeds size cap ({_BIOS_MAX_BYTES} bytes)"},
            status_code=413,
        )
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)

    store = _bios_store(request)
    try:
        record = await store.install(
            user_id=uid,
            system_id=system_id,
            canonical_filename=filename,
            data=data,
            original_filename=upload.filename or "",
            matched_by="manual",
        )
    except Exception as exc:  # noqa: BLE001 -- surface as 400, don't 500
        log.warning(
            "bios_manual_install_failed",
            user_id=uid, system=system_id, slot=filename, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=400)

    log.info(
        "bios_manual_installed",
        user_id=uid, system=system_id, slot=filename,
        size_bytes=len(data), sha1=record.sha1,
        known_slot=known_slot, verify_status=record.verify_status,
    )
    return JSONResponse(
        {"ok": True, "bios": record.to_dict()}, status_code=201,
    )


# ── Delete ───────────────────────────────────────────────────────────


@router.delete("/{system_id}/{filename}")
async def delete_bios(
    request: Request, system_id: str, filename: str,
) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = _bios_store(request)
    deleted = await store.delete(
        user_id=uid, system_id=system_id, canonical_filename=filename,
    )
    if not deleted:
        return JSONResponse(
            {"error": "BIOS not installed"}, status_code=404,
        )
    return JSONResponse({"ok": True})


