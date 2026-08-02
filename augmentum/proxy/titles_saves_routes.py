"""Save routes -- per-title save data CRUD.

Mounted at ``/api/titles/{title_id}/saves/*``. Engine-agnostic: the
same endpoints serve browser-WASM emulators, server-streamed
RetroArch, and any future runtime that produces savable state. The
runtime adapter on each side bridges its native save format to these
endpoints.

Auth: every route checks the title belongs to the user before
touching saves -- save isolation is by (user_id, title_id, kind, slot).

Size cap: per-slot bytes default 50 MB; bump via the
``emulator_save_max_per_slot_mb`` setting if a runtime needs more.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from augmentum.config import settings
from augmentum.saves import SAVE_KINDS, SaveServiceError, SaveTooLargeError
from augmentum.titles import TitleNotFound, TitleService
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/titles", tags=["titles-saves"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _save_store(request: Request):
    return getattr(request.app.state, "save_store", None)


def _title_service(request: Request) -> TitleService | None:
    return getattr(request.app.state, "title_service", None)


def _gate(request: Request) -> JSONResponse | None:
    """Same gate as titles_routes -- saves require titles to be on."""
    if not getattr(settings, "titles_enabled", False):
        return JSONResponse(
            {"error": "Titles framework is disabled"}, status_code=503,
        )
    if _save_store(request) is None or _title_service(request) is None:
        return JSONResponse(
            {"error": "Save service unavailable"}, status_code=503,
        )
    return None


async def _ensure_owned(request: Request, title_id: str, user_id: str) -> JSONResponse | None:
    """Confirm the title belongs to this user. 404 if not."""
    svc = _title_service(request)
    try:
        await svc.get_title(title_id, user_id=user_id)
    except TitleNotFound:
        return JSONResponse({"error": "Title not found"}, status_code=404)
    return None


def _max_per_slot_bytes() -> int:
    mb = int(getattr(settings, "emulator_save_max_per_slot_mb", 50) or 50)
    return max(1, mb) * 1024 * 1024


async def _resolve_guest_profile_id(
    request: Request, host_user_id: str, raw: str | None,
) -> tuple[str, JSONResponse | None]:
    """Validate an optional ``?guest_profile_id=gp_*`` query value.

    Returns ``(profile_id, error_response)``. Empty string is the
    host's-own-saves path and is always valid. A non-empty value
    must resolve to a guest profile under THIS host; cross-host
    access returns 403 to avoid leaking the existence of profiles
    at other hosts.

    Phase 4 hook — Phase 1-3 callers never pass guest_profile_id
    so the route stays backward-compat (host-saves path).
    """
    if not raw:
        return "", None
    gp = raw.strip()
    if not gp:
        return "", None
    guest_store = getattr(request.app.state, "guest_store", None)
    if guest_store is None:
        # Guest substrate not wired in test mode — accept ID without
        # validation. The save row still gets the field; the route's
        # ownership-by-host guarantee covers the auth case.
        return gp, None
    try:
        profile = await guest_store.get(gp, host_user_id=host_user_id)
    except Exception:
        profile = None
    if profile is None:
        return "", JSONResponse(
            {"error": "Guest profile not found at this host"},
            status_code=403,
        )
    return gp, None


# ── Routes ───────────────────────────────────────────────────────────


@router.get("/{title_id}/saves")
async def list_saves(
    request: Request,
    title_id: str,
    kind: str | None = None,
    guest_profile_id: str | None = None,
    include_guests: bool = False,
) -> JSONResponse:
    """List the user's saves for a title.

    ``kind`` (optional): 'sram' | 'state' | 'screenshot'.

    Save-ownership filters (mutually exclusive, default is host-only):
      * default: only the host's own saves (NULL guest_profile_id)
      * ``?guest_profile_id=gp_*``: only that guest's saves
      * ``?include_guests=true``: every save for the title, host's
        plus all guests — used by the host's "manage saves" view
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if (err := await _ensure_owned(request, title_id, uid)) is not None:
        return err

    if kind is not None and kind not in SAVE_KINDS:
        return JSONResponse(
            {"error": f"unknown kind: {kind!r} (known: {sorted(SAVE_KINDS)})"},
            status_code=400,
        )

    gp, err = await _resolve_guest_profile_id(
        request, uid, guest_profile_id,
    )
    if err is not None:
        return err

    store = _save_store(request)
    # Three modes for the guest_profile_id filter: None=all, ""=host,
    # "gp_*"=specific guest.
    list_gp: str | None
    if include_guests:
        list_gp = None
    elif gp:
        list_gp = gp
    else:
        list_gp = ""
    records = await store.list_for_title(
        user_id=uid, artifact_id=title_id, kind=kind,
        guest_profile_id=list_gp,
    )
    return JSONResponse({"saves": [r.to_dict() for r in records]})


@router.put("/{title_id}/saves/{kind}/{slot}")
async def put_save(
    request: Request, title_id: str, kind: str, slot: int,
    guest_profile_id: str | None = None,
) -> JSONResponse:
    """Write a save slot.

    Body shape:
        {
          "data":     "<base64-encoded bytes>",
          "core_id":  "fceumm",        # required for state kind
          "label":    "boss fight"     # optional
        }

    State saves require ``core_id``. SRAM and screenshots leave
    core_id empty.

    ``?guest_profile_id=gp_*`` (optional): write the save to a named
    guest's slot rather than the host's. Per-guest slot uniqueness is
    independent — alice's slot 0 and the host's slot 0 are different
    saves of the same title. Phase 4 substrate.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if (err := await _ensure_owned(request, title_id, uid)) is not None:
        return err

    if kind not in SAVE_KINDS:
        return JSONResponse(
            {"error": f"unknown kind: {kind!r}"}, status_code=400,
        )

    gp, err = await _resolve_guest_profile_id(
        request, uid, guest_profile_id,
    )
    if err is not None:
        return err

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "JSON body required"}, status_code=400,
        )

    raw = body.get("data")
    if not isinstance(raw, str) or not raw:
        return JSONResponse(
            {"error": "data (base64 string) is required"}, status_code=400,
        )
    try:
        import base64
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        return JSONResponse(
            {"error": f"data is not valid base64: {exc}"}, status_code=400,
        )

    core_id = str(body.get("core_id", ""))
    label = str(body.get("label", ""))[:200]

    if kind == "state" and not core_id:
        return JSONResponse(
            {"error": "core_id is required for state saves"},
            status_code=400,
        )

    store = _save_store(request)
    try:
        record = await store.put(
            user_id=uid,
            artifact_id=title_id,
            kind=kind,
            slot=int(slot),
            data=data,
            core_id=core_id,
            label=label,
            max_per_slot_bytes=_max_per_slot_bytes(),
            guest_profile_id=gp,
        )
    except SaveTooLargeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    except SaveServiceError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse({"save": record.to_dict()}, status_code=201)


@router.get("/{title_id}/saves/{kind}/{slot}")
async def get_save(
    request: Request, title_id: str, kind: str, slot: int,
    metadata_only: bool = False,
    guest_profile_id: str | None = None,
) -> Response:
    """Read a save slot.

    Default: returns the raw bytes as ``application/octet-stream``
    (or ``image/png`` for screenshots) so the client can stream it
    straight into the runtime.

    With ``?metadata_only=true``: returns just the index row as JSON
    (id, sha, size, core_id, label, timestamps).

    ``?guest_profile_id=gp_*`` (optional): read a named guest's save
    slot. Default reads the host's slot.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if (err := await _ensure_owned(request, title_id, uid)) is not None:
        return err

    if kind not in SAVE_KINDS:
        return JSONResponse(
            {"error": f"unknown kind: {kind!r}"}, status_code=400,
        )

    gp, err = await _resolve_guest_profile_id(
        request, uid, guest_profile_id,
    )
    if err is not None:
        return err

    store = _save_store(request)
    if metadata_only:
        record = await store.get(
            user_id=uid, artifact_id=title_id,
            kind=kind, slot=int(slot),
            guest_profile_id=gp,
        )
        if record is None:
            return JSONResponse({"error": "Save not found"}, status_code=404)
        return JSONResponse({"save": record.to_dict()})

    pair = await store.get_with_bytes(
        user_id=uid, artifact_id=title_id,
        kind=kind, slot=int(slot),
        guest_profile_id=gp,
    )
    if pair is None:
        return JSONResponse({"error": "Save not found"}, status_code=404)
    record, data = pair
    media_type = "image/png" if kind == "screenshot" else "application/octet-stream"
    return Response(content=data, media_type=media_type, headers={
        "X-Save-SHA256": record.sha256,
        "X-Save-Size-Bytes": str(record.size_bytes),
        "X-Save-Core-Id": record.core_id,
        "X-Save-Guest-Profile-Id": record.guest_profile_id,
    })


@router.delete("/{title_id}/saves/{kind}/{slot}")
async def delete_save(
    request: Request, title_id: str, kind: str, slot: int,
    guest_profile_id: str | None = None,
) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if (err := await _ensure_owned(request, title_id, uid)) is not None:
        return err
    if kind not in SAVE_KINDS:
        return JSONResponse(
            {"error": f"unknown kind: {kind!r}"}, status_code=400,
        )
    gp, err = await _resolve_guest_profile_id(
        request, uid, guest_profile_id,
    )
    if err is not None:
        return err
    store = _save_store(request)
    ok = await store.delete(
        user_id=uid, artifact_id=title_id,
        kind=kind, slot=int(slot),
        guest_profile_id=gp,
    )
    if not ok:
        return JSONResponse({"error": "Save not found"}, status_code=404)
    return JSONResponse({"ok": True})
