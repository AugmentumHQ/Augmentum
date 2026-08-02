"""Surface session API.

Surface sessions are Augmentum's device-agnostic control plane: a phone,
TV browser, Cast target, game stream, XR room, or native Augmentum panel
can all participate in one durable session and follow the same state.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.surfaces.recipes import get_recipe, list_recipes
from augmentum.surfaces.store import SurfaceConflictError
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/surfaces", tags=["surfaces"])
public_router = APIRouter(prefix="/api/surface-public", tags=["surface-public"])


class _SurfaceTokenUser:
    __slots__ = ("id",)

    def __init__(self, user_id: str) -> None:
        self.id = user_id


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _store(request: Request):
    return getattr(request.app.state, "surface_store", None)


def _store_or_503(request: Request):
    store = _store(request)
    if store is None:
        raise HTTPException(503, "surface store not initialized")
    return store


def _runtime(request: Request):
    return getattr(request.app.state, "surface_runtime", None)


def _token_store(request: Request):
    return getattr(request.app.state, "surface_access_token_store", None)


def _token_store_or_503(request: Request):
    store = _token_store(request)
    if store is None:
        raise HTTPException(503, "surface token store not initialized")
    return store


def _client_ip(request: Request) -> str:
    from augmentum.config import settings as _settings

    if getattr(_settings, "auth_trust_forwarded_for", False):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


def _browser_scheme(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    if forwarded in {"http", "https"}:
        return forwarded
    return request.url.scheme or "https"


def _public_url(request: Request, path: str) -> str:
    resolver = getattr(request.app.state, "public_host_resolver", None)
    if resolver is not None:
        try:
            resolved = resolver.public_url(path, request=request, scheme=_browser_scheme(request))
            if resolved:
                return resolved
        except Exception as exc:
            log.debug("surface_public_host_resolve_failed", error=str(exc))
    scheme = _browser_scheme(request)
    host = request.headers.get("host", "") or request.url.netloc
    path_part = path if path.startswith("/") else f"/{path}"
    return f"{scheme}://{host}{path_part}"


def _public_session(session: dict[str, Any] | None) -> dict[str, Any] | None:
    if session is None:
        return None
    result = dict(session)
    result.pop("user_id", None)
    result.pop("pairing_code", None)
    return result


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    result = dict(event)
    result.pop("user_id", None)
    return result


def _merge_query_params(request: Request, params: dict[str, str]) -> None:
    if not params:
        return
    existing = dict(parse_qsl(
        request.scope.get("query_string", b"").decode("latin-1", errors="ignore"),
        keep_blank_values=True,
    ))
    for key, value in params.items():
        existing.setdefault(key, value)
    request.scope["query_string"] = urlencode(existing).encode("latin-1")


def _lookup_public_token(request: Request, token: str, *, required_scope: str):
    store = _token_store_or_503(request)
    entry = store.lookup(
        token,
        client_ip=_client_ip(request),
        required_scope=required_scope,
    )
    if entry is None:
        raise HTTPException(404, "surface token expired or invalid")
    return entry


async def _notify(request: Request, *, user_id: str, session_id: str) -> None:
    runtime = _runtime(request)
    if runtime is not None:
        await runtime.notify(user_id=user_id, session_id=session_id)


async def _wait_for_events(
    request: Request,
    *,
    user_id: str,
    session_id: str,
    timeout_ms: int,
) -> None:
    runtime = _runtime(request)
    if runtime is None or timeout_ms <= 0:
        return
    await runtime.wait(
        user_id=user_id,
        session_id=session_id,
        timeout_s=min(30.0, max(0.0, float(timeout_ms) / 1000.0)),
    )


# --- Request shapes --------------------------------------------------------


class CreateSurfaceSessionRequest(BaseModel):
    kind: str = "surface.generic"
    title: str = ""
    content_ref: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    participants: list[dict[str, Any]] = Field(default_factory=list)


class JoinSurfaceRequest(BaseModel):
    participant_id: str = ""
    device_id: str = ""
    role: str = "observer"
    label: str = ""
    capabilities: list[str] = Field(default_factory=list)
    transport: str = "browser"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HeartbeatSurfaceRequest(BaseModel):
    participant_id: str


class PatchSurfaceStateRequest(BaseModel):
    patch: dict[str, Any] = Field(default_factory=dict)
    base_revision: int | None = None
    source_participant_id: str = ""
    replace: bool = False


class SurfaceAccessTokenRequest(BaseModel):
    file_id: str = ""
    scopes: list[str] = Field(default_factory=list)
    ttl_s: int = 6 * 60 * 60
    lock_to_client: bool = False
    query_params: dict[str, str] = Field(default_factory=dict)


class SurfaceHandoffRequest(SurfaceAccessTokenRequest):
    target_role: str = "display"
    target_label: str = ""
    target_ip: str = ""
    target_capabilities: list[str] = Field(default_factory=list)
    bluetooth_mtu: int = 185


def _default_scopes(session: dict[str, Any], requested: list[str], *, file_id: str) -> list[str]:
    if requested:
        return requested
    scopes = {"session:read", "session:join"}
    kind = str(session.get("kind") or "")
    content_ref = session.get("content_ref") if isinstance(session.get("content_ref"), dict) else {}
    content_kind = str(content_ref.get("kind") or content_ref.get("media_kind") or "").lower()
    if kind.startswith("comic.") or content_kind == "comic":
        scopes.add("comic:read")
    if kind == "media.watch" or content_kind in {"video", "audio", "movie", "episode"}:
        scopes.add("media:stream")
    if file_id and not (kind.startswith("comic.") or content_kind == "comic"):
        scopes.add("media:stream")
    return sorted(scopes)


def _surface_access_payload(request: Request, token_entry) -> dict[str, Any]:
    token_q = quote(token_entry.token)
    public_base_path = f"/api/surface-public/{token_q}"
    public_base_url = _public_url(request, public_base_path)
    receiver_url = _public_url(request, f"/ui/surface-receiver.html?token={token_q}")
    access: dict[str, Any] = {
        "token": token_entry.token,
        "expires_at": token_entry.expires_at,
        "scopes": list(token_entry.scopes),
        "receiver_url": receiver_url,
        "session_url": f"{public_base_url}/session",
        "events_url": f"{public_base_url}/events",
        "join_url": f"{public_base_url}/join",
    }
    if token_entry.allows("comic:read"):
        access["comic_manifest_url"] = f"{public_base_url}/comic/manifest"
        access["comic_page_url_template"] = f"{public_base_url}/comic/page?page={{page}}"
    if token_entry.allows("media:stream"):
        access["media_stream_url"] = f"{public_base_url}/media/stream"
    return access


# --- Authenticated routes --------------------------------------------------


@router.get("/recipes")
async def surface_recipes() -> dict[str, Any]:
    return {"recipes": list_recipes()}


@router.get("/recipes/{kind}")
async def surface_recipe(kind: str) -> dict[str, Any]:
    recipe = get_recipe(kind)
    if recipe is None:
        raise HTTPException(404, "surface recipe not found")
    return {"recipe": recipe}


@router.get("/sessions")
async def list_surface_sessions(
    request: Request,
    include_ended: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    uid = _user_id(request)
    sessions = await _store_or_503(request).list_for_user(
        user_id=uid,
        include_ended=include_ended,
        limit=limit,
    )
    return {"sessions": sessions}


@router.post("/sessions")
async def create_surface_session(
    body: CreateSurfaceSessionRequest,
    request: Request,
) -> dict[str, Any]:
    uid = _user_id(request)
    session = await _store_or_503(request).create(
        user_id=uid,
        kind=body.kind,
        title=body.title,
        content_ref=body.content_ref,
        state=body.state,
        participants=body.participants,
    )
    await _notify(request, user_id=uid, session_id=session["id"])
    return {"session": session, "recipe": get_recipe(session["kind"])}


@router.get("/sessions/{session_id}")
async def get_surface_session(session_id: str, request: Request) -> dict[str, Any]:
    uid = _user_id(request)
    session = await _store_or_503(request).get(session_id, user_id=uid)
    if session is None:
        raise HTTPException(404, "surface session not found")
    return {"session": session, "recipe": get_recipe(session["kind"])}


@router.post("/sessions/{session_id}/join")
async def join_surface_session(
    session_id: str,
    body: JoinSurfaceRequest,
    request: Request,
) -> dict[str, Any]:
    uid = _user_id(request)
    session = await _store_or_503(request).join(
        session_id,
        user_id=uid,
        participant=body.model_dump(),
    )
    if session is None:
        raise HTTPException(404, "surface session not found")
    await _notify(request, user_id=uid, session_id=session_id)
    return {"session": session}


@router.post("/sessions/{session_id}/heartbeat")
async def heartbeat_surface_session(
    session_id: str,
    body: HeartbeatSurfaceRequest,
    request: Request,
) -> dict[str, Any]:
    uid = _user_id(request)
    session = await _store_or_503(request).heartbeat(
        session_id,
        user_id=uid,
        participant_id=body.participant_id,
    )
    if session is None:
        raise HTTPException(404, "surface session not found")
    return {"session": session}


@router.post("/sessions/{session_id}/state")
async def patch_surface_state(
    session_id: str,
    body: PatchSurfaceStateRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    try:
        session = await _store_or_503(request).patch_state(
            session_id,
            user_id=uid,
            patch=body.patch,
            source_participant_id=body.source_participant_id,
            base_revision=body.base_revision,
            replace=body.replace,
        )
    except SurfaceConflictError as exc:
        current = await _store_or_503(request).get(session_id, user_id=uid)
        return JSONResponse(
            {
                "error": "revision_conflict",
                "expected": exc.expected,
                "actual": exc.actual,
                "session": current,
            },
            status_code=409,
        )
    if session is None:
        raise HTTPException(404, "surface session not found")
    await _notify(request, user_id=uid, session_id=session_id)
    return JSONResponse({"session": session})


@router.get("/sessions/{session_id}/events")
async def surface_events(
    session_id: str,
    request: Request,
    after_revision: int = -1,
    timeout_ms: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    uid = _user_id(request)
    store = _store_or_503(request)
    session = await store.get(session_id, user_id=uid)
    if session is None:
        raise HTTPException(404, "surface session not found")

    events = await store.events_after(
        session_id,
        user_id=uid,
        after_revision=after_revision,
        limit=limit,
    )
    if not events and timeout_ms > 0:
        await _wait_for_events(
            request,
            user_id=uid,
            session_id=session_id,
            timeout_ms=timeout_ms,
        )
        events = await store.events_after(
            session_id,
            user_id=uid,
            after_revision=after_revision,
            limit=limit,
        )
        session = await store.get(session_id, user_id=uid)
    return {"events": events, "session": session}


@router.post("/sessions/{session_id}/access-token")
async def issue_surface_access_token(
    session_id: str,
    body: SurfaceAccessTokenRequest,
    request: Request,
) -> dict[str, Any]:
    uid = _user_id(request)
    store = _store_or_503(request)
    session = await store.get(session_id, user_id=uid)
    if session is None:
        raise HTTPException(404, "surface session not found")

    content_ref = session.get("content_ref") if isinstance(session.get("content_ref"), dict) else {}
    file_id = str(body.file_id or content_ref.get("file_id") or content_ref.get("id") or "").strip()
    scopes = _default_scopes(session, body.scopes, file_id=file_id)
    token_entry = _token_store_or_503(request).issue(
        user_id=uid,
        session_id=session_id,
        file_id=file_id,
        scopes=scopes,
        allowed_client_ip=_client_ip(request) if body.lock_to_client else "",
        query_params=body.query_params,
        ttl_s=body.ttl_s,
    )

    return {"access": _surface_access_payload(request, token_entry), "session": session}


@router.post("/sessions/{session_id}/handoff")
async def issue_surface_handoff(
    session_id: str,
    body: SurfaceHandoffRequest,
    request: Request,
) -> dict[str, Any]:
    """Mint a Bluetooth-to-IP handoff bundle.

    The phone sends ``ble_payload_json`` over BLE. The TV receiver does
    not keep using Bluetooth; it switches to the included HTTPS URLs and
    joins the surface through the normal public receiver token routes.
    """
    uid = _user_id(request)
    store = _store_or_503(request)
    session = await store.get(session_id, user_id=uid)
    if session is None:
        raise HTTPException(404, "surface session not found")

    content_ref = session.get("content_ref") if isinstance(session.get("content_ref"), dict) else {}
    file_id = str(body.file_id or content_ref.get("file_id") or content_ref.get("id") or "").strip()
    scopes = _default_scopes(session, body.scopes, file_id=file_id)
    token_entry = _token_store_or_503(request).issue(
        user_id=uid,
        session_id=session_id,
        file_id=file_id,
        scopes=scopes,
        allowed_client_ip=str(body.target_ip or "").strip(),
        query_params=body.query_params,
        ttl_s=body.ttl_s,
        extra={"handoff": "bluetooth_to_ip"},
    )
    access = _surface_access_payload(request, token_entry)
    role = str(body.target_role or "display").strip() or "display"
    capabilities = list(body.target_capabilities or [])
    if not capabilities and role == "display":
        capabilities = ["surface.follow_state@1"]
        if token_entry.allows("comic:read"):
            capabilities.append("display.comic_read@1")

    ble_payload = {
        "v": 1,
        "type": "augmentum.surface.handoff",
        "mode": "bluetooth_to_ip",
        "session_id": session_id,
        "kind": session.get("kind", ""),
        "title": session.get("title", ""),
        "token": token_entry.token,
        "expires_at": token_entry.expires_at,
        "receiver_url": access["receiver_url"],
        "session_url": access["session_url"],
        "events_url": access["events_url"],
        "join_url": access["join_url"],
        "join": {
            "role": role,
            "label": str(body.target_label or "").strip(),
            "capabilities": capabilities,
            "transport": "bluetooth_handoff_https",
        },
    }
    ble_payload_json = json.dumps(ble_payload, separators=(",", ":"))
    mtu = max(64, min(512, int(body.bluetooth_mtu or 185)))
    return {
        "handoff": {
            "version": "augmentum.surface.handoff@1",
            "transport": "bluetooth_to_ip",
            "handoff_id": f"surfhand_{token_entry.token[:12]}",
            "ble_payload": ble_payload,
            "ble_payload_json": ble_payload_json,
            "estimated_bytes": len(ble_payload_json.encode("utf-8")),
            "bluetooth": {
                "service_uuid": "9b7e0001-4d8f-4f42-9a7a-6f675f000001",
                "payload_characteristic_uuid": "9b7e0002-4d8f-4f42-9a7a-6f675f000001",
                "mtu": mtu,
                "write_format": "utf8-json-concat",
                "chunking": "json-fragments" if len(ble_payload_json.encode("utf-8")) > mtu else "single-write",
            },
            "ip": access,
        },
        "session": session,
    }


@router.delete("/sessions/{session_id}")
async def end_surface_session(
    session_id: str,
    request: Request,
    source_participant_id: str = "",
) -> dict[str, Any]:
    uid = _user_id(request)
    session = await _store_or_503(request).end(
        session_id,
        user_id=uid,
        source_participant_id=source_participant_id,
    )
    if session is None:
        raise HTTPException(404, "surface session not found")
    token_store = _token_store(request)
    revoked = token_store.revoke_session(session_id) if token_store is not None else 0
    await _notify(request, user_id=uid, session_id=session_id)
    return {"session": session, "revoked_tokens": revoked}


# --- Public receiver routes ------------------------------------------------


@public_router.get("/{token}/session")
async def public_surface_session(token: str, request: Request) -> dict[str, Any]:
    entry = _lookup_public_token(request, token, required_scope="session:read")
    session = await _store_or_503(request).get(entry.session_id, user_id=entry.user_id)
    if session is None:
        raise HTTPException(404, "surface session not found")
    return {"session": _public_session(session)}


@public_router.post("/{token}/join")
async def public_join_surface_session(
    token: str,
    body: JoinSurfaceRequest,
    request: Request,
) -> dict[str, Any]:
    entry = _lookup_public_token(request, token, required_scope="session:join")
    session = await _store_or_503(request).join(
        entry.session_id,
        user_id=entry.user_id,
        participant=body.model_dump(),
    )
    if session is None:
        raise HTTPException(404, "surface session not found")
    await _notify(request, user_id=entry.user_id, session_id=entry.session_id)
    return {"session": _public_session(session)}


@public_router.get("/{token}/events")
async def public_surface_events(
    token: str,
    request: Request,
    after_revision: int = -1,
    timeout_ms: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    entry = _lookup_public_token(request, token, required_scope="session:read")
    store = _store_or_503(request)
    session = await store.get(entry.session_id, user_id=entry.user_id)
    if session is None:
        raise HTTPException(404, "surface session not found")
    events = await store.events_after(
        entry.session_id,
        user_id=entry.user_id,
        after_revision=after_revision,
        limit=limit,
    )
    if not events and timeout_ms > 0:
        await _wait_for_events(
            request,
            user_id=entry.user_id,
            session_id=entry.session_id,
            timeout_ms=timeout_ms,
        )
        events = await store.events_after(
            entry.session_id,
            user_id=entry.user_id,
            after_revision=after_revision,
            limit=limit,
        )
        session = await store.get(entry.session_id, user_id=entry.user_id)
    return {
        "events": [_public_event(e) for e in events],
        "session": _public_session(session),
    }


@public_router.get("/{token}/comic/manifest")
async def public_comic_manifest(token: str, request: Request):
    entry = _lookup_public_token(request, token, required_scope="comic:read")
    if not entry.file_id:
        raise HTTPException(404, "surface token has no file")
    request.scope["user"] = _SurfaceTokenUser(entry.user_id)
    _merge_query_params(request, entry.query_params)
    from augmentum.proxy.media_routes import comic_manifest

    return await comic_manifest(entry.file_id, request)


@public_router.get("/{token}/comic/page")
async def public_comic_page(token: str, request: Request):
    entry = _lookup_public_token(request, token, required_scope="comic:read")
    if not entry.file_id:
        raise HTTPException(404, "surface token has no file")
    request.scope["user"] = _SurfaceTokenUser(entry.user_id)
    _merge_query_params(request, entry.query_params)
    from augmentum.proxy.media_routes import comic_page

    return await comic_page(entry.file_id, request)


@public_router.get("/{token}/media/stream")
async def public_media_stream(token: str, request: Request):
    entry = _lookup_public_token(request, token, required_scope="media:stream")
    if not entry.file_id:
        raise HTTPException(404, "surface token has no file")
    request.scope["user"] = _SurfaceTokenUser(entry.user_id)
    _merge_query_params(request, entry.query_params)
    from augmentum.proxy.media_routes import stream_media

    return await stream_media(entry.file_id, request)
