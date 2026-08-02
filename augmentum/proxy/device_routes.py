"""Device substrate HTTP surface — driver-agnostic.

One URL pattern (`/api/devices/{id}/{capability}/{action}`) for every
device kind. Adding a new driver doesn't add new routes; it lights up
existing ones for that driver's capabilities.

All routes are user-scoped. Auth secrets in saved-device payloads are
never returned to the client (encrypted at rest, redacted on the wire).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from augmentum.devices.capabilities import get_capability, list_capabilities
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/devices", tags=["devices"])

# Separate router for the public cast-blob endpoint, mounted at
# /api/cast/blob/{token}. Auth middleware exempts this prefix; access is
# gated by the short-lived token validated inside the handler.
cast_blob_router = APIRouter(prefix="/api/cast", tags=["cast"])


# --- Helpers ---------------------------------------------------------------


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _registry(request: Request):
    """Pull the DeviceRegistry off app state — None during shutdown / tests."""
    return getattr(request.app.state, "device_registry", None)


def _registry_or_503(request: Request):
    reg = _registry(request)
    if reg is None:
        raise HTTPException(503, "device registry not initialized")
    return reg


def _device_dict(device, *, include_auth: bool = False) -> dict[str, Any]:
    return device.to_dict(include_auth=include_auth)


# --- Request shapes --------------------------------------------------------


class AddDeviceRequest(BaseModel):
    driver: str
    host: str = ""
    port: int | None = None
    label: str = ""
    hint: dict[str, Any] = Field(default_factory=dict)


class UpdateDeviceRequest(BaseModel):
    label: str | None = None
    config: dict[str, Any] | None = None


class InvokeRequest(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)


class PairCompleteRequest(BaseModel):
    code: str = ""


class FavoriteRequest(BaseModel):
    content_key: str
    is_favorite: bool


class SweepCandidate(BaseModel):
    host: str
    port: int | None = None


class SweepCandidatesRequest(BaseModel):
    candidates: list[SweepCandidate] = Field(default_factory=list)


def _client_ip(request: Request) -> str:
    """Best-effort client IP — honors X-Forwarded-For if the deployment
    trusts it, otherwise uses the raw socket peer."""
    from augmentum.config import settings as _settings
    if getattr(_settings, "auth_trust_forwarded_for", False):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


def _is_private_ipv4(ip: str) -> bool:
    """RFC 1918 + link-local. Anything we'd actually sweep."""
    try:
        import ipaddress
        addr = ipaddress.IPv4Address(ip)
    except (ValueError, ipaddress.AddressValueError):
        return False
    return addr.is_private and not addr.is_loopback


def _browser_scheme(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    if forwarded in {"http", "https"}:
        return forwarded
    return request.url.scheme or "https"


def _subnet_from_ip(ip: str, prefix: int = 24) -> str:
    """Return the /24 (or other prefix) subnet containing the IP."""
    try:
        import ipaddress
        net = ipaddress.IPv4Network(f"{ip}/{prefix}", strict=False)
    except (ValueError, ipaddress.AddressValueError):
        return ""
    return str(net)


async def _tokenize_content_url(
    request: Request,
    args: dict[str, Any],
    *,
    user_id: str,
    action: str,
) -> None:
    """Swap an auth-protected content_url for a public token-blob URL.

    The picker sets ``requires_auth=True`` when the URL points at
    augmentum's own origin; this helper looks for a `file_id` in the
    args, issues a short-lived cast token tied to that file_id, and
    rewrites ``content_url`` (or ``image_url`` for image displays) to
    the public blob endpoint that the TV can reach without auth.
    """
    store = getattr(request.app.state, "cast_token_store", None)
    if store is None:
        return  # token store not initialized; nothing to do

    file_id = str(args.get("file_id") or "").strip()
    if not file_id:
        # Without a file_id we can't bind the token to anything resolvable.
        # Pass through unchanged — the TV may still succeed if the URL is
        # already public (LibriVox archive.org, etc).
        return

    # Optional per-cast query params (LibriVox multi-file uses ?file=N).
    query_params: dict[str, str] = {}
    # Heuristic: if the original content_url has a query string, copy
    # the well-known params we know about into the token so the blob
    # path can replay them when calling the upstream stream route.
    if args.get("content_url"):
        from urllib.parse import parse_qsl, urlsplit
        parsed = urlsplit(str(args["content_url"]))
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            if k in ("file", "episode_id", "subtitle_stream_index", "audio_stream_index"):
                query_params[k] = v

    # Bind the token to the device's address host when possible — limits
    # blast radius if the token leaks. Falls back to no IP allowlist for
    # devices behind NAT or with multiple interfaces.
    allowed_ip = ""
    try:
        device_id = request.path_params.get("device_id", "")
        if device_id:
            reg = _registry(request)
            if reg is not None:
                device = await reg.get(device_id, user_id=user_id)
                if device is not None:
                    allowed_ip = str((device.address or {}).get("host") or "").strip()
    except Exception as exc:
        log.debug("device_token_allowed_ip_lookup_failed", error=str(exc))

    token = store.issue(
        user_id=user_id,
        file_id=file_id,
        allowed_client_ip=allowed_ip,
        query_params=query_params,
    )

    # Build the public URL. The TV needs a hostname/port reachable from
    # the LAN. Prefer the PublicHostResolver because localhost means the
    # TV's own loopback, not augmentum.
    blob_path = f"/api/cast/blob/{token.token}"
    blob_url = ""
    resolver = getattr(request.app.state, "public_host_resolver", None)
    if resolver is not None:
        try:
            blob_url = resolver.public_url(
                blob_path,
                request=request,
                scheme=_browser_scheme(request),
            )
        except Exception as exc:
            log.debug("cast_public_host_resolve_failed", error=str(exc))
            blob_url = ""
    if not blob_url:
        scheme = _browser_scheme(request)
        host = request.headers.get("host", "")
        if not host:
            # Fall back to the request URL netloc.
            host = request.url.netloc
        blob_url = f"{scheme}://{host}{blob_path}"

    # Patch the appropriate field in args. The cast picker maps content
    # to either `content_url` (audio/video) or `image_url` (display).
    if "content_url" in args:
        args["content_url"] = blob_url
    if "image_url" in args:
        args["image_url"] = blob_url


# --- Catalog routes --------------------------------------------------------


@router.get("/capabilities")
async def list_capabilities_route(request: Request) -> dict[str, Any]:
    """Capability catalog — drives UI capability-aware action menus."""
    return {
        "capabilities": [cap.to_dict() for cap in list_capabilities()],
    }


@router.get("/drivers")
async def list_drivers_route(request: Request) -> dict[str, Any]:
    """Registered drivers + their capability declarations."""
    reg = _registry_or_503(request)
    return {
        "drivers": [
            {
                "id": d.id,
                "label": d.label,
                "description": getattr(d, "description", ""),
                "capabilities": list(d.capabilities or ()),
                "discovery_modes": list(d.discovery_modes or ()),
                "requires_pairing": bool(d.requires_pairing),
                "supports_passive_discovery": bool(d.supports_passive_discovery),
            }
            for d in reg.list_drivers()
        ],
    }


# --- Device CRUD -----------------------------------------------------------


@router.get("")
async def list_devices(request: Request, only_online: bool = False) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    devices = await reg.list(user_id=uid, only_online=only_online)
    return {"devices": [_device_dict(d) for d in devices]}


@router.post("")
async def add_device(request: Request, body: AddDeviceRequest) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)

    if body.driver not in {d.id for d in reg.list_drivers()}:
        raise HTTPException(400, f"unknown driver: {body.driver}")

    if not body.host:
        raise HTTPException(400, "host is required")

    # Provider-bridged drivers (emby_remote, jellyfin_remote, …) can only
    # be added via discovery — the "device" is an active session in the
    # provider's session list, not something we can manual-knock on. Bail
    # early with a clear message rather than saving an invalid row.
    driver_obj = reg.get_driver(body.driver)
    driver_modes = set(getattr(driver_obj, "discovery_modes", ()) or ())
    if driver_obj is not None and driver_modes == {"via_provider"}:
        hint = body.hint or {}
        has_payload = isinstance(hint, dict) and isinstance(hint.get("discovered"), dict) \
            and hint["discovered"].get("native_id")
        if not has_payload:
            raise HTTPException(
                400,
                f"{driver_obj.label} devices can only be added through discovery "
                "while they're connected. Open the app on your TV (or other client), "
                "then click Search.",
            )

    try:
        discovered = await reg.probe(
            user_id=uid,
            driver=body.driver,
            host=body.host,
            port=body.port,
            hint=body.hint or {},
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Fallback: for drivers where probe isn't meaningful (provider-bridged
    # devices like Emby/Jellyfin sessions — the device lives in the
    # provider's session list, not at an IP we can knock on), re-hydrate
    # from the client's discovery payload. The native_id and address are
    # what the discover() sweep already validated; the registry will still
    # check ownership of any referenced server_id when invoking.
    if discovered is None:
        hint = body.hint or {}
        payload = hint.get("discovered") if isinstance(hint, dict) else None
        if isinstance(payload, dict) and payload.get("native_id"):
            from augmentum.devices.device import DiscoveredDevice
            discovered = DiscoveredDevice(
                driver=str(payload.get("driver") or body.driver),
                native_id=str(payload.get("native_id") or ""),
                label=str(payload.get("label") or body.label or ""),
                capabilities=list(payload.get("capabilities") or []),
                address=dict(payload.get("address") or {}),
                metadata=dict(payload.get("metadata") or {}),
            )

    if discovered is None:
        # Save as 'unverified' so the user can retest later. Fill in
        # capabilities from the driver's declared surface so the device
        # can at least be dispatched to — without this, every manual-add
        # 422s on first cast even when the driver is fully functional.
        fallback_caps = list(getattr(driver_obj, "capabilities", ()) or []) if driver_obj else []
        device = await reg.save(
            user_id=uid,
            driver=body.driver,
            native_id=f"manual:{body.host}:{body.port or ''}",
            label=body.label or body.host,
            capabilities=fallback_caps,
            address={"host": body.host, "port": body.port},
            status="unverified",
        )
        return {"device": _device_dict(device), "verified": False}

    device = await reg.save(
        user_id=uid,
        driver=body.driver,
        native_id=discovered.native_id,
        label=body.label or discovered.label,
        capabilities=discovered.capabilities,
        address=discovered.address,
        metadata=discovered.metadata,
        status="online",
    )
    return {"device": _device_dict(device), "verified": True}


@router.patch("/{device_id}")
async def update_device(
    request: Request,
    device_id: str,
    body: UpdateDeviceRequest,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    device = await reg.update(
        device_id,
        user_id=uid,
        label=body.label,
        config=body.config,
    )
    if device is None:
        raise HTTPException(404, "device not found")
    return {"device": _device_dict(device)}


@router.delete("/{device_id}")
async def delete_device(request: Request, device_id: str) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    ok = await reg.delete(device_id, user_id=uid)
    if not ok:
        raise HTTPException(404, "device not found")
    return {"deleted": True, "id": device_id}


# --- Discovery -------------------------------------------------------------


@router.get("/discover")
async def discover_route(
    request: Request,
    refresh: int = 1,
    drivers: str | None = None,
    timeout_ms: int = 3000,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    only = [d.strip() for d in drivers.split(",") if d.strip()] if drivers else None
    result = await reg.discover(
        user_id=uid,
        drivers=only,
        timeout_s=max(0.5, float(timeout_ms or 3000) / 1000.0),
    )
    return result.to_dict()


@router.get("/sweep")
async def sweep_route(
    request: Request,
    subnet: str = "",
    timeout_ms: int = 8000,
) -> dict[str, Any]:
    """TCP-based subnet sweep for environments where multicast fails.

    The user's request source IP is the strongest subnet hint we have —
    if they reached augmentum from `192.168.1.10`, their TVs are almost
    certainly in `192.168.1.0/24`. When the source IP is not in a
    private range (Tailscale, public IP, localhost), we fall back to
    common consumer-router defaults.

    Direct TCP unicast crosses Docker NAT cleanly, so this is the
    Docker-Desktop-friendly alternative to SSDP multicast.
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)

    # Caller-supplied subnet overrides everything — but only if it's in
    # a private RFC1918 range (the underlying sweep also enforces this,
    # but we reject early at the route boundary so the failure is
    # surfaced as a 400 rather than an empty result).
    chosen_subnet = (subnet or "").strip()
    inferred_from = ""
    if chosen_subnet:
        try:
            import ipaddress
            net = ipaddress.IPv4Network(chosen_subnet, strict=False)
            if not (net.is_private and not net.is_loopback):
                raise HTTPException(400, "subnet must be in a private (RFC1918) range")
            if net.num_addresses > 1024:
                raise HTTPException(400, "subnet too wide (max /22)")
        except (ValueError, ipaddress.AddressValueError) as exc:
            raise HTTPException(400, f"invalid subnet: {exc}") from exc
    else:
        client_ip = _client_ip(request)
        if client_ip and _is_private_ipv4(client_ip):
            chosen_subnet = _subnet_from_ip(client_ip, 24)
            inferred_from = f"client_ip:{client_ip}"

    # Default 15s — TCP-knock pre-filter makes a /24 complete well inside
    # this budget on a healthy LAN; gives slack for slower networks.
    timeout_s = max(2.0, float(timeout_ms or 15000) / 1000.0)
    discovered, errors, duration_s = await reg.sweep(
        user_id=uid,
        subnet=chosen_subnet or None,
        timeout_s=timeout_s,
    )

    saved = await reg.list(user_id=uid)
    from augmentum.devices.discovery.coordinator import merge_discovered_with_saved
    truly_new, online_ids, _heal_map = merge_discovered_with_saved(discovered, saved)

    return {
        "discovered": [d.to_dict() for d in truly_new],
        "online_saved_ids": online_ids,
        "errors": errors,
        "duration_s": duration_s,
        "subnet": chosen_subnet,
        "inferred_from": inferred_from,
    }


@router.post("/sweep_candidates")
async def sweep_candidates_route(
    request: Request,
    body: SweepCandidatesRequest,
) -> dict[str, Any]:
    """Validate browser-supplied (host, port) candidates.

    The browser is the only thing guaranteed to be on the user's LAN
    (the augmentum container may be behind Docker NAT, on a VPS, etc).
    Browser-side probing finds 'something speaks HTTP at this address';
    we then validate each candidate by fetching the actual UPnP
    description from the server side and parsing it into a real
    DiscoveredDevice. The two-tier split sidesteps both Docker
    multicast restrictions AND HTTPS mixed-content blocking.
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)

    # SSRF guard — even authenticated users must not be able to use
    # augmentum to probe arbitrary external/internal hosts. Restrict to
    # private (RFC1918) IPv4 only; drop everything else silently.
    candidates_payload: list[dict[str, Any]] = []
    rejected = 0
    for c in (body.candidates or []):
        host = (c.host or "").strip()
        if not host or not _is_private_ipv4(host):
            rejected += 1
            continue
        candidates_payload.append({"host": host, "port": c.port})

    # Hard cap on the number of candidates per request — even with the
    # private-IP filter, a malicious client could otherwise send tens of
    # thousands of probes. A /24 has 254 hosts × ~7 ports = ~1800 probes,
    # so cap at 4096 to leave headroom without enabling abuse.
    if len(candidates_payload) > 4096:
        candidates_payload = candidates_payload[:4096]

    discovered = await reg.sweep_candidates(
        user_id=uid,
        candidates=candidates_payload,
        timeout_s=6.0,
    )

    saved = await reg.list(user_id=uid)
    from augmentum.devices.discovery.coordinator import merge_discovered_with_saved
    truly_new, online_ids, _heal_map = merge_discovered_with_saved(discovered, saved)

    return {
        "discovered": [d.to_dict() for d in truly_new],
        "online_saved_ids": online_ids,
        "evaluated": len(candidates_payload),
    }


@router.post("/{device_id}/test")
async def test_device(request: Request, device_id: str) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    device = await reg.get(device_id, user_id=uid)
    if device is None:
        raise HTTPException(404, "device not found")
    host = str(device.address.get("host") or "").strip()
    if not host:
        raise HTTPException(400, "device has no host on file")
    discovered = await reg.probe(
        user_id=uid,
        driver=device.driver,
        host=host,
        port=device.address.get("port"),
    )
    return {
        "reachable": discovered is not None,
        "discovered": discovered.to_dict() if discovered else None,
    }


# --- Capability invocation -------------------------------------------------


@router.post("/{device_id}/{capability}/{action}")
async def invoke_capability(
    request: Request,
    device_id: str,
    capability: str,
    action: str,
    body: InvokeRequest,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)

    if get_capability(capability) is None:
        raise HTTPException(400, f"unknown capability: {capability}")

    args = dict(body.args or {})

    # Tokenize auth-protected content URLs so the TV (which doesn't
    # carry the user's auth cookie) can fetch them. The picker sets
    # `requires_auth: true` on args coming from the same origin as
    # augmentum; we honor that signal by issuing a short-lived token
    # and replacing the content_url with a public blob endpoint URL.
    if args.pop("requires_auth", False):
        await _tokenize_content_url(request, args, user_id=uid, action=action)

    result = await reg.invoke(
        user_id=uid,
        device_id=device_id,
        capability=capability,
        action=action,
        args=args,
    )
    if not result.ok:
        # 404 / 422 / 502 — surface a meaningful status code based on code.
        status_code = 502
        if result.code in {"device_not_found"}:
            status_code = 404
        elif result.code in {
            "unknown_capability",
            "unknown_action",
            "capability_not_supported",
            "driver_unavailable",
        }:
            status_code = 422
        return JSONResponse(status_code=status_code, content=result.to_dict())
    return result.to_dict()


@router.get("/{device_id}/{capability}")
async def snapshot_capability(
    request: Request,
    device_id: str,
    capability: str,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    snap = await reg.snapshot(user_id=uid, device_id=device_id, capability=capability)
    return {"snapshot": snap or {}}


# --- Sessions --------------------------------------------------------------


@router.get("/sessions/active")
async def list_active_sessions(request: Request) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    sessions = await reg.list_sessions(user_id=uid)
    return {"sessions": [s.to_dict() for s in sessions]}


@router.delete("/sessions/{session_id}")
async def end_session_route(request: Request, session_id: str) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    ok = await reg.end_session(user_id=uid, session_id=session_id)
    if not ok:
        raise HTTPException(404, "session not found")
    return {"ended": True, "id": session_id}


# --- Pairing ---------------------------------------------------------------


@router.post("/{device_id}/pair/start")
async def pair_start_route(request: Request, device_id: str) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    result = await reg.pair_start(user_id=uid, device_id=device_id)
    return result.to_dict()


@router.post("/{device_id}/pair/complete")
async def pair_complete_route(
    request: Request,
    device_id: str,
    body: PairCompleteRequest,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    result = await reg.pair_complete(user_id=uid, device_id=device_id, code=body.code or "")
    return result.to_dict()


# --- Events SSE ------------------------------------------------------------


@router.get("/events")
async def device_events_sse(
    request: Request,
    device_id: str | None = None,
    capability: str | None = None,
) -> StreamingResponse:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)

    async def event_stream():
        async for event in await reg.subscribe(
            user_id=uid,
            device_id=device_id,
            capability=capability,
        ):
            try:
                yield f"data: {json.dumps(event.to_dict())}\n\n"
            except Exception as exc:
                log.debug("device_events_serialize_failed", error=str(exc))
                continue

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# --- Play history (smart-match for voice/LLM) -----------------------------


@router.get("/history/recent")
async def recent_history(
    request: Request,
    content_kind: str = "",
    limit: int = 25,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    history_store = reg._history_store  # noqa: SLF001 — registry-internal access
    rows = await history_store.recent_for_kind(
        user_id=uid,
        content_kind=content_kind,
        limit=limit,
    )
    return {"history": rows}


@router.get("/history/favorites")
async def favorites(
    request: Request,
    content_kind: str = "",
    limit: int = 25,
) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    history_store = reg._history_store  # noqa: SLF001
    rows = await history_store.favorites_for_kind(
        user_id=uid,
        content_kind=content_kind,
        limit=limit,
    )
    return {"favorites": rows}


@router.post("/history/favorite")
async def set_favorite(request: Request, body: FavoriteRequest) -> dict[str, Any]:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "auth required")
    reg = _registry_or_503(request)
    history_store = reg._history_store  # noqa: SLF001
    affected = await history_store.set_favorite(
        user_id=uid,
        content_key=body.content_key,
        is_favorite=body.is_favorite,
    )
    return {"affected": affected}


# ===========================================================================
# Public cast-blob endpoint
# ===========================================================================
#
# Mounted on a separate router at /api/cast/blob/{token} so it bypasses the
# auth middleware (the prefix is whitelisted in augmentum/auth/middleware.py).
# Access is gated entirely by the cast token: short-lived, single-user-bound,
# optionally IP-locked. The handler delegates the actual stream resolution
# to the existing media_routes.stream_media() by injecting the token user
# into the request scope, so all the file_index / provider / range-proxy
# logic stays in one place.

class _CastTokenUser:
    """Minimal user stub matching the shape stream_media expects."""
    __slots__ = ("id",)
    def __init__(self, user_id: str) -> None:
        self.id = user_id


@cast_blob_router.get("/blob/{token}")
async def cast_blob(token: str, request: Request):
    """Token-gated public stream proxy for cast targets.

    Validates the token (existence, expiry, client-IP allowlist), then
    delegates to the same stream-resolution path the authenticated
    /api/media/stream/{file_id} route uses. Everything from Range
    forwarding through provider-specific URL building works unchanged.
    """
    from urllib.parse import parse_qsl, urlencode

    store = getattr(request.app.state, "cast_token_store", None)
    if store is None:
        raise HTTPException(503, "cast token store not initialized")

    client_ip = _client_ip(request)
    entry = store.lookup(token, client_ip=client_ip)
    if entry is None:
        raise HTTPException(404, "token expired or invalid")

    # Inject the tokens user into the request scope so stream_media
    # treats the call as that user. We also splice in any query params
    # the token recorded (e.g. file=N for LibriVox multi-file books).
    request.scope["user"] = _CastTokenUser(entry.user_id)
    if entry.query_params:
        existing = dict(parse_qsl(
            request.scope.get("query_string", b"").decode("latin-1", errors="ignore"),
            keep_blank_values=True,
        ))
        for k, v in entry.query_params.items():
            existing.setdefault(k, v)
        request.scope["query_string"] = urlencode(existing).encode("latin-1")

    # Delegate. The existing route handles file_index lookup, provider
    # selection, Range forwarding, and 206 Partial Content semantics.
    from augmentum.proxy.media_routes import stream_media
    return await stream_media(entry.file_id, request)
