"""Android/mobile pairing routes.

The flow reuses the cast receiver ceremony shape but finishes with a
device-bound Android auth session instead of a receiver cookie.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import Response

from augmentum.auth.mobile_pairing import (
    STATE_APPROVED,
    STATE_CLAIMED,
    STATE_PENDING,
    MobilePairStore,
    TrustedMobileDeviceStore,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/auth/pair", tags=["auth"])


class MobilePairClaimRequest(BaseModel):
    device_id: str = ""
    label: str = ""
    platform: str = "android"
    app_version: str = ""
    public_key: str = ""
    key_alg: str = ""
    scopes: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)


class MobilePairFinishRequest(BaseModel):
    grant_token: str = ""


def _get_user(request: Request):
    return request.scope.get("user")


def _get_sm(request: Request):
    sm = getattr(request.app.state, "session_manager", None)
    if sm is None:
        raise HTTPException(503, "auth system not initialised")
    return sm


def _pair_store(request: Request) -> MobilePairStore:
    store = getattr(request.app.state, "mobile_pair_store", None)
    if store is None:
        raise HTTPException(503, "mobile pair store not initialised")
    return store


def _device_store(request: Request) -> TrustedMobileDeviceStore:
    sm = _get_sm(request)
    return TrustedMobileDeviceStore(sm._db)


def _client_ip(request: Request) -> str:
    if request.client:
        return request.client.host
    return ""


def _server_root_url(request: Request) -> str:
    resolver = getattr(request.app.state, "public_host_resolver", None)
    if resolver is not None:
        resolved = resolver.public_url("/", request=request)
        if resolved:
            return resolved.rstrip("/")
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    return f"{scheme}://{host}".rstrip("/")


def _request_root_url(request: Request) -> str:
    """The host:port the *caller* reached us on (proxy-aware).

    When the phone pairs on the home LAN, this is the LAN address — the
    direct, fastest path. Unlike ``_server_root_url`` it does NOT prefer the
    operator's configured public host, so we can hand the LAN URL back as the
    primary endpoint and the configured public host as the remote one.
    """
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    return f"{scheme}://{host}".rstrip("/")


# CGNAT 100.64.0.0/10 — the range Tailscale assigns. Matches the filter
# start.sh/start.bat use to populate AUGMENTUM_TLS_EXTRA_SANS.
_TAILSCALE_IP_RE = re.compile(r"^100\.(?:6[4-9]|[7-9]\d|1[0-1]\d|12[0-7])\.\d{1,3}\.\d{1,3}$")


def _tailscale_host_from_sans() -> str:
    """First Tailscale (100.64/10) IP from the auto-detected TLS SAN list.

    ``start.sh`` / ``start.bat`` already discover the host's LAN + Tailscale
    addresses and write them to ``AUGMENTUM_TLS_EXTRA_SANS`` for the cert — so
    this needs ZERO operator config to work on a self-hosted install. Returns
    the bare IP (no port) or "".
    """
    sans = os.environ.get("AUGMENTUM_TLS_EXTRA_SANS", "")
    for token in sans.split(","):
        tok = token.strip()
        if not tok:
            continue
        # SAN entries look like "IP:100.64.0.1" / "DNS:host.example". Take the
        # value part; we only want a routable Tailscale IP.
        upper = tok.upper()
        if upper.startswith("IP:"):
            tok = tok[3:].strip()
        elif upper.startswith("DNS:"):
            continue
        if _TAILSCALE_IP_RE.match(tok):
            return tok
    return ""


def _request_port(request: Request) -> str:
    """The port the caller reached us on (so the remote URL mirrors it)."""
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    host = host.split(",")[0].strip()
    if ":" in host:
        return host.rsplit(":", 1)[1].strip()
    return ""


def _configured_remote_url(request: Request) -> str:
    """The off-LAN endpoint to hand the phone, resolved automatically.

    Two sources, in priority order:

    1. The operator's explicit ``AUGMENTUM_PUBLIC_HOST`` (custom domain / tunnel),
       if set. Reuses the PublicHostResolver's stored override.
    2. **Automatic** — a Tailscale (100.64/10) IP from the auto-detected TLS
       SANs. This is the OSS default path: no operator config, and (unlike
       ``AUGMENTUM_PUBLIC_HOST``) it does NOT touch the resolver, so cast URLs
       are unaffected.

    Empty when neither is available (pure-LAN install).
    """
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"

    # 1. Explicit operator override.
    resolver = getattr(request.app.state, "public_host_resolver", None)
    configured = ""
    if resolver is not None:
        try:
            configured = (resolver.state().get("configured") or "").strip()
        except Exception:
            configured = ""
    if configured:
        if "://" in configured:
            return configured.rstrip("/")
        return f"{scheme}://{configured}".rstrip("/")

    # 2. Automatic: Tailscale address from the cert SANs.
    ts_ip = _tailscale_host_from_sans()
    if ts_ip:
        port = _request_port(request)
        host = f"{ts_ip}:{port}" if port else ts_ip
        # Tailscale is served over the same HTTPS Caddy front door as LAN.
        return f"https://{host}".rstrip("/")

    return ""


def _pair_url(request: Request, pair_code: str) -> str:
    return "augmentum://pair?" + urlencode({
        "server_url": _server_root_url(request),
        "code": pair_code,
    })


@router.post("/start")
async def mobile_pair_start(request: Request) -> dict[str, Any]:
    """Authenticated web start. Returns QR material for Android setup."""
    user = _get_user(request)
    if user is None:
        raise HTTPException(401, "auth required")
    record = _pair_store(request).start(user_id=user.id)
    return {
        "pair_code": record.pair_code,
        "pair_url": _pair_url(request, record.pair_code),
        "qr_url": f"/api/auth/pair/qr/{record.pair_code}.svg",
        "status_path": f"/api/auth/pair/status/{record.pair_code}",
        "expires_in": record.expires_in(),
        "state": record.state,
    }


@router.get("/qr/{pair_code}.svg")
async def mobile_pair_qr(pair_code: str, request: Request) -> Response:
    """Public QR image. Knowledge of the short-lived code is the guard."""
    record = _pair_store(request).poll_code(pair_code)
    if record is None or record.state not in (STATE_PENDING, STATE_CLAIMED, STATE_APPROVED):
        raise HTTPException(404, "pair code not found or expired")
    try:
        import qrcode
        import qrcode.image.svg

        factory = qrcode.image.svg.SvgPathImage
        img = qrcode.make(_pair_url(request, pair_code), image_factory=factory, box_size=14, border=2)
        from io import BytesIO

        buf = BytesIO()
        img.save(buf)
        svg_bytes = buf.getvalue()
    except Exception as exc:
        log.warning("mobile_pair_qr_generate_failed", error=str(exc))
        raise HTTPException(500, "qr generation failed") from exc
    return Response(
        content=svg_bytes,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/status/{pair_code}")
async def mobile_pair_status(pair_code: str, request: Request) -> dict[str, Any]:
    """Authenticated web status for the approval dialog."""
    user = _get_user(request)
    if user is None:
        raise HTTPException(401, "auth required")
    record = _pair_store(request).get_for_user(pair_code, user_id=user.id)
    if record is None:
        raise HTTPException(404, "pair code not found or expired")
    return record.public_status()


@router.post("/claim/{pair_code}")
async def mobile_pair_claim(
    pair_code: str,
    body: MobilePairClaimRequest,
    request: Request,
) -> dict[str, Any]:
    """Unauthenticated Android claim. The pair code is the setup secret."""
    if not body.device_id.strip():
        raise HTTPException(400, "device_id required")
    record = _pair_store(request).claim(
        pair_code,
        device_id=body.device_id,
        label=body.label,
        platform=body.platform or "android",
        app_version=body.app_version,
        public_key=body.public_key,
        key_alg=body.key_alg,
        scopes=body.scopes,
        capabilities=body.capabilities,
        user_agent=request.headers.get("user-agent", ""),
    )
    if record is None:
        raise HTTPException(409, "pair code not found, expired, or already claimed")
    return {
        "pair_code": pair_code,
        "state": record.state,
        "claim_token": record.claim_token,
        "poll_path": f"/api/auth/pair/poll/{record.claim_token}",
        "expires_in": record.expires_in(),
    }


@router.get("/poll/{claim_token}")
async def mobile_pair_poll(claim_token: str, request: Request) -> dict[str, Any]:
    """Unauthenticated Android poll using the private claim token."""
    record = _pair_store(request).poll_claim(claim_token)
    if record is None:
        raise HTTPException(404, "pair claim not found or expired")
    include_grant = record.state == STATE_APPROVED
    return record.public_status(include_grant=include_grant)


@router.post("/approve/{pair_code}")
async def mobile_pair_approve(pair_code: str, request: Request) -> dict[str, Any]:
    """Authenticated web approval after the phone has claimed the code."""
    user = _get_user(request)
    if user is None:
        raise HTTPException(401, "auth required")
    record = _pair_store(request).approve(pair_code, user_id=user.id)
    if record is None:
        raise HTTPException(409, "pair code is not claimable by this user")
    return record.public_status()


@router.post("/finish")
async def mobile_pair_finish(
    body: MobilePairFinishRequest,
    request: Request,
) -> dict[str, Any]:
    """Android redeems the one-time grant for a scoped auth session."""
    grant_token = (body.grant_token or "").strip()
    if not grant_token:
        raise HTTPException(400, "grant_token required")

    store = _pair_store(request)
    record = store.consume_grant(grant_token)
    if record is None:
        raise HTTPException(401, "invalid or already-used mobile pair grant")

    sm = _get_sm(request)
    user = await sm.get_user_by_id(record.user_id)
    if user is None or not user.is_active:
        raise HTTPException(401, "paired user is unavailable")

    device = await _device_store(request).upsert_from_pair(record)
    raw_session = await sm.create_session(
        user.id,
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent", "")[:200],
        source="android",
        source_device_id=device.device_id,
    )
    # Hand back both endpoints so the phone gains LAN↔remote failover without
    # the user typing a second URL: the LAN address it reached us on (primary,
    # direct/fast) plus the operator's configured public host (remote, e.g.
    # Tailscale) when one is set and distinct.
    primary_url = _request_root_url(request)
    remote_url = _configured_remote_url(request)
    if remote_url and remote_url == primary_url:
        remote_url = ""
    return {
        "session_token": raw_session,
        "auth_type": "bearer",
        "source": "android",
        "device": device.to_dict(),
        "user": user.to_public_dict(),
        "server_url": primary_url,
        "remote_url": remote_url,
    }


@router.get("/devices")
async def mobile_pair_devices(request: Request) -> dict[str, Any]:
    """List the current user's trusted mobile devices."""
    user = _get_user(request)
    if user is None:
        raise HTTPException(401, "auth required")
    devices = await _device_store(request).list_for_user(user_id=user.id, include_revoked=True)
    return {"devices": [device.to_dict() for device in devices]}


@router.post("/devices/{mobile_id}/revoke")
async def mobile_pair_revoke_device(mobile_id: str, request: Request) -> dict[str, Any]:
    """Revoke a trusted mobile device and its active Android sessions."""
    user = _get_user(request)
    if user is None:
        raise HTTPException(401, "auth required")
    device = await _device_store(request).revoke(mobile_id, user_id=user.id)
    if device is None:
        raise HTTPException(404, "mobile device not found")
    revoked_sessions = await _get_sm(request).revoke_sessions_for_source_device(
        user.id,
        source="android",
        source_device_id=device.device_id,
    )
    return {
        "revoked": True,
        "revoked_sessions": revoked_sessions,
        "device": device.to_dict(),
    }

