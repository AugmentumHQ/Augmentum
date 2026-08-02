"""Outbound render dispatch — ship a RenderJob to a peer.

Mirrors image_client.py's pattern: build a signed peer envelope,
POST to the peer's ``/api/cast/render`` endpoint, parse the
``RenderResult`` JSON back. The receiving peer authenticates the
request via FabricPeerMiddleware, runs its own local executor, and
returns the result.

Unlike image_client there's no second fetch step today — the render
"output" is a URL (eventually a stream the TV pulls directly from
whichever node ran the work) so no bulk-bytes transfer is required.
When the real render pipeline lands, the output URL may point at
the rendering node's own HTTPS edge; consumers fetch the rendered
bytes from there, not via the orchestrator.

See pair_client.py for the trust-model writeup explaining why we
use ``verify=False`` on the httpx client.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import httpx

from augmentum.cast.render import RenderJob, RenderResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)


# Render is fast when stubbed but real renders (VRM frame, HTML
# rasterize) can take seconds. WebRTC stream setup blocks until ICE
# negotiation completes. 60s gives plenty of headroom; the caller's
# tolerance probably caps the experience anyway.
_RENDER_HTTP_TIMEOUT_S = 60.0


async def render_via_peer(
    *,
    http_client: httpx.AsyncClient,
    identity: "FabricIdentity",
    user_id: str,
    peer_node_id: str,
    peer_addr: str,
    job: RenderJob,
) -> RenderResult:
    """Ship a render job to a peer + return its RenderResult.

    Never raises for transport / status failures — returns a
    RenderResult with ok=False + a code instead. The dispatcher
    contract is "always return a RenderResult"; raising would
    force every caller to wrap in try/except.

    The peer authenticates the request via the same signed envelope
    every other fabric call uses (ed25519 over sender+user+method+
    path+ts+sha256(body)).
    """
    body_obj = {
        "kind": job.kind,
        "target_device_id": job.target_device_id,
        "payload": job.payload,
    }
    body = json.dumps(body_obj, separators=(",", ":")).encode("utf-8")

    scheme = "https" if "://" not in peer_addr else peer_addr.split("://", 1)[0]
    host = peer_addr.split("://", 1)[-1].rstrip("/")
    # Dedicated fabric data-plane endpoint. The user-facing
    # /api/cast/render endpoint accepts fabric peers via the per-peer
    # service user too, but routing fabric traffic through a separate
    # URL means operator changes to the user endpoint (rate limits,
    # content gates, audit) don't accidentally affect peer dispatch.
    path = "/api/fabric/render"
    url = f"{scheme}://{host}{path}"

    from augmentum.fabric.peer_middleware import build_signed_peer_headers

    headers = {
        "Content-Type": "application/json",
        **build_signed_peer_headers(
            identity=identity,
            user_id=user_id,
            method="POST",
            path=path,
            body=body,
        ),
    }

    try:
        resp = await http_client.post(
            url, content=body, headers=headers,
            timeout=_RENDER_HTTP_TIMEOUT_S,
        )
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
        log.info(
            "fabric_render_peer_unreachable",
            peer_node_id=peer_node_id, peer_addr=peer_addr,
            error=str(exc)[:160],
        )
        return RenderResult(
            ok=False,
            location="peer",
            node_id=peer_node_id,
            code="peer_unreachable",
            message=f"peer {peer_addr!r} unreachable: {exc}",
        )

    if resp.status_code >= 400:
        log.info(
            "fabric_render_peer_status",
            peer_node_id=peer_node_id, status=resp.status_code,
            body=resp.text[:200],
        )
        return RenderResult(
            ok=False,
            location="peer",
            node_id=peer_node_id,
            code=f"peer_status_{resp.status_code}",
            message=resp.text[:200],
        )

    try:
        data = resp.json()
    except Exception as exc:
        return RenderResult(
            ok=False,
            location="peer",
            node_id=peer_node_id,
            code="peer_non_json",
            message=f"peer returned non-JSON: {exc}",
        )

    # Hydrate the RenderResult from the peer's reply. Tolerant of
    # missing fields — peer may be running an older schema.
    return RenderResult(
        ok=bool(data.get("ok", False)),
        location=str(data.get("location") or "peer"),
        node_id=str(data.get("node_id") or peer_node_id),
        output_url=str(data.get("output_url") or ""),
        code=str(data.get("code") or ""),
        message=str(data.get("message") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def serialise_result(result: RenderResult) -> dict:
    """Convert a RenderResult to a JSON-safe dict for the wire reply.

    Pure helper — extracted so the inbound route + tests share one
    serializer instead of each rebuilding the dict.
    """
    return asdict(result)
