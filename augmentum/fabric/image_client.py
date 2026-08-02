"""Outbound image-generation client.

When the operator's chat asks for an image-gen model that only lives
on a peer, we proxy the request to that peer's local pipeline rather
than syncing the model weights (which can be multi-GB). The peer
runs its own queue + image pipeline as normal; we collect the
finished bytes over signed HTTPS.

Two-step transfer:

  1. POST /api/image/generate on the peer with the GenerateRequest
     JSON body (signed envelope, body-integrity hash). The peer
     handler waits synchronously for the image to finish then
     returns ``GenerateResponse`` with image_id + URL.
  2. GET /api/image/{image_id} on the peer to pull the rendered
     image bytes. Same signed-envelope auth applies. We hand the
     bytes back to the caller; the caller writes them into our
     local image_output store + creates a local image_id so the
     image becomes part of the user's local library, surviving the
     peer going offline.

The choice to pull bytes back rather than return a peer-prefixed
URL is deliberate: chat history references survive peer downtime,
the image library reflects everything the user has generated, and
downstream consumers (chat rendering, embed creation, library
browser) all expect local image_ids.

Authentication is the same signed-request envelope used by every
other fabric peer call — ed25519 over (sender, user, method, path,
ts, sha256(body)). The receiving peer's FabricPeerMiddleware
pre-populates ``scope["user"]`` so the existing user-auth path
through AuthMiddleware accepts the request.

See :mod:`augmentum.fabric.pair_client` for the trust-model
writeup that explains why we use ``verify=False`` on the httpx
client.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)


# Generation is the slow part of the round-trip — a Flux render on
# the peer can easily take 30-60s. We allow up to 5 minutes; if the
# peer's queue is even slower the user will give up before this
# timeout fires anyway.
_GENERATE_HTTP_TIMEOUT_S = 300.0

# Fetching the finished bytes back should be sub-second on LAN; we
# cap separately so a hung peer mid-transfer doesn't tie up the
# request for the whole 5-minute budget.
_FETCH_HTTP_TIMEOUT_S = 60.0


class RemoteImageError(Exception):
    """Any failure during cross-peer image generation. The caller
    surfaces this to the user (image gen is a foreground operation,
    not a background recall like knowledge search — silently dropping
    would leave them staring at a blank result).
    """


async def generate_image_via_peer(
    *,
    http_client: httpx.AsyncClient,
    identity: FabricIdentity,
    user_id: str,
    peer_addr: str,
    generate_request_payload: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Run an image generation on a peer + return the bytes + metadata.

    ``generate_request_payload`` is the JSON form of GenerateRequest
    (the pydantic model from augmentum/image/schemas.py). Passed as
    a dict here rather than a typed instance to keep the fabric
    layer free of image-layer imports — the caller serialises.

    Returns ``(image_bytes, metadata_dict)`` where metadata_dict is
    the full GenerateResponse body (minus the peer-relative URL,
    which won't resolve on this side). The caller is responsible
    for writing image_bytes to our local image_output and creating
    a local image_id.

    Raises :class:`RemoteImageError` on any transport, auth, or
    completion failure. Image-gen is foreground; the operator
    wants to know why it failed.
    """
    body = json.dumps(generate_request_payload, separators=(",", ":")).encode("utf-8")
    scheme = "https" if "://" not in peer_addr else peer_addr.split("://", 1)[0]
    host = peer_addr.split("://", 1)[-1].rstrip("/")
    # Single-shot dedicated fabric endpoint. The previous two-step
    # (POST /api/image/generate → GET /api/image/{id}) flow had a
    # mid-flight failure mode where generation succeeded but the
    # fetch hit a transient network blip, costing the user a 30-60s
    # render they couldn't see. The new endpoint returns a
    # multipart/form-data response with the metadata + bytes inline
    # so the transfer is atomic — either the operator gets both, or
    # neither.
    path = "/api/fabric/image/generate"
    url = f"{scheme}://{host}{path}"

    from augmentum.fabric.peer_middleware import build_signed_peer_headers

    headers = {
        "Content-Type": "application/json",
        **build_signed_peer_headers(
            identity=identity, user_id=user_id,
            method="POST", path=path, body=body,
        ),
    }

    try:
        resp = await http_client.post(
            url, content=body, headers=headers,
            timeout=_GENERATE_HTTP_TIMEOUT_S,
        )
    except httpx.TransportError as exc:
        log.info(
            "fabric_image_generate_unreachable",
            peer_addr=peer_addr, error=str(exc)[:160],
        )
        raise RemoteImageError(
            f"peer {peer_addr!r} unreachable: {exc}"
        ) from None

    if resp.status_code >= 400:
        log.info(
            "fabric_image_generate_status",
            peer_addr=peer_addr, status=resp.status_code,
            body=resp.text[:200],
        )
        raise RemoteImageError(
            f"peer returned {resp.status_code}: {resp.text[:200]}"
        )

    # Parse the multipart response (metadata JSON part + image bytes
    # part). Receiver uses a UUID-derived boundary so we have to read
    # it from the Content-Type header rather than hardcoding.
    content_type = resp.headers.get("content-type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise RemoteImageError(
            f"peer returned unexpected content-type: {content_type[:120]}"
        )
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    image_bytes, metadata = _parse_fabric_image_multipart(resp.content, boundary)
    if not image_bytes:
        raise RemoteImageError("peer returned empty image bytes in multipart")

    log.info(
        "fabric_image_proxy_success",
        peer_addr=peer_addr,
        bytes=len(image_bytes),
    )
    return image_bytes, metadata


def _parse_fabric_image_multipart(
    body: bytes, boundary: str,
) -> tuple[bytes, dict[str, Any]]:
    """Extract (image_bytes, metadata_dict) from the multipart body.

    Receiver shape (matches augmentum/proxy/fabric_routes.py
    fabric_image_generate):

      --<boundary>\\r\\n
      Content-Type: application/json\\r\\n
      Content-Disposition: form-data; name="metadata"\\r\\n\\r\\n
      <json bytes>\\r\\n
      --<boundary>\\r\\n
      Content-Type: application/octet-stream\\r\\n
      Content-Disposition: form-data; name="image"; filename="image.png"\\r\\n\\r\\n
      <image bytes>\\r\\n
      --<boundary>--\\r\\n

    Minimal parser — we control both sides of the protocol so we
    don't pull in a full multipart library.
    """
    delimiter = f"--{boundary}".encode("latin-1")
    segments = body.split(delimiter)
    metadata: dict[str, Any] = {}
    image_bytes = b""
    for seg in segments:
        # Skip preamble (empty before first delimiter), closing token
        # ("--\r\n"), and otherwise blank parts.
        if not seg or seg.strip() in (b"", b"--", b"--\r\n"):
            continue
        # Each part: headers \r\n\r\n body
        try:
            header_blob, part_body = seg.split(b"\r\n\r\n", 1)
        except ValueError:
            continue
        # Strip the trailing CRLF that separates the part body from
        # the next delimiter line.
        if part_body.endswith(b"\r\n"):
            part_body = part_body[:-2]
        hdr_str = header_blob.decode("latin-1", errors="ignore").lower()
        if "name=\"metadata\"" in hdr_str:
            try:
                metadata = json.loads(part_body.decode("utf-8"))
            except Exception:
                metadata = {}
        elif "name=\"image\"" in hdr_str:
            image_bytes = part_body
    return image_bytes, metadata
