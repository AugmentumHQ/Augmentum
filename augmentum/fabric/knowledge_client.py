"""Outbound knowledge-pack search client.

When a chat session uses knowledge packs and some of those packs
live only on peer instances (not on this node), we proxy a search
RPC to the peer rather than syncing the pack bytes locally. The
peer runs its own ``PackManager.search`` and returns the result
chunks; we merge them with any local results.

Wire format mirrors the existing inbound POST /api/knowledge/search
endpoint added in Phase 6: JSON body with ``{q, pack_ids, limit}``,
JSON response with ``{query, pack_ids, results: [PackResult...]}``.

Authentication is the same signed-request envelope used by
:mod:`augmentum.models.fabric_backend`: ed25519 signature over
canonical bytes that include sender, user_id, method, path, ts, and
sha256(body). The receiving peer's :class:`FabricPeerMiddleware`
pre-populates ``scope["user"]`` so the existing user-auth path
through AuthMiddleware accepts the request like a normal logged-in
call.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import httpx

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)


# Search is interactive — we want either a fast result or a clean
# failure that lets the caller fall back to local-only. A noisy
# fanout to multiple peers with slow timeouts would block in-chat
# pack injection (which already has its own latency budget).
_SEARCH_HTTP_TIMEOUT_S = 8.0


class RemoteSearchError(Exception):
    """Any failure fetching pack results from a peer. Caller should
    log + continue with local-only results rather than surface the
    error to the user — a peer being slow shouldn't break chat.
    """


async def search_remote_packs(
    *,
    http_client: httpx.AsyncClient,
    identity: FabricIdentity,
    user_id: str,
    peer_addr: str,
    query: str,
    pack_ids: Iterable[str],
    limit: int,
) -> list[dict[str, Any]]:
    """POST a search to a remote peer + return the ``results`` list.

    Returns the raw deserialised dict list (the PackResult shape on
    the receiver, but we don't import that class here to keep the
    fabric layer free of knowledge-layer imports). The caller
    decides whether to wrap them back into PackResult instances or
    consume them as dicts for RRF merging.

    ``peer_addr`` is the network address from ``fabric_nodes.addr``
    (e.g. ``"192.168.1.20:6443"``). We use https with the same
    scheme assumption as FabricBackend (Caddy + TLS) and verify=False
    on the httpx client because peer identity is established via the
    signed envelope + pinned ed25519 fingerprint, not the TLS cert
    chain. See :mod:`augmentum.fabric.pair_client` for the full
    trust-model writeup.

    Raises :class:`RemoteSearchError` on any transport or HTTP
    failure. The caller catches + continues with local-only results.
    """
    target_pack_ids = [p for p in pack_ids if p]
    if not target_pack_ids:
        return []

    payload = {
        "q": query,
        "pack_ids": target_pack_ids,
        "limit": int(limit),
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    # Lazy import to avoid an unconditional cycle between fabric and
    # the fabric-internal middleware module (peer_middleware imports
    # cryptography eagerly; we want our import graph clean).
    from augmentum.fabric.peer_middleware import build_signed_peer_headers

    # Dedicated /api/fabric/knowledge/search endpoint. Searches LOCAL
    # packs only on the receiver — the recursion guard against
    # A→B→C→… fan-out loops the architecture review called out as a
    # HIGH-severity latent risk lives there.
    path = "/api/fabric/knowledge/search"
    headers = {
        "Content-Type": "application/json",
        **build_signed_peer_headers(
            identity=identity, user_id=user_id,
            method="POST", path=path, body=body,
        ),
    }

    scheme = "https" if "://" not in peer_addr else peer_addr.split("://", 1)[0]
    host = peer_addr.split("://", 1)[-1].rstrip("/")
    url = f"{scheme}://{host}{path}"

    try:
        resp = await http_client.post(
            url, content=body, headers=headers,
            timeout=_SEARCH_HTTP_TIMEOUT_S,
        )
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
        log.info(
            "fabric_knowledge_search_unreachable",
            peer_addr=peer_addr, error=str(exc)[:160],
        )
        raise RemoteSearchError(f"peer {peer_addr!r} unreachable: {exc}") from None

    if resp.status_code >= 400:
        log.info(
            "fabric_knowledge_search_status",
            peer_addr=peer_addr, status=resp.status_code,
            body=resp.text[:200],
        )
        raise RemoteSearchError(
            f"peer returned {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise RemoteSearchError(
            f"peer returned non-JSON response: {exc}"
        ) from None

    results = data.get("results") or []
    if not isinstance(results, list):
        raise RemoteSearchError("peer response missing results list")

    log.debug(
        "fabric_knowledge_search_success",
        peer_addr=peer_addr, pack_count=len(target_pack_ids),
        result_count=len(results),
    )
    return results
