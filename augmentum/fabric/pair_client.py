"""Outbound pair client.

**Trust model (applies to every peer-to-peer HTTP call in fabric/):**

Peer identity is established by a pinned ed25519 fingerprint that the
operator exchanges out-of-band (read it aloud, share a screenshot),
NOT by the TLS certificate chain. Self-signed Caddy certs on LAN are
the common case — a real CA-signed cert for an RFC 1918 address is
the exception, not the rule. So we pass ``verify=False`` to httpx
deliberately.

This is safe because:

  1. Every outbound peer request carries a signed envelope (see
     :mod:`augmentum.fabric.peer_middleware`) that the receiver
     verifies against the pinned pubkey before honouring it.
  2. Every inbound peer request is verified the same way before
     :class:`FabricPeerMiddleware` populates ``scope["user"]``.
  3. A MITM presenting a different TLS cert can intercept bytes but
     CANNOT forge the ed25519 signature without the private key. Body
     integrity (Phase 3.y) covers the body content too, so a MITM
     also can't swap the body bytes mid-flight.

What we'd lose by setting verify=True: pairing across self-signed
LAN edges would require either a CA bundle ship + per-peer cert
import, or operators routinely pasting ``-k`` equivalents. The
fingerprint flow already proves identity at the application layer;
forcing TLS-layer proof on top would be ceremony without security
gain.

What we'd lose by setting verify=False AND skipping the signed
envelope: anyone on the network path could impersonate a peer. The
signed envelope is what makes verify=False acceptable — not laziness.

----

The operator-driven *initiator* side of the pair handshake. Pairs to
the *inbound* side handled in :mod:`augmentum.proxy.fabric_routes`
(``POST /api/fabric/pair``):

  Local operator opens the Fabric tab in Model Manager, pastes the
  remote peer's URL + fingerprint, clicks Pair. The UI POSTs to our
  ``/api/fabric/pair-with-remote`` route, which:

  1. Asks ``build_pair_request`` (already in peer_auth.py) to produce
     a signed ``PairRequest`` from our local identity.
  2. POSTs the serialised dataclass to the remote's
     ``/api/fabric/pair`` endpoint over HTTPS.
  3. Confirms the response identifies the remote at the fingerprint
     the operator typed (defensive cross-check -- the remote would
     have failed on its side if the operator pasted the wrong one,
     but a hostile remote could return a 200 with a different
     ``this_node.fingerprint`` to attempt a confused-deputy swap).
  4. Persists the remote into our ``fabric_nodes`` table via
     ``persist_remote_node`` so we have its identity for later
     signature verification on inbound peer requests.

What we deliberately do NOT do here:

  - Open a WebSocket. The reconnect supervisor in ``client.py`` does
    that as part of its periodic loop; once the new ``fabric_nodes``
    row exists the next supervisor pass picks it up.
  - Mutate any in-RAM coordinator state directly. The caller route
    handler does ``coordinator.register_paired_peer`` after we
    return, mirroring what the inbound /pair endpoint does.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import httpx

from augmentum.fabric.peer_auth import (
    build_pair_request,
    persist_remote_node,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

    from augmentum.fabric.identity import FabricIdentity
    from augmentum.fabric.peer_auth import PairedPeer

log = get_logger(__name__)


# Outbound pair POST has a short HTTP timeout. The receiver writes one
# DB row + does an ed25519 verify -- both sub-millisecond. Anything
# more than this is a stalled network path; surface it as an error
# rather than holding the operator-facing UI hanging.
_PAIR_HTTP_TIMEOUT_S = 10.0


class OutboundPairError(Exception):
    """Any failure during the outbound pair handshake. Mapped to a
    user-facing error in the route layer. Message is operator-facing
    -- it should make sense to a human reading the Fabric tab in the UI.
    """


async def initiate_pair_with_remote(
    *,
    identity: FabricIdentity,
    hostname: str,
    remote_url: str,
    expected_fingerprint: str,
    remote_addr: str,
    own_addr: str = "",
    role: str = "peer",
    icon: str = "",
    db: aiosqlite.Connection,
    http_client: httpx.AsyncClient | None = None,
) -> PairedPeer:
    """Run one half of the pair handshake from THIS node out to the remote.

    On success, the remote has us in their fabric_nodes AND we have
    them in ours. Both sides are now ready for the inter-peer
    WebSocket; the reconnect supervisor in ``client.py`` will open
    that connection on its next pass.

    ``remote_addr`` is the address WE will reach the remote at (typed by
    the operator into the pair UI). ``own_addr`` is the address the
    remote will reach US at — derived by the route handler from the
    operator's local request context (Host header on the pair-with-
    remote call, plus the canonical Caddy port 6443). The route is the
    correct authority for this because the pair_client module has no
    request-side context.

    Raises :class:`OutboundPairError` on any failure. The exception
    message is intended to be shown to the operator (e.g. "remote
    unreachable", "fingerprint mismatch — got X expected Y").
    """
    # 1) Build a signed request from our local identity. The target
    # fingerprint hint is what the operator typed in the UI; the
    # receiver will reject if it doesn't match THEIR own fingerprint.
    pair_request = build_pair_request(
        identity=identity, hostname=hostname,
        target_fingerprint_hint=expected_fingerprint, role=role,
    )

    # 2) POST the request to the remote /api/fabric/pair endpoint. We
    # use the existing dataclass serialisation rather than a fresh
    # schema so any future field additions in PairRequest flow through
    # both sides without a wire-format split.
    #
    # The ``addr`` field is THIS node's accessible address (so the
    # remote can connect back to us). Pre-fix the code shipped
    # ``remote_addr`` here — i.e. the REMOTE's address — so the
    # remote stored ITS OWN ip:port against OUR node_id. Result: the
    # remote's outbound supervisor tried to connect to itself and the
    # back-channel never came up, manifesting as "peer offline" on the
    # remote even though our outbound was healthy. Asymmetric WS bug.
    payload = dataclasses.asdict(pair_request)
    if own_addr:
        payload["addr"] = own_addr

    pair_endpoint = remote_url.rstrip("/") + "/api/fabric/pair"
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(verify=False)
    try:
        try:
            resp = await http_client.post(
                pair_endpoint, json=payload, timeout=_PAIR_HTTP_TIMEOUT_S,
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
            # str(exc) is often empty for httpx ConnectError on a closed
            # port — that produced "remote '...' unreachable:" with no
            # explanation. Always include the exception class name as a
            # minimum hint so the operator gets ConnectError / ReadError /
            # TimeoutException to grep for.
            exc_class = type(exc).__name__
            exc_detail = str(exc) or "no error message (likely connection refused or TLS handshake closed)"
            log.info(
                "fabric_outbound_pair_unreachable",
                remote=remote_url, exc_class=exc_class, error=exc_detail[:200],
            )
            raise OutboundPairError(
                f"remote {remote_url!r} unreachable: {exc_class}: {exc_detail}"
            ) from None

        # 3) Validate response. The remote returns 4xx with a JSON
        # ``detail`` message when verify_pair_request raised; surface
        # that text to the operator unchanged so they see WHY (e.g.
        # "fingerprint mismatch: the requesting peer thinks they are
        # pairing with a different node…").
        if resp.status_code >= 400:
            detail = _extract_error_detail(resp)
            log.info(
                "fabric_outbound_pair_rejected",
                remote=remote_url, status=resp.status_code, detail=detail[:200],
            )
            raise OutboundPairError(
                f"remote rejected pair ({resp.status_code}): {detail}"
            )

        try:
            body = resp.json()
        except Exception as exc:
            raise OutboundPairError(
                f"remote returned non-JSON response: {exc}"
            ) from None

        this_node = body.get("this_node") or {}
        remote_node_id = str(this_node.get("node_id", "") or "")
        remote_pubkey = str(this_node.get("public_key", "") or "")
        remote_fp = str(this_node.get("fingerprint", "") or "")
        if not remote_node_id or not remote_pubkey or not remote_fp:
            raise OutboundPairError(
                "remote response missing this_node identity fields"
            )

        # 4) Defensive cross-check: the remote MUST identify itself
        # with the same fingerprint the operator typed. If it doesn't,
        # something is wrong (hostile remote, MITM, or the operator
        # pasted a different node's URL and the SAME node's fingerprint
        # by mistake). Either way, refuse to persist a mismatched pair.
        if remote_fp != expected_fingerprint:
            raise OutboundPairError(
                f"fingerprint mismatch in remote response: "
                f"got {remote_fp!r} expected {expected_fingerprint!r}. "
                f"Verify you pasted the right node's fingerprint."
            )

        # 5) Persist remote in our fabric_nodes. We don't have a
        # PairRequest for them (only their identity from the response);
        # the helper handles that path.
        remote_hostname = ""  # not in response payload by design; we
        # don't trust remote-supplied hostnames anyway. Falls back to
        # node_id in the UI peer list when empty.
        paired = await persist_remote_node(
            db, node_id=remote_node_id, hostname=remote_hostname,
            role=str(body.get("paired", {}).get("role", "peer") or "peer"),
            pubkey_b64=remote_pubkey, addr=remote_addr, icon=icon,
        )
        log.info(
            "fabric_outbound_pair_success",
            remote_node_id=remote_node_id, addr=remote_addr,
        )
        return paired
    finally:
        if own_client:
            await http_client.aclose()


def _extract_error_detail(resp: httpx.Response) -> str:
    """Best-effort: pull the ``detail`` field from a JSON error body,
    or fall back to the raw response text. Strings only -- nested
    structures aren't useful to surface to the operator.
    """
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str) and detail:
                return detail
    except Exception:
        log.debug("fabric_pair_error_parse_failed", exc_info=True)
    return resp.text[:300] if resp.text else f"HTTP {resp.status_code}"
