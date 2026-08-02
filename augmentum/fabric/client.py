"""FabricClient: outbound WebSocket connections to paired peers.

A background asyncio task running for the lifetime of the process
when fabric is enabled. For each paired peer, opens and maintains a
persistent WebSocket connection, sends 5-second heartbeats, and
reconnects with exponential backoff on failure.

Phase 1 minimum: keep connections alive, exchange ``hello`` +
``heartbeat`` envelopes. Phase 2 will piggyback capability
advertisements on the same heartbeat cadence; Phase 3 will route
inference requests through these connections.

The client is symmetric to the server-side WebSocket endpoint -- a
paired pair of Augmentums will both have their client tasks trying
to connect to the other. One connection wins per direction; the
coordinator's ``attach_connection`` rule (close-old, install-new)
handles the dedupe. Wasteful but harmless; we'll add a
primary-preferred initiator later if it matters.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import TYPE_CHECKING

import httpx
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from augmentum.fabric.capabilities import serialise
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.fabric.protocol import (
    MSG_HEARTBEAT,
    MSG_HELLO,
    FabricEnvelope,
    FabricProtocolError,
    make_hello_payload,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.fabric.coordinator import FabricCoordinator
    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)

# Heartbeat cadence. 5s is what the design plan called out; on LAN
# the bandwidth cost is rounding error. Higher phases may push to
# 1-2s when capability deltas need to propagate fast, but Phase 1
# only carries liveness pings.
_HEARTBEAT_INTERVAL_S = 5.0

# Reconnection backoff. Starts short (peer briefly offline / Caddy
# reloading), caps to 30s (avoid hammering a long-down peer).
_RECONNECT_BACKOFF_INITIAL_S = 1.0
_RECONNECT_BACKOFF_MAX_S = 30.0
_RECONNECT_BACKOFF_FACTOR = 2.0

# Hostname identifier sent in the hello message. Best-effort -- the
# value is informational only (used in the operator dashboard for
# "this peer is desktop") and never affects auth.
_HOSTNAME_FALLBACK = "augmentum"


def _peer_ssl_context() -> ssl.SSLContext:
    """SSL context used for wss:// connections to paired peers.

    Same trust model as ``pair_client.py``'s ``verify=False`` httpx
    client: peer identity is established by a pinned Ed25519 fingerprint
    that the operator exchanged out-of-band, NOT by the TLS certificate
    chain. Self-signed Caddy certs on LAN are the common case — a real
    CA-signed cert for an RFC 1918 address is the exception. Every WS
    envelope is signed by the peer's pinned private key, and
    ``handle_inbound_envelope`` verifies the signature against the
    pinned pubkey before honouring the message. A MITM presenting a
    different TLS cert can intercept bytes but CANNOT forge the
    signature without the private key.

    Cert pinning at the TLS layer is the proper Phase-1+ follow-up
    (record the cert SPKI hash at first connect, refuse to talk to a
    different cert later). Until that lands, ``verify=False`` for both
    HTTPS (pair_client) and WSS (here) is the consistent posture.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# Built once at import. Re-creating it per-connect is wasteful and
# (for ssl.create_default_context) loads the system CA bundle each
# time — fine here since we then disable validation, but still.
_PEER_SSL_CONTEXT = _peer_ssl_context()


def _local_hostname() -> str:
    """Best-effort host string for the hello payload."""
    import socket

    try:
        return socket.gethostname() or _HOSTNAME_FALLBACK
    except Exception:
        return _HOSTNAME_FALLBACK


class FabricClient:
    """Maintains outbound WebSocket connections to each paired peer.

    Lifecycle: ``run()`` is awaited as a long-lived task. It walks the
    coordinator's known peers, spawns one connection task per peer,
    and respawns any task that returns (graceful disconnect or error).
    Calling ``stop()`` cancels every child task and returns.
    """

    def __init__(
        self,
        identity: FabricIdentity,
        coordinator: FabricCoordinator,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._identity = identity
        self._coordinator = coordinator
        self._http_client = http_client  # reserved for Phase 3 dispatch path
        self._tasks: dict[str, asyncio.Task] = {}
        self._stopping = asyncio.Event()
        self._supervisor_task: asyncio.Task | None = None
        self._hostname = _local_hostname()

    async def run(self) -> None:
        """Supervisor loop. Spawn per-peer connection tasks; restart
        any that exit while we're not stopping.

        Exits cleanly when ``stop()`` is called or when the surrounding
        task is cancelled (e.g. lifespan shutdown).
        """
        try:
            while not self._stopping.is_set():
                self._spawn_missing_tasks()
                # Wake every second to pick up newly-paired peers
                # without needing an explicit signal. Cheap polling.
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=1.0)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Cancel every per-peer connection task and wait for them
        to unwind. Safe to call multiple times.
        """
        self._stopping.set()
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # ── Internals ─────────────────────────────────────────────────

    def _spawn_missing_tasks(self) -> None:
        """For each known peer with no active task, launch one."""
        for node_id in self._coordinator.known_peer_ids():
            existing = self._tasks.get(node_id)
            if existing is None or existing.done():
                state = self._coordinator.peer_state(node_id)
                if state is None or not state.paired.addr:
                    # Skip peers with no known address (paired but
                    # never reachable). Operator must edit their
                    # entry in fabric_nodes.addr to give us a target.
                    continue
                self._tasks[node_id] = asyncio.create_task(
                    self._maintain_connection(state.paired),
                    name=f"fabric_client:{node_id[:8]}",
                )

    async def _maintain_connection(self, peer: PairedPeer) -> None:
        """Connect-loop for a single peer. Reconnects on failure with
        exponential backoff. Returns only when the supervisor cancels
        us OR the peer is unregistered.
        """
        backoff = _RECONNECT_BACKOFF_INITIAL_S
        # Circuit-breaker-style logging. A peer that's simply offline (powered
        # down, not yet provisioned, behind a dead Caddy) refuses every connect
        # at the 30s backoff cap — ~2880 failures/day. Logging each at WARNING
        # buries real signal. Warn ONCE on the up->down transition, drop
        # subsequent identical failures to debug, and emit a single INFO when
        # the peer comes back. Net: one warning per outage, not one per retry.
        consecutive_failures = 0
        while not self._stopping.is_set():
            try:
                await self._one_session(peer)
                if consecutive_failures:
                    log.info(
                        "fabric_client_peer_recovered",
                        peer_node_id=peer.node_id,
                        after_failures=consecutive_failures,
                    )
                # Clean disconnect -- reset backoff + breaker for the next retry.
                consecutive_failures = 0
                backoff = _RECONNECT_BACKOFF_INITIAL_S
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_failures += 1
                # First failure (transition into the down state) is worth a
                # warning; while it stays down, demote to debug so a long-dead
                # peer doesn't spam the operator log every backoff window.
                _emit = log.warning if consecutive_failures == 1 else log.debug
                _emit(
                    "fabric_client_session_failed",
                    peer_node_id=peer.node_id,
                    error=str(exc)[:200],
                    consecutive_failures=consecutive_failures,
                )

            # Backoff window. Sleep is interruptible via _stopping.
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=backoff)
            except TimeoutError:
                pass
            backoff = min(_RECONNECT_BACKOFF_MAX_S, backoff * _RECONNECT_BACKOFF_FACTOR)

    async def _one_session(self, peer: PairedPeer) -> None:
        """Open one WS to a peer, send hello, heartbeat in a loop
        until disconnect. Exceptions propagate to the supervisor's
        backoff branch.
        """
        # Address format follows Caddy's reverse-proxy public surface
        # for this peer: scheme is wss (Caddy terminates TLS), path
        # is /api/fabric/connect (the WS endpoint we're about to
        # register). For Phase 1 we trust the operator-supplied addr;
        # cert pinning lives in Phase 1 still-to-do but is sequenced
        # after the basic connect works.
        #
        # SSL context with verify=False for wss:// — same trust model
        # as pair_client.py (peer identity is the pinned Ed25519
        # fingerprint, every envelope signed). Without this, every
        # peer connect attempt logs CERTIFICATE_VERIFY_FAILED forever
        # because no operator has a CA-signed cert for their LAN IP.
        url = _make_ws_url(peer.addr)
        ssl_ctx = _PEER_SSL_CONTEXT if url.startswith("wss://") else None
        async with ws_connect(url, open_timeout=10.0, ssl=ssl_ctx) as ws:
            await self._send_hello(ws)
            await self._heartbeat_loop(ws, peer)

    async def _send_hello(self, ws) -> None:
        """First message on every new connection. Carries the local
        identity for the server to verify against its pinned pubkey.
        """
        envelope = FabricEnvelope.build(
            msg_type=MSG_HELLO,
            seq=self._coordinator.next_local_seq(),
            sender_node_id=self._identity.node_id,
            payload=make_hello_payload(
                hostname=self._hostname,
                public_key_b64=self._identity.public_key_b64,
            ),
            signing_key=self._identity.private_key,
        )
        await ws.send(envelope.to_wire())

    async def _heartbeat_loop(self, ws, peer: PairedPeer) -> None:
        """Send a heartbeat every N seconds; read any inbound traffic
        (acks, errors) until the server disconnects.
        """
        recv_task = asyncio.create_task(
            self._receive_loop(ws, peer),
            name=f"fabric_client_recv:{peer.node_id[:8]}",
        )
        try:
            while not self._stopping.is_set():
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                # Phase 2: refresh + advertise local capabilities. Errors
                # in extractors are absorbed inside build_local_capabilities
                # itself, so a broken extractor degrades to "smaller
                # capability list this tick" rather than blocking
                # heartbeats entirely.
                caps = await self._coordinator.build_local_capabilities()
                envelope = FabricEnvelope.build(
                    msg_type=MSG_HEARTBEAT,
                    seq=self._coordinator.next_local_seq(),
                    sender_node_id=self._identity.node_id,
                    payload={"capabilities": [serialise(c) for c in caps]},
                    signing_key=self._identity.private_key,
                )
                try:
                    await ws.send(envelope.to_wire())
                except ConnectionClosed:
                    return
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _receive_loop(self, ws, peer: PairedPeer) -> None:
        """Read inbound frames from the server side of this WS.

        From Phase 9.4, the server side (the other peer) can push
        messages back through this socket — cancellation envelopes,
        lifecycle events, etc. We delegate dispatch to the
        coordinator's shared handle_inbound_envelope so the logic
        matches the OTHER read loop in fabric_routes.fabric_connect.
        Symmetric topology means either socket can carry any type.
        """
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    envelope = FabricEnvelope.from_wire(
                        raw, expected_sender_pubkey_b64=peer.pubkey_b64,
                    )
                except FabricProtocolError as exc:
                    log.warning(
                        "fabric_client_bad_envelope",
                        peer_node_id=peer.node_id,
                        error=str(exc)[:160],
                    )
                    continue
                self._coordinator.handle_inbound_envelope(envelope)
        except ConnectionClosed:
            return
        except asyncio.CancelledError:
            raise


def _make_ws_url(addr: str) -> str:
    """Translate a paired-peer addr into the wss URL of the fabric WS.

    Accepts:
      - "example.com:6443"           -> wss://example.com:6443/api/fabric/connect
      - "https://example.com:6443"   -> wss://example.com:6443/api/fabric/connect
      - "wss://example.com:6443/x"   -> as-is (operator override)
    """
    s = addr.strip()
    if s.startswith("wss://") or s.startswith("ws://"):
        # Trust full URL forms verbatim. Operator opted in to the path.
        return s
    if s.startswith("https://"):
        s = "wss://" + s[len("https://"):]
    elif s.startswith("http://"):
        s = "ws://" + s[len("http://"):]
    else:
        s = "wss://" + s
    # Append the canonical path if the operator only supplied host:port.
    if "/" not in s.split("://", 1)[1]:
        s = s.rstrip("/") + "/api/fabric/connect"
    return s
