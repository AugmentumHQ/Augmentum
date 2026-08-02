"""FabricCoordinator: in-memory hub for peer connections + state.

Lives on ``app.state.fabric_coordinator`` when fabric is enabled.
Owns the runtime registry of known peers (loaded from fabric_nodes
at startup, refreshed on pair) AND the live WebSocket connections
to each connected peer. Everything in this module is RAM-only; the
SQLite layer holds durable identity only, and is touched here only
for the startup load and on pair events.

Phase 1 scope is connectivity-only -- no capability advertisement
(Phase 2) and no routing (Phase 3). When two peers are connected,
they exchange ``hello`` + ``heartbeat`` envelopes and the coordinator
tracks (node_id, connected, last_seen). Higher phases will extend
``PeerLiveState`` with capability + load fields.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.fabric.capabilities import CapabilityBase, deserialise_list
from augmentum.fabric.peer_auth import PairedPeer, load_paired_peers
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite
    from fastapi import WebSocket

    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)


@dataclass
class PeerLiveState:
    """Live, RAM-only state for a single peer.

    Populated by the WS handlers as messages arrive. Phase 1 added
    connection tracking; Phase 2 adds ``capabilities`` (what this peer
    advertises it can do, refreshed on every heartbeat). Phase 3 will
    add ``active_requests`` / ``queue_depth`` / ``recent_tps`` for
    load-aware routing.
    """

    paired: PairedPeer
    connected: bool = False
    last_seen_monotonic: float = 0.0
    last_seq_received: int = -1  # -1 = no message received yet
    # Capabilities the peer most recently advertised. Replaced wholesale
    # on every capability-bearing heartbeat -- additive merging would
    # leave stale entries if a peer unloaded a model. Empty list while
    # we haven't heard a heartbeat yet OR the peer advertises nothing.
    capabilities: list[CapabilityBase] = field(default_factory=list)
    # The active inbound or outbound WebSocket. ``None`` when the
    # peer is registered but no transport is up. We hold a reference
    # so broadcast/dispatch code can find the right socket without
    # walking a separate connection map.
    socket: "WebSocket | None" = field(default=None, repr=False)


class FabricCoordinator:
    """Hub of in-process fabric state.

    Initialised once at lifespan start when ``settings.fabric_enabled``
    is True. Holds:

      - ``_peers``: dict[node_id, PeerLiveState] -- all known peers,
        connected or not. Seeded at startup from fabric_nodes.
      - ``_local_seq``: monotonic counter for envelopes we send out.

    Reads are lock-free; writes take ``_lock`` for the dict mutations
    only (the dict mutation itself is a single bytecode op in CPython,
    but the lock is held across attach/detach to keep the
    "registered + connected" invariants consistent).
    """

    # Heartbeat timeout sweep — a peer whose ``last_seen_monotonic``
    # is more than this many seconds old gets auto-detached even if
    # its WS socket is still nominally open. Closes the "hung peer
    # with open socket" failure mode the architecture review flagged
    # as a Tier 0 time bomb: pre-fix, a peer whose process hard-froze
    # would stay "Connected" in the UI indefinitely and the router
    # would keep trying to dispatch to it.
    #
    # 15s = 3× the 5s heartbeat cadence in client.py. Leaves headroom
    # for one missed heartbeat (network blip) without flapping
    # connections, while still catching real hangs within a routing
    # decision window.
    HEARTBEAT_TIMEOUT_S = 15.0

    # How often the sweeper scans. 5s = same cadence as the heartbeat
    # itself; a fresh heartbeat resets the per-peer clock and the
    # next sweep sees them as live again.
    HEARTBEAT_SWEEP_INTERVAL_S = 5.0

    def __init__(self, identity: "FabricIdentity", db: "aiosqlite.Connection") -> None:
        self._identity = identity
        self._db = db
        self._peers: dict[str, PeerLiveState] = {}
        self._local_seq: int = 0
        self._lock = asyncio.Lock()
        # Background sweeper that detaches peers whose last-heartbeat
        # is stale beyond HEARTBEAT_TIMEOUT_S. Started by
        # ``start_heartbeat_sweeper()`` from lifespan; stopped on
        # shutdown. None when never started or when explicitly stopped.
        self._heartbeat_sweep_task: "asyncio.Task | None" = None
        # Capability extractors registered by lifespan startup. Each
        # exposes async collect() -> list[CapabilityBase]. We don't
        # type-narrow on the extractor class because the discipline is
        # "anything with .collect()" -- keeps tests simple.
        self._extractors: list[Any] = []
        # Cached snapshot of this node's own capabilities, refreshed
        # whenever we send a heartbeat. Stored so the read API can
        # serve "what does THIS node advertise" without re-running
        # extractors on every request.
        self._local_capabilities: list[CapabilityBase] = []
        # Phase 9.3: in-flight cross-peer request registry. Maps
        # X-Fabric-Request-Id → asyncio.Task running the handler, so
        # an inbound MSG_CANCEL_REQUEST envelope can cancel the right
        # in-flight task. Plain dict; the middleware (sole writer)
        # registers + deregisters under request-task-locality, no
        # lock needed.
        self._inflight_peer_requests: dict[str, Any] = {}
        # Phase 10 — per-(peer, kind) latency EMA. Updated by
        # FabricBackend after each successful proxy call; consumed
        # by RoutingDirector's scoring function. Bounded memory: a
        # single float per pair. Missing entries default to a
        # neutral "unknown" sentinel (-1.0) so the scorer can avoid
        # penalising un-measured peers on first request.
        self._peer_latency_ms_ema: dict[tuple[str, str], float] = {}
        # Cross-peer state transparency: model-load + prefill progress
        # snapshots a peer reports while serving an outbound request we
        # made. Keyed by model_id; each value is the already-wire-shaped
        # payload (from models/load_progress.py) plus a monotonic
        # ``recorded_at`` so reads can expire stale entries. Populated by
        # FabricBackend's load-poll loop; surfaced through the local
        # /api/engine/v2/load_progress + /prefill_progress endpoints so
        # the existing UI poller renders a cross-peer load identically to
        # a local one. Bounded: one entry per distinct remote model.
        self._peer_load_progress: dict[str, dict] = {}
        self._peer_prefill_progress: dict[str, dict] = {}
        # Phase 9-lifecycle: per-request lifecycle event log.
        # Populated when this node is the ORIGINATOR (we sent a
        # request to a peer; the peer sends job_started / completed /
        # failed events back). The chat UI + dispatch layer can poll
        # this to render "peer-X is working on your request" + final
        # status. Capped at last-N events per request_id to bound
        # memory; expired entries cleaned when the request completes.
        self._peer_call_events: dict[str, list[dict]] = {}
        # Wedge B — Connect over fabric. Set by lifespan after the
        # ConnectHub + NotificationHub are wired so inbound
        # MSG_CONNECT_ENVELOPE frames can re-route to the right local
        # user's signaling WS + drop a notification. None while we're
        # still booting or when Connect is disabled.
        self.connect_hub: Any = None
        self.notification_hub: Any = None

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialise(self) -> None:
        """Load paired peers from SQLite into ``_peers``. Called once
        at lifespan startup AFTER the identity is loaded and the DB
        connection is available.
        """
        rows = await load_paired_peers(self._db)
        async with self._lock:
            for row in rows:
                # Don't overwrite an existing entry (would clobber a
                # live ``socket`` reference if initialise() is called
                # twice for any reason).
                if row.node_id not in self._peers:
                    self._peers[row.node_id] = PeerLiveState(paired=row)
        log.info(
            "fabric_coordinator_initialised",
            paired_peer_count=len(self._peers),
            local_node_id=self._identity.node_id,
        )

    async def shutdown(self) -> None:
        """Close every active WebSocket. Called at lifespan shutdown.

        Tolerant of already-closed sockets -- we may be shutting down
        because the network died, in which case ``close()`` is going
        to raise. We log and continue so a stuck socket doesn't block
        process exit.
        """
        # Cancel the heartbeat sweeper first so it doesn't fight us
        # for the lock during close.
        self.stop_heartbeat_sweeper()

        async with self._lock:
            sockets_to_close = [
                (node_id, state.socket)
                for node_id, state in self._peers.items()
                if state.connected and state.socket is not None
            ]
            for state in self._peers.values():
                state.connected = False
                state.socket = None

        for node_id, sock in sockets_to_close:
            try:
                await sock.close()
            except Exception:
                log.debug("fabric_shutdown_close_failed", peer_node_id=node_id, exc_info=True)

    # ── Heartbeat timeout sweeper ─────────────────────────────────

    def start_heartbeat_sweeper(self) -> None:
        """Start the background task that auto-detaches stale peers.

        Called once from lifespan after ``initialise()``. Idempotent —
        repeated calls are no-ops while the task is running. Cancelled
        + restarted on subsequent calls if the previous task crashed
        (defensive against a fault in the sweeper loop itself).
        """
        if self._heartbeat_sweep_task is not None and not self._heartbeat_sweep_task.done():
            return
        self._heartbeat_sweep_task = asyncio.create_task(
            self._heartbeat_sweep_loop(),
            name="fabric_heartbeat_sweeper",
        )

    def stop_heartbeat_sweeper(self) -> None:
        """Cancel the sweeper task. Idempotent."""
        task = self._heartbeat_sweep_task
        self._heartbeat_sweep_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _heartbeat_sweep_loop(self) -> None:
        """Periodically detach peers whose last_seen is stale.

        Runs forever until cancelled. Errors in a single sweep are
        absorbed (logged + continue) so a transient fault doesn't
        kill the sweeper — leaving the system worse than if we'd
        never started it.
        """
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_SWEEP_INTERVAL_S)
                try:
                    await self._sweep_stale_peers_once()
                except Exception:
                    log.warning(
                        "fabric_heartbeat_sweep_iter_failed",
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            log.debug("fabric_heartbeat_sweep_cancelled")
            raise

    async def _sweep_stale_peers_once(self) -> None:
        """One pass of the stale-peer sweep. Extracted so tests can
        call it deterministically without waiting for the interval.

        For each currently-connected peer whose ``last_seen_monotonic``
        is more than ``HEARTBEAT_TIMEOUT_S`` ago, calls
        ``detach_connection()`` to flip them offline + clear
        capabilities + write durable last_seen_at. Subsequent dispatch
        attempts skip them; reconnect via the client supervisor pulls
        them back live.
        """
        now = time.monotonic()
        cutoff = now - self.HEARTBEAT_TIMEOUT_S
        # Snapshot stale peers + their sockets under the lock. We need
        # the socket reference NOW because detach_connection clears
        # state.socket inside its own lock — by the time we call
        # close() after detach_connection returns, the reference is
        # gone.
        stale: list[tuple[str, float, Any]] = []
        async with self._lock:
            for node_id, state in self._peers.items():
                if not state.connected:
                    continue
                # last_seen_monotonic == 0.0 is the "never seen" sentinel
                # — treat as stale only if we've been connected long
                # enough to expect at least one heartbeat. attach
                # sets last_seen, so 0.0 here means a programming bug.
                if state.last_seen_monotonic == 0.0:
                    continue
                if state.last_seen_monotonic < cutoff:
                    stale.append((node_id, state.last_seen_monotonic, state.socket))

        for node_id, last_seen, socket in stale:
            log.warning(
                "fabric_peer_heartbeat_timeout",
                peer_node_id=node_id,
                stale_for_s=round(now - last_seen, 1),
                timeout_s=self.HEARTBEAT_TIMEOUT_S,
            )
            # detach_connection acquires its own lock — must be called
            # outside the snapshot-block lock above. It flips
            # connected→False, clears capabilities, writes durable
            # last_seen_at, but does NOT close the socket (the
            # caller is responsible).
            await self.detach_connection(node_id)
            # Close the captured socket so the WS read loop on this
            # side unblocks (it's still blocked in iter_text() waiting
            # for the next frame from the hung peer). The peer's own
            # supervisor will see the close and reconnect on a fresh
            # socket — at which point heartbeats resume and the
            # router routes to them again.
            if socket is not None:
                try:
                    await socket.close()
                except Exception:
                    log.debug(
                        "fabric_stale_peer_close_failed",
                        peer_node_id=node_id, exc_info=True,
                    )

    # ── Pairing integration ───────────────────────────────────────

    async def register_paired_peer(self, peer: PairedPeer) -> None:
        """Add or refresh a peer after a successful pair handshake.

        Called by the routes layer once the SQLite write has landed.
        If the peer is already known and currently connected, we keep
        the live socket; only the identity fields refresh.
        """
        async with self._lock:
            existing = self._peers.get(peer.node_id)
            if existing is not None:
                # Refresh paired metadata; keep connection state.
                existing.paired = peer
            else:
                self._peers[peer.node_id] = PeerLiveState(paired=peer)
        log.info("fabric_peer_registered", node_id=peer.node_id, role=peer.role)

    async def unregister_peer(self, node_id: str) -> None:
        """Remove a peer from the registry. Closes any active socket.

        Called when an operator un-pairs a peer in the UI (Phase 4).
        """
        async with self._lock:
            state = self._peers.pop(node_id, None)
        if state is None:
            return
        if state.connected and state.socket is not None:
            try:
                await state.socket.close()
            except Exception:
                log.debug(
                    "fabric_unregister_close_failed",
                    peer_node_id=node_id,
                    exc_info=True,
                )

    # ── Connection management ─────────────────────────────────────

    async def attach_connection(self, node_id: str, websocket: "WebSocket") -> bool:
        """Bind a freshly-authenticated WebSocket to a peer.

        Returns True on success; False if the node_id isn't registered
        (caller should close with 4401). When a peer reconnects while
        we still think we have a live socket for them, the old one is
        closed politely first.
        """
        async with self._lock:
            state = self._peers.get(node_id)
            if state is None:
                return False
            old_socket = state.socket if state.connected else None
            state.socket = websocket
            state.connected = True
            state.last_seen_monotonic = time.monotonic()

        if old_socket is not None:
            try:
                await old_socket.close()
            except Exception:
                log.debug(
                    "fabric_replace_close_failed",
                    peer_node_id=node_id,
                    exc_info=True,
                )

        log.info("fabric_peer_connected", peer_node_id=node_id)

        # Wedge B Phase 5: drain any queued Connect envelopes destined
        # for this peer. Fire-and-forget — we don't want to block the
        # connection attach path on the drain, and the drain handles
        # its own per-row errors. No-op when no rows are queued.
        try:
            from augmentum.connect.fabric_transport import drain_outbox_for_peer
            asyncio.create_task(
                drain_outbox_for_peer(
                    self._db, coordinator=self, node_id=node_id,
                ),
            )
        except Exception as exc:
            log.warning(
                "fabric_connect_outbox_drain_schedule_failed",
                peer_node_id=node_id, error=str(exc)[:160],
            )

        return True

    async def detach_connection(self, node_id: str) -> None:
        """Mark a peer offline + clear ephemeral state.

        Updates touch three things:

          1. ``state.connected`` flips to False; ``state.socket`` is
             cleared. Subsequent :meth:`find_peers_with_capability`
             calls (connected_only=True by default) skip this peer
             until reconnect.
          2. ``state.capabilities`` is cleared. Stale capabilities for
             an offline peer have no consumer that benefits from
             keeping them (the router already filters by connection;
             the UI matrix shouldn't show "loaded · 4 slots" for a
             peer that's gone). Clearing here means a reconnect MUST
             advertise fresh capabilities before they re-appear --
             which is exactly what heartbeats are for.
          3. ``fabric_nodes.last_seen_at`` is written to the ISO
             timestamp of THIS detach. Best signal of "when was the
             peer last reachable" available without per-second SQL
             churn. UI peer rows surface this so operators can spot
             a peer that's been offline for hours.

        The SQL update is logged but never raised — a write failure
        shouldn't prevent in-memory cleanup. Worst case the UI shows
        a stale last_seen_at; the more important state (connected
        flag, capabilities) is already correct in RAM.
        """
        async with self._lock:
            state = self._peers.get(node_id)
            if state is None:
                return
            state.connected = False
            state.socket = None
            # Drop stale advertisements. A reconnect will repopulate.
            state.capabilities = []

        # Stamp last_seen_at in the durable peer row. Outside the lock
        # because the DB has its own concurrency model + busy_timeout;
        # holding the asyncio lock across the SQL would serialise
        # disconnect events unnecessarily.
        from datetime import datetime, timezone
        import dataclasses
        now_iso = datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")
        try:
            await self._db.execute(
                "UPDATE fabric_nodes SET last_seen_at = ? WHERE id = ?",
                (now_iso, node_id),
            )
            await self._db.commit()
        except Exception:
            log.warning(
                "fabric_peer_detach_last_seen_write_failed",
                peer_node_id=node_id, exc_info=True,
            )

        # Reflect the update in our in-memory PairedPeer (frozen
        # dataclass → replace). Without this the /peers route would
        # keep returning the pre-update value until the coordinator
        # next reloads from disk.
        async with self._lock:
            state = self._peers.get(node_id)
            if state is not None and state.paired is not None:
                state.paired = dataclasses.replace(
                    state.paired, last_seen_at=now_iso,
                )

        log.info("fabric_peer_disconnected", peer_node_id=node_id)

    # ── Read API (lock-free; safe to call from any task) ──────────

    def peer_state(self, node_id: str) -> PeerLiveState | None:
        """Look up a peer by node_id. None when unknown."""
        return self._peers.get(node_id)

    def known_peer_ids(self) -> list[str]:
        """Snapshot of registered peer node_ids (paired, not necessarily connected)."""
        return list(self._peers.keys())

    def connected_peer_ids(self) -> list[str]:
        """Snapshot of currently-connected peer node_ids."""
        return [n for n, s in self._peers.items() if s.connected]

    def peer_count(self) -> dict[str, int]:
        """Summary counts. Useful for the operator dashboard later."""
        total = len(self._peers)
        connected = sum(1 for s in self._peers.values() if s.connected)
        return {"total": total, "connected": connected, "offline": total - connected}

    def next_local_seq(self) -> int:
        """Increment + return our outbound sequence number. Used when
        the coordinator builds envelopes to send to peers. Wraps at
        2**31-1 (a fabric process is unlikely to reach that, but the
        guard is cheap).
        """
        self._local_seq = (self._local_seq + 1) % (2**31 - 1)
        return self._local_seq

    # ── Heartbeat bookkeeping ─────────────────────────────────────

    def record_heartbeat(self, node_id: str, seq: int) -> None:
        """Update last-seen + last-seq for a peer on incoming msg.

        Called from the WS read loop on every successfully-verified
        envelope (not just heartbeats). Updates the in-memory state
        for the routing/dashboard layers to consume.
        """
        state = self._peers.get(node_id)
        if state is None:
            return
        state.last_seen_monotonic = time.monotonic()
        if seq > state.last_seq_received:
            state.last_seq_received = seq

    # ── Capability advertisement ──────────────────────────────────

    def register_extractor(self, extractor: Any) -> None:
        """Attach a capability extractor. Called by lifespan startup
        with one extractor per kind (LLM, image, knowledge, ...).

        The contract: ``extractor.collect()`` is async and returns
        ``list[CapabilityBase]``. Failures inside an extractor are
        absorbed at collect time -- a broken extractor must not
        prevent other capabilities from being advertised.
        """
        self._extractors.append(extractor)

    async def build_local_capabilities(self) -> list[CapabilityBase]:
        """Run all extractors + return the merged capability list.

        Caches the result on ``_local_capabilities`` for the read API
        + heartbeat consumers. Errors in individual extractors are
        logged and skipped, never raised.
        """
        merged: list[CapabilityBase] = []
        for extractor in self._extractors:
            try:
                caps = await extractor.collect()
                merged.extend(caps)
            except Exception:
                log.warning(
                    "fabric_extractor_collect_failed",
                    extractor=type(extractor).__name__,
                    exc_info=True,
                )
        self._local_capabilities = merged
        return merged

    def local_capabilities(self) -> list[CapabilityBase]:
        """Most-recent cached snapshot of this node's capabilities.

        Synchronous read for the routes layer. Call
        ``build_local_capabilities()`` first to populate; subsequent
        calls return the cached list until the next heartbeat refresh.
        """
        return list(self._local_capabilities)

    def invalidate_peer_capability(
        self,
        node_id: str,
        *,
        kind: str,
        model_id: str = "",
        pack_id: str = "",
    ) -> bool:
        """Drop a single stale capability advertisement from a peer.

        Called when a peer's actual response contradicts its
        heartbeat — usually a 404 "model not found" on a model the
        peer advertised. The matching capability is removed from the
        peer's in-memory list so the next dispatch decision doesn't
        re-pick the same dead match. The peer's NEXT heartbeat
        refreshes the list authoritatively, so this is self-healing.

        Returns True when a matching capability was found + dropped,
        False when nothing matched (already gone, race with a fresh
        heartbeat, etc.). Either outcome is safe; the caller doesn't
        need to retry.

        Match rules: ``kind`` is always required; ``model_id`` and
        ``pack_id`` are kind-specific (LLM + image use model_id,
        knowledge uses pack_id). Empty string matches "any".
        """
        state = self._peers.get(node_id)
        if state is None:
            return False
        before = len(state.capabilities)
        new_caps = []
        for cap in state.capabilities:
            if cap.kind != kind:
                new_caps.append(cap)
                continue
            if model_id and getattr(cap, "model_id", "") != model_id:
                new_caps.append(cap)
                continue
            if pack_id and getattr(cap, "pack_id", "") != pack_id:
                new_caps.append(cap)
                continue
            # Match — drop.
            log.info(
                "fabric_capability_invalidated",
                peer_node_id=node_id, kind=kind,
                model_id=model_id, pack_id=pack_id,
            )
        # Rebuild keeping only the survivors.
        state.capabilities = [
            c for c in state.capabilities
            if not (
                c.kind == kind
                and (not model_id or getattr(c, "model_id", "") == model_id)
                and (not pack_id or getattr(c, "pack_id", "") == pack_id)
            )
        ]
        return len(state.capabilities) < before

    def record_remote_capabilities(
        self,
        node_id: str,
        capabilities_payload: list[Any],
    ) -> None:
        """Store the capability list a peer just advertised.

        Replaces the prior list wholesale. Called from the WS read
        loop when a heartbeat carries a ``capabilities`` field.
        Unknown kinds in the payload are silently dropped (forward
        compat -- the peer may be running a newer Augmentum).
        """
        state = self._peers.get(node_id)
        if state is None:
            return
        state.capabilities = deserialise_list(capabilities_payload)

    # ── Routing-helper read API (used in Phase 3) ─────────────────

    def find_peers_with_capability(
        self,
        kind: str,
        *,
        connected_only: bool = True,
    ) -> list[tuple[str, CapabilityBase]]:
        """Return (node_id, capability) tuples for every peer advertising
        the given kind. Phase 3 will compose this with scoring rules.
        """
        out: list[tuple[str, CapabilityBase]] = []
        for node_id, state in self._peers.items():
            if connected_only and not state.connected:
                continue
            for cap in state.capabilities:
                if cap.kind == kind:
                    out.append((node_id, cap))
        return out

    def capability_summary(self) -> dict[str, int]:
        """Count of advertised capabilities by kind across all peers
        (including local). Cheap to compute; useful for the operator
        dashboard.
        """
        counts: dict[str, int] = {}
        for cap in self._local_capabilities:
            counts[cap.kind] = counts.get(cap.kind, 0) + 1
        for state in self._peers.values():
            for cap in state.capabilities:
                counts[cap.kind] = counts.get(cap.kind, 0) + 1
        return counts

    # ── Per-peer latency tracking (Phase 10) ──────────────────────

    _LATENCY_EMA_ALPHA = 0.3  # smoothing factor; 0.3 weights ~3 recent calls heavily

    def record_peer_latency(
        self, node_id: str, *, kind: str, latency_ms: float,
    ) -> None:
        """Update the EMA of observed latency for a peer/kind pair.

        Called by FabricBackend (LLM) / image_client / knowledge_client
        right after a successful proxy call. Bounded memory; a single
        float per (peer, kind). Errors / cancellations don't update;
        only successes contribute to the rolling mean.
        """
        if latency_ms < 0 or not node_id or not kind:
            return
        key = (node_id, kind)
        prev = self._peer_latency_ms_ema.get(key)
        if prev is None:
            self._peer_latency_ms_ema[key] = latency_ms
            return
        alpha = self._LATENCY_EMA_ALPHA
        self._peer_latency_ms_ema[key] = (
            alpha * latency_ms + (1.0 - alpha) * prev
        )

    def peer_latency_ms(self, node_id: str, kind: str) -> float | None:
        """Return the EMA latency for a peer/kind pair, or None when
        we've never measured it (first request — scorer should treat
        as neutral, not bad).
        """
        return self._peer_latency_ms_ema.get((node_id, kind))

    # ── Cross-peer progress transparency ──────────────────────────
    #
    # How long a cached peer-progress snapshot is considered live. Past
    # this, a read returns None so the UI bar clears instead of freezing
    # at the last-seen value (e.g. the peer finished loading, or the
    # poll loop stopped). 8s mirrors the local prefill staleness ceiling
    # and comfortably exceeds the FabricBackend load-poll cadence (250ms).
    _PEER_PROGRESS_STALE_AFTER_S = 8.0

    def record_peer_load_progress(self, model_id: str, payload: dict) -> None:
        """Store a peer's model-load progress snapshot (already in the
        wire shape from ``build_load_progress_payload``). Keyed by
        model_id so concurrent loads of different models on different
        peers don't clobber each other. No-op on empty inputs.
        """
        if not model_id or not isinstance(payload, dict):
            return
        self._peer_load_progress[model_id] = {
            **payload,
            "recorded_at": time.monotonic(),
        }

    def record_peer_prefill_progress(self, model_id: str, payload: dict) -> None:
        """Store a peer's prefill progress snapshot (wire shape from
        ``build_prefill_progress_payload``). See record_peer_load_progress.
        """
        if not model_id or not isinstance(payload, dict):
            return
        self._peer_prefill_progress[model_id] = {
            **payload,
            "recorded_at": time.monotonic(),
        }

    def _read_peer_progress(self, store: dict, model_id: str) -> dict | None:
        entry = store.get(model_id)
        if entry is None:
            return None
        age = time.monotonic() - float(entry.get("recorded_at", 0.0))
        if age > self._PEER_PROGRESS_STALE_AFTER_S:
            # Expired — drop it so the store stays bounded and the bar
            # clears rather than freezing at the last value.
            store.pop(model_id, None)
            return None
        return {k: v for k, v in entry.items() if k != "recorded_at"}

    def peer_load_progress(self, model_id: str) -> dict | None:
        """Latest non-stale model-load snapshot for ``model_id``, in wire
        shape (sans the internal ``recorded_at``), or None."""
        return self._read_peer_progress(self._peer_load_progress, model_id)

    def peer_prefill_progress(self, model_id: str) -> dict | None:
        """Latest non-stale prefill snapshot for ``model_id``, or None."""
        return self._read_peer_progress(self._peer_prefill_progress, model_id)

    # ── Per-request lifecycle event log (Phase 9-lifecycle) ───────

    def peer_call_events(self, request_id: str) -> list[dict]:
        """Snapshot of lifecycle events received for an outbound
        request we made to a peer. Non-destructive read — events
        stay until drain_peer_call_events is called.
        """
        return list(self._peer_call_events.get(request_id, []))

    def drain_peer_call_events(self, request_id: str) -> list[dict]:
        """Pop + return all events for a request_id. Used when a
        chat turn finishes and the UI surfaces a single rollup
        ("flux-dev on 🏎 took 8.4s"). After draining the entry is
        gone — subsequent reads return [].
        """
        return self._peer_call_events.pop(request_id, [])

    # ── In-flight cross-peer request registry (Phase 9.3) ─────────

    def register_inflight(self, request_id: str, task: "asyncio.Task | Any") -> None:
        """Track an in-flight cross-peer request by request_id.

        Called by FabricPeerMiddleware right after the signed
        envelope is verified, with the asyncio.Task that's serving
        the request. A later MSG_CANCEL_REQUEST envelope can look
        the task up and cancel it (see cancel_inflight).

        Registration is bound to the asyncio.Task running the
        request handler. Cancelling that task propagates through
        the StreamingResponse generator → backend.chat_stream →
        underlying LLM cancellation (the existing Link 3 chain
        the audit confirmed works).

        Synchronous (no lock): single-writer (the middleware path)
        per request_id; readers via cancel_inflight tolerate a
        torn-down or already-finished entry.
        """
        if not request_id:
            return
        self._inflight_peer_requests[request_id] = task

    def unregister_inflight(self, request_id: str) -> None:
        """Drop a request from the registry. Called from the
        middleware's finally block when the downstream handler
        returns (success, failure, or already-cancelled).
        """
        if not request_id:
            return
        self._inflight_peer_requests.pop(request_id, None)

    def cancel_inflight(self, request_id: str) -> bool:
        """Cancel a registered in-flight request. Returns True when
        a task was found + ``cancel()`` was called, False when the
        request_id was unknown (already completed / never started /
        cancelled before registration).
        """
        task = self._inflight_peer_requests.pop(request_id, None)
        if task is None:
            return False
        try:
            task.cancel()
            return True
        except Exception:
            log.debug(
                "fabric_inflight_cancel_failed",
                request_id=request_id, exc_info=True,
            )
            return False

    # ── Inbound message dispatch (shared by client.py + fabric_routes.fabric_connect) ──

    def handle_inbound_envelope(self, envelope) -> None:
        """Apply side-effects from an inbound peer envelope.

        Called from BOTH WS read loops — fabric_routes.fabric_connect
        (server-side; messages B sent to A via the socket B opened
        to A) AND client.py._receive_loop (client-side; messages B
        sent to A via the socket A opened to B). Symmetric topology
        means either socket can carry any message type; we want both
        receive loops to apply the same logic.

        Today this handles:
          - heartbeat seq (always)
          - capability advertisement (any envelope can carry caps)
          - MSG_CANCEL_REQUEST → cancel the matching in-flight task
          - MSG_JOB_* lifecycle events (Phase 9.6 — reserved)

        Side-effects only. The read loop is responsible for the
        async iteration; this is sync because every operation is
        either dict mutation or a synchronous task.cancel() call.
        """
        from augmentum.fabric.protocol import (
            MSG_CANCEL_REQUEST,
            MSG_CONNECT_ENVELOPE,
            MSG_JOB_COMPLETED,
            MSG_JOB_FAILED,
            MSG_JOB_PROGRESS,
            MSG_JOB_STARTED,
        )

        # Heartbeat seq + capability advertisement apply to every
        # envelope. Mirrors the original fabric_connect logic.
        self.record_heartbeat(envelope.sender_node_id, envelope.seq)
        caps_payload = envelope.payload.get("capabilities")
        if isinstance(caps_payload, list):
            self.record_remote_capabilities(
                envelope.sender_node_id, caps_payload,
            )

        if envelope.msg_type == MSG_CANCEL_REQUEST:
            rid = str(envelope.payload.get("request_id", "") or "")
            if rid:
                cancelled = self.cancel_inflight(rid)
                log.info(
                    "fabric_inbound_cancel",
                    peer_node_id=envelope.sender_node_id,
                    request_id=rid, cancelled=cancelled,
                )
        elif envelope.msg_type in (
            MSG_JOB_STARTED, MSG_JOB_PROGRESS,
            MSG_JOB_COMPLETED, MSG_JOB_FAILED,
        ):
            rid = str(envelope.payload.get("request_id", "") or "")
            if rid:
                event = {
                    "msg_type": envelope.msg_type,
                    "sender_node_id": envelope.sender_node_id,
                    "ts": time.time(),
                    **{k: v for k, v in envelope.payload.items()
                       if k != "request_id"},
                }
                # Cap per-request log at 64 events; protects against
                # a misbehaving peer flooding us with progress events.
                lst = self._peer_call_events.setdefault(rid, [])
                if len(lst) >= 64:
                    lst.pop(0)
                lst.append(event)
                # Drop the entry after a terminal event — caller has
                # one read window to pick it up. Race-safe because
                # the chat path queries synchronously per turn end.
                if envelope.msg_type in (MSG_JOB_COMPLETED, MSG_JOB_FAILED):
                    # Mark for cleanup; actual eviction happens via
                    # drain_peer_call_events when the consumer reads.
                    pass
            log.debug(
                "fabric_inbound_lifecycle_event",
                peer_node_id=envelope.sender_node_id,
                msg_type=envelope.msg_type,
                request_id=rid,
            )
        elif envelope.msg_type == MSG_CONNECT_ENVELOPE:
            # Wedge B: re-inject a Connect-over-fabric frame into the
            # local hub. Fire-and-forget — the WS read loop stays
            # tight; the per-verb work happens on its own task.
            if self.connect_hub is None:
                log.warning(
                    "fabric_inbound_connect_envelope_hub_missing",
                    peer_node_id=envelope.sender_node_id,
                )
            else:
                from augmentum.connect.fabric_inbound import (
                    apply_inbound_fabric_envelope,
                )
                asyncio.create_task(
                    apply_inbound_fabric_envelope(
                        self._db,
                        connect_hub=self.connect_hub,
                        notification_hub=self.notification_hub,
                        fabric_payload=envelope.payload,
                        coordinator=self,
                        sender_node_id=envelope.sender_node_id,
                    ),
                )

    # ── Bidirectional WS: outbound to a peer ──────────────────────

    async def send_to_peer(
        self,
        node_id: str,
        *,
        msg_type: str,
        payload: dict,
    ) -> bool:
        """Send a signed envelope to a connected peer over its WS.

        Foundational for the server→peer direction added in Phase 9.
        The WebSocket sockets we hold in ``state.socket`` were opened
        by the peer (B initiated → A registered via attach_connection
        in fabric_connect), but TCP is full-duplex — we can write
        through the same socket. The receiver's read loop in
        fabric_connect handles the message type after envelope
        verification.

        Returns True on success, False when:
          - the peer isn't registered
          - the peer's socket isn't currently attached / connected
          - the underlying send raised (logged + absorbed; the
            caller treats "couldn't push" the same as "peer gone")

        Sends happen outside any lock — the WebSocket has its own
        write serialization, and holding the asyncio.Lock across an
        await on a network call would create avoidable contention.
        We pull the socket reference under the lock; if the peer
        disconnects between lookup + send, the send raises and we
        return False (correct).
        """
        from augmentum.fabric.protocol import FabricEnvelope

        async with self._lock:
            state = self._peers.get(node_id)
            socket = state.socket if (state and state.connected) else None

        if socket is None:
            log.debug(
                "fabric_send_to_peer_no_socket",
                peer_node_id=node_id, msg_type=msg_type,
            )
            return False

        try:
            envelope = FabricEnvelope.build(
                msg_type=msg_type,
                seq=self.next_local_seq(),
                sender_node_id=self._identity.node_id,
                payload=payload,
                signing_key=self._identity.private_key,
            )
            await socket.send_text(envelope.to_wire())
            return True
        except Exception as exc:
            # Common cases: ConnectionClosed (peer dropped while we
            # were composing), TypeError on a torn-down mock socket
            # in tests, etc. Don't propagate — the contract is
            # best-effort delivery; the caller already has a False
            # path for "couldn't reach peer".
            log.info(
                "fabric_send_to_peer_failed",
                peer_node_id=node_id, msg_type=msg_type,
                error=str(exc)[:160],
            )
            return False
