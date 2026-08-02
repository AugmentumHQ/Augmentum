"""Tests for Tier 0 fabric safety guards (architecture review follow-up).

Two time bombs the architecture review flagged as latent-but-real:

1. Knowledge fan-out had no recursion guard. With 3+ peers an A→B→C→A
   triangle was possible. Fixed via X-Fabric-Hop-Count header signed
   in the canonical bytes, refused at 508 LOOP_DETECTED in
   FabricPeerMiddleware when ≥ max.

2. Connected-but-hung peers (open socket, frozen process) stayed
   marked "Connected" indefinitely. Router kept dispatching to them.
   Fixed via FabricCoordinator's heartbeat sweeper that auto-detaches
   peers whose last_seen is > HEARTBEAT_TIMEOUT_S stale.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── X-Fabric-Hop-Count ────────────────────────────────────────────


def test_build_signed_peer_headers_includes_hop_count_zero_default():
    """Default hop_count=0 for an originating request. Most callers
    don't pass anything, so default-to-0 is load-bearing."""
    import aiosqlite

    async def _run():
        from augmentum.fabric.identity import FabricIdentity
        from augmentum.fabric.peer_middleware import build_signed_peer_headers
        from augmentum.state.settings_store import SettingsStore

        conn = await aiosqlite.connect(":memory:")
        try:
            await conn.execute(
                "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " updated_at TEXT DEFAULT (datetime('now')))"
            )
            await conn.commit()
            identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
            headers = build_signed_peer_headers(
                identity=identity, user_id="u",
                method="POST", path="/api/fabric/inference",
                body=b'{"hi": 1}',
            )
            assert headers["X-Fabric-Hop-Count"] == "0"
            return headers
        finally:
            await conn.close()

    asyncio.run(_run())


def test_build_signed_peer_headers_explicit_hop_count_propagates():
    """When a future re-dispatch site passes hop_count=1, the header
    + signature must both reflect that value."""
    import aiosqlite

    async def _run():
        from augmentum.fabric.identity import FabricIdentity
        from augmentum.fabric.peer_middleware import build_signed_peer_headers
        from augmentum.state.settings_store import SettingsStore

        conn = await aiosqlite.connect(":memory:")
        try:
            await conn.execute(
                "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " updated_at TEXT DEFAULT (datetime('now')))"
            )
            await conn.commit()
            identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

            zero_headers = build_signed_peer_headers(
                identity=identity, user_id="u",
                method="POST", path="/x", body=b"", hop_count=0,
            )
            one_headers = build_signed_peer_headers(
                identity=identity, user_id="u",
                method="POST", path="/x", body=b"", hop_count=1,
            )
            # Different hop count → different signature (because it's
            # in the canonical bytes — an attacker can't decrement to
            # defeat the loop guard).
            assert one_headers["X-Fabric-Hop-Count"] == "1"
            assert zero_headers["X-Fabric-Signature"] != one_headers["X-Fabric-Signature"]
        finally:
            await conn.close()

    asyncio.run(_run())


def test_canonical_bytes_at_v3_with_hop():
    """Wire-format pin: canonical bytes are version-tagged v3 (bumped
    from v2 when hop was added). Both sides upgrade in lockstep — a
    mismatch in this format breaks every cross-peer call."""
    from augmentum.fabric.peer_middleware import _peer_request_canonical_bytes

    canonical = _peer_request_canonical_bytes(
        sender="node-a", user_id="u", method="POST",
        path="/api/fabric/inference", timestamp=1234567890,
        body_sha256="abc", hop_count=0,
    )
    decoded = canonical.decode("utf-8")
    assert decoded.startswith("v3\n")
    assert "hop=0\n" in decoded or decoded.endswith("hop=0")


@pytest.mark.asyncio
async def test_middleware_508s_when_hop_count_at_max():
    """A peer request with X-Fabric-Hop-Count == _FABRIC_MAX_HOP_COUNT
    must be refused with 508 LOOP_DETECTED before any handler runs.
    No backend dispatch, no auth user provisioning, just the refusal.
    """
    from augmentum.fabric.peer_middleware import (
        _FABRIC_MAX_HOP_COUNT,
        FabricPeerMiddleware,
    )

    # downstream app shouldn't be invoked
    inner_called = False

    async def _downstream(scope, receive, send):
        nonlocal inner_called
        inner_called = True

    middleware = FabricPeerMiddleware(_downstream)

    sends: list[dict] = []

    async def _send(msg):
        sends.append(msg)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/fabric/inference",
        "query_string": b"",
        "headers": [
            (b"x-fabric-sender", b"peer-x"),
            (b"x-fabric-signature", b"sig"),
            (b"x-fabric-timestamp", str(int(time.time())).encode()),
            (b"x-fabric-user-id", b"u"),
            (b"x-fabric-hop-count", str(_FABRIC_MAX_HOP_COUNT).encode()),
        ],
        "app": MagicMock(),
    }
    # Stub the app.state lookups so we reach the hop-count check.
    scope["app"].state.fabric_coordinator = MagicMock()
    scope["app"].state.session_manager = MagicMock()
    sm_backend = MagicMock()
    sm_backend.backend.conn = MagicMock()
    scope["app"].state.state_manager = sm_backend

    with patch(
        "augmentum.fabric.peer_auth.lookup_peer_pubkey",
        new_callable=AsyncMock,
        return_value="dGVzdHB1YmtleQ==",  # would be load-bearing but check happens before this
    ):
        await middleware(scope, _receive, _send)

    # Inner handler must NOT have been called
    assert inner_called is False
    # Response is 508
    start_messages = [m for m in sends if m["type"] == "http.response.start"]
    assert len(start_messages) == 1
    assert start_messages[0]["status"] == 508
    body_messages = [m for m in sends if m["type"] == "http.response.body"]
    body_bytes = b"".join(m.get("body", b"") for m in body_messages)
    assert b"loop_detected" in body_bytes


@pytest.mark.asyncio
async def test_middleware_passes_through_when_hop_count_zero():
    """Sanity: the hop-count guard only fires at the max. Zero
    (the normal case) flows through to the downstream handler."""
    from augmentum.fabric.peer_middleware import FabricPeerMiddleware

    inner_called = False

    async def _downstream(scope, receive, send):
        nonlocal inner_called
        inner_called = True

    middleware = FabricPeerMiddleware(_downstream)

    async def _send(msg):
        pass

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/fabric/inference",
        "query_string": b"",
        "headers": [
            (b"x-fabric-sender", b"peer-x"),
            (b"x-fabric-signature", b"badsig"),  # will fail verify, but that's not what we're testing
            (b"x-fabric-timestamp", str(int(time.time())).encode()),
            (b"x-fabric-hop-count", b"0"),
        ],
        "app": MagicMock(),
    }
    scope["app"].state.fabric_coordinator = MagicMock()
    scope["app"].state.session_manager = MagicMock()
    sm_backend = MagicMock()
    sm_backend.backend.conn = MagicMock()
    scope["app"].state.state_manager = sm_backend

    with patch(
        "augmentum.fabric.peer_auth.lookup_peer_pubkey",
        new_callable=AsyncMock,
        return_value="dGVzdHB1YmtleQ==",
    ):
        await middleware(scope, _receive, _send)

    # Downstream called (request fell through after signature
    # verification failed — middleware passes through unauth, doesn't
    # 508).
    assert inner_called is True


# ── Heartbeat timeout sweeper ─────────────────────────────────────


@pytest.mark.asyncio
async def test_sweeper_detaches_stale_peer():
    """A peer whose last_seen is older than HEARTBEAT_TIMEOUT_S gets
    auto-detached (state.connected → False, capabilities cleared)."""
    import aiosqlite

    from augmentum.fabric.coordinator import FabricCoordinator, PeerLiveState
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.fabric.peer_auth import PairedPeer
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.execute(
            "CREATE TABLE fabric_nodes (id TEXT PRIMARY KEY, last_seen_at TEXT)"
        )
        await conn.execute(
            "INSERT INTO fabric_nodes (id, last_seen_at) VALUES ('stale-peer', '')"
        )
        await conn.commit()

        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        coord = FabricCoordinator(identity, conn)
        paired = PairedPeer(
            node_id="stale-peer", hostname="stale", role="peer",
            pubkey_b64="", fingerprint="", addr="192.168.1.99:6443",
            tier="local", fabric_share_enabled=True,
            paired_at="2026-01-01", last_seen_at=None,
        )
        # Install the peer as currently connected but with a very old
        # last_seen — older than the timeout threshold.
        socket = MagicMock()
        socket.close = AsyncMock()
        coord._peers["stale-peer"] = PeerLiveState(
            paired=paired,
            connected=True,
            socket=socket,
            last_seen_monotonic=time.monotonic() - (coord.HEARTBEAT_TIMEOUT_S + 5.0),
        )

        # One synchronous sweep iteration — no waiting for the
        # interval.
        await coord._sweep_stale_peers_once()

        # Peer should now be marked disconnected with capabilities cleared.
        state = coord._peers["stale-peer"]
        assert state.connected is False
        assert state.capabilities == []
        # Socket should have been closed
        socket.close.assert_awaited_once()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sweeper_leaves_fresh_peer_alone():
    """A peer whose last_seen is fresh (within timeout) must NOT be
    detached. Pin so the sweeper doesn't flap connections on the
    normal heartbeat cadence."""
    import aiosqlite

    from augmentum.fabric.coordinator import FabricCoordinator, PeerLiveState
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.fabric.peer_auth import PairedPeer
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.execute(
            "CREATE TABLE fabric_nodes (id TEXT PRIMARY KEY, last_seen_at TEXT)"
        )
        await conn.commit()

        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        coord = FabricCoordinator(identity, conn)
        paired = PairedPeer(
            node_id="fresh-peer", hostname="fresh", role="peer",
            pubkey_b64="", fingerprint="", addr="192.168.1.50:6443",
            tier="local", fabric_share_enabled=True,
            paired_at="2026-01-01", last_seen_at=None,
        )
        socket = MagicMock()
        socket.close = AsyncMock()
        coord._peers["fresh-peer"] = PeerLiveState(
            paired=paired,
            connected=True,
            socket=socket,
            # Within timeout: heartbeat 1s ago
            last_seen_monotonic=time.monotonic() - 1.0,
        )

        await coord._sweep_stale_peers_once()

        state = coord._peers["fresh-peer"]
        assert state.connected is True
        socket.close.assert_not_awaited()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sweeper_ignores_never_seen_peers():
    """A peer with last_seen_monotonic == 0.0 ("never seen" sentinel)
    must NOT be detached. attach_connection sets the field, so 0.0
    here means we never got a hello — possibly registered but no
    socket yet. The detach would be redundant."""
    import aiosqlite

    from augmentum.fabric.coordinator import FabricCoordinator, PeerLiveState
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.fabric.peer_auth import PairedPeer
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.execute(
            "CREATE TABLE fabric_nodes (id TEXT PRIMARY KEY, last_seen_at TEXT)"
        )
        await conn.commit()

        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        coord = FabricCoordinator(identity, conn)
        paired = PairedPeer(
            node_id="never-seen", hostname="x", role="peer",
            pubkey_b64="", fingerprint="", addr="192.168.1.10:6443",
            tier="local", fabric_share_enabled=True,
            paired_at="2026-01-01", last_seen_at=None,
        )
        coord._peers["never-seen"] = PeerLiveState(
            paired=paired,
            connected=True,
            last_seen_monotonic=0.0,  # sentinel
        )

        await coord._sweep_stale_peers_once()

        # Should still be connected — sentinel means defer judgment.
        assert coord._peers["never-seen"].connected is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sweeper_skips_disconnected_peers():
    """Already-disconnected peers don't need to be detached again."""
    import aiosqlite

    from augmentum.fabric.coordinator import FabricCoordinator, PeerLiveState
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.fabric.peer_auth import PairedPeer
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.execute(
            "CREATE TABLE fabric_nodes (id TEXT PRIMARY KEY, last_seen_at TEXT)"
        )
        await conn.commit()

        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        coord = FabricCoordinator(identity, conn)
        paired = PairedPeer(
            node_id="off-peer", hostname="x", role="peer",
            pubkey_b64="", fingerprint="", addr="192.168.1.11:6443",
            tier="local", fabric_share_enabled=True,
            paired_at="2026-01-01", last_seen_at=None,
        )
        coord._peers["off-peer"] = PeerLiveState(
            paired=paired,
            connected=False,
            last_seen_monotonic=time.monotonic() - 1000.0,
        )

        # Should not raise, should not touch the peer's state.
        await coord._sweep_stale_peers_once()
        assert coord._peers["off-peer"].connected is False


    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_start_stop_sweeper_lifecycle():
    """start_heartbeat_sweeper is idempotent; stop_heartbeat_sweeper
    cleans up the background task on shutdown."""
    import aiosqlite

    from augmentum.fabric.coordinator import FabricCoordinator
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        coord = FabricCoordinator(identity, conn)

        # Initially: no task
        assert coord._heartbeat_sweep_task is None

        coord.start_heartbeat_sweeper()
        task1 = coord._heartbeat_sweep_task
        assert task1 is not None
        assert not task1.done()

        # Idempotent: second call doesn't replace a healthy task
        coord.start_heartbeat_sweeper()
        assert coord._heartbeat_sweep_task is task1

        # Stop: cancels + clears
        coord.stop_heartbeat_sweeper()
        assert coord._heartbeat_sweep_task is None
        # Give the cancellation a turn to propagate
        try:
            await asyncio.wait_for(task1, timeout=0.5)
        except (TimeoutError, asyncio.CancelledError):
            pass
        assert task1.cancelled() or task1.done()
    finally:
        await conn.close()
