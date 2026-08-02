"""Tests for FabricPeerMiddleware: cross-peer request authentication.

Invariants pinned:

  - No fabric headers → pass through unchanged (AuthMiddleware handles auth)
  - Valid signed request → scope["user"] + scope["fabric_peer"] populated
  - Invalid signature → pass through (AuthMiddleware will 401)
  - Unknown sender → pass through (don't reveal which peers we know)
  - Stale timestamp → pass through (replay defense)
  - Missing claimed user → pass through (peer can't claim non-existent users)
  - WebSocket / lifespan → pass through (HTTP only)
  - build_signed_peer_headers + canonical bytes verify roundtrip
"""
from __future__ import annotations

import base64
import time
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_middleware import (
    FabricPeerMiddleware,
    _peer_request_canonical_bytes,
    build_signed_peer_headers,
)
from augmentum.state.settings_store import SettingsStore

# ── Test fixtures ─────────────────────────────────────────────────


async def _make_db_with_peer(peer_node_id: str, peer_pubkey_b64: str):
    """Create an in-memory DB with the fabric_nodes table + a paired peer."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        """CREATE TABLE fabric_nodes (
            id TEXT PRIMARY KEY, hostname TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'peer',
            pubkey_ed25519 TEXT NOT NULL, pubkey_fingerprint TEXT NOT NULL,
            addr TEXT NOT NULL DEFAULT '', tier TEXT NOT NULL DEFAULT 'local',
            fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
            paired_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT, icon TEXT NOT NULL DEFAULT '')"""
    )
    await conn.execute(
        """INSERT INTO fabric_nodes
            (id, pubkey_ed25519, pubkey_fingerprint, addr)
            VALUES (?, ?, 'fp', '192.168.1.10:6443')""",
        (peer_node_id, peer_pubkey_b64),
    )
    await conn.commit()
    return conn


def _make_app_state(*, db_conn=None, has_session_manager: bool = True,
                   has_user: bool = True, user_id: str = "user-1"):
    """Build a mock app object for the middleware to read from."""
    app = MagicMock()
    app.state = MagicMock()
    # Coordinator: any non-None object satisfies the check.
    app.state.fabric_coordinator = MagicMock() if db_conn else None
    # State manager + backend chain.
    if db_conn is not None:
        sm = MagicMock()
        sm.backend = MagicMock()
        sm.backend.conn = db_conn
        app.state.state_manager = sm
    else:
        app.state.state_manager = None
    # Session manager
    if has_session_manager:
        session_mgr = MagicMock()
        if has_user:
            user_obj = MagicMock()
            user_obj.id = user_id
            user_obj.is_active = True
            session_mgr.get_user_by_id = AsyncMock(return_value=user_obj)
            # The middleware now calls get_or_create_fabric_peer_user
            # (per-peer service user model, post-2026-05-23). Stub it
            # to return the same user object so tests work whether
            # the middleware takes the old per-user lookup path or
            # the new per-peer-service-user path.
            session_mgr.get_or_create_fabric_peer_user = AsyncMock(return_value=user_obj)
        else:
            session_mgr.get_user_by_id = AsyncMock(return_value=None)
            session_mgr.get_or_create_fabric_peer_user = AsyncMock(return_value=None)
        app.state.session_manager = session_mgr
    else:
        app.state.session_manager = None
    return app


def _build_scope(*, headers=None, method="POST", path="/v1/chat/completions",
                  type_="http", app=None) -> dict:
    """Build an ASGI scope dict."""
    return {
        "type": type_,
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or {}).items()
        ],
        "app": app,
    }


def _make_receive(body: bytes = b""):
    """Build an async receive() callable that yields a single
    http.request event carrying ``body``. After the first call it
    returns http.disconnect events. Used by tests that exercise the
    middleware's body-drain + replay path (Phase 3.y+).
    """
    served = {"first": True}

    async def receive():
        if served["first"]:
            served["first"] = False
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


# ── Pass-through invariants ───────────────────────────────────────


@pytest.mark.asyncio
async def test_no_fabric_headers_passes_through():
    """A normal authenticated user request has no X-Fabric-* headers.
    The middleware must NOT touch scope["user"] (the downstream
    AuthMiddleware sets it). Pass-through behavior is invisible.
    """
    downstream = AsyncMock()
    mw = FabricPeerMiddleware(downstream)
    scope = _build_scope()  # no headers
    await mw(scope, MagicMock(), MagicMock())
    downstream.assert_called_once()
    # scope["user"] was never set by our middleware.
    forwarded_scope = downstream.call_args[0][0]
    assert "user" not in forwarded_scope
    assert "fabric_peer" not in forwarded_scope


@pytest.mark.asyncio
async def test_websocket_passes_through():
    """Fabric peer auth is HTTP-only. WebSocket upgrades must pass
    through unchanged (the inter-peer WS uses different auth via
    its own envelope-signing).
    """
    downstream = AsyncMock()
    mw = FabricPeerMiddleware(downstream)
    scope = _build_scope(
        type_="websocket",
        headers={"X-Fabric-Sender": "peer-abc"},  # would otherwise trigger
    )
    await mw(scope, MagicMock(), MagicMock())
    downstream.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_passes_through():
    """ASGI lifespan events don't have method/path."""
    downstream = AsyncMock()
    mw = FabricPeerMiddleware(downstream)
    scope = {"type": "lifespan"}
    await mw(scope, MagicMock(), MagicMock())
    downstream.assert_called_once()


@pytest.mark.asyncio
async def test_partial_fabric_headers_passes_through():
    """A malformed request with only some fabric headers should be
    treated like a normal request (no user pre-populated). The
    downstream AuthMiddleware will 401 if the request also lacks
    real auth.
    """
    downstream = AsyncMock()
    mw = FabricPeerMiddleware(downstream)
    scope = _build_scope(headers={"X-Fabric-Sender": "peer-abc"})  # missing sig/ts/user
    await mw(scope, MagicMock(), MagicMock())
    downstream.assert_called_once()
    forwarded_scope = downstream.call_args[0][0]
    assert "user" not in forwarded_scope


@pytest.mark.asyncio
async def test_missing_app_state_passes_through():
    """Defensive: app or app.state unset → pass through. Don't crash
    during startup race windows.
    """
    downstream = AsyncMock()
    mw = FabricPeerMiddleware(downstream)
    scope = _build_scope(
        headers={
            "X-Fabric-Sender": "x", "X-Fabric-User-Id": "u",
            "X-Fabric-Timestamp": str(int(time.time())),
            "X-Fabric-Signature": "abc",
        },
        app=None,  # no app
    )
    await mw(scope, MagicMock(), MagicMock())
    downstream.assert_called_once()


# ── Valid-signature path ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_signature_populates_scope():
    """The happy path: peer signs correctly, claimed user exists.
    scope["user"] is set, scope["fabric_peer"] carries the sender id.
    """
    # Build a peer identity.
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    peer_identity = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        # Build the receiver-side DB with the peer paired.
        recv_db = await _make_db_with_peer(
            peer_identity.node_id, peer_identity.public_key_b64,
        )
        try:
            # Build signed headers over a small JSON body.
            body = b'{"model":"x","messages":[]}'
            headers = build_signed_peer_headers(
                identity=peer_identity, user_id="user-42",
                method="POST", path="/v1/chat/completions", body=body,
            )

            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            app = _make_app_state(db_conn=recv_db, user_id="user-42")
            scope = _build_scope(headers=headers, app=app)

            await mw(scope, _make_receive(body), MagicMock())

            downstream.assert_called_once()
            forwarded = downstream.call_args[0][0]
            # scope["user"] was populated.
            assert forwarded.get("user") is not None
            assert forwarded["user"].id == "user-42"
            # scope["fabric_peer"] carries the sender.
            assert forwarded["fabric_peer"]["sender_node_id"] == peer_identity.node_id
            assert forwarded["fabric_peer"]["trust_tier"] == "local"
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


# ── Rejection paths (all pass through, never crash) ───────────────


@pytest.mark.asyncio
async def test_unknown_sender_passes_through():
    """A signed request claiming a sender we've never paired with.
    Pass through so AuthMiddleware can apply normal auth. (Don't
    return early with a fabric-specific error -- gives no info to a
    probe.)
    """
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    rogue_identity = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        # Receiver DB has NO peer rows.
        recv_db = await aiosqlite.connect(":memory:")
        await recv_db.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await recv_db.execute(
            """CREATE TABLE fabric_nodes (
                id TEXT PRIMARY KEY, hostname TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'peer',
                pubkey_ed25519 TEXT NOT NULL, pubkey_fingerprint TEXT NOT NULL,
                addr TEXT NOT NULL DEFAULT '', tier TEXT NOT NULL DEFAULT 'local',
                fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
                paired_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen_at TEXT)"""
        )
        await recv_db.commit()
        try:
            # Rogue sender — pubkey lookup fails BEFORE body drain so
            # the middleware never reads receive(); MagicMock OK.
            headers = build_signed_peer_headers(
                identity=rogue_identity, user_id="user-42",
                method="POST", path="/v1/chat/completions", body=b"",
            )
            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            app = _make_app_state(db_conn=recv_db)
            scope = _build_scope(headers=headers, app=app)
            await mw(scope, MagicMock(), MagicMock())
            downstream.assert_called_once()
            forwarded = downstream.call_args[0][0]
            assert "user" not in forwarded  # rejection = no scope mutation
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


@pytest.mark.asyncio
async def test_tampered_signature_passes_through():
    """Signature bytes mutated after signing -- verify fails, pass through."""
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    peer = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        recv_db = await _make_db_with_peer(peer.node_id, peer.public_key_b64)
        try:
            body = b'{"model":"x"}'
            headers = build_signed_peer_headers(
                identity=peer, user_id="u",
                method="POST", path="/v1/chat/completions", body=body,
            )
            # Tamper the signature -- flip a byte in the base64.
            sig = headers["X-Fabric-Signature"]
            tampered = base64.b64encode(b"x" * 64).decode("ascii")
            headers["X-Fabric-Signature"] = tampered

            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            app = _make_app_state(db_conn=recv_db)
            scope = _build_scope(headers=headers, app=app)
            await mw(scope, _make_receive(body), MagicMock())
            downstream.assert_called_once()
            assert "user" not in downstream.call_args[0][0]
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


@pytest.mark.asyncio
async def test_stale_timestamp_passes_through():
    """A valid signed request but the timestamp is hours old -- replay
    defense. Pass through (AuthMiddleware will 401).
    """
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    peer = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        recv_db = await _make_db_with_peer(peer.node_id, peer.public_key_b64)
        try:
            # Hand-build a stale signed header set. Stale-ts is
            # rejected BEFORE body drain so receive() is never called.
            stale_ts = int(time.time()) - 3600  # 1h old, way outside window
            canonical = _peer_request_canonical_bytes(
                sender=peer.node_id, user_id="u",
                method="POST", path="/v1/chat/completions",
                timestamp=stale_ts,
                body_sha256="0" * 64,
            )
            sig = peer.sign(canonical)
            headers = {
                "X-Fabric-Sender": peer.node_id,
                "X-Fabric-User-Id": "u",
                "X-Fabric-Timestamp": str(stale_ts),
                "X-Fabric-Signature": base64.b64encode(sig).decode("ascii"),
            }
            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            app = _make_app_state(db_conn=recv_db)
            scope = _build_scope(headers=headers, app=app)
            await mw(scope, MagicMock(), MagicMock())
            downstream.assert_called_once()
            assert "user" not in downstream.call_args[0][0]
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


@pytest.mark.asyncio
async def test_tampered_body_passes_through():
    """Phase 3.y body integrity: sign over body bytes X but transmit
    body bytes Y. The receiver re-hashes Y, builds canonical bytes
    with Y's hash, signature fails verification → pass through. This
    closes the gap where a MITM could swap the body while keeping the
    signed headers intact (the v1 contract didn't cover bodies).
    """
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    peer = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        recv_db = await _make_db_with_peer(peer.node_id, peer.public_key_b64)
        try:
            signed_body = b'{"model":"x","messages":[{"role":"user","content":"hi"}]}'
            headers = build_signed_peer_headers(
                identity=peer, user_id="u",
                method="POST", path="/v1/chat/completions",
                body=signed_body,
            )
            # Receiver gets a DIFFERENT body than what was signed.
            transmitted_body = b'{"model":"x","messages":[{"role":"user","content":"exfiltrate"}]}'
            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            app = _make_app_state(db_conn=recv_db)
            scope = _build_scope(headers=headers, app=app)
            await mw(scope, _make_receive(transmitted_body), MagicMock())
            downstream.assert_called_once()
            # Body mismatch ⇒ no user scope ⇒ AuthMiddleware will 401.
            assert "user" not in downstream.call_args[0][0]
            # And the downstream still received the (tampered) body
            # via replay, so it can do its own decision-making about it
            # — we don't drop the request, we just refuse to authenticate it.
            replay_receive = downstream.call_args[0][1]
            first_event = await replay_receive()
            assert first_event["type"] == "http.request"
            assert first_event["body"] == transmitted_body
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


@pytest.mark.asyncio
async def test_body_replayed_to_downstream_on_happy_path():
    """The downstream handler MUST see the same body bytes we drained.
    Otherwise the proxied request body would arrive empty at the
    OpenAI route handler.
    """
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    peer = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        recv_db = await _make_db_with_peer(peer.node_id, peer.public_key_b64)
        try:
            body = b'{"model":"test","messages":[{"role":"user","content":"hello"}]}'
            headers = build_signed_peer_headers(
                identity=peer, user_id="u",
                method="POST", path="/v1/chat/completions", body=body,
            )
            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            app = _make_app_state(db_conn=recv_db, user_id="u")
            scope = _build_scope(headers=headers, app=app)
            await mw(scope, _make_receive(body), MagicMock())
            downstream.assert_called_once()
            # scope["user"] was set (happy path).
            assert downstream.call_args[0][0].get("user") is not None
            # Replay receive() yields the exact bytes we drained.
            replay_receive = downstream.call_args[0][1]
            event = await replay_receive()
            assert event["type"] == "http.request"
            assert event["body"] == body
            assert event["more_body"] is False
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


@pytest.mark.asyncio
async def test_unknown_user_passes_through():
    """Sender is paired and signature is valid, but the claimed user
    doesn't exist on this node. Pass through -- peer can't claim to
    dispatch for users we don't have.
    """
    conn_a = await aiosqlite.connect(":memory:")
    await conn_a.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn_a.commit()
    peer = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
    try:
        recv_db = await _make_db_with_peer(peer.node_id, peer.public_key_b64)
        try:
            body = b'{"model":"x"}'
            headers = build_signed_peer_headers(
                identity=peer, user_id="ghost-user",
                method="POST", path="/v1/chat/completions", body=body,
            )
            downstream = AsyncMock()
            mw = FabricPeerMiddleware(downstream)
            # session_manager.get_user_by_id returns None for this user
            app = _make_app_state(db_conn=recv_db, has_user=False)
            scope = _build_scope(headers=headers, app=app)
            await mw(scope, _make_receive(body), MagicMock())
            downstream.assert_called_once()
            assert "user" not in downstream.call_args[0][0]
        finally:
            await recv_db.close()
    finally:
        await conn_a.close()


# ── Helper-function tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_signed_peer_headers_shape():
    """Header set has the four expected keys, signature is base64,
    timestamp is plausible (within 1 second of now).
    """
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    try:
        headers = build_signed_peer_headers(
            identity=identity, user_id="u-42",
            method="POST", path="/v1/chat/completions", body=b"",
        )
        assert set(headers.keys()) == {
            "X-Fabric-Sender", "X-Fabric-User-Id",
            "X-Fabric-Timestamp", "X-Fabric-Signature",
            # Added in the Tier 0 hop-count guard: every signed
            # envelope carries the hop count so the receiver can
            # refuse forwarding loops at the middleware layer.
            "X-Fabric-Hop-Count",
        }
        assert headers["X-Fabric-Sender"] == identity.node_id
        assert headers["X-Fabric-User-Id"] == "u-42"
        # Originator default — most callers don't pass hop_count.
        assert headers["X-Fabric-Hop-Count"] == "0"
        # Timestamp is a recent unix-seconds value.
        ts = int(headers["X-Fabric-Timestamp"])
        assert abs(ts - int(time.time())) <= 2
        # Signature is decodable base64 of 64 bytes (ed25519).
        sig = base64.b64decode(headers["X-Fabric-Signature"])
        assert len(sig) == 64
    finally:
        await conn.close()


def test_canonical_bytes_is_deterministic():
    """Two calls with identical inputs produce identical bytes.
    Otherwise sign/verify would be flaky across peers.
    """
    a = _peer_request_canonical_bytes(
        sender="x", user_id="u", method="POST",
        path="/v1/x", timestamp=1700000000, body_sha256="a" * 64,
    )
    b = _peer_request_canonical_bytes(
        sender="x", user_id="u", method="POST",
        path="/v1/x", timestamp=1700000000, body_sha256="a" * 64,
    )
    assert a == b
    # And a different input produces different bytes.
    c = _peer_request_canonical_bytes(
        sender="x", user_id="u", method="POST",
        path="/v1/x", timestamp=1700000001, body_sha256="a" * 64,
    )
    assert a != c
    # Different body hash → different bytes (body integrity coverage).
    d = _peer_request_canonical_bytes(
        sender="x", user_id="u", method="POST",
        path="/v1/x", timestamp=1700000000, body_sha256="b" * 64,
    )
    assert a != d
