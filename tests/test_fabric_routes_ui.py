"""Tests for the Phase 4 UI-facing fabric endpoints.

  - GET /api/fabric/peers (detailed peer inventory for the UI)
  - DELETE /api/fabric/peers/{node_id} (unpair)

Both are admin-only and gated on settings.fabric_enabled. The
existing /status and /capabilities endpoints already have coverage
via test_fabric_capabilities; this file focuses on the new surface.

We don't go through the full FastAPI test client here because the
fabric coordinator + state-manager singletons aren't trivially
mockable through TestClient. Instead we exercise the route handlers
directly with hand-built request mocks -- enough to pin the
contract.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.fabric.capabilities import (
    LLMInferenceCapability,
    serialise,
)
from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.proxy.fabric_routes import fabric_peers, fabric_unpair
from augmentum.state.settings_store import SettingsStore


async def _make_env() -> tuple[aiosqlite.Connection, FabricCoordinator]:
    """Build an in-memory DB + coordinator. Returns (conn, coordinator)."""
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
    await conn.commit()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    return conn, FabricCoordinator(identity, conn)


def _peer(node_id: str) -> PairedPeer:
    return PairedPeer(
        node_id=node_id, hostname=f"h-{node_id[:4]}", role="peer",
        pubkey_b64="dGVzdA==", fingerprint=f"SHA256:{node_id[:8]}",
        addr="192.168.1.10:6443", tier="local",
        fabric_share_enabled=True, paired_at="2026-05-16 00:00:00",
        last_seen_at=None,
    )


def _admin_request(coordinator=None, db_conn=None):
    """Mock Request object with an admin user + the bits the route reads."""
    req = MagicMock()
    req.scope = {"user": MagicMock(is_admin=True)}
    req.app.state = MagicMock()
    req.app.state.fabric_coordinator = coordinator
    if db_conn is not None:
        sm = MagicMock()
        sm.backend = MagicMock()
        sm.backend.conn = db_conn
        req.app.state.state_manager = sm
    else:
        req.app.state.state_manager = None
    return req


def _non_admin_request():
    req = MagicMock()
    req.scope = {"user": MagicMock(is_admin=False)}
    return req


# ── GET /api/fabric/peers ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_peers_returns_disabled_when_fabric_off(monkeypatch):
    """When fabric_enabled=False, return `{"enabled": false, "peers": []}`
    rather than 503 -- the UI renders "fabric is off" without erroring.
    """
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", False)
    result = await fabric_peers(_admin_request())
    assert result == {"enabled": False, "peers": []}


@pytest.mark.asyncio
async def test_peers_returns_empty_when_no_peers_paired(monkeypatch):
    """Fabric enabled but no peers paired -- empty list, not error."""
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    conn, coord = await _make_env()
    try:
        result = await fabric_peers(_admin_request(coord, conn))
        assert result["enabled"] is True
        assert result["peers"] == []
        assert result["this_node_id"] == coord._identity.node_id
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_peers_returns_detailed_info_for_each_paired(monkeypatch):
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    conn, coord = await _make_env()
    try:
        await coord.register_paired_peer(_peer("peer-1"))
        await coord.register_paired_peer(_peer("peer-2"))
        # peer-1 is connected + advertises a capability.
        ws = MagicMock()
        ws.close = AsyncMock()
        await coord.attach_connection("peer-1", ws)
        coord.record_remote_capabilities("peer-1", [
            serialise(LLMInferenceCapability(model_id="some-model")),
        ])

        result = await fabric_peers(_admin_request(coord, conn))
        assert result["enabled"] is True
        assert len(result["peers"]) == 2

        by_id = {p["node_id"]: p for p in result["peers"]}
        # peer-1 connected with a capability.
        assert by_id["peer-1"]["connected"] is True
        assert by_id["peer-1"]["capability_count"] == 1
        assert len(by_id["peer-1"]["capabilities"]) == 1
        assert by_id["peer-1"]["capabilities"][0]["model_id"] == "some-model"
        # peer-2 offline, no caps.
        assert by_id["peer-2"]["connected"] is False
        assert by_id["peer-2"]["capability_count"] == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_peers_admin_only(monkeypatch):
    """Non-admin requests are rejected."""
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    result = await fabric_peers(_non_admin_request())
    # require_admin returns a JSONResponse on rejection; we expect
    # SOME response indicating denial (the exact shape depends on
    # the guard implementation; the contract here is "not the
    # normal success dict").
    if hasattr(result, "status_code"):
        assert result.status_code in (401, 403)
    elif isinstance(result, dict):
        # Some implementations return error dicts; either way "peers"
        # shouldn't be present as a normal list of paired peers.
        assert "peers" not in result or "error" in result


# ── DELETE /api/fabric/peers/{node_id} ────────────────────────────


@pytest.mark.asyncio
async def test_unpair_removes_peer_and_closes_connection(monkeypatch):
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    conn, coord = await _make_env()
    try:
        # Persist a peer + attach a (mock) connection.
        await conn.execute(
            """INSERT INTO fabric_nodes
               (id, pubkey_ed25519, pubkey_fingerprint, addr)
               VALUES ('p1', 'pk', 'fp', '192.168.1.10:6443')""",
        )
        await conn.commit()
        await coord.register_paired_peer(_peer("p1"))
        ws = MagicMock()
        ws.close = AsyncMock()
        await coord.attach_connection("p1", ws)

        result = await fabric_unpair(_admin_request(coord, conn), "p1")
        assert result == {"ok": True, "unpaired": "p1"}

        # Peer gone from coordinator + DB.
        assert coord.peer_state("p1") is None
        cur = await conn.execute("SELECT id FROM fabric_nodes WHERE id='p1'")
        assert (await cur.fetchone()) is None
        # Socket was closed by unregister_peer.
        ws.close.assert_called_once()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_unpair_404_when_peer_unknown(monkeypatch):
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    conn, coord = await _make_env()
    try:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await fabric_unpair(_admin_request(coord, conn), "never-paired")
        assert excinfo.value.status_code == 404
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_unpair_503_when_fabric_disabled(monkeypatch):
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", False)
    conn, _ = await _make_env()
    try:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await fabric_unpair(_admin_request(), "any-id")
        assert excinfo.value.status_code == 503
    finally:
        await conn.close()
