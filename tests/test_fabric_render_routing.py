"""Tests for cast-render routing through the RoutingDirector.

Critical invariants (mirror maybe_route_llm tests):

  - Local-first: when local has the capability flag, NEVER return peer.
  - No-capable-node: when nobody can serve, return None (don't fail
    silently into a half-finished route).
  - Connected-only: an offline peer with the capability isn't selected.
  - Tier preference: when multiple peers can serve, prefer the higher
    tier. This is the foundation hook for future load/proximity
    scoring — keep the call surface the same.
"""

from __future__ import annotations

import aiosqlite
import httpx
import pytest

from augmentum.cast.render import (
    RENDER_HTML,
    RENDER_VIDEO,
    RENDER_VRM,
    RenderJob,
    capability_flag_for,
    tier_rank,
)
from augmentum.fabric.capabilities import (
    CastRenderCapability,
    serialise,
)
from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.director import RoutingDirector
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.state.settings_store import SettingsStore


# ── Pure helpers ──────────────────────────────────────────────────


def test_capability_flag_mapping():
    assert capability_flag_for(RENDER_HTML) == "can_render_html"
    assert capability_flag_for(RENDER_VRM) == "can_render_vrm"
    assert capability_flag_for(RENDER_VIDEO) == "can_encode_video"


def test_capability_flag_unknown_kind_returns_empty():
    assert capability_flag_for("not-a-kind") == ""
    assert capability_flag_for("") == ""


def test_tier_rank_ordering():
    assert tier_rank("heavy") > tier_rank("standard") > tier_rank("lite")
    assert tier_rank("unknown") < tier_rank("lite")


# ── Router test fixture ───────────────────────────────────────────


async def _make_env() -> tuple[aiosqlite.Connection, FabricCoordinator, RoutingDirector]:
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
    coord = FabricCoordinator(identity, conn)
    http = httpx.AsyncClient()
    director = RoutingDirector(coord, http)
    return conn, coord, director


def _peer(node_id: str, addr: str = "192.168.1.10:6443") -> PairedPeer:
    return PairedPeer(
        node_id=node_id, hostname=f"h-{node_id[:4]}", role="peer",
        pubkey_b64="dGVzdA==", fingerprint=f"SHA256:{node_id[:8]}",
        addr=addr, tier="local",
        fabric_share_enabled=True, paired_at="2026-05-19 00:00:00",
        last_seen_at=None,
    )


class _FakeWebSocket:
    def __init__(self):
        self.closed = False
    async def close(self, code=1000, reason=""):
        self.closed = True


def _local_cap(coord: FabricCoordinator, cap: CastRenderCapability) -> None:
    """Directly seed the coordinator's local-cap cache.

    Production code populates this via build_local_capabilities() ->
    extractors. Tests skip that machinery and inject the cap shape
    they want under test.
    """
    coord._local_capabilities = [cap]


# ── Local-first invariant ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_first_when_local_can_render_html():
    """Local has can_render_html → router returns local route, even
    when peers also advertise the capability."""
    conn, coord, director = await _make_env()
    try:
        _local_cap(coord, CastRenderCapability(
            tier="standard",
            can_render_html=True,
            cpu_threads=8,
        ))

        # Peer also offers HTML render.
        await coord.register_paired_peer(_peer("peer-heavy"))
        await coord.attach_connection("peer-heavy", _FakeWebSocket())
        coord.record_remote_capabilities("peer-heavy", [
            serialise(CastRenderCapability(
                tier="heavy", can_render_html=True, can_render_vrm=True,
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_HTML, target_device_id="dev_xxx"),
        )
        assert route is not None
        assert route.location == "local"
        assert route.tier == "standard"
    finally:
        await conn.close()


# ── No capable node ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_none_when_nobody_can_serve():
    """Unknown kind, no local cap, no peers → None.

    Distinct from RenderRoute('local'): None tells the caller that
    NEITHER local NOR a peer can render this, so surface a clean
    error rather than dispatching into nowhere.
    """
    conn, coord, director = await _make_env()
    try:
        _local_cap(coord, CastRenderCapability(tier="lite"))  # no flags

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_VRM),
        )
        assert route is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_unknown_kind_returns_none_even_with_capable_peers():
    """Defensive: an unrecognised render kind never routes — better
    to fail closed than dispatch work nobody understands."""
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("peer-1"))
        await coord.attach_connection("peer-1", _FakeWebSocket())
        coord.record_remote_capabilities("peer-1", [
            serialise(CastRenderCapability(
                tier="heavy", can_render_html=True, can_render_vrm=True,
                can_encode_video=True,
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind="not-a-real-kind"),
        )
        assert route is None
    finally:
        await conn.close()


# ── Peer delegation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_routes_to_peer_when_local_cant_render():
    """Local can't, one peer can → route to that peer."""
    conn, coord, director = await _make_env()
    try:
        # Local has nothing useful.
        _local_cap(coord, CastRenderCapability(tier="lite"))

        await coord.register_paired_peer(_peer("peer-vrm"))
        await coord.attach_connection("peer-vrm", _FakeWebSocket())
        coord.record_remote_capabilities("peer-vrm", [
            serialise(CastRenderCapability(
                tier="standard", can_render_vrm=True,
                gpu_vendor="nvidia", gpu_vram_gb=8.0,
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_VRM),
        )
        assert route is not None
        assert route.location == "peer"
        assert route.node_id == "peer-vrm"
        assert route.tier == "standard"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_prefers_heavy_tier_peer_when_multiple_capable():
    """Tier preference: heavy beats standard, standard beats lite.

    Foundation for future scoring — load + proximity slot in here
    without changing the call surface.
    """
    conn, coord, director = await _make_env()
    try:
        _local_cap(coord, CastRenderCapability(tier="lite"))

        # Peer A: standard tier, can render VRM
        await coord.register_paired_peer(_peer("peer-std"))
        await coord.attach_connection("peer-std", _FakeWebSocket())
        coord.record_remote_capabilities("peer-std", [
            serialise(CastRenderCapability(
                tier="standard", can_render_vrm=True,
            )),
        ])

        # Peer B: heavy tier, also can render VRM
        await coord.register_paired_peer(_peer("peer-heavy"))
        await coord.attach_connection("peer-heavy", _FakeWebSocket())
        coord.record_remote_capabilities("peer-heavy", [
            serialise(CastRenderCapability(
                tier="heavy", can_render_vrm=True,
                gpu_vendor="nvidia", gpu_vram_gb=24.0,
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_VRM),
        )
        assert route is not None
        assert route.location == "peer"
        assert route.node_id == "peer-heavy"
        assert route.tier == "heavy"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_skips_peer_whose_flag_is_false():
    """A peer advertising cast.render but with the specific flag off
    is not a candidate. Coarse 'has the kind' is not enough — flag
    must be True for the requested job kind."""
    conn, coord, director = await _make_env()
    try:
        _local_cap(coord, CastRenderCapability(tier="lite"))

        # Peer advertises cast.render but can NOT render VRM.
        await coord.register_paired_peer(_peer("peer-html-only"))
        await coord.attach_connection("peer-html-only", _FakeWebSocket())
        coord.record_remote_capabilities("peer-html-only", [
            serialise(CastRenderCapability(
                tier="standard", can_render_html=True,  # VRM = False
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_VRM),
        )
        assert route is None  # nobody can serve VRM
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_skips_offline_peer():
    """Connected-only invariant: a paired-but-disconnected peer is
    not a render candidate, even with a capability advertised before
    disconnection."""
    conn, coord, director = await _make_env()
    try:
        _local_cap(coord, CastRenderCapability(tier="lite"))

        # Register but DON'T attach a connection.
        await coord.register_paired_peer(_peer("peer-offline"))
        coord.record_remote_capabilities("peer-offline", [
            serialise(CastRenderCapability(
                tier="heavy", can_render_vrm=True,
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_VRM),
        )
        assert route is None  # offline peer doesn't count
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_local_cap_missing_falls_through_to_peer_scan():
    """When extractors haven't populated local capabilities yet
    (narrow startup window), routing should still work via peers
    rather than crashing on the missing local cap."""
    conn, coord, director = await _make_env()
    try:
        # No local cap seeded at all.

        await coord.register_paired_peer(_peer("peer-1"))
        await coord.attach_connection("peer-1", _FakeWebSocket())
        coord.record_remote_capabilities("peer-1", [
            serialise(CastRenderCapability(
                tier="standard", can_render_html=True,
            )),
        ])

        route = await director.maybe_route_render(
            job=RenderJob(kind=RENDER_HTML),
        )
        assert route is not None
        assert route.location == "peer"
        assert route.node_id == "peer-1"
    finally:
        await conn.close()
