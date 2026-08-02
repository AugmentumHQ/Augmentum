"""Phase-2 tests for the coordinator's capability extensions.

Pins:
  - register_extractor + build_local_capabilities merge extractor output
  - one broken extractor doesn't poison the others
  - record_remote_capabilities replaces wholesale (no stale residue)
  - find_peers_with_capability respects connected_only
  - capability_summary aggregates local + remote correctly
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.fabric.capabilities import (
    KIND_LLM_INFERENCE,
    ImageGenerationCapability,
    LLMInferenceCapability,
    serialise,
)
from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.state.settings_store import SettingsStore


async def _make_coordinator():
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


def _fake_peer(node_id: str) -> PairedPeer:
    return PairedPeer(
        node_id=node_id, hostname=f"h-{node_id[:4]}", role="peer",
        pubkey_b64="dGVzdA==", fingerprint=f"SHA256:{node_id[:8]}",
        addr="192.168.1.10:6443", tier="local",
        fabric_share_enabled=True, paired_at="2026-05-15 22:00:00",
        last_seen_at=None,
    )


class _FakeWebSocket:
    def __init__(self):
        self.closed = False

    async def close(self, code=1000, reason=""):
        self.closed = True


@pytest.mark.asyncio
async def test_build_local_capabilities_merges_extractors():
    conn, coord = await _make_coordinator()
    try:
        e1 = MagicMock()
        e1.collect = AsyncMock(return_value=[
            LLMInferenceCapability(model_id="model-a"),
        ])
        e2 = MagicMock()
        e2.collect = AsyncMock(return_value=[
            ImageGenerationCapability(model_id="img-x"),
        ])
        coord.register_extractor(e1)
        coord.register_extractor(e2)
        caps = await coord.build_local_capabilities()
        assert len(caps) == 2
        # Cached copy matches.
        assert coord.local_capabilities() == caps
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_broken_extractor_doesnt_block_others():
    conn, coord = await _make_coordinator()
    try:
        good = MagicMock()
        good.collect = AsyncMock(return_value=[
            LLMInferenceCapability(model_id="works"),
        ])
        bad = MagicMock()
        bad.collect = AsyncMock(side_effect=RuntimeError("extractor exploded"))
        coord.register_extractor(bad)
        coord.register_extractor(good)
        caps = await coord.build_local_capabilities()
        # The good extractor's output survived.
        assert len(caps) == 1
        assert caps[0].model_id == "works"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_remote_capabilities_replaces_wholesale():
    conn, coord = await _make_coordinator()
    try:
        await coord.register_paired_peer(_fake_peer("p1"))
        # First heartbeat: peer has model-a + model-b loaded.
        coord.record_remote_capabilities("p1", [
            serialise(LLMInferenceCapability(model_id="a")),
            serialise(LLMInferenceCapability(model_id="b")),
        ])
        assert len(coord.peer_state("p1").capabilities) == 2

        # Second heartbeat: peer has unloaded "a", now only advertises "b".
        coord.record_remote_capabilities("p1", [
            serialise(LLMInferenceCapability(model_id="b")),
        ])
        caps = coord.peer_state("p1").capabilities
        assert len(caps) == 1
        # No stale residue from the first advertisement.
        assert {c.model_id for c in caps} == {"b"}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_remote_capabilities_no_op_for_unknown_peer():
    conn, coord = await _make_coordinator()
    try:
        # Should not raise -- record for a peer we don't know about
        # is silently dropped.
        coord.record_remote_capabilities("never-paired", [
            serialise(LLMInferenceCapability(model_id="x")),
        ])
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_find_peers_with_capability_respects_connected_only():
    conn, coord = await _make_coordinator()
    try:
        # Two peers; one connected, one offline. Both advertise LLM.
        await coord.register_paired_peer(_fake_peer("p_online"))
        await coord.register_paired_peer(_fake_peer("p_offline"))
        await coord.attach_connection("p_online", _FakeWebSocket())
        coord.record_remote_capabilities("p_online", [
            serialise(LLMInferenceCapability(model_id="m1")),
        ])
        coord.record_remote_capabilities("p_offline", [
            serialise(LLMInferenceCapability(model_id="m2")),
        ])

        # Default: connected only.
        matches = coord.find_peers_with_capability(KIND_LLM_INFERENCE)
        assert len(matches) == 1
        assert matches[0][0] == "p_online"

        # With connected_only=False: see both.
        all_matches = coord.find_peers_with_capability(
            KIND_LLM_INFERENCE, connected_only=False,
        )
        assert len(all_matches) == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_capability_summary_aggregates_local_and_remote():
    conn, coord = await _make_coordinator()
    try:
        # Local: 1 LLM + 1 image.
        local_extractor = MagicMock()
        local_extractor.collect = AsyncMock(return_value=[
            LLMInferenceCapability(model_id="local-llm"),
            ImageGenerationCapability(model_id="local-img"),
        ])
        coord.register_extractor(local_extractor)
        await coord.build_local_capabilities()

        # Remote peer: 2 LLMs.
        await coord.register_paired_peer(_fake_peer("p"))
        coord.record_remote_capabilities("p", [
            serialise(LLMInferenceCapability(model_id="r1")),
            serialise(LLMInferenceCapability(model_id="r2")),
        ])

        summary = coord.capability_summary()
        assert summary == {"llm.inference": 3, "image.generation": 1}
    finally:
        await conn.close()
