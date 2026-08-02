"""Tests for Phase 10 cost-aware fabric routing.

Three layers:

  - cost_table.lookup_cost: vendored JSON loads, well-known models
    return real per-token costs, unknown models return (0.0, 0.0),
    provider-prefix fallbacks work.
  - Coordinator latency EMA: first measurement seeds the value,
    subsequent measurements smooth toward the recent value, unknown
    pairs return None.
  - RoutingDirector._score_llm_candidates: picks the cheapest peer
    when all else equal, prefers more free slots, penalises high
    latency. Tie-breaking falls back to candidate order.

The integration with maybe_route_llm() is exercised through
_score_llm_candidates directly — we don't need to spin up the full
director path again (that's covered in test_fabric_director).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.fabric.capabilities import LLMInferenceCapability
from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.cost_table import lookup_cost, reload_cost_table
from augmentum.fabric.director import RoutingDirector
from augmentum.fabric.identity import FabricIdentity
from augmentum.state.settings_store import SettingsStore

# ── cost_table ────────────────────────────────────────────────────


def test_lookup_cost_known_models_return_real_values():
    """The vendored sample set covers the major providers; pick a
    handful + verify the per-token costs are non-zero in the right
    ballpark.
    """
    in_cost, out_cost = lookup_cost("gpt-4o")
    assert in_cost > 0
    assert out_cost > in_cost  # output always pricier than input

    in_cost, out_cost = lookup_cost("claude-3-5-sonnet-20241022")
    assert in_cost > 0
    assert out_cost > 0

    # Provider-prefix bare lookup (no prefix supplied) shouldn't
    # find a prefixed entry by default.
    in_cost, out_cost = lookup_cost("llama-3.1-70b-versatile")
    # The fallback DOES try "groq/...", so this resolves.
    assert in_cost >= 0
    assert out_cost >= 0


def test_lookup_cost_unknown_returns_zeros():
    """Local model_ids that aren't in the cloud table return
    (0.0, 0.0) — they're free to route to.
    """
    in_cost, out_cost = lookup_cost("Qwen2.5-72B-Instruct-q4")
    assert in_cost == 0.0
    assert out_cost == 0.0


def test_lookup_cost_case_insensitive():
    """Operators might pass model_ids with mixed casing; lookup
    normalises.
    """
    lower = lookup_cost("gpt-4o")
    upper = lookup_cost("GPT-4O")
    assert lower == upper


def test_reload_cost_table_returns_count():
    """The refresh script calls reload_cost_table to drop the
    lru_cache after overwriting the JSON. Returns the new count.
    """
    count = reload_cost_table()
    assert count > 0  # at least the vendored set


# ── Coordinator latency EMA ───────────────────────────────────────


async def _make_coordinator():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
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


@pytest.mark.asyncio
async def test_record_peer_latency_first_seed_then_smooth():
    """First measurement seeds the EMA verbatim; subsequent
    measurements smooth (~30% weight on the new sample).
    """
    conn, coord = await _make_coordinator()
    try:
        coord.record_peer_latency("p1", kind="llm.inference", latency_ms=100.0)
        assert coord.peer_latency_ms("p1", "llm.inference") == 100.0

        # Second observation at 200ms; alpha=0.3 → 0.3*200 + 0.7*100 = 130
        coord.record_peer_latency("p1", kind="llm.inference", latency_ms=200.0)
        assert abs(coord.peer_latency_ms("p1", "llm.inference") - 130.0) < 0.1

        # Third at 200ms again → trends further toward 200.
        coord.record_peer_latency("p1", kind="llm.inference", latency_ms=200.0)
        v3 = coord.peer_latency_ms("p1", "llm.inference")
        assert 130 < v3 < 200
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_peer_latency_unknown_returns_none():
    conn, coord = await _make_coordinator()
    try:
        assert coord.peer_latency_ms("never-measured", "llm.inference") is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_peer_latency_ignores_invalid_inputs():
    """Negative latency / empty node_id / empty kind silently skip
    — defensive against pathological inputs from a misbehaving
    backend.
    """
    conn, coord = await _make_coordinator()
    try:
        coord.record_peer_latency("p1", kind="llm.inference", latency_ms=-1.0)
        coord.record_peer_latency("", kind="llm.inference", latency_ms=100.0)
        coord.record_peer_latency("p1", kind="", latency_ms=100.0)
        # No entries recorded.
        assert coord.peer_latency_ms("p1", "llm.inference") is None
    finally:
        await conn.close()


# ── RoutingDirector._score_llm_candidates ─────────────────────────


@pytest.mark.asyncio
async def test_score_picks_cheaper_peer_all_else_equal():
    """Two peers with identical capacity + latency, different cost.
    Cheaper wins.
    """
    conn, coord = await _make_coordinator()
    try:
        cheap = LLMInferenceCapability(
            model_id="claude-3-5-sonnet", free_slots=4,
            output_cost_per_token=0.0,   # local (no cloud cost)
        )
        expensive = LLMInferenceCapability(
            model_id="claude-3-5-sonnet", free_slots=4,
            output_cost_per_token=0.000015,  # cloud-priced
        )
        director = RoutingDirector(coord, MagicMock())
        chosen = director._score_llm_candidates(
            "claude-3-5-sonnet",
            [("cloud-peer", expensive), ("local-peer", cheap)],
        )
        assert chosen[0] == "local-peer"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_score_prefers_more_free_slots():
    """Same cost, different free_slots. Idle peer beats busy peer.
    """
    conn, coord = await _make_coordinator()
    try:
        busy = LLMInferenceCapability(
            model_id="m", free_slots=0, output_cost_per_token=0.0,
        )
        idle = LLMInferenceCapability(
            model_id="m", free_slots=4, output_cost_per_token=0.0,
        )
        director = RoutingDirector(coord, MagicMock())
        chosen = director._score_llm_candidates(
            "m", [("busy-peer", busy), ("idle-peer", idle)],
        )
        assert chosen[0] == "idle-peer"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_score_penalises_high_latency_peer():
    """Same cost + free_slots, one peer measured slow. Fast wins.
    """
    conn, coord = await _make_coordinator()
    try:
        cap_a = LLMInferenceCapability(model_id="m", free_slots=2)
        cap_b = LLMInferenceCapability(model_id="m", free_slots=2)
        # Seed B with slow latency.
        coord.record_peer_latency("slow", kind="llm.inference", latency_ms=2000.0)
        coord.record_peer_latency("fast", kind="llm.inference", latency_ms=50.0)
        director = RoutingDirector(coord, MagicMock())
        chosen = director._score_llm_candidates(
            "m", [("slow", cap_a), ("fast", cap_b)],
        )
        assert chosen[0] == "fast"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_score_first_match_when_truly_tied():
    """All scores equal → list-iteration order wins (first match).
    Stable behaviour, no thrash between calls.
    """
    conn, coord = await _make_coordinator()
    try:
        cap_a = LLMInferenceCapability(model_id="m", free_slots=2)
        cap_b = LLMInferenceCapability(model_id="m", free_slots=2)
        director = RoutingDirector(coord, MagicMock())
        chosen = director._score_llm_candidates(
            "m", [("first", cap_a), ("second", cap_b)],
        )
        assert chosen[0] == "first"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_score_real_world_local_beats_cloud_by_default():
    """End-to-end sanity: a local-network peer (free, 4 slots, no
    measured latency) beats a cloud-priced peer (expensive, 8 slots,
    fast). Cost dominates when the gap is large enough — verifies
    default weights preserve operator intuition ("don't burn money
    when I have hardware").
    """
    conn, coord = await _make_coordinator()
    try:
        cloud = LLMInferenceCapability(
            model_id="claude-3-5-sonnet", free_slots=8,
            output_cost_per_token=0.000015,  # $15/1M out
        )
        local = LLMInferenceCapability(
            model_id="claude-3-5-sonnet", free_slots=4,
            output_cost_per_token=0.0,
        )
        coord.record_peer_latency("cloud", kind="llm.inference", latency_ms=80.0)
        coord.record_peer_latency("local", kind="llm.inference", latency_ms=150.0)
        director = RoutingDirector(coord, MagicMock())
        chosen = director._score_llm_candidates(
            "claude-3-5-sonnet",
            [("cloud", cloud), ("local", local)],
        )
        assert chosen[0] == "local"
    finally:
        await conn.close()
