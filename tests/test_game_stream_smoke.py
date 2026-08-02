"""Smoke tests for the Game Streaming Platform (AGSP) foundation.

Covers: module imports, store CRUD round-trip + user-scoping,
lifecycle state machine, port pool, and runtime orchestration with
the stub container adapter. No Docker, no network -- this is the
"does the foundation hold together" test.
"""

from __future__ import annotations

import aiosqlite
import pytest

# Minimal schema covering migrations 120-122. Mirrors the SQL in
# augmentum/state/migrations/ -- if it drifts, the migration is the
# source of truth and this fixture stays in sync via a quick edit.
_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);

CREATE TABLE game_stream_sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    world_id        TEXT,
    profile_id      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'stopped',
    container_id    TEXT,
    stream_port     INTEGER,
    game_port       INTEGER,
    bitrate_mbps    INTEGER NOT NULL DEFAULT 4,
    resolution      TEXT NOT NULL DEFAULT '1280x720',
    encoder         TEXT NOT NULL DEFAULT 'auto',
    exit_reason     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at    TEXT NOT NULL DEFAULT (datetime('now')),
    cast_input_token TEXT,
    system_id      TEXT,
    paused_at      TEXT
);

CREATE TABLE game_stream_worlds (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    profile_id          TEXT NOT NULL,
    name                TEXT NOT NULL,
    settings_json       TEXT NOT NULL DEFAULT '{}',
    whitelist_user_ids  TEXT NOT NULL DEFAULT '[]',
    storage_path        TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_played_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE game_stream_telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL REFERENCES users(id),
    ts              TEXT NOT NULL DEFAULT (datetime('now')),
    rtt_ms          REAL,
    jitter_ms       REAL,
    packet_loss     REAL,
    bitrate_kbps    INTEGER,
    fps             REAL,
    FOREIGN KEY (session_id) REFERENCES game_stream_sessions(id) ON DELETE CASCADE
);
"""


async def _mkstore():
    from augmentum.state.game_stream_store import GameStreamStore

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    return GameStreamStore(conn), conn


# ── Imports ───────────────────────────────────────────────────────


def test_subsystem_imports():
    """Public surface imports without error."""
    from augmentum.game_stream import (  # noqa: F401
        ConcurrentStreamLimitError,
        GameProfile,
        GameStreamLifecycle,
        GameStreamRuntime,
        LifecycleTransitionError,
        PortPool,
        PortPoolExhausted,
        ProfileRegistry,
        SessionStatus,
        profile_registry,
    )
    from augmentum.proxy.game_stream_routes import router  # noqa: F401


def test_default_profile_registered():
    from augmentum.game_stream import profile_registry

    luanti = profile_registry.get("luanti")
    assert luanti is not None
    assert luanti.id == "luanti"
    assert luanti.multiplayer is True
    assert luanti.scriptable is True
    assert luanti.wants_gamepad is True
    assert luanti.input_capabilities["gamepad"]["supported"] is True
    assert luanti.input_capabilities["pointer"]["mouse_sensitivity"]["default"] == 0.2


# ── Lifecycle state machine ───────────────────────────────────────


def test_lifecycle_legal_transitions():
    from augmentum.game_stream import GameStreamLifecycle, SessionStatus

    L = GameStreamLifecycle
    # cold start path
    assert L.can_transition(SessionStatus.STOPPED, SessionStatus.STARTING)
    assert L.can_transition(SessionStatus.STARTING, SessionStatus.READY)
    assert L.can_transition(SessionStatus.READY, SessionStatus.CONNECTED)
    assert L.can_transition(SessionStatus.CONNECTED, SessionStatus.IDLE)
    assert L.can_transition(SessionStatus.IDLE, SessionStatus.READY)
    # graceful shutdown
    assert L.can_transition(SessionStatus.IDLE, SessionStatus.STOPPING)
    assert L.can_transition(SessionStatus.STOPPING, SessionStatus.STOPPED)


def test_lifecycle_rejects_illegal_transitions():
    from augmentum.game_stream import GameStreamLifecycle, LifecycleTransitionError, SessionStatus

    L = GameStreamLifecycle
    # can't skip starting
    assert not L.can_transition(SessionStatus.STOPPED, SessionStatus.CONNECTED)
    # can't restart stopped without starting
    assert not L.can_transition(SessionStatus.STOPPED, SessionStatus.READY)
    # explicit transition raises
    with pytest.raises(LifecycleTransitionError):
        L.transition(SessionStatus.STOPPED, SessionStatus.CONNECTED)


def test_lifecycle_classifies_states():
    from augmentum.game_stream import GameStreamLifecycle, SessionStatus

    L = GameStreamLifecycle
    assert L.is_running(SessionStatus.CONNECTED)
    assert L.is_running(SessionStatus.IDLE)
    assert not L.is_running(SessionStatus.STOPPED)
    assert L.is_terminal(SessionStatus.STOPPED)
    assert L.is_terminal(SessionStatus.CRASHED)


# ── Port pool ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_port_pool_allocate_and_release():
    from augmentum.game_stream import PortPool, PortPoolExhausted

    pool = PortPool(base=40000, count=4)  # 2 pairs
    a = await pool.allocate()
    b = await pool.allocate()
    assert a.stream_port == 40000 and a.game_port == 40001
    assert b.stream_port == 40002 and b.game_port == 40003
    with pytest.raises(PortPoolExhausted):
        await pool.allocate()
    await pool.release(a.stream_port)
    c = await pool.allocate()
    assert c.stream_port == 40000


@pytest.mark.asyncio
async def test_port_pool_reconcile_rebuilds_in_use():
    from augmentum.game_stream import PortPool

    pool = PortPool(base=40000, count=10)
    rows = [
        {"stream_port": 40000},
        {"stream_port": 40004},
        {"stream_port": 99999},  # out-of-range, ignored
    ]
    await pool.reconcile(rows)
    assert pool.in_use == 2
    a = await pool.allocate()
    # First free is 40002.
    assert a.stream_port == 40002


# ── Store: sessions ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_create_get_roundtrip():
    store, _ = await _mkstore()
    sid = await store.create_session(
        user_id="u1", profile_id="luanti", bitrate_mbps=6, resolution="1920x1080",
    )
    got = await store.get_session(sid, user_id="u1")
    assert got is not None
    assert got["profile_id"] == "luanti"
    assert got["bitrate_mbps"] == 6
    assert got["resolution"] == "1920x1080"
    assert got["status"] == "starting"


@pytest.mark.asyncio
async def test_session_user_scoped():
    store, _ = await _mkstore()
    sid = await store.create_session(user_id="u1", profile_id="luanti")
    # u2 can't see u1's session
    assert await store.get_session(sid, user_id="u2") is None
    # u2's listing is empty
    assert await store.list_sessions_for_user(user_id="u2") == []


@pytest.mark.asyncio
async def test_session_update_status_and_ports():
    store, _ = await _mkstore()
    sid = await store.create_session(user_id="u1", profile_id="luanti")
    ok = await store.update_session(
        sid,
        user_id="u1",
        status="ready",
        stream_port=30000,
        game_port=30001,
        container_id="cid-abc",
    )
    assert ok
    got = await store.get_session(sid, user_id="u1")
    assert got["status"] == "ready"
    assert got["stream_port"] == 30000
    assert got["container_id"] == "cid-abc"


@pytest.mark.asyncio
async def test_count_live_for_user():
    store, _ = await _mkstore()
    a = await store.create_session(user_id="u1", profile_id="luanti")
    b = await store.create_session(user_id="u1", profile_id="luanti")
    await store.update_session(a, user_id="u1", status="connected")
    await store.update_session(b, user_id="u1", status="stopped")
    assert await store.count_live_for_user(user_id="u1") == 1


# ── Store: worlds ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_world_roundtrip_and_whitelist_visibility():
    store, _ = await _mkstore()
    wid = await store.create_world(
        user_id="u1",
        profile_id="luanti",
        name="Home",
        settings={"gamemode": "survival", "seed": "42"},
        whitelist=["u2"],
    )
    # owner sees it
    own = await store.list_worlds_for_user(user_id="u1")
    assert len(own) == 1 and own[0]["id"] == wid
    # whitelisted user also sees it
    guest = await store.list_worlds_for_user(user_id="u2")
    assert len(guest) == 1 and guest[0]["id"] == wid
    # settings_json round-tripped as a dict
    assert own[0]["settings_json"] == {"gamemode": "survival", "seed": "42"}


# ── Runtime: orchestration with stub adapter ──────────────────────


@pytest.mark.asyncio
async def test_runtime_start_session_assigns_ports_and_returns_info():
    from augmentum.game_stream import GameStreamRuntime, PortPool

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=30000, count=4),
        max_concurrent_per_user=2,
    )
    info = await rt.start_session(user_id="u1", profile_id="luanti")
    assert info.session_id
    assert info.stream_port == 30000
    assert info.game_port == 30001
    # signaling path includes the session id so the WS handler can route.
    assert info.session_id in info.signaling_path
    # Underlying row has container_id set by the stub adapter.
    row = await store.get_session(info.session_id, user_id="u1")
    assert row["container_id"].startswith("stub-")


@pytest.mark.asyncio
async def test_runtime_enforces_concurrent_cap():
    from augmentum.game_stream import (
        ConcurrentStreamLimitError,
        GameStreamRuntime,
        PortPool,
    )

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=30000, count=4),
        max_concurrent_per_user=1,
    )
    await rt.start_session(user_id="u1", profile_id="luanti")
    with pytest.raises(ConcurrentStreamLimitError):
        await rt.start_session(user_id="u1", profile_id="luanti")


@pytest.mark.asyncio
async def test_runtime_stop_releases_ports():
    from augmentum.game_stream import GameStreamRuntime, PortPool

    store, _ = await _mkstore()
    pool = PortPool(base=30000, count=4)
    rt = GameStreamRuntime(store=store, port_pool=pool, max_concurrent_per_user=2)
    info = await rt.start_session(user_id="u1", profile_id="luanti")
    assert pool.in_use == 1
    ok = await rt.stop_session(info.session_id, user_id="u1")
    assert ok
    assert pool.in_use == 0
    row = await store.get_session(info.session_id, user_id="u1")
    assert row["status"] == "stopped"
    assert row["exit_reason"] == "clean"


@pytest.mark.asyncio
async def test_runtime_lifecycle_marks():
    from augmentum.game_stream import GameStreamRuntime, PortPool

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=30000, count=4),
        max_concurrent_per_user=2,
    )
    info = await rt.start_session(user_id="u1", profile_id="luanti")
    assert await rt.mark_ready(info.session_id, user_id="u1")
    assert await rt.mark_connected(info.session_id, user_id="u1")
    # Idle from connected is legal.
    assert await rt.mark_idle(info.session_id, user_id="u1")
    # Reconnect: idle -> connected is legal.
    assert await rt.mark_connected(info.session_id, user_id="u1")


# ── Credit budget admission ───────────────────────────────────────


@pytest.mark.asyncio
async def test_admit_credit_budget_caps_solo_user():
    """Solo user gets up to user_hard_cap, then is blocked."""
    from augmentum.game_stream import (
        ConcurrentStreamLimitError,
        GameStreamRuntime,
        PortPool,
    )

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=30000, count=20),
        # Loose max_concurrent so the credit cap is what fires.
        max_concurrent_per_user=10,
        active_credit_budget=8,
        resident_credit_budget=16,
        user_hard_cap=3,
    )
    # Luanti = 1 credit. Solo user should hit user_hard_cap=3 before
    # the host budget.
    for _ in range(3):
        await rt.start_session(user_id="u1", profile_id="luanti")
    with pytest.raises(ConcurrentStreamLimitError):
        await rt.start_session(user_id="u1", profile_id="luanti")


@pytest.mark.asyncio
async def test_admit_heavy_profile_costs_more():
    """A profile with cost_credits=2 fills the budget faster."""
    from augmentum.game_stream import (
        ConcurrentStreamLimitError,
        GameStreamRuntime,
        PortPool,
    )
    from augmentum.game_stream.profiles import GameProfile, ProfileRegistry

    reg = ProfileRegistry()
    reg.register(GameProfile(
        id="heavy", display_name="Heavy", image="x:latest",
        cost_credits=3,
    ))
    reg.register(GameProfile(
        id="light", display_name="Light", image="x:latest",
        cost_credits=1,
    ))
    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store, registry=reg,
        port_pool=PortPool(base=30000, count=20),
        max_concurrent_per_user=10,
        active_credit_budget=5,
        resident_credit_budget=10,
        user_hard_cap=10,
    )
    # 3 + 1 = 4 credits, fits under budget 5.
    await rt.start_session(user_id="u1", profile_id="heavy")
    await rt.start_session(user_id="u1", profile_id="light")
    # Next heavy (3) would push us to 7 > 5 budget.
    with pytest.raises(ConcurrentStreamLimitError):
        await rt.start_session(user_id="u1", profile_id="heavy")
    # But another light (1) fits — 4 + 1 = 5.
    await rt.start_session(user_id="u1", profile_id="light")


@pytest.mark.asyncio
async def test_admit_fair_share_with_two_users():
    """Two active users split the budget in half."""
    from augmentum.game_stream import (
        ConcurrentStreamLimitError,
        GameStreamRuntime,
        PortPool,
    )

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=30000, count=20),
        max_concurrent_per_user=10,
        active_credit_budget=4,
        resident_credit_budget=10,
        user_hard_cap=10,
    )
    # User 1 takes 2 Luanti sessions (cost 2). Now active_users=1 so
    # they could take more — but adding u2 shifts the fair share.
    await rt.start_session(user_id="u1", profile_id="luanti")
    await rt.start_session(user_id="u1", profile_id="luanti")
    # User 2 arrives — fair share is now budget/2 = 2 per user.
    await rt.start_session(user_id="u2", profile_id="luanti")
    await rt.start_session(user_id="u2", profile_id="luanti")
    # u2's 3rd would exceed fair share.
    with pytest.raises(ConcurrentStreamLimitError):
        await rt.start_session(user_id="u2", profile_id="luanti")
    # u1's 3rd also exceeds — budget is fully consumed regardless of who.
    with pytest.raises(ConcurrentStreamLimitError):
        await rt.start_session(user_id="u1", profile_id="luanti")


# ── Pause primitive ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pause_resume_session_round_trip():
    from augmentum.game_stream import (
        GameStreamRuntime,
        PortPool,
        SessionStatus,
    )

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=30000, count=4),
        max_concurrent_per_user=2,
    )
    info = await rt.start_session(user_id="u1", profile_id="luanti")
    await rt.mark_ready(info.session_id, user_id="u1")
    await rt.mark_connected(info.session_id, user_id="u1")
    # Pause from CONNECTED is legal.
    assert await rt.pause_session(info.session_id, user_id="u1")
    row = await store.get_session(info.session_id, user_id="u1")
    assert row["status"] == SessionStatus.PAUSED.value
    assert row.get("paused_at")  # stamp is set
    # Idempotent pause.
    assert await rt.pause_session(info.session_id, user_id="u1")
    # Resume goes back to CONNECTED.
    assert await rt.resume_session(info.session_id, user_id="u1")
    row = await store.get_session(info.session_id, user_id="u1")
    assert row["status"] == SessionStatus.CONNECTED.value
    assert not row.get("paused_at")  # cleared


@pytest.mark.asyncio
async def test_pause_frees_active_credits_for_others():
    """Pausing a heavy session frees its active credits."""
    from augmentum.game_stream import (
        ConcurrentStreamLimitError,
        GameStreamRuntime,
        PortPool,
    )
    from augmentum.game_stream.profiles import GameProfile, ProfileRegistry

    reg = ProfileRegistry()
    reg.register(GameProfile(
        id="heavy", display_name="Heavy", image="x:latest", cost_credits=3,
    ))
    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store, registry=reg,
        port_pool=PortPool(base=30000, count=20),
        max_concurrent_per_user=10,
        active_credit_budget=4,
        resident_credit_budget=20,
        user_hard_cap=10,
    )
    info1 = await rt.start_session(user_id="u1", profile_id="heavy")
    # Drive to CONNECTED — pause from STARTING isn't legal.
    await rt.mark_ready(info1.session_id, user_id="u1")
    await rt.mark_connected(info1.session_id, user_id="u1")
    # 3 credits used; another heavy (3) would overflow (6 > 4).
    with pytest.raises(ConcurrentStreamLimitError):
        await rt.start_session(user_id="u1", profile_id="heavy")
    # Pause the first; active credits drop to 0.
    assert await rt.pause_session(info1.session_id, user_id="u1")
    # Now the second heavy fits.
    await rt.start_session(user_id="u1", profile_id="heavy")


# ── Watchdog liveness ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_watchdog_reaps_dead_container():
    """sweep_idle marks a session CRASHED when the adapter says
    its container has died without going through stop_session."""
    from augmentum.game_stream import (
        GameStreamRuntime,
        PortPool,
        SessionStatus,
    )
    from augmentum.game_stream.runtime import StubContainerAdapter

    class DeadAdapter(StubContainerAdapter):
        async def is_alive(self, container_id: str) -> bool:
            return False  # container died

    store, _ = await _mkstore()
    rt = GameStreamRuntime(
        store=store, adapter=DeadAdapter(),
        port_pool=PortPool(base=30000, count=4),
        max_concurrent_per_user=2,
    )
    info = await rt.start_session(user_id="u1", profile_id="luanti")
    await rt.mark_ready(info.session_id, user_id="u1")
    # The container died after start — sweep should detect.
    stopped = await rt.sweep_idle()
    assert stopped == 1
    row = await store.get_session(info.session_id, user_id="u1")
    assert row["status"] == SessionStatus.CRASHED.value
    assert row["exit_reason"] == "watchdog_dead"


# ── Pause lifecycle transitions ───────────────────────────────────


def test_lifecycle_includes_paused_transitions():
    from augmentum.game_stream import GameStreamLifecycle, SessionStatus

    L = GameStreamLifecycle
    # PAUSED is reachable from active states
    assert L.can_transition(SessionStatus.CONNECTED, SessionStatus.PAUSED)
    assert L.can_transition(SessionStatus.IDLE, SessionStatus.PAUSED)
    assert L.can_transition(SessionStatus.READY, SessionStatus.PAUSED)
    # Resume to CONNECTED or READY
    assert L.can_transition(SessionStatus.PAUSED, SessionStatus.CONNECTED)
    assert L.can_transition(SessionStatus.PAUSED, SessionStatus.READY)
    # Can stop a paused session
    assert L.can_transition(SessionStatus.PAUSED, SessionStatus.STOPPING)
    # PAUSED still counts as running
    assert L.is_running(SessionStatus.PAUSED)
