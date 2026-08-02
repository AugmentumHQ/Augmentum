"""Tests for the general-purpose strain monitor (StrainMonitor).

Covers the sample → store → read-back round trip against an in-memory DB with
migration 272 applied, plus the multi-client correlation logic (stale clients
excluded, users deduped) that makes multi-browser contention observable.
"""

from __future__ import annotations

import time
import types

import aiosqlite
import pytest

from augmentum.health import StrainMonitor

_MIGRATION = "augmentum/state/migrations/272_strain_samples_health_monitor_table.sql"


async def _db_with_table() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    with open(_MIGRATION, encoding="utf-8") as fh:
        await conn.executescript(fh.read())
    await conn.commit()
    return conn


def _fake_app(**overrides) -> types.SimpleNamespace:
    now = time.monotonic()
    state = types.SimpleNamespace(
        last_event_loop_lag_s=0.45,
        inflight_requests=3,
        slow_request_count=2,
        active_clients={
            "tabA": (now, "u1"),
            "tabB": (now, "u1"),
            "tabC": (now, "u2"),
            "stale": (now - 999, "u9"),  # outside the freshness window
        },
        presence_pipelines={"x": 1},
        narrative_engines={"s1": 1, "s2": 1},
        agentic_handlers={},
        notification_hub=None,
        container_manager=types.SimpleNamespace(_containers={"w1": 1}),
        llama_manager=types.SimpleNamespace(model_id="qwen3.5-40b"),
        secondary_slot=None,
        resource_ledger=None,
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return types.SimpleNamespace(state=state)


@pytest.mark.asyncio
async def test_sample_round_trip_and_correlation():
    conn = await _db_with_table()
    try:
        app = _fake_app()
        mon = StrainMonitor(conn, app)
        sample = await mon.sample_and_store()
        assert sample is not None

        cur = await conn.execute(
            "SELECT active_clients, active_users, inflight_requests, slow_requests, "
            "engine_model, sessions_coder, sessions_narrative FROM strain_samples"
        )
        row = await cur.fetchone()
        assert row is not None, "sample row was not written"
        active_clients, active_users, inflight, slow, engine, coder, narr = row

        # Stale client (last seen 999s ago) is excluded; 3 fresh remain.
        assert active_clients == 3
        # u1 appears twice, u2 once -> 2 distinct users.
        assert active_users == 2
        assert inflight == 3
        assert slow == 2
        assert engine == "qwen3.5-40b"
        assert coder == 1
        assert narr == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_slow_request_counter_is_read_and_reset():
    conn = await _db_with_table()
    try:
        app = _fake_app(slow_request_count=5)
        mon = StrainMonitor(conn, app)
        await mon.sample_and_store()
        # Each sample reports the delta since the last sample, so the counter
        # must be zeroed after it's read.
        assert app.state.slow_request_count == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_no_sqlite_backend_is_a_noop():
    app = _fake_app()
    mon = StrainMonitor(None, app)  # memory backend / no conn
    assert await mon.sample_and_store() is None


@pytest.mark.asyncio
async def test_empty_active_clients_counts_zero():
    conn = await _db_with_table()
    try:
        app = _fake_app(active_clients={})
        mon = StrainMonitor(conn, app)
        sample = await mon.sample_and_store()
        assert sample["active_clients"] == 0
        assert sample["active_users"] == 0
    finally:
        await conn.close()
