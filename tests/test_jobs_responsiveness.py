"""Verify the JobRunner does not starve the asyncio event loop.

Two separate questions:
  1. Does the runner itself (poll loop + store ops) block the loop?
  2. When handlers offload CPU-bound work via ctx.run_in_thread, does
     the loop stay responsive for concurrent requests?

Both should answer "no/yes". A regression in either would manifest as
chat streams stalling while a transcription runs, which is the whole
point of building this as a job queue instead of a synchronous route.
"""

from __future__ import annotations

import asyncio
import time

import aiosqlite
import pytest

from augmentum.jobs import JobRunner, register_handler
from augmentum.state.jobs_store import JobsStore


_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);
CREATE TABLE background_jobs (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(id),
    job_type         TEXT NOT NULL,
    payload          TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending',
    progress         REAL NOT NULL DEFAULT 0.0,
    stage            TEXT NOT NULL DEFAULT '',
    result           TEXT,
    error            TEXT,
    priority         INTEGER NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL DEFAULT 3,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    started_at       INTEGER,
    updated_at       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    completed_at     INTEGER
);
"""


async def _mkstore() -> JobsStore:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.commit()
    return JobsStore(conn)


async def _count_ticks_during(duration_s: float, interval_s: float = 0.01) -> int:
    """Count how often asyncio.sleep(interval) returns within ``duration``.

    A quiet loop fires ~duration/interval times (e.g. 0.5/0.01 ≈ 50).
    A blocked loop fires ~0. The gap is the signal.
    """
    ticks = 0
    deadline = asyncio.get_event_loop().time() + duration_s
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(interval_s)
        ticks += 1
    return ticks


@pytest.mark.asyncio
async def test_idle_runner_does_not_starve_loop():
    """Baseline: a runner with nothing to do shouldn't affect other work."""
    store = await _mkstore()
    runner = JobRunner(store)
    runner.start()
    try:
        ticks = await _count_ticks_during(0.3)
    finally:
        await runner.stop()
    # At 10ms intervals over 300ms we expect ~30 ticks. Accept >= 20 to
    # tolerate CI noise but still fail if the loop is being starved.
    assert ticks >= 20, f"idle runner starved loop: only {ticks} ticks"


@pytest.mark.asyncio
async def test_async_handler_does_not_block_concurrent_work():
    """An async-clean handler should coexist with other event-loop tasks."""
    store = await _mkstore()

    async def well_behaved(ctx):
        # Simulate legitimate async work split across yields. Total: ~0.3s.
        for i in range(30):
            await asyncio.sleep(0.01)
            if i % 10 == 0:
                await ctx.update_progress(i / 30.0)

    register_handler("responsive-async", well_behaved)
    await store.create(user_id="u1", job_type="responsive-async")

    runner = JobRunner(store)
    runner.start()
    try:
        # Measure loop responsiveness while the handler runs.
        ticks = await _count_ticks_during(0.3)
    finally:
        await runner.stop()
    assert ticks >= 20, (
        f"async handler starved loop: only {ticks} ticks — handler may be "
        "doing synchronous work it shouldn't"
    )


@pytest.mark.asyncio
async def test_blocking_work_offloaded_via_run_in_thread_stays_non_blocking():
    """The documented pattern (ctx.run_in_thread) must keep the loop responsive.

    This is the dominant real-world case: Moonshine transcription, ffmpeg
    subprocesses, pytorch inference — all synchronous library calls that
    must be offloaded. If this test regresses, every blocking handler in
    the codebase silently freezes the app.
    """
    store = await _mkstore()

    async def cpu_bound(ctx):
        # time.sleep is a stand-in for any blocking library call. On the
        # event loop it would hang everything; offloaded to a thread it
        # doesn't. That's the property under test.
        await ctx.run_in_thread(time.sleep, 0.3)
        return {"done": True}

    register_handler("responsive-cpu", cpu_bound)
    await store.create(user_id="u1", job_type="responsive-cpu")

    runner = JobRunner(store)
    runner.start()
    try:
        ticks = await _count_ticks_during(0.3)
    finally:
        await runner.stop()
    assert ticks >= 20, (
        f"blocking work starved loop despite run_in_thread: only {ticks} ticks"
    )
