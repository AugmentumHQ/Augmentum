"""JobRunner dispatch + cancel + retry tests."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from augmentum.jobs import JobRunner, register_handler
from augmentum.jobs.context import JobRetryable
from augmentum.state.jobs_store import JobsStore

# Same minimal schema as test_jobs_store (intentional duplication — keeps
# each test file self-contained). See that file for the rationale.
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


async def _mkstore():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.commit()
    return JobsStore(conn)


async def _wait_for(pred, *, timeout: float = 3.0):
    """Poll an async predicate until true or we time out."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await pred():
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_handler_runs_and_receives_payload():
    store = await _mkstore()
    seen: dict = {}

    async def handler(ctx):
        seen["payload"] = ctx.payload
        seen["user_id"] = ctx.user_id
        await ctx.update_progress(0.5, "halfway")
        return {"doubled": ctx.payload["n"] * 2}

    register_handler("test-ok", handler)
    job_id = await store.create(user_id="u1", job_type="test-ok", payload={"n": 7})

    runner = JobRunner(store)
    runner.start()
    try:
        done = await _wait_for(
            lambda: _is_status(store, job_id, "completed"),
        )
        assert done, "handler never completed"
    finally:
        await runner.stop()

    got = await store.get(job_id)
    assert got["status"] == "completed"
    assert got["result"] == {"doubled": 14}
    assert got["progress"] == 1.0  # mark_completed pins to 1.0
    assert seen["payload"] == {"n": 7}
    assert seen["user_id"] == "u1"


@pytest.mark.asyncio
async def test_unknown_job_type_fails_terminally():
    store = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="nobody-registers-this")

    runner = JobRunner(store)
    runner.start()
    try:
        done = await _wait_for(lambda: _is_status(store, job_id, "failed"))
        assert done
    finally:
        await runner.stop()

    got = await store.get(job_id)
    assert "No handler registered" in (got["error"] or "")


@pytest.mark.asyncio
async def test_cooperative_cancel_marks_cancelled():
    store = await _mkstore()
    started = asyncio.Event()

    async def slow_handler(ctx):
        started.set()
        # Busy-wait for cancel signal.
        for _ in range(100):
            await asyncio.sleep(0.05)
            await ctx.check_cancel()

    register_handler("test-cancel", slow_handler)
    job_id = await store.create(user_id="u1", job_type="test-cancel")

    runner = JobRunner(store)
    runner.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=2.0)
        await store.request_cancel(job_id, user_id="u1")
        done = await _wait_for(lambda: _is_status(store, job_id, "cancelled"))
        assert done
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_retryable_failure_reverts_then_eventually_completes():
    store = await _mkstore()
    attempts = {"n": 0}

    async def flaky(ctx):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise JobRetryable("transient")
        return {"attempts": attempts["n"]}

    register_handler("test-flaky", flaky)
    job_id = await store.create(
        user_id="u1", job_type="test-flaky", max_attempts=3,
    )

    runner = JobRunner(store)
    runner.start()
    try:
        done = await _wait_for(
            lambda: _is_status(store, job_id, "completed"),
            timeout=10.0,
        )
        assert done
    finally:
        await runner.stop()

    got = await store.get(job_id)
    assert got["result"] == {"attempts": 2}


@pytest.mark.asyncio
async def test_unhandled_exception_marks_failed():
    store = await _mkstore()

    async def crasher(ctx):
        raise RuntimeError("boom")

    register_handler("test-crash", crasher)
    job_id = await store.create(user_id="u1", job_type="test-crash")

    runner = JobRunner(store)
    runner.start()
    try:
        done = await _wait_for(lambda: _is_status(store, job_id, "failed"))
        assert done
    finally:
        await runner.stop()

    got = await store.get(job_id)
    assert "boom" in (got["error"] or "")


async def _is_status(store: JobsStore, job_id: str, status: str) -> bool:
    got = await store.get(job_id)
    return got is not None and got["status"] == status
