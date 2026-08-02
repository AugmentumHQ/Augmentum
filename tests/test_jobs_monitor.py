"""Tests — augmentum.jobs.monitor reliability contract.

The contract: every job that reaches a terminal state (success,
failure, cancellation, timeout) emits exactly one
:class:`JobTerminalEvent`. Listeners that raise are isolated. Stale
jobs are force-terminated by the sweeper.

Covers:
  * Successful job → completed event with result
  * Hard-failing job → failed event with error
  * Cancelled job → cancelled event
  * Retryable-with-attempts-remaining does NOT emit (the next attempt
    will; or the final terminal will)
  * Per-job follow-ups fire exactly once
  * Global subscribers fire for every terminal event
  * Stale sweeper marks long-running jobs as failed + emits timed_out
  * Listener exceptions are isolated (other listeners still run)
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.jobs.context import JobCancelled, JobContext, JobRetryable
from augmentum.jobs.monitor import JobMonitor, JobTerminalEvent
from augmentum.jobs.runner import JobRunner, register_handler


@pytest.fixture
async def store():
    """In-memory backend with the background_jobs schema migrated."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.jobs_store import JobsStore

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield JobsStore(backend._conn)
    await backend.close()


@pytest.fixture
def monitor(store):
    return JobMonitor(store, runtime_deadline_s=2, sweep_interval_s=1)


@pytest.fixture
def runner_with_monitor(store, monitor):
    return JobRunner(store, monitor=monitor)


# ─── Listener helpers ──────────────────────────────────────────────────────


class _CollectingListener:
    """Async listener that records every event it receives."""

    def __init__(self):
        self.events: list[JobTerminalEvent] = []

    async def __call__(self, event: JobTerminalEvent) -> None:
        self.events.append(event)


# ─── Terminal-state contract ───────────────────────────────────────────────


class TestTerminalContract:

    @pytest.mark.asyncio
    async def test_successful_job_emits_completed(self, store, monitor, runner_with_monitor):
        listener = _CollectingListener()
        monitor.subscribe(listener)

        async def _ok_handler(ctx: JobContext) -> dict:
            return {"value": 42}

        register_handler("test_ok", _ok_handler)
        job_id = await store.create(
            user_id="u1", job_type="test_ok", payload={"x": 1},
        )
        # Pull the row and drive the runner manually so we can await
        # the terminal cleanly without spinning the worker loop.
        row = await _claim_one(store, job_id)
        await runner_with_monitor._run_one(row)

        assert len(listener.events) == 1
        event = listener.events[0]
        assert event.job_id == job_id
        assert event.user_id == "u1"
        assert event.job_type == "test_ok"
        assert event.outcome == "completed"
        assert event.result == {"value": 42}
        assert event.payload == {"x": 1}

    @pytest.mark.asyncio
    async def test_failing_job_emits_failed(self, store, monitor, runner_with_monitor):
        listener = _CollectingListener()
        monitor.subscribe(listener)

        async def _fail_handler(ctx: JobContext) -> dict:
            raise RuntimeError("kaboom")

        register_handler("test_fail", _fail_handler)
        job_id = await store.create(user_id="u1", job_type="test_fail")
        row = await _claim_one(store, job_id)
        await runner_with_monitor._run_one(row)

        assert len(listener.events) == 1
        assert listener.events[0].outcome == "failed"
        assert "kaboom" in listener.events[0].error

    @pytest.mark.asyncio
    async def test_cancelled_job_emits_cancelled(self, store, monitor, runner_with_monitor):
        listener = _CollectingListener()
        monitor.subscribe(listener)

        async def _cancel_handler(ctx: JobContext) -> dict:
            raise JobCancelled("user requested")

        register_handler("test_cancel", _cancel_handler)
        job_id = await store.create(user_id="u1", job_type="test_cancel")
        row = await _claim_one(store, job_id)
        await runner_with_monitor._run_one(row)

        assert len(listener.events) == 1
        assert listener.events[0].outcome == "cancelled"

    @pytest.mark.asyncio
    async def test_retryable_does_not_emit_when_attempts_remaining(
        self, store, monitor, runner_with_monitor,
    ):
        """Retryable failure with attempts left goes back to pending —
        no terminal event yet. The next attempt's outcome is what
        triggers the contract."""
        listener = _CollectingListener()
        monitor.subscribe(listener)

        async def _retry_handler(ctx: JobContext) -> dict:
            raise JobRetryable("transient")

        register_handler("test_retry", _retry_handler)
        # Enqueue with multiple attempts so the first retry stays in
        # pending instead of terminating.
        job_id = await store.create(
            user_id="u1", job_type="test_retry", max_attempts=3,
        )
        row = await _claim_one(store, job_id)
        await runner_with_monitor._run_one(row)

        assert listener.events == []  # no emit on retryable-with-attempts


# ─── Follow-ups ────────────────────────────────────────────────────────────


class TestFollowUps:

    @pytest.mark.asyncio
    async def test_follow_up_fires_once_for_named_job(self, store, monitor, runner_with_monitor):
        async def _noop(ctx: JobContext) -> dict:
            return {}

        register_handler("test_followup_target", _noop)
        job_id = await store.create(user_id="u1", job_type="test_followup_target")

        followup = _CollectingListener()
        monitor.add_follow_up(job_id, followup)

        row = await _claim_one(store, job_id)
        await runner_with_monitor._run_one(row)

        assert len(followup.events) == 1
        # Follow-up registry pruned after fire.
        assert job_id not in monitor._follow_ups

    @pytest.mark.asyncio
    async def test_follow_up_not_fired_for_other_jobs(self, store, monitor, runner_with_monitor):
        async def _noop(ctx: JobContext) -> dict:
            return {}

        register_handler("test_unrelated", _noop)
        followup = _CollectingListener()
        monitor.add_follow_up("nonexistent_job_id", followup)

        unrelated_job = await store.create(user_id="u1", job_type="test_unrelated")
        row = await _claim_one(store, unrelated_job)
        await runner_with_monitor._run_one(row)

        assert followup.events == []
        # Follow-up for nonexistent job remains registered (waiting
        # for a job that may never run).
        assert "nonexistent_job_id" in monitor._follow_ups


# ─── Listener isolation ────────────────────────────────────────────────────


class TestListenerIsolation:

    @pytest.mark.asyncio
    async def test_raising_listener_does_not_break_others(
        self, store, monitor, runner_with_monitor,
    ):
        async def _ok(ctx: JobContext) -> dict:
            return {}

        register_handler("test_iso", _ok)

        async def _evil(_event):
            raise RuntimeError("listener broke")

        good = _CollectingListener()
        monitor.subscribe(_evil)
        monitor.subscribe(good)

        job_id = await store.create(user_id="u1", job_type="test_iso")
        row = await _claim_one(store, job_id)
        await runner_with_monitor._run_one(row)

        # The good listener still received the event despite the
        # evil one raising. This is the contract — "user always
        # hears back" means no single listener can break the chain.
        assert len(good.events) == 1


# ─── Stale sweeper ─────────────────────────────────────────────────────────


class TestStaleSweeper:

    @pytest.mark.asyncio
    async def test_stale_running_job_force_terminated(self, store, monitor):
        """A job that has been running past the runtime deadline is
        force-terminated and emits a timed_out event."""
        listener = _CollectingListener()
        monitor.subscribe(listener)

        # Enqueue + mark running with an artificially old updated_at.
        job_id = await store.create(user_id="u1", job_type="test_stale")
        # Reach into the store to set running + back-dated updated_at.
        import time as _time
        old = int(_time.time()) - 10_000  # well past the 2s deadline
        await store._conn.execute(
            """UPDATE background_jobs
                  SET status = 'running', updated_at = ?
                WHERE id = ?""",
            (old, job_id),
        )
        await store._conn.commit()

        await monitor._sweep_once()

        # Row was marked failed.
        cur = await store._conn.execute(
            "SELECT status, error FROM background_jobs WHERE id = ?",
            (job_id,),
        )
        row = await cur.fetchone()
        assert row[0] == "failed"
        assert "force-terminated" in row[1]

        # Event was emitted with outcome=timed_out.
        assert len(listener.events) == 1
        assert listener.events[0].outcome == "timed_out"
        assert listener.events[0].job_id == job_id

    @pytest.mark.asyncio
    async def test_fresh_running_job_not_terminated(self, store, monitor):
        """A job whose updated_at is fresh stays running."""
        listener = _CollectingListener()
        monitor.subscribe(listener)

        job_id = await store.create(user_id="u1", job_type="test_fresh")
        import time as _time
        fresh = int(_time.time())
        await store._conn.execute(
            """UPDATE background_jobs
                  SET status = 'running', updated_at = ?
                WHERE id = ?""",
            (fresh, job_id),
        )
        await store._conn.commit()

        await monitor._sweep_once()

        cur = await store._conn.execute(
            "SELECT status FROM background_jobs WHERE id = ?",
            (job_id,),
        )
        row = await cur.fetchone()
        assert row[0] == "running"
        assert listener.events == []


# ─── Helpers ───────────────────────────────────────────────────────────────


async def _claim_one(store, job_id):
    """Pull a job into running state so the runner's terminal-state
    branches behave like they do in the real worker loop."""
    job = await store.claim_next_pending()
    # claim_next_pending picks oldest pending; we expect this is the
    # job we just created in single-test setups.
    assert job is not None and job["id"] == job_id, (
        f"expected {job_id}, got {job!r}"
    )
    return job
