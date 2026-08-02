"""JobsStore round-trip + dispatch tests (pure SQLite, no app stack)."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

# Minimal schema mirroring migration 102. Duplicating here (rather than
# wiring the full migration runner) keeps this a unit test: one file, one
# table, no app-state assembly. If the migration drifts from this, the
# real schema is the source of truth — this just keeps the test fast.
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
    from augmentum.state.jobs_store import JobsStore

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    return JobsStore(conn), conn


@pytest.mark.asyncio
async def test_create_and_get_roundtrip():
    store, _ = await _mkstore()
    job_id = await store.create(
        user_id="u1", job_type="demo", payload={"x": 1}, priority=5,
    )
    got = await store.get(job_id, user_id="u1")
    assert got is not None
    assert got["job_type"] == "demo"
    assert got["payload"] == {"x": 1}
    assert got["status"] == "pending"
    assert got["priority"] == 5


@pytest.mark.asyncio
async def test_get_is_user_scoped():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo")
    # User u2 can't see u1's job.
    assert await store.get(job_id, user_id="u2") is None
    # No user_id passed = bypass filter (trusted internal calls).
    assert await store.get(job_id) is not None


@pytest.mark.asyncio
async def test_claim_respects_priority_and_age():
    store, _ = await _mkstore()
    low = await store.create(user_id="u1", job_type="demo", priority=1)
    # Small sleep so created_at differs from the next row.
    await asyncio.sleep(1.1)
    high = await store.create(user_id="u1", job_type="demo", priority=10)
    # Even though ``low`` was created first, high priority wins.
    claimed = await store.claim_next_pending()
    assert claimed["id"] == high
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1
    # Next claim picks the remaining one.
    nxt = await store.claim_next_pending()
    assert nxt["id"] == low


@pytest.mark.asyncio
async def test_claim_skips_cancel_requested_pending():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo")
    await store.request_cancel(job_id, user_id="u1")
    # request_cancel on a pending job terminates it immediately, so the
    # next claim has nothing to hand back.
    assert await store.claim_next_pending() is None
    final = await store.get(job_id, user_id="u1")
    assert final["status"] == "cancelled"


@pytest.mark.asyncio
async def test_mark_completed_stores_result():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo")
    await store.claim_next_pending()
    await store.mark_completed(job_id, result={"ok": True, "count": 3})
    got = await store.get(job_id, user_id="u1")
    assert got["status"] == "completed"
    assert got["progress"] == 1.0
    assert got["result"] == {"ok": True, "count": 3}


@pytest.mark.asyncio
async def test_mark_failed_retryable_reverts_to_pending():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo", max_attempts=3)
    await store.claim_next_pending()  # attempts → 1
    await store.mark_failed(job_id, error="flake", retryable=True)
    got = await store.get(job_id, user_id="u1")
    assert got["status"] == "pending"
    assert got["error"] == "flake"


@pytest.mark.asyncio
async def test_mark_failed_retryable_terminates_after_max_attempts():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo", max_attempts=2)
    await store.claim_next_pending()  # attempts 1
    await store.mark_failed(job_id, error="still flake", retryable=True)
    await store.claim_next_pending()  # attempts 2
    await store.mark_failed(job_id, error="still flake", retryable=True)
    got = await store.get(job_id, user_id="u1")
    assert got["status"] == "failed"


@pytest.mark.asyncio
async def test_requeue_crashed_resets_running_jobs():
    store, conn = await _mkstore()
    j1 = await store.create(user_id="u1", job_type="demo", max_attempts=3)
    j2 = await store.create(user_id="u1", job_type="demo", max_attempts=1)
    # Simulate a worker that marked both running then died.
    await store.claim_next_pending()  # one becomes running
    await store.claim_next_pending()  # both now running, both attempts=1
    # Now restart: j1 still has retries (max 3, attempts 1), j2 is at its
    # max (max 1, attempts 1) so it should terminate.
    requeued = await store.requeue_crashed()
    assert requeued == 1
    got1 = await store.get(j1)
    got2 = await store.get(j2)
    # One of them (whichever was picked first) re-queues, the other fails.
    statuses = {got1["status"], got2["status"]}
    assert statuses == {"pending", "failed"}


@pytest.mark.asyncio
async def test_cancel_running_job_flips_flag():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo")
    await store.claim_next_pending()
    await store.request_cancel(job_id, user_id="u1")
    got = await store.get(job_id, user_id="u1")
    # Running jobs stay running until the handler cooperates.
    assert got["status"] == "running"
    assert got["cancel_requested"] is True
    assert await store.is_cancel_requested(job_id) is True


@pytest.mark.asyncio
async def test_cancel_other_users_job_rejected():
    store, _ = await _mkstore()
    job_id = await store.create(user_id="u1", job_type="demo")
    assert await store.request_cancel(job_id, user_id="u2") is False
    got = await store.get(job_id)
    assert got["status"] == "pending"


@pytest.mark.asyncio
async def test_list_for_user_respects_filters():
    store, _ = await _mkstore()
    await store.create(user_id="u1", job_type="typeA")
    await store.create(user_id="u1", job_type="typeB")
    await store.create(user_id="u2", job_type="typeA")

    u1_all = await store.list_for_user(user_id="u1")
    assert len(u1_all) == 2
    u1_typeA = await store.list_for_user(user_id="u1", job_type="typeA")
    assert len(u1_typeA) == 1
    assert u1_typeA[0]["job_type"] == "typeA"


@pytest.mark.asyncio
async def test_delete_older_than_prunes_terminal_rows():
    store, conn = await _mkstore()
    j = await store.create(user_id="u1", job_type="demo")
    await store.claim_next_pending()
    await store.mark_completed(j)
    # Back-date it so the prune catches it.
    await conn.execute(
        "UPDATE background_jobs SET completed_at = 0 WHERE id = ?", (j,),
    )
    await conn.commit()
    removed = await store.delete_older_than(seconds=3600)
    assert removed == 1
    assert await store.get(j) is None


# ── Read-connection split (2026-05-26) ─────────────────────────────────


@pytest.mark.asyncio
async def test_attach_read_conn_routes_claim_select():
    """claim_next_pending's SELECT runs on the read connection when
    attached. The UPDATE-to-claim still goes through the main conn so
    the row gets flipped to 'running' atomically.

    We can't directly inspect which connection executed which query
    in aiosqlite, but we CAN attach a sentinel read_conn that points
    at a SEPARATE in-memory DB lacking the row — then claim returns
    None, proving the SELECT was routed to the sentinel and not the
    main conn (which has the row).
    """
    from augmentum.state.jobs_store import JobsStore

    main_conn = await aiosqlite.connect(":memory:")
    await main_conn.executescript(_SCHEMA_SQL)
    await main_conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await main_conn.commit()
    store = JobsStore(main_conn)
    await store.create(user_id="u1", job_type="demo")

    # Sanity: with no read_conn attached, claim sees the row.
    claimed = await store.claim_next_pending()
    assert claimed is not None
    # Reset for the attached-read-conn test.
    await main_conn.execute("UPDATE background_jobs SET status = 'pending'")
    await main_conn.commit()

    # Sentinel: empty schema, no rows. After attach, claim_next_pending
    # should query this conn and find nothing.
    sentinel_conn = await aiosqlite.connect(":memory:")
    await sentinel_conn.executescript(_SCHEMA_SQL)
    await sentinel_conn.commit()
    store.attach_read_conn(sentinel_conn)

    assert await store.claim_next_pending() is None


@pytest.mark.asyncio
async def test_attach_read_conn_none_falls_back_to_main():
    """attach_read_conn(None) reverts to the main conn — tests +
    early-lifespan paths rely on this fallback."""
    from augmentum.state.jobs_store import JobsStore

    main_conn = await aiosqlite.connect(":memory:")
    await main_conn.executescript(_SCHEMA_SQL)
    await main_conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await main_conn.commit()
    store = JobsStore(main_conn)
    await store.create(user_id="u1", job_type="demo")

    store.attach_read_conn(None)
    claimed = await store.claim_next_pending()
    assert claimed is not None
