"""pi_run_store roundtrip against the real migration SQL (in-memory DB).

Covers the invariants the pushed-mirror design depends on:
* idempotent event batches (INSERT OR IGNORE on (run_id, seq))
* strict user scoping (user B can never read user A's run)
* incremental reads via since_seq (the SSE poller's contract)
* re-attach flips a finished run back to 'running'
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.coder.external import pi_run_store as s

MIGRATION = Path(__file__).resolve().parents[1] / "augmentum" / "state" / "migrations" / "311_pi_runs.sql"


@pytest.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT,"
        " applied_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.executescript(MIGRATION.read_text(encoding="utf-8"))
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_roundtrip_idempotent_and_scoped(db):
    await s.upsert_run(
        db, run_id="r1", user_id="u1", project="augmentum",
        session_file="C:/x.jsonl", title="test session", model="d/deepseek-v4-flash",
    )
    n = await s.add_events(db, run_id="r1", user_id="u1", events=[
        {"seq": 0, "kind": "message", "text": "hello"},
        {"seq": 1, "kind": "tool_call", "tool": "bash", "text": "ls"},
    ])
    assert n == 2

    # Idempotent replay: same seq inserts nothing.
    n2 = await s.add_events(
        db, run_id="r1", user_id="u1",
        events=[{"seq": 1, "kind": "tool_call", "tool": "bash", "text": "ls"}],
    )
    assert n2 == 0

    # User isolation.
    assert await s.get_run(db, run_id="r1", user_id="u2") is None

    run = await s.get_run(db, run_id="r1", user_id="u1")
    assert run["title"] == "test session"
    assert len(run["events"]) == 2

    # Incremental read (SSE poller contract).
    inc = await s.get_run(db, run_id="r1", user_id="u1", since_seq=0)
    assert [e["seq"] for e in inc["events"]] == [1]


@pytest.mark.asyncio
async def test_finish_and_reattach(db):
    await s.upsert_run(db, run_id="r1", user_id="u1", project="augmentum")
    await s.finish_run(
        db, run_id="r1", user_id="u1", status="done",
        outcome="ok", files_changed=["a.py"], num_turns=3,
    )
    runs = await s.list_runs(db, user_id="u1", project="augmentum")
    assert runs[0]["status"] == "done"
    assert runs[0]["files_changed"] == ["a.py"]
    assert runs[0]["engine"] == "pi"

    # Host re-attaches (same session resumed) → status back to running.
    await s.upsert_run(db, run_id="r1", user_id="u1", project="augmentum")
    runs = await s.list_runs(db, user_id="u1")
    assert runs[0]["status"] == "running"

    # Unknown finish status is clamped to detached, never invented.
    await s.finish_run(db, run_id="r1", user_id="u1", status="exploded")
    runs = await s.list_runs(db, user_id="u1")
    assert runs[0]["status"] == "detached"
