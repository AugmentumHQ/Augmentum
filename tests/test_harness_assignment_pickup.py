"""Tier-1 "My machine" assignment pickup — the server lifecycle.

Exercises the provider-neutral path both clients (claude-aug hook + pi
extension) drive through POST /api/harness/checkin:

    dispatch → coding_runs row (queued) + assignment linked to it
    agent check-in → assignment delivered ONCE + run advances to 'working'
    agent check-in (status=done) → run finalized to 'done' + summary

This is the load-bearing new logic (agent_bridge.checkin lifecycle +
coding_driver link); the client halves live on the user's machine.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import aiosqlite

from augmentum.coder import coding_driver
from augmentum.proxy import agent_bridge

_MIGRATIONS = Path(__file__).resolve().parents[1] / "augmentum" / "state" / "migrations"


async def _mk_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)")
    for mig in (
        "316_harness_agent_bridge",
        "319_coding_runs",
        "320_coding_runs_review",
        "322_harness_assignment_run_link",
    ):
        await conn.executescript((_MIGRATIONS / f"{mig}.sql").read_text(encoding="utf-8"))
    await conn.commit()
    return conn


def _app(conn: aiosqlite.Connection) -> SimpleNamespace:
    return SimpleNamespace(
        state_manager=SimpleNamespace(backend=SimpleNamespace(conn=conn)))


async def test_assignment_lifecycle_queued_working_done():
    conn = await _mk_db()
    app = _app(conn)
    uid = "u1"
    try:
        # 1. A bare-machine agent registers its session.
        first = await agent_bridge.checkin(app, user_id=uid, harness="pi", project="proj")
        agent_id = first["agent_id"]
        assert agent_id

        # 2. Dispatch: status row (queued) + assignment linked to it (mirrors
        #    HarnessCoderDriver.dispatch: run first, then linked assignment).
        run_id = await coding_driver.create_run(
            app, user_id=uid, driver="harness", workspace_id="",
            task="do X", engine_ref=agent_id, status="queued")
        assert run_id
        assigned = await agent_bridge.create_assignment(
            app, user_id=uid, agent_session_id=agent_id, task="do X",
            linked_run_id=run_id)
        assert assigned and assigned.get("request_id")
        assert (await coding_driver.get_run(app, user_id=uid, run_id=run_id))["status"] == "queued"

        # 3. Next check-in DELIVERS the task (with its run_id) and advances the
        #    status row queued → working.
        second = await agent_bridge.checkin(
            app, user_id=uid, harness="pi", project="proj", agent_id=agent_id)
        assert any(a["task"] == "do X" and a["run_id"] == run_id
                   for a in second["assignments"])
        assert (await coding_driver.get_run(app, user_id=uid, run_id=run_id))["status"] == "working"

        # 4. Delivered exactly once — a further check-in returns no assignments.
        third = await agent_bridge.checkin(
            app, user_id=uid, harness="pi", project="proj", agent_id=agent_id)
        assert third["assignments"] == []

        # 5. Reporting done finalizes the run (status + summary).
        await agent_bridge.checkin(
            app, user_id=uid, harness="pi", project="proj", agent_id=agent_id,
            status="done", summary="did X")
        final = await coding_driver.get_run(app, user_id=uid, run_id=run_id)
        assert final["status"] == "done"
        assert final["summary"] == "did X"
    finally:
        await conn.close()


async def test_assignment_only_delivered_to_its_own_agent():
    """An assignment for agent A is never handed to agent B (isolation)."""
    conn = await _mk_db()
    app = _app(conn)
    uid = "u1"
    try:
        a = (await agent_bridge.checkin(app, user_id=uid, harness="pi", project="p"))["agent_id"]
        b = (await agent_bridge.checkin(app, user_id=uid, harness="claude_code", project="q"))["agent_id"]
        run_id = await coding_driver.create_run(
            app, user_id=uid, driver="harness", workspace_id="", task="t",
            engine_ref=a, status="queued")
        await agent_bridge.create_assignment(
            app, user_id=uid, agent_session_id=a, task="t", linked_run_id=run_id)

        # B checks in — must NOT receive A's assignment, and A's run stays queued.
        b_res = await agent_bridge.checkin(
            app, user_id=uid, harness="claude_code", project="q", agent_id=b)
        assert b_res["assignments"] == []
        assert (await coding_driver.get_run(app, user_id=uid, run_id=run_id))["status"] == "queued"
    finally:
        await conn.close()
