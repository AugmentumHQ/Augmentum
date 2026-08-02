"""Archive-store tests — the never-pruned self-edit lineage."""

from __future__ import annotations

import pathlib

import aiosqlite

from augmentum.selfedit import store

_MIGRATION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "288_self_edit_attempts.sql"
)


async def _db():
    from augmentum.selfedit.growth_db import _ensure_columns
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)"
    )
    await conn.executescript(_MIGRATION.read_text())
    await _ensure_columns(conn)  # mirror the live growth-DB open (post-288 columns)
    await conn.commit()
    return conn


async def test_attempt_lifecycle_and_isolation():
    conn = await _db()
    try:
        await store.create_attempt(
            conn, attempt_id="a1", user_id="u1",
            objective="make the agents panel cleaner", surface="frontend",
            tier="green", base_ref="abc123", target="code_quality.dead_css",
        )
        await store.set_candidate(
            conn, attempt_id="a1", user_id="u1",
            candidate_ref="selfedit/a1", run_id="run-xyz",
        )
        await store.set_gate(
            conn, attempt_id="a1", user_id="u1", passed=True,
            verdict={"passed": True, "summary": "PASS"},
            files_changed=["ui/styles/coder.css"],
        )
        await store.finalize(
            conn, attempt_id="a1", user_id="u1", status="promoted",
            outcome="shipped", lesson="accent var tidies the panel",
            promoted_commit="def456",
        )

        got = await store.get_attempt(conn, attempt_id="a1", user_id="u1")
        assert got["status"] == "promoted"
        assert got["surface"] == "frontend" and got["tier"] == "green"
        assert got["target"] == "code_quality.dead_css"  # debt class survives round-trip
        assert got["run_id"] == "run-xyz"
        assert got["gate_passed"] is True
        assert got["files_changed"] == ["ui/styles/coder.css"]
        assert got["lesson"] == "accent var tidies the panel"
        assert got["promoted_commit"] == "def456"

        runs = await store.list_attempts(conn, user_id="u1")
        assert len(runs) == 1

        # user isolation
        assert await store.get_attempt(conn, attempt_id="a1", user_id="u2") is None
        assert await store.list_attempts(conn, user_id="u2") == []
    finally:
        await conn.close()


async def test_lesson_survives_rollback():
    conn = await _db()
    try:
        await store.create_attempt(
            conn, attempt_id="a2", user_id="u1", objective="risky backend tweak",
            surface="backend", tier="yellow",
        )
        await store.finalize(
            conn, attempt_id="a2", user_id="u1", status="rolled_back",
            outcome="reverted: broke boot",
            lesson="that import order deadlocks the resource sampler",
        )
        got = await store.get_attempt(conn, attempt_id="a2", user_id="u1")
        assert got["status"] == "rolled_back"
        # the pillar: code reverted, lesson kept
        assert "deadlocks" in got["lesson"]
    finally:
        await conn.close()


def test_store_has_no_delete_path():
    # The archive is sacred — there must be no way to wipe an attempt.
    public = [n for n in dir(store) if not n.startswith("_")]
    assert not any("delete" in n.lower() or "purge" in n.lower()
                   or "prune" in n.lower() or "wipe" in n.lower() for n in public)
