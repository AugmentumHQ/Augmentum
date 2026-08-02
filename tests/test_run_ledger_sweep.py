"""Startup run-ledger orphan sweep (state/run_ledger_sweep.py)."""

from __future__ import annotations

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.run_ledger_sweep import finalize_orphan_runs


async def _mk_backend(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "sweep.db"))
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, role) "
        "VALUES ('u1', 'u1', 'U', 'pw', 'user')",
    )
    await backend.conn.commit()
    return backend


@pytest.mark.asyncio
async def test_sweep_finalizes_running_coder_turn_runs(tmp_path):
    backend = await _mk_backend(tmp_path)
    try:
        await backend.conn.execute(
            "INSERT INTO coder_turn_runs (id, user_id, project_id, session_id, "
            "status, started_at, updated_at) "
            "VALUES ('r1', 'u1', 'ws', 's', 'running', 1, 1), "
            "       ('r2', 'u1', 'ws', 's', 'completed', 1, 1)",
        )
        await backend.conn.commit()

        finalized = await finalize_orphan_runs(backend.conn)
        assert finalized.get("coder_turn_runs") == 1

        cur = await backend.conn.execute(
            "SELECT status, finish_reason, completed_at FROM coder_turn_runs "
            "WHERE id = 'r1'",
        )
        status, reason, completed = await cur.fetchone()
        assert status == "error"
        assert reason == "interrupted_by_restart"
        assert completed is not None
        # Terminal rows untouched.
        cur = await backend.conn.execute(
            "SELECT status FROM coder_turn_runs WHERE id = 'r2'",
        )
        assert (await cur.fetchone())[0] == "completed"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sweep_finalizes_claude_pi_and_xr(tmp_path):
    backend = await _mk_backend(tmp_path)
    try:
        await backend.conn.execute(
            "INSERT INTO claude_runs (id, user_id, workspace_id, session_id, "
            "task, permission, status, created_at, updated_at) "
            "VALUES ('c1', 'u1', 'ws', 's', 't', 'p', 'running', "
            "datetime('now'), datetime('now'))",
        )
        await backend.conn.execute(
            "INSERT INTO pi_runs (id, user_id, project, session_file, status, "
            "created_at, updated_at) "
            "VALUES ('p1', 'u1', 'proj', 'f', 'running', "
            "datetime('now'), datetime('now'))",
        )
        await backend.conn.execute(
            "INSERT INTO xr_sessions (id, user_id, surface, status, "
            "created_at, updated_at) "
            "VALUES ('x1', 'u1', 'vr', 'running', datetime('now'), datetime('now')), "
            "       ('x2', 'u1', 'vr', 'preflight', datetime('now'), datetime('now')), "
            "       ('x3', 'u1', 'vr', 'ended', datetime('now'), datetime('now'))",
        )
        await backend.conn.commit()

        finalized = await finalize_orphan_runs(backend.conn)
        assert finalized.get("claude_runs") == 1
        assert finalized.get("pi_runs") == 1
        assert finalized.get("xr_sessions") == 2

        for table, ident, want in (
            ("claude_runs", "c1", "failed"),
            ("pi_runs", "p1", "failed"),
            ("xr_sessions", "x1", "ended"),
            ("xr_sessions", "x2", "ended"),
            ("xr_sessions", "x3", "ended"),
        ):
            cur = await backend.conn.execute(
                f"SELECT status FROM {table} WHERE id = ?", (ident,),
            )
            assert (await cur.fetchone())[0] == want, (table, ident)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sweep_isolates_missing_table(tmp_path):
    """One broken table (partial migration) must not stop the rest."""
    backend = await _mk_backend(tmp_path)
    try:
        await backend.conn.execute(
            "INSERT INTO coder_turn_runs (id, user_id, project_id, session_id, "
            "status, started_at, updated_at) "
            "VALUES ('r1', 'u1', 'ws', 's', 'running', 1, 1)",
        )
        await backend.conn.commit()
        await backend.conn.execute("ALTER TABLE pi_runs RENAME TO pi_runs_gone")
        await backend.conn.commit()

        finalized = await finalize_orphan_runs(backend.conn)
        # pi_runs failed silently (warning logged); coder sweep still ran.
        assert finalized.get("coder_turn_runs") == 1
        assert "pi_runs" not in finalized
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sweep_noop_on_clean_db(tmp_path):
    backend = await _mk_backend(tmp_path)
    try:
        assert await finalize_orphan_runs(backend.conn) == {}
    finally:
        await backend.close()
