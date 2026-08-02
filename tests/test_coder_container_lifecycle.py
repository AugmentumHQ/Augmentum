"""Tests for the coder container lifecycle policy + idle reaper.

Covers migration 211 + ``ContainerManager.set_always_on`` +
``mark_active`` (debounce) + ``sweep_idle`` (selection rules).
Docker is fully mocked — the reaper's ``stop()`` calls go through
the same mock path the rest of test_coder_containers.py uses.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.coder.containers import (
    ContainerManager,
    _assemble_keepalive_cmd,
    _resolve_memory_swap,
)


# Minimal project_checkouts schema — just the columns the reaper +
# always-on setter touch. The real migration adds many more, but a
# focused subset keeps the test fast and the schema readable. Mirrors
# the column types from migrations 200, 138, 207, 209, 211.
_SCHEMA = """
CREATE TABLE project_checkouts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    container_id TEXT,
    status TEXT NOT NULL DEFAULT 'stopped',
    template_id TEXT,
    git_url TEXT,
    created_at REAL,
    last_active REAL,
    resources_cpu REAL DEFAULT 2.0,
    resources_memory TEXT DEFAULT '2g',
    safeguards_enabled INTEGER DEFAULT 1,
    tooling_profile TEXT DEFAULT 'browser',
    user_id TEXT DEFAULT '',
    kind TEXT DEFAULT 'regular',
    bug_finder_verifier_model TEXT,
    project_id TEXT,
    planning_mode TEXT DEFAULT 'auto',
    always_on INTEGER NOT NULL DEFAULT 0
);
"""


async def _insert_workspace(
    conn: aiosqlite.Connection,
    *,
    ws_id: str,
    status: str = "running",
    always_on: int = 0,
    last_active: float | None = None,
    kind: str = "regular",
) -> None:
    await conn.execute(
        "INSERT INTO project_checkouts "
        "(id, name, container_id, status, last_active, always_on, kind) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ws_id, ws_id, f"docker-{ws_id}", status, last_active, always_on, kind),
    )
    await conn.commit()


@pytest.fixture
async def conn():
    db = await aiosqlite.connect(":memory:")
    await db.executescript(_SCHEMA)
    yield db
    await db.close()


@pytest.fixture
def mock_docker():
    """Docker mock for the reaper's lifecycle ops.

    Reaper calls ``stop`` / ``pause`` / ``unpause`` -> ``docker.containers.get``
    -> ``container.{stop,pause,unpause}()``; the mock chain returns a
    coroutine-friendly stub for each.
    """
    container = MagicMock()
    container.start = AsyncMock()
    container.stop = AsyncMock()
    container.pause = AsyncMock()
    container.unpause = AsyncMock()
    container.delete = AsyncMock()
    # Default state probe used by start() to branch start vs unpause.
    container.show = AsyncMock(return_value={"State": {"Status": "exited"}})
    docker = MagicMock()
    docker.containers = MagicMock()
    docker.containers.get = AsyncMock(return_value=container)
    return docker


@pytest.fixture
def stop_only_settings(monkeypatch):
    """Pin the reaper to single-stage stop (legacy behavior).

    Tests that pre-date the two-stage policy assert ``status='stopped'``
    after one sweep — that still holds when pause_idle is off.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_pause_idle", False)
    yield


# ----------------------------------------------------------------------
# set_always_on
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_always_on_persists_and_returns_info(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-toggle", always_on=0)
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.set_always_on("ws-toggle", always_on=True)
    assert info.always_on is True

    row = await (await conn.execute(
        "SELECT always_on FROM project_checkouts WHERE id=?", ("ws-toggle",),
    )).fetchone()
    assert row[0] == 1

    info = await mgr.set_always_on("ws-toggle", always_on=False)
    assert info.always_on is False
    row = await (await conn.execute(
        "SELECT always_on FROM project_checkouts WHERE id=?", ("ws-toggle",),
    )).fetchone()
    assert row[0] == 0


# ----------------------------------------------------------------------
# mark_active debounce
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_active_writes_first_call(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-bump", last_active=0)
    mgr = ContainerManager(docker=mock_docker, db=conn)
    before = time.time()
    await mgr.mark_active("ws-bump")
    row = await (await conn.execute(
        "SELECT last_active FROM project_checkouts WHERE id=?", ("ws-bump",),
    )).fetchone()
    assert row[0] >= before


@pytest.mark.asyncio
async def test_mark_active_debounces_within_window(conn, mock_docker):
    import asyncio as _asyncio

    await _insert_workspace(conn, ws_id="ws-debounce", last_active=0)
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr.mark_active("ws-debounce")
    row1 = await (await conn.execute(
        "SELECT last_active FROM project_checkouts WHERE id=?", ("ws-debounce",),
    )).fetchone()
    first_ts = row1[0]

    # Second call inside the debounce window must NOT update the DB.
    await mgr.mark_active("ws-debounce")
    row2 = await (await conn.execute(
        "SELECT last_active FROM project_checkouts WHERE id=?", ("ws-debounce",),
    )).fetchone()
    assert row2[0] == first_ts

    # Force the cache past the window by rewinding the cached entry,
    # and sleep a tick so ``time.time()`` increments past Windows'
    # 15.6 ms clock resolution — otherwise the third write can land
    # at the same wall-clock as the first and the ``>`` assertion
    # below flakes on fast machines.
    mgr._activity_last_seen["ws-debounce"] -= mgr._ACTIVITY_DEBOUNCE_S + 1
    await _asyncio.sleep(0.05)
    await mgr.mark_active("ws-debounce")
    row3 = await (await conn.execute(
        "SELECT last_active FROM project_checkouts WHERE id=?", ("ws-debounce",),
    )).fetchone()
    assert row3[0] > first_ts


@pytest.mark.asyncio
async def test_mark_active_no_op_on_missing_workspace(conn, mock_docker):
    mgr = ContainerManager(docker=mock_docker, db=conn)
    # Must not raise — UPDATE … WHERE id=? matches zero rows cleanly.
    await mgr.mark_active("does-not-exist")
    await mgr.mark_active("")  # empty id short-circuit


# ----------------------------------------------------------------------
# sweep_idle selection rules
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sweep_idle_pauses_idle_on_demand_workspace(conn, mock_docker):
    # Default policy (pause_idle=True): first sweep moves a running idle
    # workspace to 'paused' (cgroup freeze, RAM held). It only becomes
    # 'stopped' after the deeper coder_pause_stop_after_seconds window.
    cutoff_ago = time.time() - 9999
    await _insert_workspace(
        conn, ws_id="ws-stale", status="running",
        always_on=0, last_active=cutoff_ago,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    acted = await mgr.sweep_idle(timeout_seconds=600)
    assert acted == 1

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-stale",),
    )).fetchone()
    assert row[0] == "paused"


@pytest.mark.asyncio
async def test_sweep_idle_stops_when_pause_disabled(
    conn, mock_docker, stop_only_settings,
):
    # Legacy single-stage behavior: with coder_pause_idle=False the
    # reaper goes straight to stop, matching the pre-2026-06-02
    # contract.
    cutoff_ago = time.time() - 9999
    await _insert_workspace(
        conn, ws_id="ws-legacy", status="running",
        always_on=0, last_active=cutoff_ago,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    acted = await mgr.sweep_idle(timeout_seconds=600)
    assert acted == 1

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-legacy",),
    )).fetchone()
    assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_sweep_idle_skips_always_on(conn, mock_docker):
    cutoff_ago = time.time() - 9999
    await _insert_workspace(
        conn, ws_id="ws-persistent", status="running",
        always_on=1, last_active=cutoff_ago,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    stopped = await mgr.sweep_idle(timeout_seconds=600)
    assert stopped == 0

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-persistent",),
    )).fetchone()
    assert row[0] == "running"


@pytest.mark.asyncio
async def test_sweep_idle_skips_recently_active(conn, mock_docker):
    await _insert_workspace(
        conn, ws_id="ws-fresh", status="running",
        always_on=0, last_active=time.time() - 30,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    stopped = await mgr.sweep_idle(timeout_seconds=600)
    assert stopped == 0


@pytest.mark.asyncio
async def test_sweep_idle_skips_already_stopped(conn, mock_docker):
    cutoff_ago = time.time() - 9999
    await _insert_workspace(
        conn, ws_id="ws-already-down", status="stopped",
        always_on=0, last_active=cutoff_ago,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    stopped = await mgr.sweep_idle(timeout_seconds=600)
    assert stopped == 0


@pytest.mark.asyncio
async def test_sweep_idle_skips_bug_finder_workspaces(conn, mock_docker):
    # Bug Finder runs have their own lifecycle; reaping them mid-audit
    # would corrupt the bundle output. Reaper must skip kind='bug_finder'.
    cutoff_ago = time.time() - 9999
    await _insert_workspace(
        conn, ws_id="ws-bf", status="running",
        always_on=0, last_active=cutoff_ago, kind="bug_finder",
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    stopped = await mgr.sweep_idle(timeout_seconds=600)
    assert stopped == 0


@pytest.mark.asyncio
async def test_sweep_idle_partial_failure_continues(conn, mock_docker):
    # Two idle workspaces; first pause() raises. Second should still
    # be processed. The single-workspace failure must not abort the
    # sweep — the next tick retries.
    cutoff_ago = time.time() - 9999
    await _insert_workspace(
        conn, ws_id="ws-fail", status="running",
        always_on=0, last_active=cutoff_ago,
    )
    await _insert_workspace(
        conn, ws_id="ws-ok", status="running",
        always_on=0, last_active=cutoff_ago,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    real_pause = mgr.pause

    async def _flaky_pause(ws_id: str):
        if ws_id == "ws-fail":
            raise RuntimeError("docker daemon hiccup")
        return await real_pause(ws_id)

    mgr.pause = _flaky_pause  # type: ignore[assignment]
    acted = await mgr.sweep_idle(timeout_seconds=600)
    assert acted == 1

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-ok",),
    )).fetchone()
    assert row[0] == "paused"


# ----------------------------------------------------------------------
# pause / unpause + two-stage reaper
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pause_unpause_round_trip(conn, mock_docker):
    await _insert_workspace(
        conn, ws_id="ws-rt", status="running",
        always_on=0, last_active=time.time(),
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.pause("ws-rt")
    assert info.status == "paused"
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-rt",),
    )).fetchone()
    assert row[0] == "paused"

    info = await mgr.unpause("ws-rt")
    assert info.status == "running"
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-rt",),
    )).fetchone()
    assert row[0] == "running"


@pytest.mark.asyncio
async def test_pause_reconciles_when_container_not_running(conn, mock_docker):
    # Reproduces the post-Docker-restart cascade: container exited (137)
    # but DB row still says 'running'. Reaper calls pause() and Docker
    # returns 409 "container is not running". pause() must reconcile the
    # row to 'stopped' so the reaper doesn't retry every tick forever.
    await _insert_workspace(
        conn, ws_id="ws-zombie", status="running",
        always_on=0, last_active=time.time() - 9999,
    )

    container = mock_docker.containers.get.return_value
    container.pause.side_effect = Exception(
        "[409] container abc123 is not running"
    )

    mgr = ContainerManager(docker=mock_docker, db=conn)
    info = await mgr.pause("ws-zombie")
    assert info.status == "stopped"

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-zombie",),
    )).fetchone()
    assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_sweep_idle_does_not_lie_when_pause_409s(conn, mock_docker):
    # When pause() hits 409-not-running and reconciles to 'stopped',
    # sweep_idle must NOT count it as an actioned reap, and the DB row
    # must flip to 'stopped' so the next tick doesn't re-pick it. The
    # prior bug emitted a warning AND a 'reaped_paused' info line on
    # every tick AND left the row dirty, looping forever.
    await _insert_workspace(
        conn, ws_id="ws-carcass", status="running",
        always_on=0, last_active=time.time() - 9999,
    )

    container = mock_docker.containers.get.return_value
    container.pause.side_effect = Exception(
        "[409] container abc123 is not running"
    )

    mgr = ContainerManager(docker=mock_docker, db=conn)
    acted = await mgr.sweep_idle(timeout_seconds=600)

    assert acted == 0  # reconciled, not reaped
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-carcass",),
    )).fetchone()
    assert row[0] == "stopped"

    # Second tick: DB now says 'stopped' so the row falls out of the
    # stage-1 query entirely. No further pause attempts.
    container.pause.reset_mock()
    acted2 = await mgr.sweep_idle(timeout_seconds=600)
    assert acted2 == 0
    assert container.pause.call_count == 0


@pytest.mark.asyncio
async def test_sweep_idle_stage2_stops_paused_past_deep_cutoff(
    conn, mock_docker, monkeypatch,
):
    # Pin the deeper threshold low so the test runs in real time.
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_pause_idle", True)
    monkeypatch.setattr(settings, "coder_pause_stop_after_seconds", 100)

    # Already paused, last_active way past timeout + stop_after.
    deep_ago = time.time() - 10_000
    await _insert_workspace(
        conn, ws_id="ws-deep", status="paused",
        always_on=0, last_active=deep_ago,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    acted = await mgr.sweep_idle(timeout_seconds=600)
    assert acted == 1

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-deep",),
    )).fetchone()
    assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_sweep_idle_stage2_skips_paused_under_deep_cutoff(
    conn, mock_docker, monkeypatch,
):
    # Paused recently — under timeout + stop_after — must not be stopped.
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_pause_idle", True)
    monkeypatch.setattr(settings, "coder_pause_stop_after_seconds", 100_000)

    not_so_old = time.time() - 1200  # past timeout, but well under stop_after
    await _insert_workspace(
        conn, ws_id="ws-recent-paused", status="paused",
        always_on=0, last_active=not_so_old,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    acted = await mgr.sweep_idle(timeout_seconds=600)
    assert acted == 0

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?",
        ("ws-recent-paused",),
    )).fetchone()
    assert row[0] == "paused"


@pytest.mark.asyncio
async def test_mark_active_unpauses_paused_workspace(conn, mock_docker):
    # A returning user touches the workspace; mark_active should both
    # bump last_active and thaw the cgroup-frozen container so the
    # next exec doesn't hang.
    await _insert_workspace(
        conn, ws_id="ws-resume", status="paused",
        always_on=0, last_active=time.time() - 9999,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    await mgr.mark_active("ws-resume")

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-resume",),
    )).fetchone()
    assert row[0] == "running"
    # The container.unpause stub should have been awaited exactly once.
    container = await mock_docker.containers.get("docker-ws-resume")
    assert container.unpause.await_count >= 1


# ----------------------------------------------------------------------
# list_workspaces drift writeback
# ----------------------------------------------------------------------

def _docker_with_live_containers(*workspace_ids: str):
    """Build a docker mock whose containers.list returns the given ws ids
    as live containers (so anything NOT in the list is treated as drift)."""
    live = []
    for ws_id in workspace_ids:
        c = MagicMock()
        c._container = {
            "Labels": {"augmentum.workspace": "true", "augmentum.id": ws_id},
            "State": "running",
        }
        live.append(c)
    docker = MagicMock()
    docker.containers = MagicMock()
    docker.containers.list = AsyncMock(return_value=live)
    docker.containers.get = AsyncMock()
    return docker


@pytest.mark.asyncio
async def test_list_workspaces_writes_back_drift(conn):
    # DB has two rows claiming running. Docker only knows about the first.
    # The second's container was docker-rm'd externally — list_workspaces
    # must persist status='stopped' for it, not just mutate the response.
    await _insert_workspace(conn, ws_id="ws-live", status="running")
    await _insert_workspace(conn, ws_id="ws-phantom", status="running")

    docker = _docker_with_live_containers("ws-live")
    mgr = ContainerManager(docker=docker, db=conn)

    await mgr.list_workspaces()

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-phantom",),
    )).fetchone()
    assert row[0] == "stopped", "drift must be persisted, not just mutated in memory"

    # Live row is untouched.
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-live",),
    )).fetchone()
    assert row[0] == "running"


@pytest.mark.asyncio
async def test_list_workspaces_drift_skips_already_stopped_rows(conn):
    # Already-stopped rows must not be touched (avoid no-op write traffic).
    await _insert_workspace(conn, ws_id="ws-already-stopped", status="stopped")
    docker = _docker_with_live_containers()  # nothing live
    mgr = ContainerManager(docker=docker, db=conn)

    await mgr.list_workspaces()

    # The drifted set should have been empty -> no UPDATE issued. Verify
    # status didn't get re-written (still 'stopped'), and that ContainerInfo
    # carries the stopped status too.
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-already-stopped",),
    )).fetchone()
    assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_list_workspaces_drift_picks_up_phantom_paused(conn):
    # Paused rows with missing containers are also drift.
    await _insert_workspace(conn, ws_id="ws-phantom-paused", status="paused")
    docker = _docker_with_live_containers()
    mgr = ContainerManager(docker=docker, db=conn)

    await mgr.list_workspaces()

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-phantom-paused",),
    )).fetchone()
    assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_sweep_idle_warns_stale_always_on(conn, mock_docker, caplog):
    # A workspace that has always_on=1 and hasn't been touched in 30 days
    # should emit a workspace_stale_always_on advisory (info-level — nothing
    # is wrong; the reaper exempts always_on=1, this is just a leftover-flag
    # heads-up) and record the throttle stamp, without being reaped.
    import logging
    caplog.set_level(logging.INFO)
    await _insert_workspace(
        conn, ws_id="ws-stale-flag",
        always_on=1, last_active=time.time() - 30 * 86400,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr.sweep_idle(timeout_seconds=600)

    # Row still running (always_on exempts from reaper)
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-stale-flag",),
    )).fetchone()
    assert row[0] == "running"
    # But the warning fired (throttle dict records the warn timestamp)
    assert "ws-stale-flag" in mgr._stale_always_on_warned
    assert mgr._stale_always_on_warned["ws-stale-flag"] > 0


@pytest.mark.asyncio
async def test_sweep_idle_skips_recently_active_always_on(conn, mock_docker):
    # always_on=1 but recent activity -> no warning.
    await _insert_workspace(
        conn, ws_id="ws-recent-flag",
        always_on=1, last_active=time.time() - 60,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)
    await mgr.sweep_idle(timeout_seconds=600)
    assert "ws-recent-flag" not in mgr._stale_always_on_warned


# ----------------------------------------------------------------------
# reconcile_with_docker — startup drift correction
# ----------------------------------------------------------------------

def _docker_with_states(*pairs: tuple[str, str]):
    """Build a docker mock whose containers.list returns the (ws_id, state)
    pairs as live workspace containers. ``state`` is a raw Docker state
    string (running / paused / exited / etc.)."""
    live = []
    for ws_id, state in pairs:
        c = MagicMock()
        c._container = {
            "Labels": {"augmentum.workspace": "true", "augmentum.id": ws_id},
            "State": state,
        }
        live.append(c)
    docker = MagicMock()
    docker.containers = MagicMock()
    docker.containers.list = AsyncMock(return_value=live)
    docker.containers.get = AsyncMock()
    return docker


@pytest.mark.asyncio
async def test_reconcile_daemon_restart_cohort(conn):
    # The scenario from the original incident: Docker daemon restarted and
    # SIGKILL'd every workspace container. DB still says running. Reconcile
    # must flip these to stopped so the idle sweeper doesn't try to pause
    # phantom containers.
    await _insert_workspace(conn, ws_id="ws-a", status="running")
    await _insert_workspace(conn, ws_id="ws-b", status="paused")
    await _insert_workspace(conn, ws_id="ws-c", status="running")
    docker = _docker_with_states()  # nothing alive
    mgr = ContainerManager(docker=docker, db=conn)

    summary = await mgr.reconcile_with_docker()

    assert summary == {"reconciled": 3, "orphans": 0, "ok": 0}
    for ws_id in ("ws-a", "ws-b", "ws-c"):
        row = await (await conn.execute(
            "SELECT status FROM project_checkouts WHERE id=?", (ws_id,),
        )).fetchone()
        assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_reconcile_picks_up_out_of_band_state_changes(conn):
    # Operator ran ``docker pause`` / ``docker start`` from outside
    # Augmentum. Reconcile must converge DB to Docker truth.
    await _insert_workspace(conn, ws_id="ws-thought-running", status="running")
    await _insert_workspace(conn, ws_id="ws-thought-stopped", status="stopped")
    docker = _docker_with_states(
        ("ws-thought-running", "paused"),   # externally paused
        ("ws-thought-stopped", "running"),  # externally started
    )
    mgr = ContainerManager(docker=docker, db=conn)

    summary = await mgr.reconcile_with_docker()
    assert summary["reconciled"] == 2
    assert summary["orphans"] == 0

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-thought-running",),
    )).fetchone()
    assert row[0] == "paused"
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-thought-stopped",),
    )).fetchone()
    assert row[0] == "running"


@pytest.mark.asyncio
async def test_reconcile_aligned_state_is_noop(conn):
    # When everyone agrees, the reconciler must not issue UPDATEs and
    # must return ok counters reflecting alignment.
    await _insert_workspace(conn, ws_id="ws-up", status="running")
    await _insert_workspace(conn, ws_id="ws-zz", status="stopped")
    docker = _docker_with_states(("ws-up", "running"))
    mgr = ContainerManager(docker=docker, db=conn)

    summary = await mgr.reconcile_with_docker()
    assert summary == {"reconciled": 0, "orphans": 0, "ok": 2}


@pytest.mark.asyncio
async def test_reconcile_detects_orphan_containers(conn, caplog):
    # Docker has a workspace container with no DB row (manual row delete,
    # backup restore, etc.). Reconcile must log the orphan + count it
    # without auto-removing — the bind-mounted workspace volume may
    # still hold user data the operator wants to recover.
    import logging
    caplog.set_level(logging.WARNING)
    await _insert_workspace(conn, ws_id="ws-known", status="running")
    docker = _docker_with_states(
        ("ws-known", "running"),
        ("ws-orphan-1", "running"),
        ("ws-orphan-2", "exited"),
    )
    mgr = ContainerManager(docker=docker, db=conn)

    summary = await mgr.reconcile_with_docker()
    assert summary == {"reconciled": 0, "orphans": 2, "ok": 1}
    # No DB row was created for the orphans — they stay as Docker-only.
    row = await (await conn.execute(
        "SELECT COUNT(*) FROM project_checkouts WHERE id LIKE 'ws-orphan%'",
    )).fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_reconcile_tolerates_docker_unreachable(conn):
    # When Docker is down at startup, reconcile must return zeros (not
    # crash, not flip every row to stopped — we genuinely don't know
    # what's running).
    await _insert_workspace(conn, ws_id="ws-x", status="running")
    docker = MagicMock()
    docker.containers = MagicMock()
    docker.containers.list = AsyncMock(side_effect=RuntimeError("docker down"))
    mgr = ContainerManager(docker=docker, db=conn)

    summary = await mgr.reconcile_with_docker()
    assert summary == {"reconciled": 0, "orphans": 0, "ok": 0}
    # Row is unchanged — we did NOT pessimistically mark it stopped.
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-x",),
    )).fetchone()
    assert row[0] == "running"


@pytest.mark.asyncio
async def test_reconcile_maps_exited_dead_to_stopped(conn):
    # Docker has many "not running" states (created/restarting/exited/
    # dead/removing). All of them collapse to ``stopped`` in our DB.
    await _insert_workspace(conn, ws_id="ws-exited", status="running")
    await _insert_workspace(conn, ws_id="ws-dead", status="running")
    await _insert_workspace(conn, ws_id="ws-created", status="running")
    docker = _docker_with_states(
        ("ws-exited", "exited"),
        ("ws-dead", "dead"),
        ("ws-created", "created"),
    )
    mgr = ContainerManager(docker=docker, db=conn)
    summary = await mgr.reconcile_with_docker()
    assert summary["reconciled"] == 3
    for ws_id in ("ws-exited", "ws-dead", "ws-created"):
        row = await (await conn.execute(
            "SELECT status FROM project_checkouts WHERE id=?", (ws_id,),
        )).fetchone()
        assert row[0] == "stopped"


# ----------------------------------------------------------------------
# per-action drift reconciliation — start/stop/pause/unpause
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_reconciles_when_container_gone(conn, mock_docker):
    # docker rm'd externally → containers.get raises 404. stop() must
    # treat 'gone' as 'already stopped' (the goal state), clear the
    # stale container_id, return success.
    await _insert_workspace(conn, ws_id="ws-evaporated", status="running")
    mock_docker.containers.get.side_effect = Exception(
        "[404] No such container: docker-ws-evaporated"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.stop("ws-evaporated")
    assert info.status == "stopped"
    assert info.container_id is None

    row = await (await conn.execute(
        "SELECT status, container_id FROM project_checkouts WHERE id=?",
        ("ws-evaporated",),
    )).fetchone()
    assert row[0] == "stopped"
    assert row[1] is None  # stale id cleared


@pytest.mark.asyncio
async def test_stop_reconciles_when_already_stopped(conn, mock_docker):
    # Container exists in Docker but already stopped — stop() raises 409.
    # No-op + reconcile (container_id stays, row says stopped).
    await _insert_workspace(conn, ws_id="ws-stale", status="running")
    container = mock_docker.containers.get.return_value
    container.stop.side_effect = Exception(
        "[409] container abc is not running"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.stop("ws-stale")
    assert info.status == "stopped"

    row = await (await conn.execute(
        "SELECT status, container_id FROM project_checkouts WHERE id=?",
        ("ws-stale",),
    )).fetchone()
    assert row[0] == "stopped"
    assert row[1] == "docker-ws-stale"  # id retained


@pytest.mark.asyncio
async def test_pause_reconciles_when_container_gone(conn, mock_docker):
    # Extends the original pause-fix to cover the 404 case alongside 409.
    await _insert_workspace(conn, ws_id="ws-vanished", status="running")
    mock_docker.containers.get.side_effect = Exception(
        "[404] No such container"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.pause("ws-vanished")
    assert info.status == "stopped"
    assert info.container_id is None

    row = await (await conn.execute(
        "SELECT status, container_id FROM project_checkouts WHERE id=?",
        ("ws-vanished",),
    )).fetchone()
    assert row[0] == "stopped"
    assert row[1] is None


@pytest.mark.asyncio
async def test_unpause_reconciles_when_container_gone(conn, mock_docker):
    # The silent-loop bug: mark_active sees DB='paused', calls unpause(),
    # Docker says container is gone. Pre-fix, unpause just logged and
    # left DB='paused' — next mark_active tick (30s later) tried again,
    # forever. Now: clear container_id, flip to stopped, break the loop.
    await _insert_workspace(conn, ws_id="ws-thaw-fail", status="paused")
    mock_docker.containers.get.side_effect = Exception(
        "[404] No such container"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.unpause("ws-thaw-fail")
    assert info.status == "stopped"
    assert info.container_id is None

    row = await (await conn.execute(
        "SELECT status, container_id FROM project_checkouts WHERE id=?",
        ("ws-thaw-fail",),
    )).fetchone()
    assert row[0] == "stopped"
    assert row[1] is None


@pytest.mark.asyncio
async def test_unpause_reconciles_when_already_stopped(conn, mock_docker):
    # Container exited between DB read and the unpause RPC (idle reaper
    # stage-2 stop, OOM, etc.). Reconcile to stopped, no loop.
    await _insert_workspace(conn, ws_id="ws-raced", status="paused")
    container = mock_docker.containers.get.return_value
    container.unpause.side_effect = Exception(
        "[409] container abc is not running"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    info = await mgr.unpause("ws-raced")
    assert info.status == "stopped"

    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-raced",),
    )).fetchone()
    assert row[0] == "stopped"


@pytest.mark.asyncio
async def test_start_reconciles_and_raises_when_container_gone(conn, mock_docker):
    # User clicks Start on a workspace whose container was wiped (manual
    # rm, image rebuild, etc.). start() must clear the stale container_id
    # and raise a clear error so the route surfaces "needs recreate"
    # instead of bubbling a raw 404 + leaving stale state.
    await _insert_workspace(conn, ws_id="ws-deleted", status="stopped")
    mock_docker.containers.get.side_effect = Exception(
        "[404] No such container: docker-ws-deleted"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    with pytest.raises(RuntimeError, match="recreate"):
        await mgr.start("ws-deleted")

    row = await (await conn.execute(
        "SELECT status, container_id FROM project_checkouts WHERE id=?",
        ("ws-deleted",),
    )).fetchone()
    assert row[0] == "stopped"
    assert row[1] is None  # stale id cleared


@pytest.mark.asyncio
async def test_mark_active_no_longer_loops_when_unpause_404s(conn, mock_docker):
    # End-to-end of the unpause silent loop: mark_active picks up
    # DB='paused', calls unpause(), Docker 404s. Reconcile must flip
    # the row to 'stopped' so the NEXT mark_active call sees 'stopped'
    # and skips the unpause branch entirely.
    await _insert_workspace(conn, ws_id="ws-loop", status="paused")
    mock_docker.containers.get.side_effect = Exception(
        "[404] No such container"
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    # Bypass the 30s debounce so we can simulate two ticks in a row.
    await mgr.mark_active("ws-loop")
    mgr._activity_last_seen["ws-loop"] = 0

    # After first call, DB should be 'stopped' (reconciled).
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-loop",),
    )).fetchone()
    assert row[0] == "stopped"

    # Second call must NOT re-enter the unpause path (DB no longer says
    # paused). containers.get should only have been touched once.
    mock_docker.containers.get.reset_mock()
    await mgr.mark_active("ws-loop")
    assert mock_docker.containers.get.call_count == 0


# ----------------------------------------------------------------------
# Docker state cache — hot-path coalescing
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_docker_state_cache_coalesces_hot_reads(conn):
    # Two list_workspaces calls within the TTL should share one Docker
    # containers.list round-trip. That's the whole point of the cache —
    # /api/coder/workspaces is hot (entry, 10s poll, every switch).
    await _insert_workspace(conn, ws_id="ws-a", status="running")
    docker = _docker_with_states(("ws-a", "running"))
    mgr = ContainerManager(docker=docker, db=conn)

    await mgr.list_workspaces()
    await mgr.list_workspaces()
    await mgr.list_workspaces()

    # Three list_workspaces calls collapse to one Docker IPC.
    assert docker.containers.list.call_count == 1


@pytest.mark.asyncio
async def test_docker_state_cache_invalidates_on_stop(conn, mock_docker):
    # User-initiated stop must drop the cache so the next read shows the
    # post-stop state, not the TTL-stale pre-stop snapshot.
    await _insert_workspace(conn, ws_id="ws-cycle", status="running")
    mock_docker.containers.list = AsyncMock(return_value=[])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    # Prime cache.
    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 1

    # stop() invalidates → next list_workspaces re-fetches.
    await mgr.stop("ws-cycle")
    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 2


@pytest.mark.asyncio
async def test_docker_state_cache_invalidates_on_start(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-restart", status="stopped")
    mock_docker.containers.list = AsyncMock(return_value=[])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 1

    await mgr.start("ws-restart")
    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 2


@pytest.mark.asyncio
async def test_docker_state_cache_invalidates_on_pause(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-naptime", status="running")
    mock_docker.containers.list = AsyncMock(return_value=[])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 1

    await mgr.pause("ws-naptime")
    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 2


@pytest.mark.asyncio
async def test_docker_state_cache_invalidates_on_delete(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-gone", status="running")
    mock_docker.containers.list = AsyncMock(return_value=[])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 1

    await mgr.delete("ws-gone")
    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == 2


@pytest.mark.asyncio
async def test_docker_state_cache_invalidates_on_drift_reconcile(conn, mock_docker):
    # When pause() detects "container gone" and reconciles to stopped,
    # the cache must also drop — otherwise a follow-up list_workspaces
    # would still report the row as running until TTL.
    await _insert_workspace(conn, ws_id="ws-poof", status="running")
    mock_docker.containers.list = AsyncMock(return_value=[])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr.list_workspaces()
    primed_count = mock_docker.containers.list.call_count

    mock_docker.containers.get.side_effect = Exception("[404] No such container")
    await mgr.pause("ws-poof")  # triggers _reconcile_to_stopped
    await mgr.list_workspaces()
    assert mock_docker.containers.list.call_count == primed_count + 1


@pytest.mark.asyncio
async def test_docker_state_cache_ttl_expires(conn, monkeypatch):
    # Beyond TTL the cache is stale and the next read must refresh.
    # Tested by rewinding the cached timestamp instead of sleeping.
    await _insert_workspace(conn, ws_id="ws-tick", status="running")
    docker = _docker_with_states(("ws-tick", "running"))
    mgr = ContainerManager(docker=docker, db=conn)

    await mgr.list_workspaces()
    assert docker.containers.list.call_count == 1

    # Force the cache stale by rewinding its timestamp past the TTL.
    ts, states = mgr._docker_state_cache
    mgr._docker_state_cache = (ts - mgr._DOCKER_STATE_CACHE_TTL_S - 1, states)

    await mgr.list_workspaces()
    assert docker.containers.list.call_count == 2


@pytest.mark.asyncio
async def test_docker_state_cache_concurrent_misses_share_one_fetch(conn):
    # Two coroutines that miss the cache simultaneously must share ONE
    # Docker call. Without the lock guard, both would race straight into
    # containers.list and we'd amplify load under contention (10s poller
    # + mode-entry both arriving on a cold cache).
    await _insert_workspace(conn, ws_id="ws-x", status="running")
    docker = _docker_with_states(("ws-x", "running"))

    # Slow down containers.list so we can interleave waiters before it
    # resolves. Without this, the first call completes before the second
    # arrives and the test trivially passes for the wrong reason.
    import asyncio as _asyncio
    async def _slow_list(**_):
        await _asyncio.sleep(0.05)
        return docker.containers.list.return_value
    docker.containers.list = AsyncMock(side_effect=_slow_list)

    mgr = ContainerManager(docker=docker, db=conn)
    results = await _asyncio.gather(
        mgr.list_workspaces(), mgr.list_workspaces(), mgr.list_workspaces(),
    )
    assert all(len(r) == 1 for r in results)
    assert docker.containers.list.call_count == 1


@pytest.mark.asyncio
async def test_docker_state_cache_reconcile_always_fresh(conn):
    # reconcile_with_docker MUST drop the cache before reading — startup
    # reconciliation should never trust a TTL-stale snapshot from a
    # previous server lifetime (impossible in practice since the cache
    # is in-memory, but the rule is "reconcile reads truth").
    await _insert_workspace(conn, ws_id="ws-r", status="running")
    docker = _docker_with_states(("ws-r", "running"))
    mgr = ContainerManager(docker=docker, db=conn)

    await mgr.list_workspaces()  # primes cache
    primed = docker.containers.list.call_count

    await mgr.reconcile_with_docker()
    assert docker.containers.list.call_count == primed + 1


@pytest.mark.asyncio
async def test_docker_state_map_propagates_errors(conn):
    # When docker is unreachable, callers branch on the exception (the
    # helper used to swallow → empty dict, which the reconcile path then
    # could not distinguish from "no containers" → could trigger wrong
    # writes). Verify it surfaces the error.
    docker = MagicMock()
    docker.containers = MagicMock()
    docker.containers.list = AsyncMock(side_effect=RuntimeError("docker down"))
    mgr = ContainerManager(docker=docker, db=conn)

    with pytest.raises(RuntimeError, match="docker down"):
        await mgr._docker_state_map()


# ----------------------------------------------------------------------
# import_archive_into — counterpart to workspace_archive_stream
# ----------------------------------------------------------------------

def _make_workspace_export_bytes(file_payloads: dict[str, bytes]) -> bytes:
    """Build a gzipped tar in the same shape as workspace_archive_stream
    produces: top-level ``workspace/`` directory containing the named
    files. Returns the gzip bytes.
    """
    import gzip
    import io
    import tarfile
    import time as _time

    raw = io.BytesIO()
    now = int(_time.time())
    with tarfile.open(fileobj=raw, mode="w") as tar:
        # Directory entry first so extraction works on strict tar readers.
        d = tarfile.TarInfo(name="workspace")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        d.mtime = now
        tar.addfile(d)
        for rel, data in file_payloads.items():
            ti = tarfile.TarInfo(name=f"workspace/{rel}")
            ti.size = len(data)
            ti.mtime = now
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


@pytest.mark.asyncio
async def test_import_archive_into_extracts_at_root(conn, mock_docker):
    # Round-trip the export-format archive into a workspace. The Docker
    # put_archive call should receive RAW tar bytes (gunzip'd) extracted
    # at /, because the archive's top-level entry is ``workspace/``.
    await _insert_workspace(conn, ws_id="ws-imp", status="running")
    container = mock_docker.containers.get.return_value
    container.put_archive = AsyncMock()
    mgr = ContainerManager(docker=mock_docker, db=conn)

    payload = _make_workspace_export_bytes({
        "main.py": b"print('hello from import')",
        "subdir/note.txt": b"second file",
    })
    await mgr.import_archive_into("ws-imp", payload)

    container.put_archive.assert_awaited_once()
    call = container.put_archive.await_args
    assert call.kwargs["path"] == "/"
    sent = call.kwargs["data"]
    # Verify the bytes Docker received are valid raw tar (i.e. gunzip'd).
    import io as _io, tarfile as _tarfile
    with _tarfile.open(fileobj=_io.BytesIO(sent), mode="r") as tar:
        names = tar.getnames()
    assert "workspace/main.py" in names
    assert "workspace/subdir/note.txt" in names


@pytest.mark.asyncio
async def test_import_archive_into_rejects_empty(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-empty", status="running")
    mgr = ContainerManager(docker=mock_docker, db=conn)

    with pytest.raises(ValueError, match="empty"):
        await mgr.import_archive_into("ws-empty", b"")


@pytest.mark.asyncio
async def test_import_archive_into_rejects_corrupt_gzip(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-corrupt", status="running")
    mgr = ContainerManager(docker=mock_docker, db=conn)

    with pytest.raises(ValueError, match="gzip"):
        await mgr.import_archive_into(
            "ws-corrupt", b"this is not a gzip file at all",
        )


@pytest.mark.asyncio
async def test_import_archive_into_requires_container(conn, mock_docker):
    # A workspace whose container has been wiped (start-reconcile cleared
    # the id) can't be imported into — caller must recreate first.
    await conn.execute(
        "INSERT INTO project_checkouts (id, name, container_id, status) "
        "VALUES (?, ?, NULL, ?)",
        ("ws-no-container", "no-container", "stopped"),
    )
    await conn.commit()
    mgr = ContainerManager(docker=mock_docker, db=conn)

    with pytest.raises(ValueError, match="no associated container"):
        await mgr.import_archive_into(
            "ws-no-container", _make_workspace_export_bytes({"x": b"y"}),
        )


# ----------------------------------------------------------------------
# Keep-alive decoupling — _assemble_keepalive_cmd
#
# Regression guard for the 2026-06-20 workspace-container-death incident:
# a provisioning failure (python3/npm not found mid-chain) aborted the
# `&&` chain BEFORE `tail -f /dev/null`, so PID 1 exited 127 and the
# container died under a running agent turn. The keep-alive must be
# unconditional — never gated behind the install chain's success.
# ----------------------------------------------------------------------
def test_keepalive_cmd_shape_is_sh_c():
    cmd = _assemble_keepalive_cmd(["echo hi"])
    assert cmd[0] == "sh" and cmd[1] == "-c"
    assert len(cmd) == 3


def test_keepalive_is_unconditional_not_chained_after_setup():
    # The OLD bug: `<setup> && tail -f /dev/null` — a failing setup step
    # short-circuits the chain and PID 1 exits. The fix must NOT join the
    # keep-alive to the setup with `&&`.
    script = _assemble_keepalive_cmd(["false", "touch /workspace/.augmentum/ready"])[2]
    assert "exec tail -f /dev/null" in script
    # The keep-alive must be reachable via `;` (statement separator), never
    # as the tail of the provisioning `&&` chain.
    assert "&& tail -f /dev/null" not in script
    assert "&& exec tail -f /dev/null" not in script


def test_keepalive_runs_setup_in_isolated_subshell():
    # Setup runs inside `( ... )` so its `&&` chain (and any failure) is
    # contained — it cannot reach past the subshell to the keep-alive.
    script = _assemble_keepalive_cmd(["stepA", "stepB"])[2]
    assert "( stepA && stepB )" in script
    # exec tail comes AFTER the subshell closes, separated by `;`.
    assert script.index("( stepA && stepB )") < script.index("exec tail -f /dev/null")
    sub = script.split("( stepA && stepB )")[1]
    assert sub.lstrip().startswith(">") or "provision.log" in sub


def test_keepalive_captures_diagnostics_markers():
    script = _assemble_keepalive_cmd(["do_install"])[2]
    assert "/workspace/.augmentum/provision.log" in script
    assert "/workspace/.augmentum/provision.exit" in script
    # Exit code captured AFTER the subshell so a failed install is
    # diagnosable on a still-alive container.
    assert "echo $? >" in script


def test_keepalive_hoists_mkdir_before_redirect():
    # The provision.log redirect target dir must exist before the shell
    # opens the redirect — so mkdir is hoisted OUT of the redirected subshell.
    script = _assemble_keepalive_cmd(["x"])[2]
    mkdir_at = script.index("mkdir -p /workspace/.augmentum")
    redirect_at = script.index("> /workspace/.augmentum/provision.log")
    assert mkdir_at < redirect_at


def test_keepalive_failing_step_cannot_reach_keepalive_via_shell_semantics():
    # Belt-and-suspenders: simulate POSIX sh semantics on the generated
    # script's structure. With `( a && b ) > log; exec tail`, a failing `a`
    # stops the subshell but the `;` still runs the keep-alive. With the OLD
    # `a && b && tail`, a failing `a` skips the tail. Assert we're the former.
    script = _assemble_keepalive_cmd(["FAILING", "after"])[2]
    # Everything before the first top-level `;` is the (contained) provision
    # group; the keep-alive lives after a `;`, not after the provision `&&`.
    head, _, tail = script.partition("; exec tail -f /dev/null")
    assert "exec tail -f /dev/null" not in head  # keep-alive not inside provision
    assert tail == ""  # keep-alive is the final statement


# ----------------------------------------------------------------------
# Revive-on-409 — _revive_container
#
# When docker exec hits `[409] … is not running`, the turn must self-heal
# (start the container once, reconcile the DB) instead of 409-ing on every
# subsequent tool call for the rest of the session.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revive_container_starts_and_reconciles_db(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-revive", status="running")
    container = mock_docker.containers.get.return_value
    container.show = AsyncMock(return_value={"State": {"Running": True}})
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr._revive_container(container, "ws-revive")

    container.start.assert_awaited_once()
    row = await (await conn.execute(
        "SELECT status FROM project_checkouts WHERE id=?", ("ws-revive",),
    )).fetchone()
    assert row[0] == "running"


@pytest.mark.asyncio
async def test_revive_container_tolerates_already_started_race(conn, mock_docker):
    # Another tool call racing us already started it: Docker says
    # "already started". Benign — must not raise.
    await _insert_workspace(conn, ws_id="ws-race", status="running")
    container = mock_docker.containers.get.return_value
    container.start = AsyncMock(
        side_effect=Exception("[304] container abc already started")
    )
    container.show = AsyncMock(return_value={"State": {"Running": True}})
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr._revive_container(container, "ws-race")  # no raise
    container.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_revive_container_raises_on_unrecoverable_start_failure(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-dead", status="running")
    container = mock_docker.containers.get.return_value
    container.start = AsyncMock(
        side_effect=Exception("[500] driver failed: no space left on device")
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    with pytest.raises(RuntimeError, match="could not be revived"):
        await mgr._revive_container(container, "ws-dead")


@pytest.mark.asyncio
async def test_revive_container_logs_oom_when_killed(conn, mock_docker, capsys):
    # An OOM kill (exit 137 / OOMKilled) under the no-swap cap must surface a
    # distinct `run_command_container_oom` warning so recurring OOM is
    # diagnosable rather than hidden behind a generic revive.
    await _insert_workspace(conn, ws_id="ws-oom", status="running")
    container = mock_docker.containers.get.return_value
    container.show = AsyncMock(side_effect=[
        {"State": {"OOMKilled": True, "ExitCode": 137, "Running": False}},  # pre-revive probe
        {"State": {"Running": True}},                                        # post-start wait
    ])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr._revive_container(container, "ws-oom")
    assert "run_command_container_oom" in capsys.readouterr().out
    container.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_revive_container_no_oom_log_on_clean_exit(conn, mock_docker, capsys):
    await _insert_workspace(conn, ws_id="ws-clean", status="running")
    container = mock_docker.containers.get.return_value
    container.show = AsyncMock(side_effect=[
        {"State": {"OOMKilled": False, "ExitCode": 0, "Running": False}},
        {"State": {"Running": True}},
    ])
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr._revive_container(container, "ws-clean")
    assert "run_command_container_oom" not in capsys.readouterr().out


# ----------------------------------------------------------------------
# Reaper-race prevention — _touch_last_active (exec proves liveness)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_touch_last_active_bumps_then_debounces(conn, mock_docker):
    await _insert_workspace(conn, ws_id="ws-touch", status="running", last_active=0.0)
    mgr = ContainerManager(docker=mock_docker, db=conn)

    await mgr._touch_last_active("ws-touch")
    row = await (await conn.execute(
        "SELECT last_active FROM project_checkouts WHERE id=?", ("ws-touch",),
    )).fetchone()
    first = row[0]
    assert first is not None and first > 0

    # A second call within the 30s debounce window must NOT write again.
    await mgr._touch_last_active("ws-touch")
    row = await (await conn.execute(
        "SELECT last_active FROM project_checkouts WHERE id=?", ("ws-touch",),
    )).fetchone()
    assert row[0] == first


@pytest.mark.asyncio
async def test_touch_last_active_keeps_active_ws_out_of_reaper(conn, mock_docker):
    # Stale row that WOULD be reaped — until an exec bumps it. Proves the
    # reaper-race fix: a workspace actively running commands is never idle.
    await _insert_workspace(
        conn, ws_id="ws-busy", status="running", last_active=time.time() - 99999,
    )
    mgr = ContainerManager(docker=mock_docker, db=conn)

    # Before the bump: the sweep_idle selection (running + stale) would match.
    before = await (await conn.execute(
        "SELECT id FROM project_checkouts WHERE status='running' "
        "AND (last_active IS NULL OR last_active < ?)", (time.time() - 1800,),
    )).fetchall()
    assert ("ws-busy",) in [tuple(r) for r in before]

    await mgr._touch_last_active("ws-busy")  # an exec just ran

    after = await (await conn.execute(
        "SELECT id FROM project_checkouts WHERE status='running' "
        "AND (last_active IS NULL OR last_active < ?)", (time.time() - 1800,),
    )).fetchall()
    assert ("ws-busy",) not in [tuple(r) for r in after]


# ----------------------------------------------------------------------
# Swap cushion — _resolve_memory_swap
# ----------------------------------------------------------------------


def test_resolve_memory_swap_default_adds_cushion():
    gib = 2 * 1024 * 1024 * 1024
    swap = _resolve_memory_swap(gib)
    # Default ratio 0.5 → MemorySwap is 1.5× memory (i.e. 1GB of swap on a 2GB ws).
    assert swap == gib + gib // 2
    assert swap > gib  # there IS headroom now (the whole point)


def test_resolve_memory_swap_ratio_zero_restores_no_swap(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_workspace_swap_ratio", 0.0)
    gib = 2 * 1024 * 1024 * 1024
    assert _resolve_memory_swap(gib) == gib  # MemorySwap == Memory (old behavior)
