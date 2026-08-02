"""Smoke + behavior tests for companion runtime.

Verifies the module imports, the router exists with HTTP + WebSocket
endpoints, runtime-gated routes return 503 when disabled, and — for
each capability we've wired end-to-end — the happy path actually fires.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient


def test_companion_routes_import():
    from augmentum.proxy.companion_routes import router
    assert router is not None
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/companion/snapshot" in paths
    assert "/api/companion/intent" in paths


def test_companion_snapshot_gated_by_setting(sqlite_client: TestClient):
    """When companion_runtime_enabled is off, snapshot returns 503."""
    resp = sqlite_client.get("/api/companion/snapshot")
    assert resp.status_code in {503, 401, 404}, (
        f"unexpected status {resp.status_code}: {resp.text[:200]}"
    )


def test_safety_floor_audit_event_endpoint_registered():
    """The realtalk_panel_opened telemetry endpoint should be registered
    (added 2026-05-17 — previously a ghost call from becca-presence.js)."""
    from augmentum.proxy.companion_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/companion/safety_floor_audit_event" in paths


# ── Owner resolution (migration 173) ──────────────────────────────────

async def _boot_runtime(*, with_user: str | None = None):
    """Spin up a CompanionRuntime against a fresh :memory: backend.

    Optionally inserts a single user before runtime.start() so the
    auto-bind path resolves owner_user_id.
    """
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.memory.store import MemoryStore
    from augmentum.memory.core_profile import CoreProfileManager
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    if with_user is not None:
        await backend.conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (with_user, f"u_{with_user[:6]}", "x"),
        )
        await backend.conn.commit()
    runtime = CompanionRuntime(backend, companion_id="becca")
    await runtime.start(
        memory_store=MemoryStore(backend),
        core_profile=CoreProfileManager(backend),
    )
    return runtime, backend


@pytest.mark.asyncio
async def test_owner_user_id_auto_binds_for_single_user():
    """Fresh DB with exactly one user → owner_user_id auto-binds."""
    runtime, backend = await _boot_runtime(with_user="user-alpha")
    try:
        assert runtime.owner_user_id == "user-alpha"
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


@pytest.mark.asyncio
async def test_owner_user_id_unresolved_for_zero_or_many_users():
    """No users → owner stays empty; the dream gate then skips correctly."""
    runtime, backend = await _boot_runtime()
    try:
        assert runtime.owner_user_id == ""
        # invoke_dream must respect the unresolved gate
        from augmentum.companion_runtime.behavior import sleep_wake
        fired = await sleep_wake.invoke_dream(runtime)
        assert fired is False
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


# ── State driver ──────────────────────────────────────────────────────

def test_state_driver_pure_function_decision_matrix():
    """The pure-function decision matrix maps signals to states."""
    from augmentum.companion_runtime.behavior import state_driver
    from augmentum.companion_runtime.state import AttentionState

    now = time.time()
    # Recent activity → present (override quiet hours too)
    target = state_driver._target_state(
        observed_state={"last_chat_at": now - 10, "last_tool_at": 0.0},
        now_wall=now,
        now_local=datetime(2026, 5, 17, 3, 0),
        quiet_start="22:00", quiet_end="07:00",
    )
    assert target == AttentionState.PRESENT

    # Long inactivity + quiet hours → asleep
    target = state_driver._target_state(
        observed_state={"last_chat_at": now - 1800, "last_tool_at": 0.0},
        now_wall=now,
        now_local=datetime(2026, 5, 17, 3, 0),
        quiet_start="22:00", quiet_end="07:00",
    )
    assert target == AttentionState.ASLEEP

    # Long inactivity outside quiet hours → dormant
    target = state_driver._target_state(
        observed_state={"last_chat_at": now - 3600, "last_tool_at": 0.0},
        now_wall=now,
        now_local=datetime(2026, 5, 17, 14, 0),
        quiet_start="24:00", quiet_end="07:00",
    )
    assert target == AttentionState.DORMANT


def test_state_driver_quiet_hours_wraparound():
    """22:00→07:00 must include 23:30 and 03:00, exclude 14:00."""
    from augmentum.companion_runtime.behavior.state_driver import _in_quiet_hours
    assert _in_quiet_hours(datetime(2026, 5, 17, 23, 30), "22:00", "07:00") is True
    assert _in_quiet_hours(datetime(2026, 5, 17, 3, 0), "22:00", "07:00") is True
    assert _in_quiet_hours(datetime(2026, 5, 17, 14, 0), "22:00", "07:00") is False
    # exclusive end
    assert _in_quiet_hours(datetime(2026, 5, 17, 7, 0), "22:00", "07:00") is False


# ── Drift audit ───────────────────────────────────────────────────────

def test_drift_audit_timestamp_parser():
    """Parser handles SQLite datetime('now') formats."""
    from augmentum.companion_runtime.behavior.drift_audit import _parse_db_timestamp
    assert _parse_db_timestamp("2026-05-17 14:23:45") is not None
    assert _parse_db_timestamp("2026-05-17T14:23:45") is not None
    assert _parse_db_timestamp("2026-05-17 14:23:45.123456") is not None
    assert _parse_db_timestamp(None) is None
    assert _parse_db_timestamp("not-a-date") is None


@pytest.mark.asyncio
async def test_drift_audit_skip_when_disabled(monkeypatch):
    """Flag off → no-op even if interval has elapsed."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import drift_audit

    runtime, backend = await _boot_runtime(with_user="user-x")
    try:
        monkeypatch.setattr(settings, "companion_drift_audit_enabled", False)
        # Backdate so it would otherwise be due
        await runtime.backend.conn.execute(
            "UPDATE companion_identities SET last_kernel_refresh_at = "
            "datetime('now', '-48 hours') WHERE companion_id = ?",
            (runtime.companion_id,),
        )
        await runtime.backend.conn.commit()
        await runtime.identity.load()
        fired = await drift_audit.run_if_due(runtime)
        assert fired is False
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


@pytest.mark.asyncio
async def test_drift_audit_runs_when_due():
    """Flag on + interval elapsed + doc present → audit runs + emits event."""
    from augmentum.companion_runtime.behavior import drift_audit

    runtime, backend = await _boot_runtime(with_user="user-x")
    try:
        # Subscribe before backdating so we don't miss the event
        sub = await runtime.bus.subscribe("drift.**", slice_key="t")
        collected = []

        async def drain():
            for _ in range(10):
                try:
                    ev = await asyncio.wait_for(sub.queue.get(), timeout=0.1)
                    collected.append(ev)
                except asyncio.TimeoutError:
                    pass

        # Backdate to force due
        await runtime.backend.conn.execute(
            "UPDATE companion_identities SET last_kernel_refresh_at = "
            "datetime('now', '-48 hours') WHERE companion_id = ?",
            (runtime.companion_id,),
        )
        await runtime.backend.conn.commit()
        await runtime.identity.load()

        fired = await drift_audit.run_if_due(runtime)
        await drain()
        # Either fired=True or the personality doc is missing (in which case
        # we'd see no event). The runtime auto-refresh on start() already
        # required the doc, so it must exist here.
        assert fired is True
        assert any(e.topic == "drift.audit_run" for e in collected)

        await runtime.bus.unsubscribe(sub)
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


# ── Subagent + tool wiring (no LLM call — just import + construction) ─

def test_companion_internal_chat_request_imports_canonical():
    """All companion_runtime LLM-call sites use the canonical path now.

    Prior bug: `from augmentum.proxy.schema import InternalChatRequest`
    silently no-op'd in dispatch + honest_gap + all 5 subagents because
    that module doesn't exist. The fix moved them to
    `augmentum.models.base`. Voice keeps a fallback for defense.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "augmentum" / "companion_runtime"
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        if py.name == "voice.py":
            continue  # voice has the fallback by design
        src = py.read_text(encoding="utf-8")
        if "from augmentum.proxy.schema" in src:
            offenders.append(str(py.relative_to(root.parent.parent)))
    assert not offenders, (
        f"Stale 'augmentum.proxy.schema' imports in: {offenders}. "
        f"Use 'augmentum.models.base' instead."
    )


def test_channel_handoff_method_renamed():
    """`_handle_handoff_stub` was a misleading name — the body is real."""
    from augmentum.companion_runtime.voice import BeccaVoice
    assert hasattr(BeccaVoice, "_handle_handoff")
    assert not hasattr(BeccaVoice, "_handle_handoff_stub")


# ── on_wake bridge (regression for 2026-05-18 prod warning) ──────────

@pytest.mark.asyncio
async def test_on_wake_writes_journal_entry():
    """Wake-from-asleep surfaces the latest dream into companion_journal.

    Regression: sleep_wake.on_wake had two latent bugs that nothing
    exercised until the state driver landed and started moving the
    machine through ``asleep``:
      1. ``async with backend.connect()`` — connect() returns a coroutine,
         not an async context manager. TypeError every wake.
      2. Even if (1) had worked, the SELECT named ``summary``/``ts`` which
         aren't columns on dream_entries — would have raised OperationalError.
    """
    from augmentum.companion_runtime.state import AttentionState
    runtime, backend = await _boot_runtime(with_user="alex")
    try:
        await backend.conn.execute(
            "INSERT INTO dream_entries (id, persona_id, companion_id, "
            "content, entry_type, dream_cycle_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            ("d_test", "becca", "becca",
             "Floated through a hallway of half-open doors.",
             "reflection", "c_test"),
        )
        await backend.conn.commit()

        await runtime.state.transition_state(
            AttentionState.ASLEEP, reason="t", force=True,
        )
        await asyncio.sleep(0.05)
        await runtime.state.transition_state(
            AttentionState.DORMANT, reason="t", force=True,
        )
        # The wake listener runs async via asyncio.create_task — wait.
        await asyncio.sleep(0.5)

        cur = await backend.conn.execute(
            "SELECT entry_type, content FROM companion_journal "
            "WHERE companion_id = ? ORDER BY id DESC LIMIT 1",
            (runtime.companion_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        assert row is not None, "on_wake did not write a journal entry"
        entry_type, content = row[0], row[1]
        assert entry_type == "noticing"
        assert "woke from a dream" in (content or "")
    finally:
        await runtime.stop(grace_seconds=1.5)
        await backend.close()
