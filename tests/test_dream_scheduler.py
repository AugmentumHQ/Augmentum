"""Tests for the per-user dream scheduler.

The scheduler is a process singleton that keys every piece of state
(counters, last-dream timestamp, running-set) by ``user_id``. Each
test sets counters for a specific user, then asserts the gate decision
for that user only.

Pre-Stage-B tests in this file treated the scheduler's state as
scalars — that interface is gone. The tests below exercise the
post-multi-tenant scheduler plus the two new gates added in this
refactor:

* ``_user_opted_in`` — consults ``ui.dreamEnabled`` per user.
* ``_user_thresholds`` — resolves per-user message/idle/cooldown
  overrides with constructor fallbacks.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from augmentum.dream.scheduler import DreamScheduler, DreamsDisabledError
from augmentum.state.settings_store import SettingsStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        "CREATE TABLE user_settings ("
        "  user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')),"
        "  PRIMARY KEY (user_id, key))"
    )
    await conn.commit()
    return conn, SettingsStore(conn)


def _build_scheduler(store: SettingsStore | None) -> DreamScheduler:
    """Scheduler with a mock engine and no running task."""
    return DreamScheduler(
        engine=AsyncMock(),
        settings_store=store,
        enabled=True,
        message_threshold=10,
        idle_minutes=30,
        cooldown_minutes=60,
    )


def _set_idle(scheduler: DreamScheduler, user_id: str, minutes: int) -> None:
    """Mark ``user_id`` as idle for ``minutes`` minutes."""
    scheduler._last_request_at[user_id] = datetime.now(UTC) - timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# _is_eligible — deterministic trigger gate (per user)
# ---------------------------------------------------------------------------


class TestIsEligible:
    async def test_below_message_threshold(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 5
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 60)
        assert await s._is_eligible("u1") is False
        await conn.close()

    async def test_no_approved_memories(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 0
        _set_idle(s, "u1", 60)
        assert await s._is_eligible("u1") is False
        await conn.close()

    async def test_user_still_active(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 0)  # just requested
        assert await s._is_eligible("u1") is False
        await conn.close()

    async def test_all_conditions_met(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 35)
        assert await s._is_eligible("u1") is True
        await conn.close()

    async def test_cooldown_blocks(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 35)
        s._last_dream_at["u1"] = datetime.now(UTC) - timedelta(minutes=30)
        assert await s._is_eligible("u1") is False
        await conn.close()

    async def test_running_blocks(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 35)
        s._running_for.add("u1")
        assert await s._is_eligible("u1") is False
        await conn.close()

    async def test_isolated_per_user(self):
        """u1 eligible, u2 not — state is keyed by user."""
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 35)
        s._messages_since["u2"] = 2  # below threshold
        assert await s._is_eligible("u1") is True
        assert await s._is_eligible("u2") is False
        await conn.close()


# ---------------------------------------------------------------------------
# _user_thresholds — per-user override lookup
# ---------------------------------------------------------------------------


class TestUserThresholds:
    async def test_fallback_to_constructor_without_store(self):
        s = _build_scheduler(None)
        assert await s._user_thresholds("u1") == (10, 30, 60)

    async def test_fallback_when_user_unset(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        assert await s._user_thresholds("u1") == (10, 30, 60)
        await conn.close()

    async def test_user_override_honoured(self):
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamMessageThreshold", "25")
        await store.set_user("u1", "ui.dreamIdleMinutes", "15")
        await store.set_user("u1", "ui.dreamCooldownMinutes", "120")
        s = _build_scheduler(store)
        assert await s._user_thresholds("u1") == (25, 15, 120)
        await conn.close()

    async def test_malformed_value_falls_back(self):
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamMessageThreshold", "not-a-number")
        s = _build_scheduler(store)
        threshold, idle, cooldown = await s._user_thresholds("u1")
        assert threshold == 10  # constructor default
        assert idle == 30
        assert cooldown == 60
        await conn.close()

    async def test_partial_override(self):
        """One field overridden, others fall back."""
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamIdleMinutes", "5")
        s = _build_scheduler(store)
        assert await s._user_thresholds("u1") == (10, 5, 60)
        await conn.close()

    async def test_eligibility_uses_user_threshold(self):
        """Lower threshold on one user should flip eligibility."""
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 5
        s._approved_since["u1"] = 1
        _set_idle(s, "u1", 35)
        # With default threshold 10, not eligible
        assert await s._is_eligible("u1") is False
        # User lowers threshold to 3 → eligible
        await store.set_user("u1", "ui.dreamMessageThreshold", "3")
        assert await s._is_eligible("u1") is True
        await conn.close()


# ---------------------------------------------------------------------------
# _user_opted_in — per-user dreamEnabled gate
# ---------------------------------------------------------------------------


class TestUserOptedIn:
    async def test_defaults_true_without_store(self):
        """Test harness with no settings store — permissive fallback."""
        s = _build_scheduler(None)
        assert await s._user_opted_in("u1") is True

    async def test_false_when_user_unset(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        assert await s._user_opted_in("u1") is False
        await conn.close()

    async def test_true_when_user_opts_in(self):
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamEnabled", "true")
        s = _build_scheduler(store)
        assert await s._user_opted_in("u1") is True
        await conn.close()

    async def test_false_when_user_opts_out(self):
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamEnabled", "false")
        s = _build_scheduler(store)
        assert await s._user_opted_in("u1") is False
        await conn.close()

    async def test_global_fallback(self):
        """When the user hasn't set their own value, the install-wide
        default applies — matches ``get_user_or_global`` semantics."""
        conn, store = await _make_store()
        await store.set("ui.dreamEnabled", "true")
        s = _build_scheduler(store)
        assert await s._user_opted_in("u1") is True
        await conn.close()


# ---------------------------------------------------------------------------
# Notifications — per-user counters
# ---------------------------------------------------------------------------


class TestNotify:
    def test_notify_message_increments_per_user(self):
        s = _build_scheduler(None)
        s.notify_message(user_id="u1")
        s.notify_message(user_id="u1")
        s.notify_message(user_id="u2")
        assert s._messages_since["u1"] == 2
        assert s._messages_since["u2"] == 1

    def test_notify_approval_increments_per_user(self):
        s = _build_scheduler(None)
        s.notify_approval("mem_1", user_id="u1")
        s.notify_approval("mem_2", user_id="u2")
        s.notify_approval("mem_3", user_id="u2")
        assert s._approved_since["u1"] == 1
        assert s._approved_since["u2"] == 2

    def test_reset_counters_scoped(self):
        s = _build_scheduler(None)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 10
        s._messages_since["u2"] = 50
        s._reset_counters("u1")
        assert s._messages_since["u1"] == 0
        assert s._approved_since["u1"] == 0
        assert s._messages_since["u2"] == 50  # untouched


# ---------------------------------------------------------------------------
# get_status — async, per-user
# ---------------------------------------------------------------------------


class TestGetStatus:
    async def test_returns_per_user_counters(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 7
        s._approved_since["u1"] = 2
        status = await s.get_status("u1")
        assert status["enabled"] is True
        assert status["messages_since_dream"] == 7
        assert status["approved_memories_since_dream"] == 2
        assert status["running"] is False
        await conn.close()

    async def test_eligible_reflects_user_state(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        s._messages_since["u1"] = 100
        s._approved_since["u1"] = 5
        _set_idle(s, "u1", 35)
        status = await s.get_status("u1")
        assert status["next_dream_eligible"] is True
        await conn.close()


# ---------------------------------------------------------------------------
# trigger_manual — rejects opted-out users
# ---------------------------------------------------------------------------


class TestTriggerManual:
    async def test_raises_when_user_opted_out(self):
        conn, store = await _make_store()
        s = _build_scheduler(store)
        with pytest.raises(DreamsDisabledError):
            await s.trigger_manual(user_id="u1")
        # Engine must NOT have been called — we refuse before running.
        s._engine.run_cycle.assert_not_called()
        await conn.close()

    async def test_runs_when_user_opted_in(self):
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamEnabled", "true")
        s = _build_scheduler(store)
        s._engine.run_cycle = AsyncMock(
            return_value=type("Cycle", (), {"id": "cyc1", "entries_count": 3}),
        )
        cycle_id = await s.trigger_manual(user_id="u1")
        assert cycle_id == "cyc1"
        s._engine.run_cycle.assert_called_once_with("default", "manual", user_id="u1")
        # Counters reset post-cycle
        assert s._messages_since["u1"] == 0
        assert s._approved_since["u1"] == 0
        await conn.close()

    async def test_already_running_is_idempotent(self):
        conn, store = await _make_store()
        await store.set_user("u1", "ui.dreamEnabled", "true")
        s = _build_scheduler(store)
        s._running_for.add("u1")
        result = await s.trigger_manual(user_id="u1")
        assert result == "already_running"
        s._engine.run_cycle.assert_not_called()
        await conn.close()

    async def test_runs_without_store(self):
        """Legacy single-tenant path with no store wired — permissive fallback."""
        s = _build_scheduler(None)
        s._engine.run_cycle = AsyncMock(
            return_value=type("Cycle", (), {"id": "cyc2", "entries_count": 1}),
        )
        cycle_id = await s.trigger_manual(user_id="")
        assert cycle_id == "cyc2"
