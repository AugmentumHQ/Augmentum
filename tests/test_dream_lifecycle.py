"""Tests for the dream subsystem lifecycle.

Covers the decision layer: who wants dreams on, who wants them off, and
whether the process-level singleton should be booted or torn down in
response. The heavy machinery (journal/portrait/engine/scheduler) is
tested elsewhere — this file verifies only the gating.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from augmentum.dream.lifecycle import should_dream_run
from augmentum.state.settings_store import SettingsStore


async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    """Fresh in-memory SettingsStore with both settings tables."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now'))"
        ")"
    )
    await conn.execute(
        "CREATE TABLE user_settings ("
        "  user_id TEXT NOT NULL,"
        "  key TEXT NOT NULL,"
        "  value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')),"
        "  PRIMARY KEY (user_id, key)"
        ")"
    )
    await conn.commit()
    return conn, SettingsStore(conn)


class TestShouldDreamRun:
    """``should_dream_run`` is the single source of truth for "is the
    dream singleton wanted?". Boot path and teardown path both consult
    it — divergence here was the original bug."""

    async def test_no_store(self):
        """Without a settings store we can't know — default to off."""
        assert await should_dream_run(None) is False

    async def test_fresh_install_no_opt_in(self):
        conn, store = await _make_store()
        assert await should_dream_run(store) is False
        await conn.close()

    async def test_install_default_on(self):
        """Legacy single-tenant installs stored the flag globally."""
        conn, store = await _make_store()
        await store.set("ui.dreamEnabled", "true")
        assert await should_dream_run(store) is True
        await conn.close()

    async def test_install_default_explicitly_off(self):
        conn, store = await _make_store()
        await store.set("ui.dreamEnabled", "false")
        assert await should_dream_run(store) is False
        await conn.close()

    async def test_single_user_opted_in(self):
        """The exact symptom this refactor fixes: global is unset but a
        user has it on via personalization."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        assert await should_dream_run(store) is True
        await conn.close()

    async def test_user_opted_out_but_another_opted_in(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "false")
        await store.set_user("bob", "ui.dreamEnabled", "true")
        assert await should_dream_run(store) is True
        await conn.close()

    async def test_all_users_opted_out(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "false")
        await store.set_user("bob", "ui.dreamEnabled", "false")
        assert await should_dream_run(store) is False
        await conn.close()

    async def test_last_user_flips_off(self):
        """Simulates the teardown decision after one user flips off."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        assert await should_dream_run(store) is True
        # alice disables
        await store.set_user("alice", "ui.dreamEnabled", "false")
        assert await should_dream_run(store) is False
        await conn.close()

    async def test_global_wins_when_all_users_off(self):
        """Even if every user said no, an install default of true keeps
        the subsystem alive. Admin override."""
        conn, store = await _make_store()
        await store.set("ui.dreamEnabled", "true")
        await store.set_user("alice", "ui.dreamEnabled", "false")
        assert await should_dream_run(store) is True
        await conn.close()


class TestReconcileDreamLifecycle:
    """Exercise the ``_reconcile_dream_lifecycle`` helper on config_routes.

    The helper wraps setup/teardown with the ``should_dream_run``
    predicate; verify it no-ops in-state and flips when out of state.
    Uses a fake ``request.app`` with a patched ``setup_dream_system`` /
    ``teardown_dream_system`` so we don't need real DB/memory plumbing.
    """

    @pytest.fixture
    async def store(self):
        conn, store = await _make_store()
        yield store
        await conn.close()

    @staticmethod
    def _fake_request(store, scheduler=None):
        app = SimpleNamespace(state=SimpleNamespace(
            settings_store=store, dream_scheduler=scheduler,
        ))
        return SimpleNamespace(app=app)

    async def test_boots_when_wanted_and_not_running(self, store, monkeypatch):
        from augmentum.proxy import config_routes

        calls: list[str] = []

        async def _fake_setup(app):
            calls.append("setup")
            app.state.dream_scheduler = object()  # simulate successful boot

        async def _fake_teardown(app):
            calls.append("teardown")

        monkeypatch.setattr(
            "augmentum.dream.lifecycle.setup_dream_system", _fake_setup,
        )
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.teardown_dream_system", _fake_teardown,
        )

        await store.set_user("alice", "ui.dreamEnabled", "true")
        req = self._fake_request(store, scheduler=None)
        await config_routes._reconcile_dream_lifecycle(req)

        assert calls == ["setup"], f"expected single setup, got {calls}"

    async def test_no_op_when_wanted_and_running(self, store, monkeypatch):
        from augmentum.proxy import config_routes

        calls: list[str] = []
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.setup_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("setup")),
        )
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.teardown_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("teardown")),
        )

        await store.set_user("alice", "ui.dreamEnabled", "true")
        req = self._fake_request(store, scheduler=object())
        await config_routes._reconcile_dream_lifecycle(req)

        assert calls == [], f"expected no-op, got {calls}"

    async def test_tears_down_when_last_user_flips_off(self, store, monkeypatch):
        from augmentum.proxy import config_routes

        calls: list[str] = []
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.setup_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("setup")),
        )

        async def _fake_teardown(app):
            calls.append("teardown")
            app.state.dream_scheduler = None

        monkeypatch.setattr(
            "augmentum.dream.lifecycle.teardown_dream_system", _fake_teardown,
        )

        # System is currently running for alice; she disables.
        await store.set_user("alice", "ui.dreamEnabled", "false")
        req = self._fake_request(store, scheduler=object())
        await config_routes._reconcile_dream_lifecycle(req)

        assert calls == ["teardown"], f"expected teardown, got {calls}"

    async def test_stays_running_when_other_user_still_wants_it(
        self, store, monkeypatch,
    ):
        """Alice flips off but Bob still has it on — no teardown."""
        from augmentum.proxy import config_routes

        calls: list[str] = []
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.setup_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("setup")),
        )
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.teardown_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("teardown")),
        )

        await store.set_user("alice", "ui.dreamEnabled", "false")
        await store.set_user("bob", "ui.dreamEnabled", "true")
        req = self._fake_request(store, scheduler=object())
        await config_routes._reconcile_dream_lifecycle(req)

        assert calls == [], (
            "teardown fired despite another user still opted in — this is the "
            "multi-tenant regression"
        )

    async def test_no_op_when_no_store(self, monkeypatch):
        from augmentum.proxy import config_routes

        calls: list[str] = []
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.setup_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("setup")),
        )
        monkeypatch.setattr(
            "augmentum.dream.lifecycle.teardown_dream_system",
            AsyncMock(side_effect=lambda _app: calls.append("teardown")),
        )

        app = SimpleNamespace(state=SimpleNamespace(
            settings_store=None, dream_scheduler=None,
        ))
        req = SimpleNamespace(app=app)
        await config_routes._reconcile_dream_lifecycle(req)
        assert calls == []
