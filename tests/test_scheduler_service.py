"""SchedulerService + generalized standing-task dispatch tests.

Covers the app-level scheduling substrate (augmentum/scheduling/):
multi-user dispatch, the owner-lane skip when the companion dispatcher
is live, headless-context construction with the companion OFF, the
explicit-user step() override, the presence-gate carve-out, and the
standing_gate fallback that keeps chat/voice schedule tools working
without the companion runtime.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


async def _fresh_backend(*user_ids: str):
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    for i, uid in enumerate(user_ids):
        await backend.conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (uid, f"tester{i}", "x"),
        )
    await backend.conn.commit()
    return backend


def _runtime_for(backend, owner_user_id: str):
    app_state = MagicMock()
    store = MagicMock()
    store.get_user_or_global = AsyncMock(return_value="")
    app_state.settings_store = store
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    runtime.owner_user_id = owner_user_id
    runtime._app_state = app_state
    runtime.memory = MagicMock()
    runtime.memory.safe_journal = AsyncMock()
    return runtime


async def _seed_due_task(backend, *, user_id: str, title: str, kind: str,
                         params: str = "{}") -> int:
    cur = await backend.conn.execute(
        "INSERT INTO companion_standing_tasks "
        "(user_id, companion_id, title, kind, params, interval_seconds, "
        " next_run_at, enabled, consecutive_error_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, 3600, "
        "        datetime('now', '-1 minute'), 1, 0, datetime('now'))",
        (user_id, "becca", title, kind, params),
    )
    await backend.conn.commit()
    return int(cur.lastrowid or 0)


async def _next_run_at(backend, task_id: int) -> str | None:
    cur = await backend.conn.execute(
        "SELECT next_run_at FROM companion_standing_tasks WHERE id = ?",
        (task_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    return row[0] if row else None


# ── step(): explicit user + presence carve-out ──────────────────────────


@pytest.mark.asyncio
async def test_step_explicit_user_id_overrides_owner(monkeypatch):
    """step(user_id=...) dispatches THAT user's due task even though the
    runtime is bound to a different owner — the multi-tenant unlock."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged", raising=False)

    backend = await _fresh_backend("owner", "guest")
    runtime = _runtime_for(backend, "owner")

    ran_for: list[str] = []

    async def probe(_runtime, *, user_id, params):
        ran_for.append(user_id)
        return {"summary": "ok", "noteworthy": False, "refs": []}

    standing_tasks._TASK_KINDS["_svc_probe"] = probe
    try:
        await _seed_due_task(
            backend, user_id="guest", title="guest task", kind="_svc_probe")
        ran = await standing_tasks.step(runtime, user_id="guest")
        assert ran is not None
        assert ran_for == ["guest"]
    finally:
        del standing_tasks._TASK_KINDS["_svc_probe"]
        await backend.close()


@pytest.mark.asyncio
async def test_step_presence_gate_carve_out(monkeypatch):
    """Silent presence mode gates the companion tick path (default) but
    NOT the generalized dispatcher (respect_presence_gate=False) — an
    explicit user schedule is the user's ask, not companion initiative."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "silent", raising=False)

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")

    async def probe(_runtime, *, user_id, params):
        return {"summary": "ok", "noteworthy": False, "refs": []}

    standing_tasks._TASK_KINDS["_svc_probe2"] = probe
    try:
        task_id = await _seed_due_task(
            backend, user_id="u1", title="reminder", kind="_svc_probe2")
        # Companion tick path: gated.
        assert await standing_tasks.step(runtime) is None
        # Generalized dispatcher: fires.
        ran = await standing_tasks.step(
            runtime, user_id="u1", respect_presence_gate=False)
        assert ran == task_id
    finally:
        del standing_tasks._TASK_KINDS["_svc_probe2"]
        await backend.close()


# ── SchedulerService scan ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_dispatches_all_users_but_skips_companion_owner(monkeypatch):
    """With the companion dispatcher live, the service covers every
    OTHER user (no double-fire on the owner's lane)."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.scheduling import SchedulerService

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    backend = await _fresh_backend("owner", "u2", "u3")
    companion_rt = _runtime_for(backend, "owner")

    ran_for: list[str] = []

    async def probe(_runtime, *, user_id, params):
        ran_for.append(user_id)
        return {"summary": "ok", "noteworthy": False, "refs": []}

    standing_tasks._TASK_KINDS["_svc_probe3"] = probe
    try:
        owner_task = await _seed_due_task(
            backend, user_id="owner", title="t", kind="_svc_probe3")
        await _seed_due_task(backend, user_id="u2", title="t", kind="_svc_probe3")
        await _seed_due_task(backend, user_id="u3", title="t", kind="_svc_probe3")

        service = SchedulerService(
            backend=backend, app_state=companion_rt._app_state,
            companion_runtime=companion_rt,
        )
        # Use the companion runtime as ctx (the ON path) directly.
        await service._scan_once()

        assert sorted(ran_for) == ["u2", "u3"]
        # Owner's row untouched — still due for the tick verb's lane.
        nra = await _next_run_at(backend, owner_task)
        cur = await backend.conn.execute("SELECT datetime('now')")
        now_row = await cur.fetchone()
        await cur.close()
        assert nra is not None and nra <= now_row[0]
    finally:
        del standing_tasks._TASK_KINDS["_svc_probe3"]
        await backend.close()


@pytest.mark.asyncio
async def test_scan_headless_when_companion_off(monkeypatch):
    """No companion runtime at all: the service builds a headless ctx
    (unstarted CompanionRuntime) and every user's due tasks fire."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.scheduling import SchedulerService

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    backend = await _fresh_backend("u1", "u2")

    ran_for: list[str] = []

    async def probe(_runtime, *, user_id, params):
        ran_for.append(user_id)
        return {"summary": "ok", "noteworthy": False, "refs": []}

    standing_tasks._TASK_KINDS["_svc_probe4"] = probe
    try:
        await _seed_due_task(backend, user_id="u1", title="t", kind="_svc_probe4")
        await _seed_due_task(backend, user_id="u2", title="t", kind="_svc_probe4")

        app_state = MagicMock()
        store = MagicMock()
        store.get_user_or_global = AsyncMock(return_value="")
        app_state.settings_store = store
        app_state.companion_runtime = None

        service = SchedulerService(
            backend=backend, app_state=app_state, companion_runtime=None,
        )
        await service._scan_once()
        assert sorted(ran_for) == ["u1", "u2"]
        # Headless ctx is a real (unstarted) CompanionRuntime.
        from augmentum.companion_runtime.runtime import CompanionRuntime
        assert isinstance(service.ctx, CompanionRuntime)
    finally:
        del standing_tasks._TASK_KINDS["_svc_probe4"]
        await backend.close()


@pytest.mark.asyncio
async def test_scan_respects_kill_switch(monkeypatch):
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.scheduling import SchedulerService

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", False, raising=False)

    backend = await _fresh_backend("u1")
    ran_for: list[str] = []

    async def probe(_runtime, *, user_id, params):
        ran_for.append(user_id)
        return {"summary": "ok", "noteworthy": False, "refs": []}

    standing_tasks._TASK_KINDS["_svc_probe5"] = probe
    try:
        await _seed_due_task(backend, user_id="u1", title="t", kind="_svc_probe5")
        service = SchedulerService(
            backend=backend, app_state=MagicMock(), companion_runtime=None,
        )
        await service._scan_once()
        assert ran_for == []
    finally:
        del standing_tasks._TASK_KINDS["_svc_probe5"]
        await backend.close()


# ── standing_gate fallback ──────────────────────────────────────────────


def test_standing_gate_falls_back_to_scheduler_ctx(monkeypatch):
    """Schedule tools keep working with the companion OFF: the gate
    returns the SchedulerService's headless ctx."""
    from augmentum.config import settings
    from augmentum.tools._standing_common import standing_gate

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    headless = MagicMock(name="headless_ctx")
    service = MagicMock()
    service.ctx = headless

    app_state = MagicMock()
    app_state.companion_runtime = None
    app_state.scheduler_service = service

    ok, err, runtime = standing_gate(app_state)
    assert ok and err is None
    assert runtime is headless


def test_standing_gate_refuses_when_no_dispatcher(monkeypatch):
    from augmentum.config import settings
    from augmentum.tools._standing_common import standing_gate

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    app_state = MagicMock()
    app_state.companion_runtime = None
    app_state.scheduler_service = None

    ok, err, runtime = standing_gate(app_state)
    assert not ok and runtime is None
    assert err.metadata.get("reason") == "scheduling_disabled"


def test_standing_gate_prefers_companion_runtime(monkeypatch):
    from augmentum.config import settings
    from augmentum.tools._standing_common import standing_gate

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    rt = MagicMock(name="companion_rt")
    app_state = MagicMock()
    app_state.companion_runtime = rt

    ok, _err, runtime = standing_gate(app_state)
    assert ok and runtime is rt


# ── read-before-create duplicate review ─────────────────────────────────


@pytest.mark.asyncio
async def test_find_similar_tasks_same_target_same_kind():
    """Two watches on the same URL are the same watch regardless of
    title — the strongest duplicate signal."""
    from augmentum.companion_runtime import standing_tasks

    backend = await _fresh_backend("u1")
    try:
        await backend.conn.execute(
            "INSERT INTO companion_standing_tasks "
            "(user_id, companion_id, title, kind, params, interval_seconds,"
            " enabled, created_at) VALUES (?, ?, ?, ?, ?, 3600, 1, "
            " datetime('now'))",
            ("u1", "becca", "BTC price",
             "url_watch", '{"url": "https://example.com/btc"}'),
        )
        await backend.conn.commit()
        similar = await standing_tasks.find_similar_tasks(
            backend.conn, user_id="u1", companion_id="becca",
            kind="url_watch", title="watch bitcoin",
            params={"url": "https://EXAMPLE.com/btc"},
        )
        assert len(similar) == 1 and similar[0]["title"] == "BTC price"
        # Different URL, unrelated title → no match.
        clear = await standing_tasks.find_similar_tasks(
            backend.conn, user_id="u1", companion_id="becca",
            kind="url_watch", title="watch gold",
            params={"url": "https://example.com/gold"},
        )
        assert clear == []
        # Same title on a DIFFERENT kind still flags (title signal).
        cross = await standing_tasks.find_similar_tasks(
            backend.conn, user_id="u1", companion_id="becca",
            kind="feed_digest", title="BTC price",
            params={"topic": "bitcoin"},
        )
        assert len(cross) == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_watch_for_duplicate_review_and_confirm_bypass(monkeypatch):
    """watch_for refuses a same-URL duplicate with a review naming the
    match, and confirm_replace=true bypasses it."""
    from augmentum.config import settings
    from augmentum.tools.watch_for import WatchForTool

    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")
    app_state = MagicMock(companion_runtime=runtime)
    try:
        await backend.conn.execute(
            "INSERT INTO companion_standing_tasks "
            "(user_id, companion_id, title, kind, params, interval_seconds,"
            " enabled, created_at) VALUES (?, ?, ?, ?, ?, 3600, 1, "
            " datetime('now'))",
            ("u1", "becca", "BTC price",
             "url_watch", '{"url": "https://example.com/btc"}'),
        )
        await backend.conn.commit()

        tool = WatchForTool(app_state)
        r = await tool.execute(
            title="bitcoin watcher", kind="url",
            target="https://example.com/btc", _user_id="u1",
        )
        assert r.success is False
        assert (r.metadata or {}).get("reason") == "duplicate_review"
        assert "BTC price" in r.error

        # Explicit confirmation proceeds past the review (the probe then
        # runs against the mocked runtime; we only assert it got past
        # the dup gate, i.e. the failure reason is no longer the review).
        r2 = await tool.execute(
            title="bitcoin watcher", kind="url",
            target="https://example.com/btc", confirm_replace=True,
            _user_id="u1",
        )
        assert (r2.metadata or {}).get("reason") != "duplicate_review"
    finally:
        await backend.close()
