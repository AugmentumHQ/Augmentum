"""Standing-tasks unit tests — scheduler math, CRUD, briefing kind, chat tool.

Companion to ``test_companion_curator.py``. Where the curator surfaces
"I noticed this in the world," standing tasks are "I went and checked
because you asked me to."
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


def _disable_user_tz_lookup(app_state):
    """Help: many tests construct a MagicMock app_state — the auto-mock
    settings_store satisfies _resolve_user_timezone's attribute checks but
    returns a MagicMock from get_user_or_global, which the resolver wraps
    in try/except and treats as empty. Tests that need an explicit empty
    TZ can use this to be unambiguous."""
    store = MagicMock()
    store.get_user_or_global = AsyncMock(return_value="")
    app_state.settings_store = store
    return app_state


async def _fresh_backend(user_id: str = "usr_s"):
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (user_id, "tester", "x"),
    )
    await backend.conn.commit()
    return backend


# ── _compute_next_run_at ─────────────────────────────────────────────────


def test_next_run_at_anchored_lands_on_minute_boundary():
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    # An anchored result is always on a minute boundary (seconds == 0)
    # because we replace(second=0, microsecond=0). Interval mode has
    # seconds drifting with the wall clock.
    assert parsed.second == 0


def test_next_run_at_just_fired_guard():
    """Firing at the current local HH:MM must schedule the next run at
    least one day later — not re-fire on the same minute boundary."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    now_local = datetime.now().astimezone()
    hh = now_local.strftime("%H:%M")
    s = _compute_next_run_at(
        params={"local_time": hh}, interval_seconds=86400,
    )
    next_dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    # Strictly in the future.
    assert next_dt > datetime.now(UTC)


def test_next_run_at_negative_jitter_never_schedules_in_past():
    """Regression (4x-fire incident): deterministic jitter is a SIGNED
    ±window nudge applied AFTER the today/tomorrow decision, which only
    proves the UN-jittered candidate is in the future. A negative offset
    can pull next_run_at back across "now"; the row then stays "due now"
    and the dispatcher re-runs the (expensive gather+synthesis) task every
    tick until the wall clock crawls past the raw anchor. next_run_at must
    be strictly in the future for EVERY jitter seed.

    Anchored ~1 minute ahead so the un-jittered candidate lands TODAY —
    the exact condition under which a negative offset underflows. Scans
    many seeds so the negative-offset branch is actually exercised (the
    final assert guards against a vacuous pass)."""
    from augmentum.companion_runtime.standing_tasks import (
        _JITTER_RECURRING_S,
        _compute_next_run_at,
        _deterministic_jitter_seconds,
    )
    now_local = datetime.now().astimezone()
    anchor = (now_local + timedelta(minutes=1)).strftime("%H:%M")

    exercised_negative = False
    for i in range(200):
        seed = f"jit-{i}"
        offset = _deterministic_jitter_seconds(seed, _JITTER_RECURRING_S)
        # Offsets more negative than the ~60s until the anchor are the ones
        # that would underflow past "now" without the post-jitter guard.
        if offset < -90:
            exercised_negative = True
        s = _compute_next_run_at(
            params={"local_time": anchor}, interval_seconds=86400,
            jitter_seed=seed,
        )
        next_dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        assert next_dt > datetime.now(UTC), (
            f"seed={seed} offset={offset}s produced a past next_run_at {s}"
        )
    assert exercised_negative, "scan never hit a strong negative offset"


def test_next_run_at_negative_jitter_preserves_weekday():
    """The post-jitter future-guard bumps whole days; when weekdays are
    restricted it must re-hop so the bumped candidate still lands on a
    valid day (Saturday here), not just any future day."""
    from augmentum.companion_runtime.standing_tasks import (
        _JITTER_RECURRING_S,
        _compute_next_run_at,
        _deterministic_jitter_seconds,
    )
    now_local = datetime.now().astimezone()
    anchor = (now_local + timedelta(minutes=1)).strftime("%H:%M")
    # Pick a seed with a strong negative offset so the guard fires.
    seed = next(
        s for s in (f"wd-{i}" for i in range(500))
        if _deterministic_jitter_seconds(s, _JITTER_RECURRING_S) < -120
    )
    out = _compute_next_run_at(
        params={"local_time": anchor, "weekdays": [6]},
        interval_seconds=86400, jitter_seed=seed,
    )
    parsed = datetime.strptime(out, "%Y-%m-%d %H:%M:%S")
    local = parsed.replace(tzinfo=UTC).astimezone()
    assert local.isoweekday() == 6
    assert parsed.replace(tzinfo=UTC) > datetime.now(UTC)


def test_next_run_at_malformed_local_time_falls_back_to_interval():
    """Garbage in local_time → interval-mode fallback rather than crash."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s_anchor = _compute_next_run_at(
        params={"local_time": "25:99"}, interval_seconds=120,
    )
    s_interval = _compute_next_run_at(params=None, interval_seconds=120)
    a = datetime.strptime(s_anchor, "%Y-%m-%d %H:%M:%S")
    b = datetime.strptime(s_interval, "%Y-%m-%d %H:%M:%S")
    # Both ≈ now + 120s; small drift between invocations.
    assert abs((a - b).total_seconds()) < 5


def test_next_run_at_weekday_restriction_skips_to_valid_day():
    """Anchor at 09:00 with weekdays = Saturday only → next_run_at,
    interpreted in the server's local zone, must be a Saturday."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(
        params={"local_time": "09:00", "weekdays": [6]},
        interval_seconds=86400,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    local = parsed.replace(tzinfo=UTC).astimezone()
    assert local.isoweekday() == 6


def test_next_run_at_interval_only_uses_interval_seconds():
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(params={}, interval_seconds=600)
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    expected = datetime.now(UTC) + timedelta(seconds=600)
    assert abs((parsed - expected).total_seconds()) < 5


def test_next_run_at_invalid_weekday_values_ignored():
    """weekdays = ['x', 99, -1] → treated as no restriction."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s_unrestricted = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
    )
    s_bad_restriction = _compute_next_run_at(
        params={"local_time": "09:00", "weekdays": ["x", 99, -1]},
        interval_seconds=86400,
    )
    # Same day chosen — bad restrictions equivalent to no restriction.
    a = datetime.strptime(s_unrestricted, "%Y-%m-%d %H:%M:%S")
    b = datetime.strptime(s_bad_restriction, "%Y-%m-%d %H:%M:%S")
    assert a == b


# ── add_task ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_anchored_uses_local_time_for_initial_next():
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    task = await standing_tasks.add_task(
        backend.conn,
        user_id="usr_s", companion_id="becca",
        title="Morning briefing", kind="briefing",
        params={"topics": ["news"], "local_time": "09:00"},
    )
    assert task is not None
    assert task.next_run_at is not None
    parsed = datetime.strptime(task.next_run_at, "%Y-%m-%d %H:%M:%S")
    # Strictly in the future.
    assert parsed.replace(tzinfo=UTC) > datetime.now(UTC)
    # Anchored creation applies the deterministic creation-seed jitter
    # (user:companion:title:kind) — mirror it for exact equality.
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    expected = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
        jitter_seed="usr_s:becca:Morning briefing:briefing",
    )
    assert task.next_run_at == expected


@pytest.mark.asyncio
async def test_add_task_interval_only_runs_immediately():
    """No local_time anchor → next_run_at is "datetime('now')" so the
    first tick picks it up."""
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    task = await standing_tasks.add_task(
        backend.conn,
        user_id="usr_s", companion_id="becca",
        title="hourly", kind="recurring_search",
        params={"query": "anything"},
        interval_seconds=3600,
    )
    assert task is not None
    assert task.next_run_at is not None
    parsed = datetime.strptime(task.next_run_at, "%Y-%m-%d %H:%M:%S")
    # Within a few seconds of "now" — runs on the next tick.
    assert abs(
        (parsed.replace(tzinfo=UTC) - datetime.now(UTC)).total_seconds()
    ) < 5


@pytest.mark.asyncio
async def test_add_task_unknown_kind_raises():
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    with pytest.raises(ValueError, match="unknown task kind"):
        await standing_tasks.add_task(
            backend.conn,
            user_id="usr_s", companion_id="becca",
            title="bogus", kind="not_a_kind", params={},
        )


@pytest.mark.asyncio
async def test_add_task_missing_required_raises():
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    with pytest.raises(ValueError):
        await standing_tasks.add_task(
            backend.conn,
            user_id="", companion_id="becca",
            title="t", kind="briefing", params={},
        )


@pytest.mark.asyncio
async def test_add_task_interval_floor_enforced():
    """interval_seconds < 300 (5min) is raised to the floor to stop
    a misconfigured task from hammering the engine."""
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    task = await standing_tasks.add_task(
        backend.conn,
        user_id="usr_s", companion_id="becca",
        title="t", kind="recurring_search",
        params={"query": "x"}, interval_seconds=10,
    )
    assert task is not None
    assert task.interval_seconds == 300


# ── CRUD ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_get_set_enabled_remove_round_trip():
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    task = await standing_tasks.add_task(
        backend.conn,
        user_id="usr_s", companion_id="becca",
        title="t", kind="recurring_search", params={"query": "x"},
    )
    listed = await standing_tasks.list_tasks(
        backend.conn, user_id="usr_s", companion_id="becca",
    )
    assert len(listed) == 1

    got = await standing_tasks.get_task(
        backend.conn, task_id=task.id,
        user_id="usr_s", companion_id="becca",
    )
    assert got is not None and got.id == task.id

    ok = await standing_tasks.set_enabled(
        backend.conn, task_id=task.id,
        user_id="usr_s", companion_id="becca", enabled=False,
    )
    assert ok
    refetched = await standing_tasks.get_task(
        backend.conn, task_id=task.id,
        user_id="usr_s", companion_id="becca",
    )
    assert refetched is not None and refetched.enabled is False

    removed = await standing_tasks.remove_task(
        backend.conn, task_id=task.id,
        user_id="usr_s", companion_id="becca",
    )
    assert removed
    listed_after = await standing_tasks.list_tasks(
        backend.conn, user_id="usr_s", companion_id="becca",
    )
    assert listed_after == []


@pytest.mark.asyncio
async def test_remove_task_wrong_user_no_op():
    """Tenant isolation: another user's task_id is invisible."""
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend("usr_owner")
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("usr_attacker", "att", "x"),
    )
    await backend.conn.commit()
    task = await standing_tasks.add_task(
        backend.conn,
        user_id="usr_owner", companion_id="becca",
        title="private", kind="recurring_search", params={"query": "x"},
    )
    assert task is not None
    ok = await standing_tasks.remove_task(
        backend.conn, task_id=task.id,
        user_id="usr_attacker", companion_id="becca",
    )
    assert ok is False


# ── Briefing kind ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_briefing_no_topics_raises():
    from augmentum.companion_runtime import standing_tasks
    runner = standing_tasks._TASK_KINDS["briefing"]
    with pytest.raises(ValueError, match="topics"):
        await runner(MagicMock(), user_id="usr_s", params={"topics": []})


@pytest.mark.asyncio
async def test_briefing_no_searxng_returns_deferred():
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    runner = standing_tasks._TASK_KINDS["briefing"]
    runtime = MagicMock()
    runtime._app_state = MagicMock(http_client=MagicMock())
    saved = getattr(settings, "searxng_base_url", "")
    settings.searxng_base_url = ""
    try:
        result = await runner(
            runtime, user_id="usr_s", params={"topics": ["news"]},
        )
        assert result["noteworthy"] is False
        assert "searxng" in result["summary"].lower()
    finally:
        settings.searxng_base_url = saved


@pytest.mark.asyncio
async def test_briefing_no_http_client_returns_deferred():
    from augmentum.companion_runtime import standing_tasks
    runner = standing_tasks._TASK_KINDS["briefing"]
    runtime = MagicMock()
    runtime._app_state = MagicMock(http_client=None)
    result = await runner(
        runtime, user_id="usr_s", params={"topics": ["news"]},
    )
    assert result["noteworthy"] is False
    assert "http_client" in result["summary"]


@pytest.mark.asyncio
async def test_briefing_happy_path_fanout_over_topics(monkeypatch):
    """Mock SearXNG responses; confirm we get one section per topic with
    refs, plus the title header and the location anchor."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    runner = standing_tasks._TASK_KINDS["briefing"]

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Headline A",
                        "content": "Snippet about A",
                    },
                    {
                        "url": "https://example.com/b",
                        "title": "Headline B",
                        "content": "Snippet about B",
                    },
                ],
            }

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=FakeResp())

    runtime = MagicMock()
    runtime._app_state = MagicMock(http_client=fake_client)

    monkeypatch.setattr(settings, "searxng_base_url", "http://searx:8080")

    result = await runner(
        runtime, user_id="usr_s",
        params={
            "topics": ["news", "weather"],
            "location": "Seattle WA",
            "title": "Morning briefing",
        },
    )
    assert result["noteworthy"] is True
    # One URL per topic (max_per_topic defaults to 1).
    assert len(result["refs"]) == 2
    content = result["details"]["content"]
    assert "Morning briefing" in content
    assert "Seattle WA" in content
    assert "— news" in content
    assert "— weather" in content
    assert "Headline A" in content
    # SearXNG was called once per topic, each with location appended.
    assert fake_client.get.await_count == 2


@pytest.mark.asyncio
async def test_briefing_topic_query_failure_isolated(monkeypatch):
    """If one topic's SearXNG query throws, other topics still surface
    and the briefing remains noteworthy."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    runner = standing_tasks._TASK_KINDS["briefing"]

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [
                {"url": "https://e.com/x", "title": "X", "content": "x"},
            ]}

    calls = {"n": 0}

    async def flaky_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("searxng exploded")
        return FakeResp()

    fake_client = MagicMock()
    fake_client.get = flaky_get

    runtime = MagicMock()
    runtime._app_state = MagicMock(http_client=fake_client)
    monkeypatch.setattr(settings, "searxng_base_url", "http://s:8080")

    result = await runner(
        runtime, user_id="usr_s",
        params={"topics": ["broken", "ok"]},
    )
    # 1 of 2 topics surfaced — briefing is still noteworthy.
    assert result["noteworthy"] is True
    assert len(result["refs"]) == 1


# ── ScheduleBriefingTool — coercion helpers ──────────────────────────────


def test_coerce_topic_list_accepts_list():
    from augmentum.tools.schedule_briefing import _coerce_topic_list
    assert _coerce_topic_list(["a", "b", " c "]) == ["a", "b", "c"]


def test_coerce_topic_list_accepts_string_and_semicolons():
    from augmentum.tools.schedule_briefing import _coerce_topic_list
    assert _coerce_topic_list("a, b ; c") == ["a", "b", "c"]


def test_coerce_topic_list_drops_empty_chunks():
    from augmentum.tools.schedule_briefing import _coerce_topic_list
    assert _coerce_topic_list(["", " ", "x"]) == ["x"]
    assert _coerce_topic_list(",,  ,x") == ["x"]
    assert _coerce_topic_list(None) == []


def test_normalize_weekdays_mixed_input():
    from augmentum.tools.schedule_briefing import _normalize_weekdays
    assert _normalize_weekdays(["mon", "tue", 5]) == [1, 2, 5]
    assert _normalize_weekdays([7, "sunday"]) == [7]
    # Single non-list value gets wrapped.
    assert _normalize_weekdays("monday") == [1]


def test_normalize_weekdays_bad_input_dropped():
    from augmentum.tools.schedule_briefing import _normalize_weekdays
    assert _normalize_weekdays(["xxx", 99, 0, -1, "8"]) == []


# ── ScheduleBriefingTool — execute() gates ───────────────────────────────


@pytest.mark.asyncio
async def test_schedule_briefing_tool_kill_switch_standing_tasks_disabled():
    """The standing-tasks kill switch refuses tool creation. (The old
    companion_runtime_enabled gate is GONE by design — scheduling is an
    app-level substrate; with the companion off the tool falls back to
    the SchedulerService context instead of refusing.)"""
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool
    saved = settings.companion_standing_tasks_enabled
    settings.companion_standing_tasks_enabled = False
    try:
        tool = ScheduleBriefingTool(MagicMock())
        r = await tool.execute(
            title="x", topics=["news"], local_time="09:00", _user_id="u1",
        )
        assert r.success is False
        assert "disabled" in r.error.lower()
    finally:
        settings.companion_standing_tasks_enabled = saved


@pytest.mark.asyncio
async def test_schedule_briefing_tool_no_dispatcher_at_all():
    """Neither the companion runtime nor the SchedulerService exists —
    the one remaining refusal shape for missing infrastructure."""
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool
    settings.companion_standing_tasks_enabled = True
    app_state = MagicMock(companion_runtime=None, scheduler_service=None)
    tool = ScheduleBriefingTool(app_state)
    r = await tool.execute(
        title="x", topics=["news"], local_time="09:00", _user_id="u1",
    )
    assert r.success is False
    assert "scheduling" in r.error.lower()


@pytest.mark.asyncio
async def test_schedule_briefing_tool_validation_errors():
    """Genuine schedule-parse failures surface a validation_error rather
    than a runtime crash. (Missing title/topics are NOT errors — the tool
    defaults them and flags the assumption, by design; the schedule is
    the only thing it can't invent.)"""
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True
    tool = ScheduleBriefingTool(MagicMock(companion_runtime=MagicMock()))

    # Unparseable time, no cron → refused before any DB touch.
    r = await tool.execute(
        title="t", topics=["x"], local_time="not-a-time", _user_id="u1",
    )
    assert r.success is False and r.validation_error is True

    # No schedule at all.
    r = await tool.execute(title="t", topics=["x"], _user_id="u1")
    assert r.success is False and r.validation_error is True
    assert "schedule" in r.error.lower()

    # Malformed cron.
    r = await tool.execute(
        title="t", topics=["x"], cron="61 * * * *", _user_id="u1",
    )
    assert r.success is False and r.validation_error is True
    assert "cron" in r.error.lower()


@pytest.mark.asyncio
async def test_schedule_briefing_tool_missing_user_id():
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool
    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True
    tool = ScheduleBriefingTool(MagicMock(companion_runtime=MagicMock()))
    r = await tool.execute(
        title="t", topics=["x"], local_time="09:00",  # no _user_id
    )
    assert r.success is False
    assert "user" in r.error.lower()


@pytest.mark.asyncio
async def test_schedule_briefing_tool_happy_path_creates_task():
    """End-to-end: tool execute() calls add_task and returns a summary
    with the first occurrence in it."""
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    app_state = MagicMock(companion_runtime=runtime)

    tool = ScheduleBriefingTool(app_state)
    r = await tool.execute(
        title="Morning briefing",
        topics=["news", "weather", "traffic"],
        local_time="09:00",
        weekdays=["mon", "tue", "wed", "thu", "fri"],
        location="Seattle WA",
        _user_id="u1",
    )
    assert r.success is True
    assert "Morning briefing" in r.output
    assert "09:00" in r.output
    assert "Seattle WA" in r.output
    md = r.metadata or {}
    assert md.get("ok") is True
    assert md.get("task_id")
    assert md.get("topics") == ["news", "weather", "traffic"]
    assert md.get("weekdays") == [1, 2, 3, 4, 5]
    assert md.get("location") == "Seattle WA"
    assert md.get("next_run_at")


@pytest.mark.asyncio
async def test_schedule_briefing_tool_topics_as_string():
    """LLMs sometimes pass topics as a comma-separated string. The tool
    must coerce, not reject."""
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    app_state = MagicMock(companion_runtime=runtime)

    tool = ScheduleBriefingTool(app_state)
    r = await tool.execute(
        title="Briefing",
        topics="news, weather, traffic",
        local_time="09:00",
        _user_id="u1",
    )
    assert r.success is True
    assert (r.metadata or {}).get("topics") == ["news", "weather", "traffic"]


# ── Per-user timezone substrate ──────────────────────────────────────────


def test_compute_next_run_at_user_timezone_distinguishes_zones():
    """Tokyo 09:00 ≠ Los_Angeles 09:00 — both anchor to their respective
    wall-clock targets, which yield different UTC instants."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s_tokyo = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
        user_timezone="Asia/Tokyo",
    )
    s_la = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
        user_timezone="America/Los_Angeles",
    )
    assert s_tokyo != s_la


def test_compute_next_run_at_bad_timezone_falls_back_silent():
    """Bogus TZ name → fall back to server-local anchoring, don't crash."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s_bad = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
        user_timezone="Mars/Olympus",
    )
    s_server = _compute_next_run_at(
        params={"local_time": "09:00"}, interval_seconds=86400,
    )
    assert s_bad == s_server


@pytest.mark.asyncio
async def test_resolve_user_timezone_reads_settings_store():
    from augmentum.companion_runtime.standing_tasks import (
        _resolve_user_timezone,
    )
    store = MagicMock()
    store.get_user_or_global = AsyncMock(return_value="Europe/Berlin")
    app_state = MagicMock(settings_store=store)
    tz = await _resolve_user_timezone(app_state, "u1")
    assert tz == "Europe/Berlin"
    store.get_user_or_global.assert_awaited_once_with("u1", "timezone")


@pytest.mark.asyncio
async def test_resolve_user_timezone_handles_missing_store():
    from augmentum.companion_runtime.standing_tasks import (
        _resolve_user_timezone,
    )
    # No settings_store attribute → empty string fallback.
    app_state = MagicMock(spec=[])
    tz = await _resolve_user_timezone(app_state, "u1")
    assert tz == ""


@pytest.mark.asyncio
async def test_schedule_briefing_tool_threads_user_tz_to_persisted_task():
    """End-to-end: the tool resolves user TZ and the persisted task's
    next_run_at reflects the user-zone anchor, not server-local."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    from augmentum.config import settings
    from augmentum.tools.schedule_briefing import ScheduleBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    store = MagicMock()
    store.get_user_or_global = AsyncMock(return_value="Asia/Tokyo")
    app_state = MagicMock(companion_runtime=runtime, settings_store=store)

    tool = ScheduleBriefingTool(app_state)
    try:
        r = await tool.execute(
            title="Tokyo briefing", topics=["news"], local_time="09:00",
            _user_id="u1",
        )
        assert r.success
        task_id = (r.metadata or {}).get("task_id")
        task = await standing_tasks.get_task(
            backend.conn, task_id=task_id,
            user_id="u1", companion_id="becca",
        )
        # add_task anchors the first fire with the creation jitter seed
        # (user:companion:title:kind) so the initial occurrence lands in
        # the same slot as every future fire — mirror it exactly.
        expected = _compute_next_run_at(
            params={"local_time": "09:00"}, interval_seconds=86400,
            user_timezone="Asia/Tokyo",
            jitter_seed="u1:becca:Tokyo briefing:briefing",
        )
        assert task.next_run_at == expected
    finally:
        await backend.close()


# ── last_error round-trip ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_last_error_round_trips_through_dataclass_and_dict():
    """last_error must survive: DB → _row_to_task → StandingTask →
    as_dict(). Without this, the UI can never surface failures."""
    from augmentum.companion_runtime import standing_tasks
    backend = await _fresh_backend()
    task = await standing_tasks.add_task(
        backend.conn, user_id="usr_s", companion_id="becca",
        title="t", kind="briefing",
        params={"topics": ["x"], "local_time": "09:00"},
    )
    # Newly created → no error.
    assert task.last_error is None
    assert task.consecutive_error_count == 0
    assert "last_error" in task.as_dict()

    # Write a failure and re-fetch.
    await backend.conn.execute(
        "UPDATE companion_standing_tasks SET last_error = ?, "
        "consecutive_error_count = ? WHERE id = ?",
        ("searxng down: connection refused", 2, task.id),
    )
    await backend.conn.commit()
    refetched = await standing_tasks.get_task(
        backend.conn, task_id=task.id,
        user_id="usr_s", companion_id="becca",
    )
    assert refetched.last_error == "searxng down: connection refused"
    assert refetched.consecutive_error_count == 2
    d = refetched.as_dict()
    assert d["last_error"] == refetched.last_error
    assert d["consecutive_error_count"] == 2


# ── ListBriefingsTool ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_briefings_empty():
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import ListBriefingsTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = ListBriefingsTool(MagicMock(companion_runtime=runtime))
    r = await tool.execute(_user_id="u1")
    assert r.success
    assert (r.metadata or {}).get("count") == 0
    assert "No briefings" in r.output


@pytest.mark.asyncio
async def test_list_briefings_filters_to_kind_briefing():
    """Other kinds (recurring_search, etc.) must NOT appear in the
    briefing list."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import ListBriefingsTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    await standing_tasks.add_task(
        backend.conn, user_id="u1", companion_id="becca",
        title="Morning", kind="briefing",
        params={"topics": ["news"], "local_time": "09:00"},
    )
    await standing_tasks.add_task(
        backend.conn, user_id="u1", companion_id="becca",
        title="Some other watch", kind="recurring_search",
        params={"query": "x"},
    )
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = ListBriefingsTool(MagicMock(companion_runtime=runtime))
    r = await tool.execute(_user_id="u1")
    md = r.metadata or {}
    assert r.success
    assert md["count"] == 1
    assert md["briefings"][0]["title"] == "Morning"


@pytest.mark.asyncio
async def test_list_briefings_renders_human_readable_lines():
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import ListBriefingsTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    await standing_tasks.add_task(
        backend.conn, user_id="u1", companion_id="becca",
        title="Weekday briefing", kind="briefing",
        params={
            "topics": ["news"],
            "local_time": "09:00",
            "weekdays": [1, 2, 3, 4, 5],
            "location": "Seattle WA",
        },
    )
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = ListBriefingsTool(MagicMock(companion_runtime=runtime))
    r = await tool.execute(_user_id="u1")
    assert r.success
    assert "Weekday briefing" in r.output
    assert "09:00" in r.output
    assert "Mon" in r.output
    assert "Fri" in r.output
    assert "Seattle WA" in r.output


# ── CancelBriefingTool ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_briefing_requires_selector():
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import CancelBriefingTool
    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True
    tool = CancelBriefingTool(MagicMock(companion_runtime=MagicMock()))
    r = await tool.execute(_user_id="u1")
    assert not r.success and r.validation_error


@pytest.mark.asyncio
async def test_cancel_briefing_by_title_substring():
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import CancelBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    await standing_tasks.add_task(
        backend.conn, user_id="u1", companion_id="becca",
        title="Morning briefing", kind="briefing",
        params={"topics": ["news"], "local_time": "09:00"},
    )
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = CancelBriefingTool(MagicMock(companion_runtime=runtime))

    # Case-insensitive substring works.
    r = await tool.execute(_user_id="u1", title="MORNING")
    assert r.success
    assert "Morning briefing" in r.output

    # Already gone.
    r2 = await tool.execute(_user_id="u1", title="morning")
    assert not r2.success
    assert "no briefing" in r2.error.lower()


@pytest.mark.asyncio
async def test_cancel_briefing_ambiguous_title_refuses():
    """Two briefings, one substring match → refuse with ambiguity
    metadata so the LLM can disambiguate by id."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import CancelBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    for title in ("Morning news", "Morning weather"):
        await standing_tasks.add_task(
            backend.conn, user_id="u1", companion_id="becca",
            title=title, kind="briefing",
            params={"topics": ["x"], "local_time": "09:00"},
        )
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = CancelBriefingTool(MagicMock(companion_runtime=runtime))
    r = await tool.execute(_user_id="u1", title="morning")
    assert not r.success
    assert (r.metadata or {}).get("reason") == "ambiguous"
    matches = (r.metadata or {}).get("matches") or []
    assert len(matches) == 2


@pytest.mark.asyncio
async def test_cancel_briefing_by_task_id_bypasses_ambiguity():
    """When task_id is supplied, title fuzzy match is skipped."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import CancelBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    t1 = await standing_tasks.add_task(
        backend.conn, user_id="u1", companion_id="becca",
        title="Morning news", kind="briefing",
        params={"topics": ["x"], "local_time": "09:00"},
    )
    await standing_tasks.add_task(
        backend.conn, user_id="u1", companion_id="becca",
        title="Morning weather", kind="briefing",
        params={"topics": ["x"], "local_time": "09:30"},
    )
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = CancelBriefingTool(MagicMock(companion_runtime=runtime))
    r = await tool.execute(_user_id="u1", task_id=t1.id)
    assert r.success
    assert (r.metadata or {}).get("task_id") == t1.id

    # Second briefing still exists.
    remaining = await standing_tasks.list_tasks(
        backend.conn, user_id="u1", companion_id="becca",
    )
    assert len(remaining) == 1
    assert remaining[0].title == "Morning weather"


@pytest.mark.asyncio
async def test_cancel_briefing_unknown_id_returns_not_found():
    from augmentum.config import settings
    from augmentum.tools.manage_briefings import CancelBriefingTool

    settings.companion_runtime_enabled = True
    settings.companion_standing_tasks_enabled = True

    backend = await _fresh_backend("u1")
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    tool = CancelBriefingTool(MagicMock(companion_runtime=runtime))
    r = await tool.execute(_user_id="u1", task_id=99999)
    assert not r.success
    assert (r.metadata or {}).get("reason") == "not_found"


# ── step() cancellation handling ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_runner_cancellation_advances_next_run_at(monkeypatch):
    """A runner that exceeds the verb's wallclock budget gets cancelled
    by asyncio.wait_for. Without the CancelledError handler in step(),
    next_run_at stays at "due now" and the same task gets re-picked on
    every tick — five repicks trip the verb's auto-pause. Regression
    pinned 2026-06-08 after tick_scheduler stayed paused for ~7 hours
    when a briefing went over the 5s budget.
    """
    import asyncio

    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    # monkeypatch auto-restores so we don't leak presence_mode to other tests.
    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged", raising=False)
    # Restore the briefing runner after the test so other tests aren't
    # affected by our hung-runner monkey-patch.
    original_briefing = standing_tasks._TASK_KINDS.get("briefing")

    backend = await _fresh_backend("u1")

    # Seed a "due now" task with a kind whose runner we'll monkey-patch
    # to hang. Using "briefing" because it's a real registered kind.
    await backend.conn.execute(
        "INSERT INTO companion_standing_tasks "
        "(user_id, companion_id, title, kind, params, interval_seconds, "
        " next_run_at, enabled, consecutive_error_count, created_at) "
        "VALUES (?, ?, 'hangs', 'briefing', '{}', 3600, "
        "        datetime('now', '-1 minute'), 1, 0, datetime('now'))",
        ("u1", "becca"),
    )
    await backend.conn.commit()

    # Replace the briefing runner with one that hangs forever — the
    # wait_for at the call site will cancel it.
    async def hung_runner(_runtime, *, user_id, params):
        await asyncio.sleep(60)  # well past the 0.1s wait_for below
        return {"summary": "should never reach"}

    standing_tasks._TASK_KINDS["briefing"] = hung_runner

    app_state = MagicMock()
    _disable_user_tz_lookup(app_state)
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    runtime.owner_user_id = "u1"
    runtime._app_state = app_state

    # Run step() under a 100ms wallclock — mimics the verb's wait_for,
    # which raises TimeoutError on the caller side after cancelling the
    # inner coroutine. The inner step() sees CancelledError.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(standing_tasks.step(runtime), timeout=0.1)

    # Give the shielded _persist_run a moment to drain on the loop.
    await asyncio.sleep(0.05)

    # The shielded _persist_run must have run. Verify next_run_at
    # advanced into the future. A budget timeout (slow, not broken) must
    # increment the BUDGET counter, NOT consecutive_error_count, so a
    # merely-slow task isn't auto-paused like a failing one (audit
    # 2026-06-17).
    cur = await backend.conn.execute(
        "SELECT next_run_at, consecutive_error_count, "
        "       consecutive_budget_timeout_count, last_error "
        "FROM companion_standing_tasks WHERE user_id = 'u1' AND title = 'hangs'"
    )
    row = await cur.fetchone()
    assert row is not None
    next_run_at, errs, budget_errs, last_error = row
    assert errs == 0
    assert budget_errs == 1
    assert "cancelled" in (last_error or "")
    # next_run_at must NOT still be "due now" — it should have rolled
    # forward by interval_seconds (≈ 1 hour).
    import datetime as _dt
    nrt = _dt.datetime.fromisoformat(next_run_at)
    now = _dt.datetime.utcnow()
    assert nrt > now + _dt.timedelta(seconds=30), (
        f"next_run_at={nrt} should be >30s in the future; "
        f"otherwise the verb will re-pick the task on the next tick"
    )

    # Restore the original briefing runner so subsequent tests get the
    # real one. (monkeypatch only handles attributes, not dict entries.)
    if original_briefing is not None:
        standing_tasks._TASK_KINDS["briefing"] = original_briefing


# ── run history (migration 258) ──────────────────────────────────────────


def _runtime_for(backend, user_id: str):
    """MagicMock runtime wired to a real backend — the step()/run_now()
    shape used across this suite."""
    app_state = MagicMock()
    _disable_user_tz_lookup(app_state)
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    runtime.owner_user_id = user_id
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


@pytest.mark.asyncio
async def test_record_run_round_trip_and_user_scoping():
    from augmentum.companion_runtime import standing_tasks

    backend = await _fresh_backend("u1")
    task_id = await _seed_due_task(
        backend, user_id="u1", title="t", kind="url_watch")

    await standing_tasks._record_run(
        backend.conn, task_id=task_id, user_id="u1",
        status="silent", summary="unchanged",
    )
    await standing_tasks._record_run(
        backend.conn, task_id=task_id, user_id="u1",
        status="fired", summary="changed!", details={"elapsed_ms": 42},
    )

    runs = await standing_tasks.list_runs(
        backend.conn, task_id=task_id, user_id="u1")
    assert [r["status"] for r in runs] == ["fired", "silent"]  # newest first
    assert runs[0]["details"]["elapsed_ms"] == 42

    # User scoping: another user sees nothing for this task.
    assert await standing_tasks.list_runs(
        backend.conn, task_id=task_id, user_id="someone_else") == []


@pytest.mark.asyncio
async def test_record_run_retention_keeps_newest_20():
    from augmentum.companion_runtime import standing_tasks

    backend = await _fresh_backend("u1")
    task_id = await _seed_due_task(
        backend, user_id="u1", title="t", kind="url_watch")

    for i in range(25):
        await standing_tasks._record_run(
            backend.conn, task_id=task_id, user_id="u1",
            status="silent", summary=f"run {i}",
        )

    runs = await standing_tasks.list_runs(
        backend.conn, task_id=task_id, user_id="u1", limit=100)
    assert len(runs) == 20
    assert runs[0]["summary"] == "run 24"   # newest kept
    assert runs[-1]["summary"] == "run 5"   # oldest five trimmed


@pytest.mark.asyncio
async def test_step_records_silent_fired_and_error_rows(monkeypatch):
    """Every step() outcome leaves a run row — the trust surface. A
    quiet watch ('checked, nothing new') must be distinguishable from a
    dead one (no rows) and a broken one (error rows)."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged", raising=False)

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")

    outcomes = {
        "silent one": {"summary": "nothing new", "noteworthy": False, "refs": []},
        "fired one": {"summary": "it changed", "noteworthy": True, "refs": []},
    }

    async def scripted_runner(_runtime, *, user_id, params):
        title = params.get("_title", "")
        if title == "error one":
            raise ValueError("boom")
        return outcomes[title]

    standing_tasks._TASK_KINDS["_test_scripted"] = scripted_runner
    try:
        ids = {}
        for title in ("silent one", "fired one", "error one"):
            ids[title] = await _seed_due_task(
                backend, user_id="u1", title=title, kind="_test_scripted",
                params=f'{{"_title": "{title}"}}',
            )
            await standing_tasks.step(runtime)

        for title, want_status in (
            ("silent one", "silent"),
            ("fired one", "fired"),
            ("error one", "error"),
        ):
            runs = await standing_tasks.list_runs(
                backend.conn, task_id=ids[title], user_id="u1")
            assert len(runs) == 1, f"{title}: expected exactly one run row"
            assert runs[0]["status"] == want_status
        # The error row carries the exception text.
        err_runs = await standing_tasks.list_runs(
            backend.conn, task_id=ids["error one"], user_id="u1")
        assert "boom" in err_runs[0]["summary"]
    finally:
        del standing_tasks._TASK_KINDS["_test_scripted"]


@pytest.mark.asyncio
async def test_one_shot_afterlife_still_records_run(monkeypatch):
    """One-shot rows get disabled after firing but their run row must
    exist — the historical artifact includes the execution trace."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged", raising=False)

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")

    async def one_hit(_runtime, *, user_id, params):
        return {"summary": "delivered", "noteworthy": True, "refs": []}

    standing_tasks._TASK_KINDS["_test_oneshot"] = one_hit
    try:
        task_id = await _seed_due_task(
            backend, user_id="u1", title="once", kind="_test_oneshot",
            params='{"one_shot": true}',
        )
        await standing_tasks.step(runtime)

        runs = await standing_tasks.list_runs(
            backend.conn, task_id=task_id, user_id="u1")
        assert len(runs) == 1 and runs[0]["status"] == "fired"

        cur = await backend.conn.execute(
            "SELECT enabled FROM companion_standing_tasks WHERE id = ?",
            (task_id,),
        )
        row = await cur.fetchone()
        assert row is not None and row[0] == 0  # disabled, not deleted
    finally:
        del standing_tasks._TASK_KINDS["_test_oneshot"]


# ── Scheduled requests & watches: entry points + verification ───────────


def _watch_tool(backend):
    app_state = MagicMock()
    _disable_user_tz_lookup(app_state)
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    runtime._app_state = app_state
    app_state.companion_runtime = runtime
    from augmentum.tools.watch_for import WatchForTool
    return WatchForTool(app_state), runtime


@pytest.mark.asyncio
async def test_watch_for_kind_mapping_intent_condition(monkeypatch):
    """kind=search → recurring_search row; intent + condition land in
    params; probe result is reflected in the reply."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    backend = await _fresh_backend("u1")
    tool, runtime = _watch_tool(backend)

    async def fake_run_now(_runtime, *, task_id, user_id, surface=True):
        assert surface is False  # probe must never notify
        return {"summary": "\"gpu prices\": 3 results", "noteworthy": False}

    monkeypatch.setattr(standing_tasks, "run_now", fake_run_now)

    r = await tool.execute(
        _user_id="u1", title="gpu watch", kind="search",
        target="gpu prices",
        intent="only price drops",
        condition={"op": "<", "value": 500, "unit": "USD"},
    )
    assert r.success, r.error
    assert "Right now:" in r.output
    tasks = await standing_tasks.list_tasks(
        backend.conn, user_id="u1", companion_id="becca")
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == "recurring_search"
    assert t.params["query"] == "gpu prices"
    assert t.params["intent"] == "only price drops"
    assert t.params["condition"] == {"op": "<", "value": 500.0, "unit": "USD"}


@pytest.mark.asyncio
async def test_watch_for_url_probe_fetch_fail_refuses(monkeypatch):
    """A URL watch whose probe can't fetch is refused and leaves no row
    — never silently watch nothing."""
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    backend = await _fresh_backend("u1")
    tool, runtime = _watch_tool(backend)

    async def fake_run_now(_runtime, *, task_id, user_id, surface=True):
        return {"summary": "error: fetch failed: 404", "noteworthy": False}

    monkeypatch.setattr(standing_tasks, "run_now", fake_run_now)

    r = await tool.execute(
        _user_id="u1", title="dead page", kind="url",
        target="https://example.invalid/404",
    )
    assert not r.success
    assert (r.metadata or {}).get("reason") == "probe_fetch_failed"
    tasks = await standing_tasks.list_tasks(
        backend.conn, user_id="u1", companion_id="becca")
    assert tasks == []


@pytest.mark.asyncio
async def test_watch_for_unknown_kind_refused(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    backend = await _fresh_backend("u1")
    tool, _ = _watch_tool(backend)
    r = await tool.execute(_user_id="u1", title="x", kind="rss", target="y")
    assert not r.success and (r.metadata or {}).get("reason") == "bad_kind"


def test_validate_future_date_rules():
    from datetime import datetime, timedelta

    from augmentum.tools._standing_common import validate_future_date

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    norm, err = validate_future_date(tomorrow, "09:00")
    assert err is None and norm == tomorrow

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    norm, err = validate_future_date(yesterday, "09:00")
    assert norm is None and "past" in err

    norm, err = validate_future_date("2026-02-30", "09:00")
    assert norm is None and err  # not a real calendar day

    norm, err = validate_future_date("soonish", "09:00")
    assert norm is None and err


def test_compute_next_run_at_dated_anchor():
    """params.date pins the fire to that day; a stale date falls
    through to the regular next-occurrence rules."""
    from datetime import datetime, timedelta

    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at

    in_three_days = datetime.now() + timedelta(days=3)
    s = _compute_next_run_at(
        params={"local_time": "09:00", "date": in_three_days.strftime("%Y-%m-%d")},
        interval_seconds=86400,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    from datetime import UTC
    local = parsed.replace(tzinfo=UTC).astimezone()
    assert local.date() == in_three_days.date()

    # Stale date → falls back to normal anchored behavior (future, not
    # the dead day).
    long_past = "2020-01-01"
    s2 = _compute_next_run_at(
        params={"local_time": "09:00", "date": long_past},
        interval_seconds=86400,
    )
    parsed2 = datetime.strptime(s2, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    assert parsed2 > datetime.now(UTC)


@pytest.mark.asyncio
async def test_schedule_request_creates_prompt_fire_row(monkeypatch):
    from augmentum.companion_runtime import standing_tasks
    from augmentum.config import settings
    from augmentum.tools.schedule_request import ScheduleRequestTool

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)

    backend = await _fresh_backend("u1")
    app_state = MagicMock()
    _disable_user_tz_lookup(app_state)
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    app_state.companion_runtime = runtime
    tool = ScheduleRequestTool(app_state)

    r = await tool.execute(
        _user_id="u1",
        prompt="Summarize what happened with the llama.cpp release "
               "in the last 24 hours",
        local_time="9am",
    )
    assert r.success, r.error
    tasks = await standing_tasks.list_tasks(
        backend.conn, user_id="u1", companion_id="becca")
    assert len(tasks) == 1
    assert tasks[0].kind == "prompt_fire"
    assert tasks[0].params["one_shot"] is True
    assert "llama.cpp" in tasks[0].params["prompt"]

    # No prompt → refused; anon → refused.
    r2 = await tool.execute(_user_id="u1", prompt="", local_time="9am")
    assert not r2.success
    r3 = await tool.execute(prompt="do a thing", local_time="9am")
    assert not r3.success


@pytest.mark.asyncio
async def test_url_watch_condition_pipeline_through_step(monkeypatch):
    """End-to-end numeric watch: fixture page → extraction → state
    machine → observation rows → confirmed condition fire. The judge is
    NOT consulted for condition fires without intent."""
    from pathlib import Path

    from augmentum.companion_runtime import curator, standing_tasks
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged", raising=False)

    fixtures = Path(__file__).parent / "fixtures" / "watch_pages"
    pages = {"current": (fixtures / "product_jsonld.html").read_text(encoding="utf-8")}

    class FakeResp:
        status_code = 200
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass

    class FakeClient:
        async def get(self, url, **kw): return FakeResp(pages["current"])

    monkeypatch.setattr(curator, "_resolve_http_client", lambda _r: FakeClient())

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")

    task_id = await _seed_due_task(
        backend, user_id="u1", title="widget price", kind="url_watch",
        params='{"url": "https://shop.example/widget", '
               '"condition": {"op": "<", "value": 500, "unit": "USD"}}',
    )

    async def fire():
        await backend.conn.execute(
            "UPDATE companion_standing_tasks SET next_run_at = "
            "datetime('now', '-1 minute') WHERE id = ?", (task_id,))
        await backend.conn.commit()
        await standing_tasks.step(runtime)

    await fire()                      # 549.99 — baseline, above threshold
    pages["current"] = pages["current"].replace("549.99", "467.00")
    await fire()                      # 467 — condition met 1/2, confirming
    await fire()                      # 467 — confirmed → fires

    runs = await standing_tasks.list_runs(
        backend.conn, task_id=task_id, user_id="u1", limit=10)
    assert [r["status"] for r in runs] == ["fired", "silent", "silent"]
    assert "condition < 500 met" in runs[0]["summary"]

    cur = await backend.conn.execute(
        "SELECT value, status, method FROM companion_metric_observations "
        "WHERE task_id = ? ORDER BY id", (task_id,))
    obs = await cur.fetchall()
    assert [(o[0], o[1]) for o in obs] == [
        (54999, "ok"), (46700, "ok"), (46700, "ok"),
    ]
    assert obs[0][2] == "json-ld"


@pytest.mark.asyncio
async def test_judge_suppression_records_suppressed_row(monkeypatch):
    """An intent-carrying watch whose change the judge rules unimportant:
    no surfacing, run row 'suppressed' with the verdict attached, and
    the baseline still advances (no re-fire next tick)."""
    from augmentum.companion_runtime import standing_tasks, watch_judge
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_runtime_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_standing_tasks_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged", raising=False)

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")

    async def changed_runner(_runtime, *, user_id, params):
        return {
            "summary": "page changed", "noteworthy": True, "refs": [],
            "details": {"content": "footer date rotated"},
        }

    async def fake_judge(_runtime, *, intent, diff_content):
        return watch_judge.JudgeVerdict(
            important=False, reason="footer date, not a price",
            evidence="footer date rotated", evidence_verified=True,
            consulted=True, raw_important=False,
        )

    monkeypatch.setattr(watch_judge, "judge_change", fake_judge)
    standing_tasks._TASK_KINDS["_test_changed"] = changed_runner
    try:
        task_id = await _seed_due_task(
            backend, user_id="u1", title="judged", kind="_test_changed",
            params='{"intent": "only price changes"}',
        )
        await standing_tasks.step(runtime)

        runs = await standing_tasks.list_runs(
            backend.conn, task_id=task_id, user_id="u1")
        assert len(runs) == 1
        assert runs[0]["status"] == "suppressed"
        assert runs[0]["details"]["judge"]["reason"].startswith("footer")
        runtime.memory.safe_journal.assert_not_called()  # nothing surfaced
    finally:
        del standing_tasks._TASK_KINDS["_test_changed"]


@pytest.mark.asyncio
async def test_run_now_records_row():
    from augmentum.companion_runtime import standing_tasks

    backend = await _fresh_backend("u1")
    runtime = _runtime_for(backend, "u1")

    async def quiet(_runtime, *, user_id, params):
        return {"summary": "manual check, nothing", "noteworthy": False, "refs": []}

    standing_tasks._TASK_KINDS["_test_manual"] = quiet
    try:
        task_id = await _seed_due_task(
            backend, user_id="u1", title="manual", kind="_test_manual")
        result = await standing_tasks.run_now(
            runtime, task_id=task_id, user_id="u1")
        assert result is not None

        runs = await standing_tasks.list_runs(
            backend.conn, task_id=task_id, user_id="u1")
        assert len(runs) == 1 and runs[0]["status"] == "silent"
    finally:
        del standing_tasks._TASK_KINDS["_test_manual"]


# ── _compute_next_run_at: cron mode ─────────────────────────────────────


def test_cron_mode_daily_expression_lands_on_expected_minute():
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(
        params={"cron": "30 6 * * *"}, interval_seconds=900,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    # No jitter seed → exact occurrence: strictly future, on :30, and it
    # ignored the 15-min interval fallback (>= 1 min out is enough proof
    # only combined with the minute check).
    assert parsed > datetime.now(UTC)
    assert parsed.minute == 30
    assert parsed.second == 0


def test_cron_mode_wins_over_local_time_and_interval():
    """Precedence: cron > anchored local_time. A row carrying both uses
    the cron occurrence, not the daily local_time anchor."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(
        params={"cron": "0 */2 * * *", "local_time": "09:00"},
        interval_seconds=86400,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    # Every-2-hours: next fire is at most ~2h out. The local_time anchor
    # could be up to 24h out; the interval fallback exactly 24h.
    assert parsed - datetime.now(UTC) <= timedelta(hours=2, minutes=1)
    assert parsed.minute == 0


def test_cron_mode_malformed_falls_back_to_interval():
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(
        params={"cron": "not a cron"}, interval_seconds=600,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    delta = parsed - datetime.now(UTC)
    assert timedelta(minutes=9) < delta <= timedelta(minutes=10, seconds=5)


def test_cron_mode_unsatisfiable_falls_back_to_interval():
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    s = _compute_next_run_at(
        params={"cron": "0 0 31 2 *"}, interval_seconds=600,
    )
    parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    delta = parsed - datetime.now(UTC)
    assert timedelta(minutes=9) < delta <= timedelta(minutes=10, seconds=5)


def test_cron_mode_jitter_never_schedules_in_past():
    """Same regression class as the anchored 4x-fire incident: negative
    jitter on a near-term cron occurrence must not land next_run_at in
    the past for ANY seed."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    now_utc = datetime.now(UTC)
    # Fires every minute — the nearest possible occurrence, so any
    # negative jitter offset would underflow without the guard.
    for seed in [str(i) for i in range(40)]:
        s = _compute_next_run_at(
            params={"cron": "* * * * *"}, interval_seconds=86400,
            jitter_seed=seed,
        )
        parsed = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        assert parsed > now_utc, f"seed {seed} scheduled in the past"


def test_cron_mode_respects_user_timezone():
    """'0 9 * * *' in Tokyo and New York must produce different UTC
    moments (unless the probe lands during the ~0 window where both are
    simultaneously before/after their local 9am — the hour spread makes
    that impossible: JST and EST 9am are 13-14h apart in UTC)."""
    from augmentum.companion_runtime.standing_tasks import _compute_next_run_at
    tokyo = _compute_next_run_at(
        params={"cron": "0 9 * * *"}, interval_seconds=86400,
        user_timezone="Asia/Tokyo",
    )
    ny = _compute_next_run_at(
        params={"cron": "0 9 * * *"}, interval_seconds=86400,
        user_timezone="America/New_York",
    )
    assert tokyo != ny


@pytest.mark.asyncio
async def test_add_task_cron_waits_for_first_occurrence():
    """A cron task must NOT run immediately at creation — it anchors to
    the next cron occurrence, same as local_time anchors."""
    from augmentum.companion_runtime import standing_tasks

    backend = await _fresh_backend()
    try:
        task = await standing_tasks.add_task(
            backend.conn, user_id="usr_s", companion_id="becca",
            title="hourly check", kind="recurring_search",
            params={"query": "x", "cron": "0 * * * *"},
            interval_seconds=86400,
        )
        assert task is not None
        assert task.next_run_at is not None
        nxt = datetime.strptime(
            task.next_run_at, "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=UTC)
        assert nxt > datetime.now(UTC)
        # Anchored creation applies the ±10min recurring jitter window
        # around the on-the-hour occurrence.
        assert nxt.minute >= 50 or nxt.minute <= 10
    finally:
        await backend.close()


# ── delivery preference (user choice over kind default) ─────────────────


def test_surface_importance_honors_user_delivery_choice():
    """params.delivery overrides the kind default in BOTH directions —
    a quiet briefing stays off the phone, a loud url_watch pushes
    everywhere. Absent a choice, kind defaults hold."""
    from augmentum.companion_runtime.standing_tasks import (
        StandingTask,
        _surface_importance,
    )
    from augmentum.notifications.catalog import (
        IMPORTANCE_DEFAULT,
        IMPORTANCE_HIGH,
    )

    def mk(kind, params):
        return StandingTask(
            id=1, user_id="u1", companion_id="becca", title="t",
            kind=kind, params=params, interval_seconds=3600,
            last_run_at=None, next_run_at=None,
            last_result_summary=None, last_error=None,
            enabled=True, consecutive_error_count=0,
        )

    # Kind defaults (no choice made).
    assert _surface_importance(mk("briefing", {})) == IMPORTANCE_HIGH
    assert _surface_importance(mk("url_watch", {})) == IMPORTANCE_DEFAULT
    # Explicit choice wins, both directions.
    assert _surface_importance(
        mk("briefing", {"delivery": "quiet"})) == IMPORTANCE_DEFAULT
    assert _surface_importance(
        mk("url_watch", {"delivery": "alert"})) == IMPORTANCE_HIGH
    # Junk value falls back to the kind default.
    assert _surface_importance(
        mk("url_watch", {"delivery": "banana"})) == IMPORTANCE_DEFAULT


def test_parse_delivery_param_vocabulary():
    from augmentum.tools._standing_common import parse_delivery_param
    assert parse_delivery_param(None) == (None, None)
    assert parse_delivery_param("") == (None, None)
    assert parse_delivery_param("alert") == ("alert", None)
    assert parse_delivery_param("QUIET") == ("quiet", None)
    assert parse_delivery_param("push") == ("alert", None)
    assert parse_delivery_param("digest") == ("quiet", None)
    norm, err = parse_delivery_param("shout")
    assert norm is None and "delivery" in err
