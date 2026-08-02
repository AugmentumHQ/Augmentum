"""Tests for the lesson registry — the learn-from-correction substrate.

The inverse of the skill graph (mig 193): where skills accumulate what
worked, lessons accumulate what the user corrected her on. Covers:

- Migration 270 creates companion_lessons
- capture creates a held lesson (strength 0.5, times_seen 1)
- capture dedups a recurring correction into a reinforce (no duplicate)
- reinforce raises strength + bumps the right counter
- retire drops a lesson from active retrieval
- query_relevant: empty for empty intent; strength-gated
- list_lessons filters by status + user
- bus emits lesson.captured
- the prompt_compose Layer 5.7 renderer
- Companion.lessons façade + view.relevant_lessons flag-gating
- the capture-layer JSON parser (_parse_lessons)
- capture_from_text is a no-op when the capture flag is off
"""

from __future__ import annotations

import asyncio

import pytest


async def _boot_backend():
    """Fresh :memory: SQLite backend with migrations applied."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


# ── Migration 270 substrate ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_270_creates_table():
    backend = await _boot_backend()
    try:
        cur = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name = 'companion_lessons'"
        )
        rows = await cur.fetchall()
        await cur.close()
        assert [r[0] for r in rows] == ["companion_lessons"]
    finally:
        await backend.close()


# ── Capture ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capture_creates_lesson():
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        lesson = await graph.capture(
            situation="he's still describing a bug",
            trap="jumping to a fix before he's finished",
            better="let him finish, then ask one question",
            user_id="u_test",
        )
        assert lesson.id > 0
        assert lesson.user_id == "u_test"
        assert lesson.status == "active"
        assert lesson.strength == 0.5
        assert lesson.times_seen == 1
        assert lesson.times_applied == 0
        assert lesson.source == "reflection"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_capture_requires_situation_and_better():
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        with pytest.raises(ValueError):
            await graph.capture(situation="", trap="t", better="b", user_id="u")
        with pytest.raises(ValueError):
            await graph.capture(situation="s", trap="t", better="", user_id="u")
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_recapture_dedups_into_reinforce():
    """The same correction recurring strengthens the lesson rather than
    inserting a duplicate — recurrence is the strongest signal a lesson
    is real."""
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        first = await graph.capture(
            situation="he asks for the news",
            trap="editorializing instead of just reporting",
            better="give the facts first, opinions only if asked",
            user_id="u",
        )
        second = await graph.capture(
            situation="He asks for the news",          # case-insensitive
            trap="Editorializing instead of just reporting",
            better="(re-stated differently)",
            user_id="u",
        )
        # Same row — recurrence reinforced rather than duplicated.
        assert first.id == second.id
        assert second.times_seen == 2
        assert second.strength > first.strength

        only = await graph.list_lessons(user_id="u")
        assert len(only) == 1
    finally:
        await backend.close()


# ── Reinforcement / retire ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reinforce_raises_strength_and_counters():
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        lesson = await graph.capture(
            situation="s", trap="t", better="b", user_id="u",
        )
        base = lesson.strength

        seen = await graph.reinforce(lesson.id, seen=True)
        assert seen.strength > base
        assert seen.times_seen == 2
        assert seen.times_applied == 0

        applied = await graph.reinforce(lesson.id, applied=True)
        assert applied.strength > seen.strength
        assert applied.times_applied == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_retire_removes_from_active():
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        lesson = await graph.capture(
            situation="s", trap="t", better="b", user_id="u",
        )
        await graph.retire(lesson.id)
        active = await graph.list_lessons(user_id="u", status="active")
        assert lesson.id not in [x.id for x in active]
        retired = await graph.list_lessons(user_id="u", status="retired")
        assert lesson.id in [x.id for x in retired]
    finally:
        await backend.close()


# ── Retrieval ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_relevant_empty_for_empty_intent():
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        assert await graph.query_relevant("", user_id="u") == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_query_relevant_strength_gated():
    """A weak lesson below min_strength never surfaces (it's still being
    learned). Embeddings may be absent in the test env — in that case the
    result is empty, which still satisfies "the weak one never returns"."""
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        firm = await graph.capture(
            situation="debugging a flaky test",
            trap="guessing", better="isolate first",
            user_id="u", strength=0.85,
        )
        weak = await graph.capture(
            situation="debugging a flaky test (variant)",
            trap="guessing", better="isolate first",
            user_id="u", strength=0.30,
        )
        results = await graph.query_relevant(
            "debugging a flaky test",
            user_id="u", min_strength=0.5, min_relevance=0.0,
        )
        for r in results:
            assert r.lesson.id != weak.id
    finally:
        await backend.close()


# ── Listing ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_lessons_filters_by_status_and_user():
    from augmentum.companion.lessons import LessonGraph

    backend = await _boot_backend()
    try:
        graph = LessonGraph(backend, companion_id="becca")
        await graph.capture(situation="a", trap="t", better="b", user_id="u")
        await graph.capture(situation="c", trap="t", better="b", user_id="other")
        mine = await graph.list_lessons(user_id="u")
        situations = [x.situation for x in mine]
        assert "a" in situations
        assert "c" not in situations
    finally:
        await backend.close()


# ── Bus emission ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_emits_on_capture():
    from augmentum.companion.lessons import LessonGraph
    from augmentum.companion_runtime.bus import PresenceBus

    backend = await _boot_backend()
    bus = PresenceBus()
    graph = LessonGraph(backend, bus=bus, companion_id="becca")
    sub = await bus.subscribe("lesson.**", slice_key="t")
    captured: list[str] = []

    async def _drain():
        for _ in range(4):
            try:
                ev = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            if ev is None:
                break
            captured.append(ev.topic)

    drain_task = asyncio.create_task(_drain())
    try:
        await graph.capture(situation="s", trap="t", better="b", user_id="u")
        await asyncio.sleep(0.15)
        await drain_task
        assert "lesson.captured" in captured
    finally:
        await bus.unsubscribe(sub)
        await backend.close()


# ── Layer 5.7 renderer ────────────────────────────────────────────────


def test_relevant_lessons_block_renders_guardrails():
    from augmentum.companion.lessons import Lesson, RelevantLesson
    from augmentum.companion_runtime.prompt_compose import _relevant_lessons_block

    lesson = Lesson(
        id=1, companion_id="becca", user_id="u",
        situation="he's still describing a bug",
        trap="jumping to a fix early",
        better="let him finish, then ask one question",
        strength=0.7, times_seen=2, times_applied=1,
        source="reflection", evidence="", status="active",
        created_at="", updated_at="",
    )
    block = _relevant_lessons_block([
        RelevantLesson(lesson=lesson, relevance=0.8, effective_score=0.56),
    ])
    assert "learned with them" in block.lower()
    assert "let him finish" in block
    assert "not: jumping to a fix early" in block
    # Prohibition-with-escape grammar (Arbor principle #1): the header frames
    # each lesson as a standing correction against the *class* of mistake,
    # with departure allowed only on genuine difference — not a soft "be aware".
    low = block.lower()
    assert "standing correction" in low
    assert "near-variation" in low
    assert "genuinely differs" in low


def test_relevant_lessons_block_empty_when_no_input():
    from augmentum.companion_runtime.prompt_compose import _relevant_lessons_block

    assert _relevant_lessons_block([]) == ""


# ── Façade wiring ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_companion_lessons_facade_returns_graph():
    from augmentum.companion import Companion
    from augmentum.companion_runtime.bus import PresenceBus

    backend = await _boot_backend()
    try:
        class _Runtime:
            companion_id = "becca"
            bus = PresenceBus()
            _started = True

        runtime = _Runtime()
        runtime.backend = backend
        c = Companion(runtime)

        graph = c.lessons
        assert graph is not None
        assert c.lessons is graph  # memoized
        lesson = await graph.capture(
            situation="s", trap="t", better="b", user_id="u",
        )
        assert lesson.id > 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_view_relevant_lessons_respects_feature_flag(monkeypatch):
    from augmentum.companion import Companion
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.config import settings as _settings

    backend = await _boot_backend()
    try:
        class _Runtime:
            companion_id = "becca"
            bus = PresenceBus()
            _started = True

        runtime = _Runtime()
        runtime.backend = backend
        c = Companion(runtime)

        monkeypatch.setattr(_settings, "companion_lessons_enabled", False)
        view = c.for_user("u")
        assert await view.relevant_lessons("anything") == []
    finally:
        await backend.close()


# ── Capture-layer parser ──────────────────────────────────────────────


def test_parse_lessons_plain_array():
    from augmentum.companion_runtime.lessons_capture import _parse_lessons

    out = _parse_lessons(
        '[{"situation": "s", "trap": "t", "better": "b"}]'
    )
    assert out == [{"situation": "s", "trap": "t", "better": "b"}]


def test_parse_lessons_strips_code_fence_and_wrapper():
    from augmentum.companion_runtime.lessons_capture import _parse_lessons

    fenced = '```json\n{"lessons": [{"situation": "s", "better": "b"}]}\n```'
    out = _parse_lessons(fenced)
    assert out == [{"situation": "s", "trap": "", "better": "b"}]


def test_parse_lessons_drops_incomplete_and_caps():
    from augmentum.companion_runtime.lessons_capture import (
        MAX_LESSONS_PER_PASS, _parse_lessons,
    )

    # One valid, one missing 'better' (dropped), plus many valid to test cap.
    items = [{"situation": f"s{i}", "trap": "t", "better": "b"} for i in range(10)]
    items.insert(1, {"situation": "no_better"})  # dropped
    import json as _json
    out = _parse_lessons(_json.dumps(items))
    assert len(out) == MAX_LESSONS_PER_PASS
    assert all("better" in x and x["better"] for x in out)


def test_parse_lessons_garbage_returns_empty():
    from augmentum.companion_runtime.lessons_capture import _parse_lessons

    assert _parse_lessons("not json at all") == []
    assert _parse_lessons("") == []


@pytest.mark.asyncio
async def test_capture_from_text_noop_when_flag_off(monkeypatch):
    from augmentum.companion_runtime import lessons_capture
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_lessons_capture_enabled", False)

    class _Runtime:
        companion_id = "becca"

    res = await lessons_capture.capture_from_text(
        _Runtime(), user_id="u", text="anything",
    )
    assert res["skipped"] == "feature_disabled"
    assert res["captured"] == []
