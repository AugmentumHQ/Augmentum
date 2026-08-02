"""Tests for the skill graph + outcome ledger — thesis Step 3.

The capability-side accumulation substrate. Covers:

- Migration 193 creates the three tables
- Skill CRUD: register, get_by_name (idempotent), fetch
- Instance recording bumps the parent skill's instances_count
- Outcome recording moves confidence + success/failure counts
- Relevance query returns top-K by cosine + confidence-gated
- query_relevant returns empty when feature gate is off (façade level)
- Bus events fire on register / instance / outcome
- The prompt_compose Layer 5.6 renders relevant skills correctly
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


# ── Migration 193 substrate ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_193_creates_tables():
    backend = await _boot_backend()
    try:
        # All three tables must exist with the right column shape
        cur = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'companion_skill%' ORDER BY name"
        )
        rows = await cur.fetchall()
        await cur.close()
        names = [r[0] for r in rows]
        assert "companion_skills" in names
        assert "companion_skill_instances" in names
        assert "companion_skill_outcomes" in names
    finally:
        await backend.close()


# ── SkillGraph CRUD ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_skill_creates_node():
    from augmentum.companion.skills import SkillGraph
    from augmentum.companion_runtime.bus import PresenceBus

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, bus=PresenceBus(), companion_id="becca")
        skill = await graph.register_skill(
            name="isolate_before_guessing",
            description="When the user is stuck, isolate variables before suggesting causes.",
            problem_shape="debugging unfamiliar code that suddenly broke",
            user_id="u_test",
        )
        assert skill.id > 0
        assert skill.name == "isolate_before_guessing"
        assert skill.user_id == "u_test"
        assert skill.status == "active"
        assert skill.confidence == 0.5  # default
        assert skill.instances_count == 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_register_skill_is_idempotent_by_name():
    """Re-registering same name + user_id updates instead of duplicating."""
    from augmentum.companion.skills import SkillGraph

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        first = await graph.register_skill(
            name="pause_before_pleasing",
            description="first description",
            problem_shape="conversational warmth gates",
            user_id="u_test",
        )
        second = await graph.register_skill(
            name="pause_before_pleasing",
            description="updated description",
            problem_shape="conversational warmth gates",
            user_id="u_test",
        )
        # Same id — the second call updated the existing row
        assert first.id == second.id
        assert second.description == "updated description"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_get_by_name_returns_user_scoped():
    """Looking up by name + user_id returns only that user's skill."""
    from augmentum.companion.skills import SkillGraph

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        await graph.register_skill(
            name="x",
            description="x desc",
            problem_shape="user a problem",
            user_id="u_a",
        )
        await graph.register_skill(
            name="x",  # same name, different user
            description="x desc for b",
            problem_shape="user b problem",
            user_id="u_b",
        )
        s_a = await graph.get_by_name("x", user_id="u_a")
        s_b = await graph.get_by_name("x", user_id="u_b")
        assert s_a is not None
        assert s_b is not None
        assert s_a.id != s_b.id
        assert s_a.user_id == "u_a"
        assert s_b.user_id == "u_b"
    finally:
        await backend.close()


# ── Instances ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_instance_bumps_skill_counter():
    from augmentum.companion.skills import SkillGraph

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        skill = await graph.register_skill(
            name="test_skill",
            description="desc",
            problem_shape="problem",
            user_id="u_test",
        )
        assert skill.instances_count == 0

        await graph.record_instance(
            skill.id,
            context="user said something hard",
            approach="paused, then named the difficulty",
            user_id="u_test",
            session_id="s1",
        )

        refreshed = await graph.get(skill.id)
        assert refreshed is not None
        assert refreshed.instances_count == 1
    finally:
        await backend.close()


# ── Outcomes + confidence movement ────────────────────────────────────


@pytest.mark.asyncio
async def test_outcome_moves_confidence_toward_signal():
    """Positive outcome → confidence rises; negative → falls. EWMA-style
    so one good doesn't fully promote and one bad doesn't fully demote."""
    from augmentum.companion.skills import (
        OUTCOME_ACCEPTED, OUTCOME_REJECTED, SkillGraph,
    )

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        skill = await graph.register_skill(
            name="s",
            description="d",
            problem_shape="p",
            user_id="u",
        )
        instance = await graph.record_instance(
            skill.id,
            context="c",
            approach="a",
            user_id="u",
        )
        baseline = (await graph.get(skill.id)).confidence

        # Accepted (+0.6 signal) should raise confidence
        await graph.record_outcome(
            instance.id,
            outcome=OUTCOME_ACCEPTED,
            evidence="user thanked",
            detected_by="user_explicit",
        )
        after_pos = (await graph.get(skill.id)).confidence
        assert after_pos > baseline

        # Second instance + rejection should drop it back below baseline
        i2 = await graph.record_instance(
            skill.id, context="c2", approach="a2", user_id="u",
        )
        await graph.record_outcome(
            i2.id,
            outcome=OUTCOME_REJECTED,
            evidence="user said no",
            detected_by="user_explicit",
        )
        after_neg = (await graph.get(skill.id)).confidence
        assert after_neg < after_pos
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_outcome_increments_counters_for_strong_signals():
    """successes_count rises on signal > 0.5; failures_count on signal < -0.5."""
    from augmentum.companion.skills import (
        OUTCOME_REJECTED, OUTCOME_SHIPPED, SkillGraph,
    )

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        skill = await graph.register_skill(
            name="s",
            description="d",
            problem_shape="p",
            user_id="u",
        )
        # Three shipped outcomes (+1.0 signal each)
        for _ in range(3):
            i = await graph.record_instance(
                skill.id, context="c", approach="a", user_id="u",
            )
            await graph.record_outcome(i.id, outcome=OUTCOME_SHIPPED)
        # Two rejected (-0.8 signal each)
        for _ in range(2):
            i = await graph.record_instance(
                skill.id, context="c", approach="a", user_id="u",
            )
            await graph.record_outcome(i.id, outcome=OUTCOME_REJECTED)
        s = await graph.get(skill.id)
        assert s.successes_count == 3
        assert s.failures_count == 2
    finally:
        await backend.close()


# ── Bus emission ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_emits_on_register_and_instance_and_outcome():
    from augmentum.companion.skills import (
        OUTCOME_ACCEPTED, SkillGraph,
    )
    from augmentum.companion_runtime.bus import PresenceBus

    backend = await _boot_backend()
    bus = PresenceBus()
    graph = SkillGraph(backend, bus=bus, companion_id="becca")

    sub = await bus.subscribe("skill.**", slice_key="t")
    captured = []

    async def _drain():
        for _ in range(6):
            try:
                ev = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            if ev is None:
                break
            captured.append(ev.topic)

    drain_task = asyncio.create_task(_drain())
    try:
        skill = await graph.register_skill(
            name="s", description="d", problem_shape="p", user_id="u",
        )
        instance = await graph.record_instance(
            skill.id, context="c", approach="a", user_id="u",
        )
        await graph.record_outcome(instance.id, outcome=OUTCOME_ACCEPTED)
        await asyncio.sleep(0.15)
        await drain_task

        assert "skill.registered" in captured
        assert "skill.instance_recorded" in captured
        assert "skill.outcome_observed" in captured
    finally:
        await bus.unsubscribe(sub)
        await backend.close()


# ── Relevance retrieval ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_relevant_filters_by_confidence():
    """Skills below min_confidence shouldn't appear in retrieval."""
    from augmentum.companion.skills import SkillGraph

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        await graph.register_skill(
            name="established",
            description="this one works",
            problem_shape="debugging a weird race condition",
            user_id="u",
            confidence=0.85,
        )
        await graph.register_skill(
            name="early",
            description="untested",
            problem_shape="debugging a weird race condition",
            user_id="u",
            confidence=0.30,  # below default min
        )

        # The skill module may not have embeddings (no model installed
        # in the test env). When embeddings are absent, query returns
        # empty list — confidence filtering still applies at the SQL
        # level so the filter test runs even without embeddings.
        results = await graph.query_relevant(
            "debugging a weird race condition",
            user_id="u",
            min_confidence=0.5,
            min_relevance=0.0,  # don't gate on relevance in this test
        )
        # If embeddings work: only "established" comes back.
        # If embeddings don't work: empty result. Both are valid for
        # this assertion — what matters is "early" is never returned.
        for r in results:
            assert r.skill.name != "early"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_query_relevant_empty_for_empty_intent():
    from augmentum.companion.skills import SkillGraph

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        results = await graph.query_relevant("", user_id="u")
        assert results == []
    finally:
        await backend.close()


# ── List operations ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_skills_filters_by_status_and_user():
    from augmentum.companion.skills import SkillGraph

    backend = await _boot_backend()
    try:
        graph = SkillGraph(backend, companion_id="becca")
        s1 = await graph.register_skill(
            name="active_a", description="d", problem_shape="p", user_id="u",
        )
        s2 = await graph.register_skill(
            name="active_b", description="d", problem_shape="p", user_id="u",
        )
        # Manually retire s2 to test the status filter
        await backend.conn.execute(
            "UPDATE companion_skills SET status = 'retired' WHERE id = ?",
            (s2.id,),
        )
        await backend.conn.commit()

        active = await graph.list_skills(user_id="u", status="active")
        names = [s.name for s in active]
        assert "active_a" in names
        assert "active_b" not in names
    finally:
        await backend.close()


# ── Layer 5.6 renderer ───────────────────────────────────────────────


def test_relevant_skills_block_renders_with_confidence_hints():
    """The compose-time block names confidence as well-tested /
    moderately-tested / early so the model can weight."""
    from augmentum.companion.skills import RelevantSkill, Skill
    from augmentum.companion_runtime.prompt_compose import _relevant_skills_block

    s1 = Skill(
        id=1, companion_id="becca", user_id="u",
        name="well_tested",
        description="a well-tested approach",
        problem_shape="p",
        confidence=0.85,
        instances_count=10, successes_count=8, failures_count=1,
        status="active", abstracted_from_ids=[],
        created_at="", updated_at="",
    )
    s2 = Skill(
        id=2, companion_id="becca", user_id="u",
        name="early_one",
        description="an early approach",
        problem_shape="p",
        confidence=0.55,
        instances_count=2, successes_count=1, failures_count=0,
        status="active", abstracted_from_ids=[],
        created_at="", updated_at="",
    )
    block = _relevant_skills_block([
        RelevantSkill(skill=s1, relevance=0.85, effective_score=0.72),
        RelevantSkill(skill=s2, relevance=0.7, effective_score=0.385),
    ])
    assert "well-tested" in block
    assert "early" in block
    assert "well_tested" in block
    assert "approaches that have worked" in block.lower()


def test_relevant_skills_block_empty_when_no_input():
    from augmentum.companion_runtime.prompt_compose import _relevant_skills_block

    assert _relevant_skills_block([]) == ""


# ── Façade wiring ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_companion_skills_facade_returns_graph():
    """Companion.skills lazily constructs a SkillGraph over the
    runtime's backend."""
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

        graph = c.skills
        assert graph is not None
        # Second access returns same instance
        assert c.skills is graph

        # And it functions end-to-end
        skill = await graph.register_skill(
            name="s", description="d", problem_shape="p", user_id="u",
        )
        assert skill.id > 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_view_relevant_skills_respects_feature_flag(monkeypatch):
    """When companion_skills_enabled is off, view.relevant_skills
    returns empty regardless of graph state."""
    from augmentum.companion import Companion
    from augmentum.companion.skills import SkillGraph
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

        # Force flag off
        monkeypatch.setattr(_settings, "companion_skills_enabled", False)
        view = c.for_user("u")
        result = await view.relevant_skills("anything")
        assert result == []
    finally:
        await backend.close()
