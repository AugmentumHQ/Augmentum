"""Knowledge store tests — round-trip + renderer + multi-tenant scoping.

The store is one of the durability surfaces for the comprehension
phase: the map needs to survive restart, compaction, and per-run
churn. We exercise: empty-fetch, upsert, re-fetch, forget, the
freshness/age metadata, the user-scoping invariant, and the
prompt renderer's branches.
"""

from __future__ import annotations

import time

import pytest

from augmentum.bug_finder.knowledge_store import (
    CodebaseKnowledge,
    EntryPoint,
    KnowledgeStore,
    Pillar,
    RiskSurface,
    Subsystem,
    render_knowledge_brief,
)
from augmentum.state.backends.sqlite import SQLiteBackend


@pytest.fixture
async def store():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield KnowledgeStore(backend.conn)
    await backend.close()


# ---------------------------------------------------------------------------
# get — empty / populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_empty_returns_unpopulated_knowledge(store) -> None:
    k = await store.get(user_id="u1", workspace_id="ws_a")
    assert k.workspace_id == "ws_a"
    assert k.user_id == "u1"
    assert not k.is_populated
    assert k.brief == ""
    assert k.subsystems == ()
    assert k.last_updated == 0
    assert k.refresh_count == 0


@pytest.mark.asyncio
async def test_upsert_then_get_round_trips(store) -> None:
    subsystems = (
        Subsystem(
            name="auth",
            purpose="multi-tenant authentication + session management",
            paths=("augmentum/auth",),
            size_files=12,
            pillars=("user_id_scoping", "argon2_passwords"),
        ),
        Subsystem(
            name="bug_finder",
            purpose="LLM-driven security audit pipeline",
            paths=("augmentum/bug_finder",),
            size_files=21,
            pillars=("disproof_oriented_verifier",),
        ),
    )
    pillars = (
        Pillar(
            name="user_id_scoping",
            statement="Every user-scoped table accepts user_id and routes scope on it.",
            evidence=("augmentum/auth/store.py:42", "augmentum/proxy/auth_routes.py:88"),
        ),
    )
    risk_surfaces = (
        RiskSurface(
            name="http_routes",
            entry_points=("augmentum/proxy/openai_routes.py:chat_completions",),
            trust_boundary="user-supplied",
            downstream_sinks=("backend resolution", "memory store"),
        ),
    )
    entry_points = (
        EntryPoint(
            kind="http", path="POST /v1/chat/completions",
            handler="augmentum/proxy/openai_routes.py:chat_completions",
        ),
    )
    await store.upsert(
        user_id="u1", workspace_id="ws_a",
        brief="## Augmentum\nFastAPI proxy + multi-modal substrate...",
        subsystems=subsystems,
        pillars=pillars,
        risk_surfaces=risk_surfaces,
        entry_points=entry_points,
        commit_sha="abc1234",
        tokens_in=12345,
        tokens_out=2345,
        wallclock_seconds=88.5,
    )

    k = await store.get(user_id="u1", workspace_id="ws_a")
    assert k.is_populated
    assert k.brief.startswith("## Augmentum")
    assert len(k.subsystems) == 2
    assert k.subsystems[0].name == "auth"
    assert k.subsystems[0].pillars == ("user_id_scoping", "argon2_passwords")
    assert k.pillars[0].evidence == (
        "augmentum/auth/store.py:42",
        "augmentum/proxy/auth_routes.py:88",
    )
    assert k.risk_surfaces[0].trust_boundary == "user-supplied"
    assert k.entry_points[0].kind == "http"
    assert k.last_commit_sha == "abc1234"
    assert k.tokens_in == 12345
    assert k.refresh_count == 1


# ---------------------------------------------------------------------------
# upsert — bumps refresh_count, preserves over-writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_bumps_refresh_count_on_repeat(store) -> None:
    for i in range(3):
        await store.upsert(
            user_id="u1", workspace_id="ws_a",
            brief=f"brief v{i}",
        )
    k = await store.get(user_id="u1", workspace_id="ws_a")
    assert k.refresh_count == 3
    assert k.brief == "brief v2"


@pytest.mark.asyncio
async def test_upsert_overwrites_old_data(store) -> None:
    await store.upsert(
        user_id="u1", workspace_id="ws_a",
        brief="old", subsystems=(Subsystem(name="old", purpose="old"),),
    )
    await store.upsert(
        user_id="u1", workspace_id="ws_a",
        brief="new", subsystems=(Subsystem(name="new", purpose="new"),),
    )
    k = await store.get(user_id="u1", workspace_id="ws_a")
    assert k.brief == "new"
    assert len(k.subsystems) == 1
    assert k.subsystems[0].name == "new"


# ---------------------------------------------------------------------------
# forget — explicit re-comprehension trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_drops_the_row(store) -> None:
    await store.upsert(user_id="u1", workspace_id="ws_a", brief="x")
    assert (await store.get(user_id="u1", workspace_id="ws_a")).is_populated
    await store.forget(user_id="u1", workspace_id="ws_a")
    k = await store.get(user_id="u1", workspace_id="ws_a")
    assert not k.is_populated
    assert k.refresh_count == 0


@pytest.mark.asyncio
async def test_forget_is_idempotent(store) -> None:
    """Forgetting a non-existent row should not raise."""
    await store.forget(user_id="u1", workspace_id="ws_nonexistent")


# ---------------------------------------------------------------------------
# Multi-tenant isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_is_user_scoped(store) -> None:
    """Two users with same workspace_id must not see each other's data."""
    await store.upsert(user_id="u1", workspace_id="ws_shared", brief="u1-data")
    await store.upsert(user_id="u2", workspace_id="ws_shared", brief="u2-data")
    k1 = await store.get(user_id="u1", workspace_id="ws_shared")
    k2 = await store.get(user_id="u2", workspace_id="ws_shared")
    assert k1.brief == "u1-data"
    assert k2.brief == "u2-data"


@pytest.mark.asyncio
async def test_forget_is_user_scoped(store) -> None:
    """Forgetting u1's row leaves u2's intact."""
    await store.upsert(user_id="u1", workspace_id="ws_shared", brief="u1")
    await store.upsert(user_id="u2", workspace_id="ws_shared", brief="u2")
    await store.forget(user_id="u1", workspace_id="ws_shared")
    assert not (await store.get(user_id="u1", workspace_id="ws_shared")).is_populated
    assert (await store.get(user_id="u2", workspace_id="ws_shared")).is_populated


# ---------------------------------------------------------------------------
# Age + freshness metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_age_seconds_reflects_freshness(store) -> None:
    await store.upsert(user_id="u1", workspace_id="ws_a", brief="x")
    k = await store.get(user_id="u1", workspace_id="ws_a")
    # Freshly upserted — age should be tiny
    assert k.age_seconds < 5
    assert k.is_populated


def test_unpopulated_knowledge_has_zero_age() -> None:
    k = CodebaseKnowledge(
        workspace_id="ws", user_id="u", brief="",
        subsystems=(), pillars=(), risk_surfaces=(), entry_points=(),
        last_updated=0, last_commit_sha="", refresh_count=0,
        tokens_in=0, tokens_out=0, wallclock_seconds=0.0,
    )
    assert k.age_seconds == 0
    assert not k.is_populated


# ---------------------------------------------------------------------------
# render_knowledge_brief
# ---------------------------------------------------------------------------


def test_render_empty_knowledge_returns_empty_string() -> None:
    """An unpopulated store renders to '' — callers concat without an if."""
    k = CodebaseKnowledge(
        workspace_id="ws", user_id="u", brief="",
        subsystems=(), pillars=(), risk_surfaces=(), entry_points=(),
        last_updated=0, last_commit_sha="", refresh_count=0,
        tokens_in=0, tokens_out=0, wallclock_seconds=0.0,
    )
    assert render_knowledge_brief(k) == ""


def test_render_uses_brief_when_present() -> None:
    """When a markdown brief exists, that's authoritative — render it
    inside the standard wrapper with age + refresh metadata."""
    k = CodebaseKnowledge(
        workspace_id="ws", user_id="u",
        brief="## Codebase: FastAPI proxy\nMulti-tenant.",
        subsystems=(), pillars=(), risk_surfaces=(), entry_points=(),
        last_updated=int(time.time()), last_commit_sha="abc",
        refresh_count=2, tokens_in=0, tokens_out=0, wallclock_seconds=0.0,
    )
    out = render_knowledge_brief(k)
    assert "Codebase knowledge" in out
    assert "refresh #2" in out
    assert "FastAPI proxy" in out


def test_render_falls_back_to_structured_when_brief_empty() -> None:
    """No brief — render the structured tables so the planner still
    gets something useful."""
    k = CodebaseKnowledge(
        workspace_id="ws", user_id="u",
        brief="",  # No prose; only structured data
        subsystems=(
            Subsystem(name="auth", purpose="multi-tenant auth"),
        ),
        pillars=(
            Pillar(name="user_id_scoping", statement="all CRUD takes user_id"),
        ),
        risk_surfaces=(
            RiskSurface(name="http_routes", trust_boundary="user-supplied"),
        ),
        entry_points=(),
        last_updated=int(time.time()), last_commit_sha="",
        refresh_count=1, tokens_in=0, tokens_out=0, wallclock_seconds=0.0,
    )
    out = render_knowledge_brief(k)
    assert "Subsystems" in out
    assert "auth" in out
    assert "user_id_scoping" in out
    assert "http_routes" in out
