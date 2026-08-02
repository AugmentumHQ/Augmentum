"""CoderProfileStore round-trip + multi-tenant + workspace-merge tests.

Pure SQLite, in-memory connection — no app stack, fast, deterministic.
The schema is duplicated here (rather than wiring the migration runner)
to keep tests self-contained; the real migration is the source of
truth at augmentum/state/migrations/136_coder_profile.sql.
"""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.coder.profile import CoderProfileStore, ProfileEntry


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS coder_profile (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    workspace_id      TEXT NOT NULL DEFAULT '',
    category          TEXT NOT NULL,
    key               TEXT NOT NULL,
    value             TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 1,
    last_observed_at  REAL NOT NULL,
    created_at        REAL NOT NULL,
    UNIQUE(user_id, workspace_id, category, key)
);
"""


async def _mkstore() -> tuple[CoderProfileStore, aiosqlite.Connection]:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    return CoderProfileStore(conn), conn


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_and_get_roundtrip():
    store, _ = await _mkstore()
    entry = await store.upsert(
        user_id="u1", category="language",
        key="python.return_type_style",
        value="explicit",
        confidence=0.7,
    )
    assert isinstance(entry, ProfileEntry)
    assert entry.user_id == "u1"
    assert entry.workspace_id == ""
    assert entry.category == "language"
    assert entry.key == "python.return_type_style"
    assert entry.value == "explicit"
    assert entry.confidence == 0.7
    assert entry.observation_count == 1


@pytest.mark.asyncio
async def test_get_returns_none_for_missing():
    store, _ = await _mkstore()
    assert await store.get(
        user_id="u1", category="language", key="nonexistent",
    ) is None


@pytest.mark.asyncio
async def test_value_can_be_complex_json():
    store, _ = await _mkstore()
    payload = {"prefer": ["a", "b"], "score": 1.5, "meta": {"x": 1}}
    entry = await store.upsert(
        user_id="u1", category="pattern", key="edit.shape", value=payload,
    )
    assert entry.value == payload
    fetched = await store.get(user_id="u1", category="pattern", key="edit.shape")
    assert fetched is not None
    assert fetched.value == payload


# ---------------------------------------------------------------------------
# Upsert semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_increments_observation_count():
    store, _ = await _mkstore()
    for _ in range(3):
        await store.upsert(
            user_id="u1", category="tool", key="prefer.grep", value=True,
        )
    fetched = await store.get(user_id="u1", category="tool", key="prefer.grep")
    assert fetched is not None
    assert fetched.observation_count == 3


@pytest.mark.asyncio
async def test_upsert_refreshes_value_and_confidence():
    """Later observations win — assumption: fresher = more correct."""
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="convention", key="naming",
        value="snake_case", confidence=0.4,
    )
    second = await store.upsert(
        user_id="u1", category="convention", key="naming",
        value="kebab-case", confidence=0.9,
    )
    assert second.value == "kebab-case"
    assert second.confidence == 0.9
    assert second.observation_count == 2


@pytest.mark.asyncio
async def test_upsert_requires_user_id():
    store, _ = await _mkstore()
    with pytest.raises(ValueError, match="user_id"):
        await store.upsert(
            user_id="", category="x", key="y", value="z",
        )


@pytest.mark.asyncio
async def test_upsert_requires_category_and_key():
    store, _ = await _mkstore()
    with pytest.raises(ValueError, match="category"):
        await store.upsert(
            user_id="u1", category="", key="y", value="z",
        )
    with pytest.raises(ValueError, match="category"):
        await store.upsert(
            user_id="u1", category="x", key="", value="z",
        )


# ---------------------------------------------------------------------------
# Multi-tenancy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_is_user_scoped():
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="language", key="x", value="u1-value",
    )
    # u2 cannot see u1's entry on the same logical key
    assert await store.get(
        user_id="u2", category="language", key="x",
    ) is None
    # u1 still sees their own
    e = await store.get(user_id="u1", category="language", key="x")
    assert e is not None and e.value == "u1-value"


@pytest.mark.asyncio
async def test_query_global_does_not_leak_across_users():
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="tool", key="t", value="u1",
    )
    await store.upsert(
        user_id="u2", category="tool", key="t", value="u2",
    )
    u1_entries = await store.query_global(user_id="u1")
    u2_entries = await store.query_global(user_id="u2")
    assert len(u1_entries) == 1
    assert len(u2_entries) == 1
    assert u1_entries[0].value == "u1"
    assert u2_entries[0].value == "u2"


# ---------------------------------------------------------------------------
# Workspace scoping + merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_local_overrides_global():
    """When both global + workspace-local entries exist for the same
    (category, key), query_for_workspace returns the workspace-local one."""
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="convention", key="naming",
        value="snake_case",  # global
    )
    await store.upsert(
        user_id="u1", workspace_id="ws-A",
        category="convention", key="naming",
        value="kebab-case",  # workspace-local override
    )
    merged = await store.query_for_workspace(
        user_id="u1", workspace_id="ws-A",
    )
    assert len(merged) == 1
    assert merged[0].value == "kebab-case"
    assert merged[0].workspace_id == "ws-A"


@pytest.mark.asyncio
async def test_workspace_local_falls_back_to_global():
    """When NO workspace-local entry exists, query_for_workspace
    returns the global one as fallback."""
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="convention", key="naming",
        value="snake_case",  # only global
    )
    merged = await store.query_for_workspace(
        user_id="u1", workspace_id="ws-B",
    )
    assert len(merged) == 1
    assert merged[0].value == "snake_case"
    assert merged[0].workspace_id == ""


@pytest.mark.asyncio
async def test_query_global_excludes_workspace_local():
    """query_global returns ONLY workspace_id='' rows."""
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="tool", key="t", value="global",
    )
    await store.upsert(
        user_id="u1", workspace_id="ws-X",
        category="tool", key="t", value="ws-local",
    )
    globals_ = await store.query_global(user_id="u1")
    assert len(globals_) == 1
    assert globals_[0].value == "global"


@pytest.mark.asyncio
async def test_query_workspace_only_excludes_global():
    """query_workspace_only is the strict per-workspace view (no merge)."""
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="tool", key="t", value="global",
    )
    await store.upsert(
        user_id="u1", workspace_id="ws-X",
        category="tool", key="t", value="ws-X-only",
    )
    only = await store.query_workspace_only(
        user_id="u1", workspace_id="ws-X",
    )
    assert len(only) == 1
    assert only[0].value == "ws-X-only"


@pytest.mark.asyncio
async def test_workspace_id_none_is_global():
    """``None`` and ``""`` for workspace_id behave identically (global)."""
    store, _ = await _mkstore()
    e1 = await store.upsert(
        user_id="u1", workspace_id=None,
        category="x", key="k", value="v",
    )
    assert e1.workspace_id == ""
    # Re-upsert with explicit "" should hit the same row
    e2 = await store.upsert(
        user_id="u1", workspace_id="",
        category="x", key="k", value="v2",
    )
    assert e2.observation_count == 2


@pytest.mark.asyncio
async def test_query_for_workspace_filters_by_category():
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", workspace_id="ws-1",
        category="language", key="a", value="A",
    )
    await store.upsert(
        user_id="u1", workspace_id="ws-1",
        category="tool", key="b", value="B",
    )
    lang = await store.query_for_workspace(
        user_id="u1", workspace_id="ws-1", category="language",
    )
    tool = await store.query_for_workspace(
        user_id="u1", workspace_id="ws-1", category="tool",
    )
    assert len(lang) == 1 and lang[0].key == "a"
    assert len(tool) == 1 and tool[0].key == "b"


@pytest.mark.asyncio
async def test_query_for_workspace_requires_workspace_id():
    store, _ = await _mkstore()
    with pytest.raises(ValueError, match="workspace_id"):
        await store.query_for_workspace(user_id="u1", workspace_id="")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_only_target_row():
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="x", key="a", value="A",
    )
    await store.upsert(
        user_id="u1", category="x", key="b", value="B",
    )
    deleted = await store.delete(user_id="u1", category="x", key="a")
    assert deleted is True
    # Only "a" gone; "b" remains
    assert await store.get(user_id="u1", category="x", key="a") is None
    b = await store.get(user_id="u1", category="x", key="b")
    assert b is not None and b.value == "B"


@pytest.mark.asyncio
async def test_delete_returns_false_for_missing():
    store, _ = await _mkstore()
    deleted = await store.delete(
        user_id="u1", category="x", key="nonexistent",
    )
    assert deleted is False


@pytest.mark.asyncio
async def test_delete_respects_workspace_scope():
    """Deleting a global entry must NOT touch workspace-local rows on
    the same logical key (and vice versa)."""
    store, _ = await _mkstore()
    await store.upsert(
        user_id="u1", category="x", key="k", value="global",
    )
    await store.upsert(
        user_id="u1", workspace_id="ws-1",
        category="x", key="k", value="ws-1",
    )
    deleted = await store.delete(user_id="u1", category="x", key="k")  # global
    assert deleted is True
    # Workspace-local entry survives
    ws_entries = await store.query_workspace_only(
        user_id="u1", workspace_id="ws-1",
    )
    assert len(ws_entries) == 1
    assert ws_entries[0].value == "ws-1"
