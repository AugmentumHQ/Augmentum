"""Tests for the native dot-named lorebook tools — lorebook.check / lorebook.create.

These are the F1/F5 tools the companion training rows emit. They MUST:

* exist and dispatch (check is read-only; create records session lore)
* tag created entries source="narrative_established" + carry branch_id
* refuse to author lore with no user_id (never write the anon row)
* tolerate the underscore-sanitized name spelling the model often returns
* round-trip through the user-scoped persistence layer (migration 304
  branch_id column applied), staying user-scoped on save AND load

Run: python -m pytest tests/test_narrative_lorebook_native.py -v
"""
from __future__ import annotations

import pytest

from augmentum.modes.narrative.lore_engine import LoreEngine
from augmentum.modes.narrative.lorebook_native_schemas import (
    LOREBOOK_NATIVE_TOOL_NAMES,
    NARRATIVE_SOURCE,
    _canonical_name,
    dispatch_lorebook_native_tool,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.state.narrative_state import NarrativeSessionState

_UID = "u-test"
_OTHER_UID = "u-other"
_SID = "s-lore"


# ---------------------------------------------------------------------------
# Dispatcher behaviour (no DB)
# ---------------------------------------------------------------------------


def test_native_names_include_both_spellings():
    # Dotted canonical + underscore-sanitized form both register so the
    # recall loop matches whichever the model returns.
    assert "lorebook.check" in LOREBOOK_NATIVE_TOOL_NAMES
    assert "lorebook_check" in LOREBOOK_NATIVE_TOOL_NAMES
    assert "lorebook.create" in LOREBOOK_NATIVE_TOOL_NAMES
    assert "lorebook_create" in LOREBOOK_NATIVE_TOOL_NAMES
    assert _canonical_name("lorebook_check") == "lorebook.check"
    assert _canonical_name("lorebook.create") == "lorebook.create"


def test_create_records_session_lore_with_branch_and_source():
    le = LoreEngine()
    txt, mutations = dispatch_lorebook_native_tool(
        le, _SID, user_id=_UID, branch_id="branch-x",
        tool_name="lorebook.create",
        raw_arguments={
            "keywords": ["Ashwander", "river"],
            "content": "The main river through the valley.",
            "category": "location",
        },
    )
    assert "Recorded session lore" in txt
    assert mutations and mutations[0]["action"] == "create"

    entries = list(le.entries.values())
    assert len(entries) == 1
    e = entries[0]
    assert e.source == NARRATIVE_SOURCE
    assert e.branch_id == "branch-x"
    assert e.keywords == ["Ashwander", "river"]
    assert "[location]" in e.comment


def test_create_rejects_bad_category():
    le = LoreEngine()
    txt, mutations = dispatch_lorebook_native_tool(
        le, _SID, user_id=_UID,
        tool_name="lorebook.create",
        raw_arguments={"keywords": ["x"], "content": "y", "category": "nonsense"},
    )
    assert "category" in txt.lower()
    assert mutations is None
    assert len(le.entries) == 0


def test_create_refuses_without_user_id():
    # Never author session lore with no owner — it would land in the anon row.
    le = LoreEngine()
    txt, mutations = dispatch_lorebook_native_tool(
        le, _SID, user_id="",
        tool_name="lorebook.create",
        raw_arguments={"keywords": ["x"], "content": "y"},
    )
    assert "user context" in txt.lower()
    assert mutations is None
    assert len(le.entries) == 0


def test_check_empty_is_not_an_error():
    le = LoreEngine()
    txt, mutations = dispatch_lorebook_native_tool(
        le, _SID, user_id=_UID,
        tool_name="lorebook.check",
        raw_arguments={"query": "anything"},
    )
    assert "No established lore" in txt
    assert mutations is None


def test_check_finds_created_entry_via_sanitized_name():
    le = LoreEngine()
    dispatch_lorebook_native_tool(
        le, _SID, user_id=_UID,
        tool_name="lorebook.create",
        raw_arguments={
            "keywords": ["Ashwander", "river"],
            "content": "The main river through the valley.",
        },
    )
    # Model returns the underscore-sanitized spelling — must still dispatch.
    txt, _ = dispatch_lorebook_native_tool(
        le, _SID, user_id=_UID,
        tool_name="lorebook_check",
        raw_arguments={"query": "river"},
    )
    assert "Ashwander" in txt
    assert "main river" in txt


# ---------------------------------------------------------------------------
# Persistence round-trip (real SQLite — exercises migration 304)
# ---------------------------------------------------------------------------


@pytest.fixture
async def backend():
    be = SQLiteBackend(":memory:")
    await be.connect()
    yield be
    await be.close()


@pytest.fixture
async def persist(backend):
    return NarrativePersistence(backend.conn)


@pytest.mark.asyncio
async def test_branch_id_column_exists(backend):
    # Migration 304 must have applied — branch_id is a real column.
    cursor = await backend.conn.execute("PRAGMA table_info(lorebook_entries)")
    cols = {dict(r)["name"] for r in await cursor.fetchall()}
    assert "branch_id" in cols


@pytest.mark.asyncio
async def test_created_lore_round_trips_user_scoped(persist):
    le = LoreEngine()
    dispatch_lorebook_native_tool(
        le, _SID, user_id=_UID, branch_id="main",
        tool_name="lorebook.create",
        raw_arguments={
            "keywords": ["Ashwander"],
            "content": "The river runs red since the battle.",
            "category": "location",
        },
    )
    # Persist via the user-scoped save path (mirrors the engine's persist).
    state = NarrativeSessionState(session_id=_SID, branch_id="main")
    state.lorebook = list(le.entries.values())
    await persist.save_session_state(_SID, state, user_id=_UID)

    # The owner sees it, with source + branch preserved.
    loaded = await persist._load_lorebook_entries(_SID, user_id=_UID)
    assert len(loaded) == 1
    assert loaded[0].source == NARRATIVE_SOURCE
    assert loaded[0].branch_id == "main"
    assert "runs red" in loaded[0].content

    # A different user sees nothing — no cross-tenant leak.
    other = await persist._load_lorebook_entries(_SID, user_id=_OTHER_UID)
    assert other == []
