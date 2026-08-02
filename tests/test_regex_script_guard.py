"""Store-level round-trip for the regex_scripts stale-write guard.

The guard unit tests (tests/test_write_guard.py) prove the comparison
logic against a synthetic table. This file proves the part that actually
breaks in practice: that ``RegexScriptStore`` really PERSISTS the client
stamp and really READS it back. A guard whose stamp never reaches the
column is a guard that silently never fires, which is worse than no guard
at all because it looks correct in review.
"""

from __future__ import annotations

import aiosqlite
import pytest

from augmentum.modes.narrative.regex_transformer import (
    RegexScript,
    RegexScriptStore,
)
from augmentum.state.write_guard import incoming_stamp, is_stale

# Mirrors migration 018 + user_id (093) + client_updated_at (325).
_SCHEMA = """
CREATE TABLE regex_scripts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    find_regex TEXT NOT NULL,
    replace_string TEXT NOT NULL DEFAULT '',
    placement TEXT NOT NULL DEFAULT 'output'
        CHECK(placement IN ('input', 'output', 'both')),
    enabled INTEGER NOT NULL DEFAULT 1,
    order_num INTEGER NOT NULL DEFAULT 100,
    character_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT,
    client_updated_at INTEGER NOT NULL DEFAULT 0
);
"""


@pytest.fixture
async def store():
    db = await aiosqlite.connect(":memory:")
    await db.executescript(_SCHEMA)
    await db.commit()
    yield RegexScriptStore(db), db
    await db.close()


async def test_stamp_survives_save_and_load(store):
    """The round-trip the guard depends on."""
    s, _db = store
    await s.save_script(
        RegexScript(id="r1", name="Trim", client_updated_at=1234),
        user_id="u1",
    )
    scripts = await s.list_scripts(user_id="u1")
    assert [x.client_updated_at for x in scripts] == [1234]


async def test_guard_sees_the_stored_stamp(store):
    """End-to-end: a client holding an older base is rejected, and one
    holding the current base is not."""
    s, db = store
    await s.save_script(
        RegexScript(id="r1", name="Trim", client_updated_at=1000),
        user_id="u1",
    )
    assert await is_stale(db, "regex_scripts", "r1", 500, user_id="u1")
    assert not await is_stale(db, "regex_scripts", "r1", 1000, user_id="u1")


async def test_resave_advances_the_stamp(store):
    """A second edit must move the stamp forward, or a third client
    holding the FIRST base would look current and clobber silently."""
    s, db = store
    await s.save_script(
        RegexScript(id="r1", name="Trim", client_updated_at=1000),
        user_id="u1",
    )
    await s.save_script(
        RegexScript(id="r1", name="Trim v2", client_updated_at=2000),
        user_id="u1",
    )
    assert await is_stale(db, "regex_scripts", "r1", 1000, user_id="u1")


async def test_unstamped_legacy_client_still_saves(store):
    """Rolling this out must not break an older tab mid-session."""
    s, db = store
    await s.save_script(RegexScript(id="r1", name="Trim"), user_id="u1")
    assert not await is_stale(db, "regex_scripts", "r1", 0, user_id="u1")


async def test_list_payload_feeds_the_next_guard_call(store):
    """The value the route serialises as ``clientUpdatedAt`` must be the
    same one ``incoming_stamp`` reads back off the next request body."""
    s, db = store
    await s.save_script(
        RegexScript(id="r1", name="Trim", client_updated_at=1000),
        user_id="u1",
    )
    loaded = (await s.list_scripts(user_id="u1"))[0]
    body = {"id": "r1", "baseUpdatedAt": loaded.client_updated_at}
    assert not await is_stale(
        db, "regex_scripts", "r1", incoming_stamp(body), user_id="u1",
    )


async def test_other_tenant_cannot_be_read_as_a_conflict(store):
    s, db = store
    await s.save_script(
        RegexScript(id="r1", name="Theirs", client_updated_at=9999),
        user_id="someone-else",
    )
    assert not await is_stale(db, "regex_scripts", "r1", 1, user_id="u1")
