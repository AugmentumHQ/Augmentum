"""Tests for the shared stale-write guard (augmentum/state/write_guard.py).

Covers both storage shapes (JSON-blob and column), both guard strengths
(base-stamp vs edit-stamp), and the fail-open contract.
"""

from __future__ import annotations

import json

import aiosqlite
import pytest

from augmentum.state.write_guard import (
    StampSource,
    edit_stamp,
    find_stale,
    incoming_stamp,
    is_stale,
    stale_payload,
    stored_stamps,
)


@pytest.fixture
async def conn():
    """In-memory DB with one JSON-blob table and one column table."""
    db = await aiosqlite.connect(":memory:")
    await db.execute(
        "CREATE TABLE ui_characters ("
        " id TEXT PRIMARY KEY, data TEXT NOT NULL, user_id TEXT)",
    )
    await db.execute(
        "CREATE TABLE prompt_presets ("
        " id TEXT PRIMARY KEY, name TEXT,"
        " client_updated_at INTEGER NOT NULL DEFAULT 0, user_id TEXT)",
    )
    await db.commit()
    yield db
    await db.close()


async def _put_card(db, cid: str, stamp: int, uid: str = "u1") -> None:
    """Cards stamp at ``clientUpdatedAt`` — NOT ``updatedAt``, which the
    characters GET overwrites with the server's ISO column."""
    await db.execute(
        "INSERT INTO ui_characters (id, data, user_id) VALUES (?, ?, ?)",
        (cid, json.dumps({"name": "Ada", "clientUpdatedAt": stamp}), uid),
    )
    await db.commit()


async def _put_preset(db, pid: str, stamp: int, uid: str = "u1") -> None:
    await db.execute(
        "INSERT INTO prompt_presets (id, name, client_updated_at, user_id) "
        "VALUES (?, ?, ?, ?)",
        (pid, "preset", stamp, uid),
    )
    await db.commit()


# ── incoming_stamp: which stamp wins ──────────────────────────────────────

def test_incoming_stamp_prefers_base_over_edit():
    """baseUpdatedAt is the strong signal and must win when both present."""
    assert incoming_stamp({"baseUpdatedAt": 100, "updatedAt": 999}) == 100


def test_incoming_stamp_falls_back_to_updated_at():
    assert incoming_stamp({"updatedAt": 500}) == 500


def test_incoming_stamp_zero_when_absent_or_junk():
    assert incoming_stamp({}) == 0
    assert incoming_stamp({"updatedAt": None}) == 0
    assert incoming_stamp({"updatedAt": ""}) == 0
    assert incoming_stamp({"updatedAt": "not-a-number"}) == 0


def test_incoming_stamp_skips_unparseable_base():
    """A junk base must not mask a usable edit stamp."""
    assert incoming_stamp({"baseUpdatedAt": "junk", "updatedAt": 42}) == 42


# ── edit_stamp: what gets PERSISTED (not what we guard against) ───────────

def test_edit_stamp_ignores_base():
    """Storing the base would pin the row in the past and make the next
    write look spuriously fresh — the guard would then never fire."""
    assert edit_stamp({"baseUpdatedAt": 100, "updatedAt": 999}) == 999


def test_edit_stamp_zero_when_absent():
    assert edit_stamp({}) == 0
    assert edit_stamp({"baseUpdatedAt": 100}) == 0


def test_edit_and_guard_stamps_diverge_on_the_same_body():
    """The two helpers must not be interchangeable — this is the whole
    reason both exist."""
    body = {"baseUpdatedAt": 100, "updatedAt": 999}
    assert incoming_stamp(body) != edit_stamp(body)


# ── JSON-blob shape ───────────────────────────────────────────────────────

async def test_json_shape_reads_stamp_from_blob(conn):
    await _put_card(conn, "c1", 1000)
    assert await stored_stamps(
        conn, "ui_characters", ["c1"], user_id="u1",
    ) == {"c1": 1000}


async def test_stale_when_stored_is_newer(conn):
    """The clobber case: client loaded at 500, someone else wrote 1000."""
    await _put_card(conn, "c1", 1000)
    assert await is_stale(conn, "ui_characters", "c1", 500, user_id="u1")


async def test_not_stale_when_client_has_current_base(conn):
    await _put_card(conn, "c1", 1000)
    assert not await is_stale(conn, "ui_characters", "c1", 1000, user_id="u1")


async def test_not_stale_when_client_is_ahead(conn):
    await _put_card(conn, "c1", 1000)
    assert not await is_stale(conn, "ui_characters", "c1", 1500, user_id="u1")


async def test_missing_row_is_never_stale(conn):
    """A first write has nothing to be stale against."""
    assert not await is_stale(conn, "ui_characters", "nope", 1, user_id="u1")


async def test_zero_stamp_is_never_stale(conn):
    """Legacy clients that send no stamp must keep saving."""
    await _put_card(conn, "c1", 9999)
    assert not await is_stale(conn, "ui_characters", "c1", 0, user_id="u1")


async def test_iso_date_stamp_reads_as_zero_not_a_crash(conn):
    """Regression: list_characters() overwrites the blob's ``updatedAt``
    with the server's ISO ``updated_at`` column. If the guard ever reads
    that key again it gets '2026-07-25T...' instead of a ms integer. That
    must degrade to 0 (unguarded) rather than raising — and it is WHY the
    card StampSource points at ``$.clientUpdatedAt`` instead."""
    await conn.execute(
        "INSERT INTO ui_characters (id, data, user_id) VALUES (?, ?, ?)",
        ("c1", json.dumps({"clientUpdatedAt": "2026-07-25T10:00:00Z"}), "u1"),
    )
    await conn.commit()
    assert await stored_stamps(
        conn, "ui_characters", ["c1"], user_id="u1",
    ) == {"c1": 0}
    assert not await is_stale(conn, "ui_characters", "c1", 500, user_id="u1")


# ── Column shape ──────────────────────────────────────────────────────────

async def test_column_shape_reads_stamp_from_column(conn):
    await _put_preset(conn, "p1", 700)
    assert await stored_stamps(
        conn, "prompt_presets", ["p1"], user_id="u1",
    ) == {"p1": 700}


async def test_column_shape_detects_stale(conn):
    await _put_preset(conn, "p1", 700)
    assert await is_stale(conn, "prompt_presets", "p1", 300, user_id="u1")
    assert not await is_stale(conn, "prompt_presets", "p1", 700, user_id="u1")


# ── Tenant scoping ────────────────────────────────────────────────────────

async def test_other_users_rows_are_invisible(conn):
    """Another tenant's row must not make our write look stale."""
    await _put_card(conn, "c1", 9999, uid="someone-else")
    assert await stored_stamps(
        conn, "ui_characters", ["c1"], user_id="u1",
    ) == {}
    assert not await is_stale(conn, "ui_characters", "c1", 1, user_id="u1")


async def test_legacy_null_user_rows_are_claimable(conn):
    """Pre-auth rows (NULL user_id) stay visible to the first owner."""
    await conn.execute(
        "INSERT INTO ui_characters (id, data, user_id) VALUES (?, ?, NULL)",
        ("c1", json.dumps({"clientUpdatedAt": 800})),
    )
    await conn.commit()
    assert await stored_stamps(
        conn, "ui_characters", ["c1"], user_id="u1",
    ) == {"c1": 800}


# ── Batch ─────────────────────────────────────────────────────────────────

async def test_find_stale_returns_only_the_stale_ids(conn):
    await _put_card(conn, "a", 1000)
    await _put_card(conn, "b", 1000)
    await _put_card(conn, "c", 1000)
    stale = await find_stale(
        conn, "ui_characters",
        {"a": 500, "b": 2000, "c": 0},   # stale, fresh, unstamped
        user_id="u1",
    )
    assert stale == ["a"]


async def test_find_stale_empty_input(conn):
    assert await find_stale(conn, "ui_characters", {}, user_id="u1") == []


# ── Fail-open contract ────────────────────────────────────────────────────

async def test_unregistered_table_fails_open(conn):
    """Never block a save because a table wasn't registered."""
    assert await stored_stamps(
        conn, "not_a_real_table", ["x"], user_id="u1",
    ) == {}


async def test_broken_query_fails_open(conn):
    """A missing column must not block saves — losing the guard is
    recoverable, refusing the user's write is not."""
    await conn.execute("DROP TABLE prompt_presets")
    await conn.commit()
    assert await stored_stamps(
        conn, "prompt_presets", ["p1"], user_id="u1",
    ) == {}


# ── Registry hygiene ──────────────────────────────────────────────────────

def test_stamp_source_rejects_ambiguous_config():
    with pytest.raises(ValueError):
        StampSource("t", json_column="data", stamp_column="client_updated_at")
    with pytest.raises(ValueError):
        StampSource("t")


def test_stale_payload_shape():
    """One wire contract across every guarded surface."""
    assert stale_payload("abc") == {"ok": False, "stale": True, "id": "abc"}
