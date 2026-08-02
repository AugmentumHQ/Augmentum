"""CollectionStore + ActivityStore + home payload tests.

Pure in-memory aiosqlite. Schema duplicated from migration 235 to keep
the test self-contained; the real migration is the source of truth.
"""
from __future__ import annotations

import json

import aiosqlite
import pytest

from augmentum.library.activity import ActivityStore
from augmentum.library.collections import CollectionStore, SlugCollision
from augmentum.library.home import build_home_payload

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    task_id         TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    filename        TEXT NOT NULL,
    display_name    TEXT NOT NULL DEFAULT '',
    format          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    path            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    metadata        TEXT NOT NULL DEFAULT '{}',
    pinned          INTEGER NOT NULL DEFAULT 0,
    last_opened_at  TEXT DEFAULT NULL,
    transient       INTEGER NOT NULL DEFAULT 0,
    tags            TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS library_collections (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'manual',
    filter_json     TEXT NOT NULL DEFAULT '{}',
    cover_url       TEXT NOT NULL DEFAULT '',
    accent_color    TEXT NOT NULL DEFAULT '',
    view_mode       TEXT NOT NULL DEFAULT 'list',
    sort_order      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_library_collections_slug
    ON library_collections(user_id, slug);

-- No artifacts(id) FK on artifact_id since migration 309 (union ids).
CREATE TABLE IF NOT EXISTS library_collection_items (
    collection_id   TEXT NOT NULL REFERENCES library_collections(id) ON DELETE CASCADE,
    artifact_id     TEXT NOT NULL,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    added_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (collection_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS library_activity (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    artifact_id     TEXT NOT NULL,
    action          TEXT NOT NULL,
    surface         TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL DEFAULT '{}',
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ``build_home_payload`` UNIONs ``library_publications.kind`` into the
-- type_counts query (so coder saves show up in the Games bucket). The
-- table has to exist for that subquery to plan; we don't seed any rows
-- in this file's tests since they target collections+home, not the
-- save-to-library merge — coverage for that lives in test_library_routes.py.
CREATE TABLE IF NOT EXISTS library_publications (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id        TEXT NOT NULL DEFAULT '',
    kind                TEXT NOT NULL,
    title               TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    screenshot_path     TEXT NOT NULL DEFAULT '',
    entry_point         TEXT NOT NULL,
    storage_path        TEXT NOT NULL,
    storage_kind        TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL DEFAULT 0,
    version             INTEGER NOT NULL DEFAULT 1,
    shared              INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    last_launched_at    REAL,
    launch_count        INTEGER NOT NULL DEFAULT 0,
    pinned              INTEGER NOT NULL DEFAULT 0,
    tags                TEXT NOT NULL DEFAULT '[]'
);
"""


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
async def setup():
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.executescript(_SCHEMA_SQL)
        await conn.execute("INSERT INTO users (id, username) VALUES (?, ?)", ("u1", "alice"))
        await conn.execute("INSERT INTO users (id, username) VALUES (?, ?)", ("u2", "bob"))
        await conn.commit()
        yield conn
    finally:
        await conn.close()


async def _mkartifact(
    conn: aiosqlite.Connection,
    aid: str,
    *,
    user_id: str = "u1",
    fmt: str = "html",
    pinned: int = 0,
    tags: list[str] | None = None,
) -> None:
    await conn.execute(
        "INSERT INTO artifacts (id, user_id, filename, display_name, format, "
        "pinned, tags, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (aid, user_id, f"{aid}.html", aid.title(), fmt, pinned,
         json.dumps(tags or [])),
    )
    await conn.commit()


# ── CollectionStore: create / read ────────────────────────────────────


@pytest.mark.asyncio
async def test_create_manual_collection(setup):
    cs = CollectionStore(setup)
    col = await cs.create(user_id="u1", name="Tower Defense Stuff")
    assert col["id"].startswith("col_")
    assert col["name"] == "Tower Defense Stuff"
    assert col["slug"] == "tower-defense-stuff"
    assert col["kind"] == "manual"
    assert col["view_mode"] == "list"
    assert col["sort_order"] == 0


@pytest.mark.asyncio
async def test_slug_collision_per_user(setup):
    cs = CollectionStore(setup)
    await cs.create(user_id="u1", name="Games")
    with pytest.raises(SlugCollision):
        await cs.create(user_id="u1", name="Games")
    # Different user: no collision.
    col = await cs.create(user_id="u2", name="Games")
    assert col["slug"] == "games"


@pytest.mark.asyncio
async def test_list_for_user_ordered(setup):
    cs = CollectionStore(setup)
    await cs.create(user_id="u1", name="A")
    await cs.create(user_id="u1", name="B")
    await cs.create(user_id="u1", name="C")
    rows = await cs.list_for_user(user_id="u1")
    assert [r["name"] for r in rows] == ["A", "B", "C"]
    assert [r["sort_order"] for r in rows] == [0, 1, 2]


@pytest.mark.asyncio
async def test_get_returns_none_for_other_tenant(setup):
    cs = CollectionStore(setup)
    col = await cs.create(user_id="u1", name="Private")
    assert await cs.get(col["id"], user_id="u2") is None


# ── CollectionStore: manual items ─────────────────────────────────────


@pytest.mark.asyncio
async def test_add_items_idempotent(setup):
    cs = CollectionStore(setup)
    col = await cs.create(user_id="u1", name="Stuff")
    await _mkartifact(setup, "a1")
    await _mkartifact(setup, "a2")

    added = await cs.add_items(col["id"], user_id="u1", artifact_ids=["a1", "a2"])
    assert added == 2

    again = await cs.add_items(col["id"], user_id="u1", artifact_ids=["a1", "a2"])
    assert again == 0  # INSERT OR IGNORE

    items = await cs.list_items(col["id"], user_id="u1")
    assert items == ["a1", "a2"]


@pytest.mark.asyncio
async def test_add_items_rejects_cross_tenant_artifacts(setup):
    cs = CollectionStore(setup)
    col = await cs.create(user_id="u1", name="Mine")
    await _mkartifact(setup, "a-other", user_id="u2")
    # u1 cannot add u2's artifact even by knowing its id.
    added = await cs.add_items(col["id"], user_id="u1", artifact_ids=["a-other"])
    assert added == 0
    assert await cs.list_items(col["id"], user_id="u1") == []


@pytest.mark.asyncio
async def test_add_items_rejected_for_dynamic_collection(setup):
    cs = CollectionStore(setup)
    col = await cs.create(
        user_id="u1", name="Pinned games",
        kind="dynamic", filter_json={"pinned_only": True, "types": ["game"]},
    )
    await _mkartifact(setup, "a1")
    with pytest.raises(ValueError):
        await cs.add_items(col["id"], user_id="u1", artifact_ids=["a1"])


@pytest.mark.asyncio
async def test_remove_item(setup):
    cs = CollectionStore(setup)
    col = await cs.create(user_id="u1", name="Stuff")
    await _mkartifact(setup, "a1")
    await cs.add_items(col["id"], user_id="u1", artifact_ids=["a1"])
    assert await cs.remove_item(col["id"], "a1", user_id="u1") is True
    assert await cs.list_items(col["id"], user_id="u1") == []


# ── CollectionStore: dynamic resolution ───────────────────────────────


@pytest.mark.asyncio
async def test_dynamic_filter_by_type(setup):
    cs = CollectionStore(setup)
    await _mkartifact(setup, "g1", fmt="game")
    await _mkartifact(setup, "g2", fmt="game")
    await _mkartifact(setup, "d1", fmt="pdf")

    col = await cs.create(
        user_id="u1", name="Games only",
        kind="dynamic", filter_json={"types": ["game"]},
    )
    items = await cs.resolve_dynamic(col["id"], user_id="u1")
    assert sorted(items) == ["g1", "g2"]


@pytest.mark.asyncio
async def test_dynamic_filter_pinned_only(setup):
    cs = CollectionStore(setup)
    await _mkartifact(setup, "p1", pinned=1)
    await _mkartifact(setup, "p2", pinned=1)
    await _mkartifact(setup, "u1art", pinned=0)

    col = await cs.create(
        user_id="u1", name="Pinned",
        kind="dynamic", filter_json={"pinned_only": True},
    )
    items = await cs.resolve_dynamic(col["id"], user_id="u1")
    assert sorted(items) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_dynamic_filter_tags_any_and_all(setup):
    cs = CollectionStore(setup)
    await _mkartifact(setup, "a1", tags=["roguelike", "indie"])
    await _mkartifact(setup, "a2", tags=["indie"])
    await _mkartifact(setup, "a3", tags=["puzzle"])

    any_col = await cs.create(
        user_id="u1", name="Any-match",
        kind="dynamic", filter_json={"tags_any": ["puzzle", "roguelike"]},
    )
    items = await cs.resolve_dynamic(any_col["id"], user_id="u1")
    assert sorted(items) == ["a1", "a3"]

    all_col = await cs.create(
        user_id="u1", name="All-match",
        kind="dynamic", slug="all-match",
        filter_json={"tags_all": ["roguelike", "indie"]},
    )
    items_all = await cs.resolve_dynamic(all_col["id"], user_id="u1")
    assert items_all == ["a1"]


@pytest.mark.asyncio
async def test_dynamic_empty_filter_returns_empty(setup):
    """Sanity guard: a misconfigured (empty) dynamic collection MUST NOT
    return every artifact."""
    cs = CollectionStore(setup)
    await _mkartifact(setup, "a1")
    col = await cs.create(
        user_id="u1", name="Empty rules",
        kind="dynamic", filter_json={},
    )
    assert await cs.resolve_dynamic(col["id"], user_id="u1") == []


# ── ActivityStore ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_and_list_activity(setup):
    acts = ActivityStore(setup)
    await _mkartifact(setup, "a1")

    await acts.record(user_id="u1", artifact_id="a1", action="open")
    await acts.record(
        user_id="u1", artifact_id="a1", action="cast", surface="tv",
        payload={"receiver_id": "r1"},
    )
    events = await acts.list_for_artifact("a1", user_id="u1")
    assert len(events) == 2
    assert events[0]["action"] == "cast"  # DESC ordering
    assert events[0]["payload"] == {"receiver_id": "r1"}
    assert events[1]["action"] == "open"


@pytest.mark.asyncio
async def test_record_rejects_cross_tenant_artifact(setup):
    acts = ActivityStore(setup)
    await _mkartifact(setup, "a-of-u1", user_id="u1")
    with pytest.raises(PermissionError):
        await acts.record(user_id="u2", artifact_id="a-of-u1", action="open")


@pytest.mark.asyncio
async def test_recent_artifact_ids_collapses_repeats(setup):
    acts = ActivityStore(setup)
    await _mkartifact(setup, "a1")
    await _mkartifact(setup, "a2")
    await _mkartifact(setup, "a3")

    # a1 first, a2 second, a1 again, then a3 - newest distinct wins.
    await acts.record(user_id="u1", artifact_id="a1", action="open")
    await acts.record(user_id="u1", artifact_id="a2", action="open")
    await acts.record(user_id="u1", artifact_id="a1", action="open")
    await acts.record(user_id="u1", artifact_id="a3", action="cast")

    recent = await acts.recent_artifact_ids(user_id="u1")
    # a3 most recent, a1 next (second open is its latest), a2 last
    assert recent == ["a3", "a1", "a2"]


# ── Home payload ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_home_payload_assembles_sections(setup):
    cs = CollectionStore(setup)
    acts = ActivityStore(setup)
    await _mkartifact(setup, "p1", pinned=1)
    await _mkartifact(setup, "r1")
    await acts.record(user_id="u1", artifact_id="r1", action="open")
    col = await cs.create(user_id="u1", name="Games")

    payload = await build_home_payload(setup, user_id="u1")
    assert [a["id"] for a in payload["pinned"]] == ["p1"]
    assert [a["id"] for a in payload["recent"]] == ["r1"]
    assert [c["name"] for c in payload["collections_summary"]] == ["Games"]
    assert payload["collections_summary"][0]["count"] == 0
    # type_counts: 2 html artifacts (p1, r1), total 2.
    assert payload["type_counts"] == {"html": 2}
    assert payload["total_count"] == 2


@pytest.mark.asyncio
async def test_home_payload_isolates_tenants(setup):
    """u1 pins something, u2 should see nothing."""
    await _mkartifact(setup, "p1", user_id="u1", pinned=1)
    u1_payload = await build_home_payload(setup, user_id="u1")
    u2_payload = await build_home_payload(setup, user_id="u2")
    assert [a["id"] for a in u1_payload["pinned"]] == ["p1"]
    assert u2_payload["pinned"] == []


@pytest.mark.asyncio
async def test_home_payload_continue_excludes_done_tag(setup):
    acts = ActivityStore(setup)
    await _mkartifact(setup, "live", tags=["wip"])
    await _mkartifact(setup, "shipped", tags=["done"])
    await acts.record(user_id="u1", artifact_id="live", action="open")
    await acts.record(user_id="u1", artifact_id="shipped", action="open")

    cont = await acts.continue_artifact_ids(user_id="u1")
    assert cont == ["live"]
