"""End-to-end route tests for ``augmentum/proxy/library_routes.py``.

Uses a hand-rolled in-memory SQLite with just the Library-relevant
tables, so the test suite stays fast (the production SQLiteBackend
runs all 234+ migrations on connect, which would dominate runtime).
Conftest ``app`` fixture authenticates ``Bearer test-token`` as
``usr_test``.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import aiosqlite
import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"
OTHER_USER_ID = "usr_other"


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

-- artifact_id holds a UNION id (artifact OR pub_) since migration 309 —
-- no artifacts(id) FK, matching production so pub_ ids can join.
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

-- Save-to-Library publications. /api/library/items + home.py UNION
-- this into the same shape as ``artifacts`` so coder-saved games show
-- in the Library UI. Mirrors production migrations 197 + 309 (pinned/tags)
-- — fields kept in sync with augmentum/state/migrations/.
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


@pytest.fixture
def library_client(app):
    """Wire a minimal in-memory SQLite to ``app.state.state_manager.backend.conn``
    and seed two users + four artifacts (three for the test user, one for the
    other user so cross-tenant guards can be exercised)."""
    conn = _run(aiosqlite.connect(":memory:"))
    _run(conn.executescript(_SCHEMA_SQL))
    _run(conn.execute("INSERT INTO users (id, username) VALUES (?, ?)", (TEST_USER_ID, "alice")))
    _run(conn.execute("INSERT INTO users (id, username) VALUES (?, ?)", (OTHER_USER_ID, "bob")))
    for aid, owner, fmt, pinned, tags in [
        ("a-mine-1", TEST_USER_ID, "html", 0, []),
        ("a-mine-2", TEST_USER_ID, "game", 1, ["roguelike"]),
        ("a-mine-3", TEST_USER_ID, "pdf",  0, ["done"]),
        ("a-others",  OTHER_USER_ID, "html", 0, []),
    ]:
        _run(conn.execute(
            "INSERT INTO artifacts (id, user_id, filename, display_name, "
            "format, pinned, tags, path) VALUES (?, ?, ?, ?, ?, ?, ?, '')",
            (aid, owner, f"{aid}.html", aid, fmt, pinned, json.dumps(tags)),
        ))
    _run(conn.commit())

    # The route reads request.app.state.state_manager.backend.conn — we
    # don't need a real StateManager for this surface, only that path.
    app.state.state_manager = SimpleNamespace(backend=SimpleNamespace(conn=conn))

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})

    yield tc, conn
    _run(conn.close())


# ── /api/library/home ────────────────────────────────────────────────


def test_home_returns_sections(library_client):
    tc, _ = library_client
    r = tc.get("/api/library/home")
    assert r.status_code == 200
    body = r.json()
    assert {"pinned", "recent", "continue", "collections_summary"} <= set(body)
    pinned_ids = {a["id"] for a in body["pinned"]}
    assert "a-mine-2" in pinned_ids
    assert "a-others" not in pinned_ids


def test_home_requires_auth(library_client):
    tc, _ = library_client
    tc.headers.pop("Authorization", None)
    r = tc.get("/api/library/home")
    assert r.status_code == 401


# ── /api/library/items ───────────────────────────────────────────────


def test_items_default_returns_user_artifacts_only(library_client):
    tc, _ = library_client
    body = tc.get("/api/library/items").json()
    ids = [it["id"] for it in body["items"]]
    assert sorted(ids) == ["a-mine-1", "a-mine-2", "a-mine-3"]
    assert "a-others" not in ids
    assert body["total"] == 3
    assert body["has_more"] is False


def test_items_filter_by_type(library_client):
    tc, _ = library_client
    body = tc.get("/api/library/items?types=game").json()
    assert [it["id"] for it in body["items"]] == ["a-mine-2"]


def test_items_filter_by_pinned(library_client):
    tc, _ = library_client
    body = tc.get("/api/library/items?pinned=1").json()
    assert [it["id"] for it in body["items"]] == ["a-mine-2"]


def test_items_search_matches_name_or_tag(library_client):
    tc, _ = library_client
    by_name = tc.get("/api/library/items?q=mine-1").json()
    assert [it["id"] for it in by_name["items"]] == ["a-mine-1"]

    by_tag = tc.get("/api/library/items?q=roguelike").json()
    assert [it["id"] for it in by_tag["items"]] == ["a-mine-2"]


def test_items_sort_name_ascending(library_client):
    tc, _ = library_client
    body = tc.get("/api/library/items?sort=name").json()
    names = [it["display_name"] for it in body["items"]]
    assert names == sorted(names, key=str.lower)


def test_items_pagination_has_more(library_client):
    tc, _ = library_client
    body = tc.get("/api/library/items?limit=2&offset=0").json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["has_more"] is True

    rest = tc.get("/api/library/items?limit=2&offset=2").json()
    assert len(rest["items"]) == 1
    assert rest["has_more"] is False


def test_items_invalid_sort_falls_back_to_recent(library_client):
    """Unknown sort key must not be reflected into SQL — falls back to
    'recent' rather than 400'ing so the URL is forgiving."""
    tc, _ = library_client
    body = tc.get("/api/library/items?sort=DROP%20TABLE").json()
    assert body["total"] == 3  # query still executed, no injection


# ── Collections CRUD ─────────────────────────────────────────────────


def test_create_then_list_collection(library_client):
    tc, _ = library_client
    r = tc.post(
        "/api/library/collections",
        json={"name": "Roguelikes", "view_mode": "grid"},
    )
    assert r.status_code == 201, r.text
    col = r.json()
    assert col["name"] == "Roguelikes"
    assert col["slug"] == "roguelikes"

    listing = tc.get("/api/library/collections").json()
    assert any(c["id"] == col["id"] for c in listing["collections"])


def test_create_collection_slug_collision_409(library_client):
    tc, _ = library_client
    tc.post("/api/library/collections", json={"name": "Games"})
    r = tc.post("/api/library/collections", json={"name": "Games"})
    assert r.status_code == 409


def test_get_collection_includes_items(library_client):
    tc, _ = library_client
    col = tc.post(
        "/api/library/collections", json={"name": "Mine"},
    ).json()
    r = tc.post(
        f"/api/library/collections/{col['id']}/items",
        json={"artifact_ids": ["a-mine-1", "a-mine-2"]},
    )
    assert r.status_code == 200
    assert r.json()["added"] == 2

    body = tc.get(f"/api/library/collections/{col['id']}").json()
    assert [it["id"] for it in body["items"]] == ["a-mine-1", "a-mine-2"]


def test_collection_cross_tenant_isolation(library_client):
    """A collection created by usr_other must 404 for usr_test."""
    tc, conn = library_client
    _run(conn.execute(
        "INSERT INTO library_collections (id, user_id, name, slug) "
        "VALUES (?, ?, ?, ?)",
        ("col_other", OTHER_USER_ID, "Hers", "hers"),
    ))
    _run(conn.commit())
    r = tc.get("/api/library/collections/col_other")
    assert r.status_code == 404


def test_add_items_rejects_cross_tenant_artifact(library_client):
    tc, _ = library_client
    col = tc.post("/api/library/collections", json={"name": "Mine"}).json()
    r = tc.post(
        f"/api/library/collections/{col['id']}/items",
        json={"artifact_ids": ["a-others", "a-mine-1"]},
    )
    assert r.status_code == 200
    assert r.json()["added"] == 1

    body = tc.get(f"/api/library/collections/{col['id']}").json()
    assert [it["id"] for it in body["items"]] == ["a-mine-1"]


def test_dynamic_collection_resolves_at_get(library_client):
    tc, _ = library_client
    col = tc.post(
        "/api/library/collections",
        json={
            "name": "Pinned games",
            "kind": "dynamic",
            "filter_json": {"pinned_only": True, "types": ["game"]},
        },
    ).json()
    body = tc.get(f"/api/library/collections/{col['id']}").json()
    assert [it["id"] for it in body["items"]] == ["a-mine-2"]


def test_delete_collection(library_client):
    tc, _ = library_client
    col = tc.post("/api/library/collections", json={"name": "Tmp"}).json()
    r = tc.delete(f"/api/library/collections/{col['id']}")
    assert r.status_code == 200
    assert tc.get(f"/api/library/collections/{col['id']}").status_code == 404


# ── Pin / activity / tags ────────────────────────────────────────────


def test_set_pin_and_unpin(library_client):
    tc, _ = library_client
    r = tc.post("/api/library/items/a-mine-1/pin", json={"pinned": True})
    assert r.status_code == 200
    assert r.json()["pinned"] is True

    r = tc.post("/api/library/items/a-mine-1/pin", json={"pinned": False})
    assert r.json()["pinned"] is False


def test_pin_404_for_cross_tenant(library_client):
    tc, _ = library_client
    r = tc.post("/api/library/items/a-others/pin", json={"pinned": True})
    assert r.status_code == 404


def test_activity_record_and_list(library_client):
    tc, _ = library_client
    tc.post(
        "/api/library/items/a-mine-1/activity",
        json={"action": "open", "surface": "desktop"},
    )
    tc.post(
        "/api/library/items/a-mine-1/activity",
        json={"action": "cast", "surface": "tv",
              "payload": {"receiver_id": "r1"}},
    )
    body = tc.get("/api/library/items/a-mine-1/activity").json()
    assert len(body["events"]) == 2
    assert body["events"][0]["action"] == "cast"
    assert body["events"][0]["payload"]["receiver_id"] == "r1"


def test_activity_cross_tenant_returns_404(library_client):
    tc, _ = library_client
    r = tc.post(
        "/api/library/items/a-others/activity",
        json={"action": "open"},
    )
    assert r.status_code == 404


def test_set_tags_normalises(library_client):
    tc, _ = library_client
    r = tc.put(
        "/api/library/items/a-mine-1/tags",
        json={"tags": ["  fun  ", "fun", "rare", ""]},
    )
    assert r.status_code == 200
    # Whitespace trimmed, duplicates collapsed, empties dropped, order preserved.
    assert r.json()["tags"] == ["fun", "rare"]


# ── Save-to-Library publication merge into /api/library/items ────────
#
# Regression coverage for the structural gap where coder-saved games
# landed in ``library_publications`` but library2 only read from
# ``artifacts`` — so saves were invisible in the Library UI even though
# the row was persisted. The fix UNION ALLs publications into the
# items query under the artifact column shape (see ``_ITEMS_UNION_SQL``
# in ``proxy/library_routes.py``).


def _seed_publication(
    conn,
    *,
    pub_id: str,
    user_id: str,
    title: str,
    kind: str = "game",
    entry_point: str = "index.html",
    size_bytes: int = 1024,
    created_at: float = 1_780_000_000.0,
    last_launched_at: float | None = None,
) -> None:
    _run(conn.execute(
        "INSERT INTO library_publications ("
        "  id, user_id, workspace_id, kind, title, description, "
        "  screenshot_path, entry_point, storage_path, storage_kind, "
        "  size_bytes, version, shared, created_at, updated_at, "
        "  last_launched_at, launch_count"
        ") VALUES (?, ?, '', ?, ?, '', '', ?, '/tmp/x', 'bundle', ?, "
        "1, 0, ?, ?, ?, 0)",
        (pub_id, user_id, kind, title, entry_point, size_bytes,
         created_at, created_at, last_launched_at),
    ))
    _run(conn.commit())


def test_items_includes_publications(library_client):
    """Save-to-library publications appear in /api/library/items.

    Direct regression: pre-fix, list_items only queried ``artifacts``
    and the saved publication was invisible despite persisting.
    """
    tc, conn = library_client
    _seed_publication(
        conn, pub_id="pub_aaa", user_id=TEST_USER_ID, title="Roguelike",
    )

    body = tc.get("/api/library/items").json()
    ids = {it["id"] for it in body["items"]}
    assert "pub_aaa" in ids, f"publication missing from items: {ids}"
    assert body["total"] == 4  # 3 artifacts + 1 publication


def test_items_publication_projected_into_artifact_shape(library_client):
    """Publication fields map cleanly to the artifact column shape:
    title→display_name, entry_point→filename, kind→format, pinned→0,
    tags→[]. The UI consumes these without knowing two tables exist.
    """
    tc, conn = library_client
    _seed_publication(
        conn, pub_id="pub_bbb", user_id=TEST_USER_ID,
        title="My Roguelike", kind="game", entry_point="game.html",
        size_bytes=4096,
    )

    body = tc.get("/api/library/items?q=Roguelike").json()
    pub = next(it for it in body["items"] if it["id"] == "pub_bbb")
    assert pub["display_name"] == "My Roguelike"
    assert pub["filename"] == "game.html"
    assert pub["format"] == "game"
    assert pub["size_bytes"] == 4096
    assert pub["pinned"] == 0  # publications can't be pinned in v1
    assert pub["tags"] == []   # publications don't carry tags


def test_items_publications_cross_tenant_isolation(library_client):
    """A publication owned by another user must not appear in this
    user's listing. Same guarantee as artifacts — the WHERE inside
    the UNION enforces it on the publications side.
    """
    tc, conn = library_client
    _seed_publication(
        conn, pub_id="pub_mine", user_id=TEST_USER_ID, title="Mine",
    )
    _seed_publication(
        conn, pub_id="pub_theirs", user_id=OTHER_USER_ID, title="Theirs",
    )

    body = tc.get("/api/library/items").json()
    ids = {it["id"] for it in body["items"]}
    assert "pub_mine" in ids
    assert "pub_theirs" not in ids


def test_items_pinned_filter_excludes_publications(library_client):
    """Pinned filter is artifact-only by construction — publications
    project ``pinned = 0`` constant, so ``pinned = 1`` excludes them.
    Documents the v1 limit; we'll lift it if publications get a
    user_publications_pinned table later.
    """
    tc, conn = library_client
    _seed_publication(
        conn, pub_id="pub_unpinned", user_id=TEST_USER_ID,
        title="Cannot Pin",
    )

    body = tc.get("/api/library/items?pinned=1").json()
    ids = [it["id"] for it in body["items"]]
    # Only the pinned artifact, no publications.
    assert ids == ["a-mine-2"]


def test_items_recent_sort_orders_publications_chronologically(library_client):
    """A publication created NOW must sort above artifacts that are
    older. Regression for the mixed-type sort gotcha — artifacts use
    TEXT ``datetime('now')`` while publications use REAL epoch
    seconds. Without the ``strftime(... 'unixepoch')`` cast on the
    publication side, SQLite's type affinity broke ordering (REAL
    1_780_000_000 < TEXT '2026' under numeric coercion).
    """
    tc, conn = library_client
    # Force the artifacts' created_at to a clearly-OLDER value so the
    # publication should win on a recent-DESC sort.
    _run(conn.execute(
        "UPDATE artifacts SET created_at = '1970-01-01 00:00:00' "
        "WHERE user_id = ?",
        (TEST_USER_ID,),
    ))
    _run(conn.commit())

    import time as _t
    _seed_publication(
        conn, pub_id="pub_newest", user_id=TEST_USER_ID,
        title="Fresh", created_at=_t.time(),
    )

    body = tc.get("/api/library/items?sort=recent").json()
    assert body["items"][0]["id"] == "pub_newest", (
        f"publication didn't sort first under recent — got {[it['id'] for it in body['items']]}"
    )


def test_home_type_counts_include_publications(library_client):
    """Home dashboard ``type_counts`` UNIONs publication ``kind`` into
    the same bucket namespace as artifacts. Drives the sidebar's
    auto-type virtual collections — a coder save should bump the
    Games bucket count.
    """
    tc, conn = library_client
    _seed_publication(
        conn, pub_id="pub_game1", user_id=TEST_USER_ID,
        title="Game 1", kind="game",
    )
    _seed_publication(
        conn, pub_id="pub_app1", user_id=TEST_USER_ID,
        title="App 1", kind="app",
    )

    body = tc.get("/api/library/home").json()
    counts = body["type_counts"]
    # Artifact 'game' format (a-mine-2) + the seeded pub_game1.
    assert counts.get("game") == 2
    # App bucket is publication-only.
    assert counts.get("app") == 1
    # Existing buckets still represented.
    assert counts.get("html") == 1
    assert counts.get("pdf") == 1
