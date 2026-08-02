"""PublicationStore + LibraryStorage + preview_kind tests.

Pure SQLite (in-memory) + tmp_path for storage. The schema is duplicated
here from migration 197 to keep tests self-contained — the real migration
at ``augmentum/state/migrations/197_library_publications.sql`` is the
source of truth.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.library.preview_kind import classify_response
from augmentum.library.publications import (
    LibraryStorage,
    PublicationStore,
    SizeBudgetExceeded,
    TitleCollision,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL
);
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
-- PublicationStore.delete sweeps these on delete (mig 309 dropped the
-- artifacts FK, so the cleanup is app-level now — the tables must exist).
CREATE TABLE IF NOT EXISTS library_activity (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    artifact_id     TEXT NOT NULL,
    action          TEXT NOT NULL,
    surface         TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL DEFAULT '{}',
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS library_collection_items (
    collection_id   TEXT NOT NULL,
    artifact_id     TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    added_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (collection_id, artifact_id)
);
"""


_BIG = 10 * 1024 * 1024     # 10 MB
_TINY = 1024                # 1 KB


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
async def setup(tmp_path: Path):
    """Fresh DB + storage per test, with deterministic teardown so the
    aiosqlite worker thread closes and pytest can exit cleanly."""
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.executescript(_SCHEMA_SQL)
        await conn.execute("INSERT INTO users (id, username) VALUES (?, ?)", ("u1", "alice"))
        await conn.execute("INSERT INTO users (id, username) VALUES (?, ?)", ("u2", "bob"))
        await conn.commit()
        storage = LibraryStorage(tmp_path / "library_published")
        yield PublicationStore(conn, storage), conn, storage
    finally:
        await conn.close()


def _mkbundle(root: Path, name: str = "game") -> Path:
    """Create a small HTML/JS bundle at ``root/{name}/`` and return its path.

    Bundle is sized at ~5KB so size-cap tests can use a sub-KB cap to
    trigger SizeBudgetExceeded without needing megabyte payloads.
    """
    src = root / name
    src.mkdir(parents=True)
    (src / "index.html").write_text("<html><body><canvas></canvas></body></html>")
    (src / "game.js").write_text("console.log('hi')\n" + "x" * 1024)
    assets = src / "assets"
    assets.mkdir()
    # ~4KB sprite so the bundle clears any sub-KB cap deterministically.
    (assets / "sprite.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096)
    return src


def _mksingle(root: Path, name: str = "snake.html") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    target.write_text("<html><body>Snake</body></html>")
    return target


# ── PublicationStore: create / get / list ──────────────────────────────


@pytest.mark.asyncio
async def test_create_bundle_roundtrip(setup, tmp_path: Path):
    store, _, storage = setup
    src = _mkbundle(tmp_path / "ws")

    row = await store.create_or_overwrite(
        user_id="u1",
        workspace_id="ws-1",
        title="Tower Defense",
        description="defend the bases",
        kind="game",
        source_path=src,
        entry_point="index.html",
        max_bytes=_BIG,
        user_budget_bytes=_BIG,
    )
    assert row["id"].startswith("pub_")
    assert row["user_id"] == "u1"
    assert row["title"] == "Tower Defense"
    assert row["kind"] == "game"
    assert row["storage_kind"] == "bundle"
    assert row["entry_point"] == "index.html"
    assert row["version"] == 1
    assert row["_action"] == "created"
    assert row["size_bytes"] > 0

    # Storage actually got snapshotted.
    pub_dir = storage.publication_dir("u1", row["id"])
    assert (pub_dir / "content" / "index.html").is_file()
    assert (pub_dir / "content" / "game.js").is_file()
    assert (pub_dir / "content" / "assets" / "sprite.png").is_file()
    assert (pub_dir / "meta.json").is_file()


@pytest.mark.asyncio
async def test_create_single_file_roundtrip(setup, tmp_path: Path):
    store, _, storage = setup
    src = _mksingle(tmp_path / "ws")

    row = await store.create_or_overwrite(
        user_id="u1",
        workspace_id="ws-1",
        title="Snake",
        description="",
        kind="game",
        source_path=src,
        entry_point="snake.html",
        max_bytes=_BIG,
        user_budget_bytes=_BIG,
    )
    assert row["storage_kind"] == "single"
    assert row["entry_point"] == "snake.html"
    pub_dir = storage.publication_dir("u1", row["id"])
    assert (pub_dir / "content" / "snake.html").is_file()


@pytest.mark.asyncio
async def test_get_and_list_for_user(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="A", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    src2 = _mkbundle(tmp_path / "ws2", "b")
    b = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="B", description="", kind="app",
        source_path=src2, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )

    got = await store.get(a["id"], user_id="u1")
    assert got and got["title"] == "A"

    rows = await store.list_for_user(user_id="u1")
    assert {r["title"] for r in rows} == {"A", "B"}

    only_apps = await store.list_for_user(user_id="u1", kind="app")
    assert [r["id"] for r in only_apps] == [b["id"]]


# ── Multi-tenant isolation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_isolation(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="A", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    # User B can't see User A's publication via either path.
    assert await store.get(a["id"], user_id="u2") is None
    assert await store.get_by_title(user_id="u2", title="A") is None
    assert await store.list_for_user(user_id="u2") == []


@pytest.mark.asyncio
async def test_same_title_different_users_both_allowed(setup, tmp_path: Path):
    store, _, _ = setup
    src1 = _mkbundle(tmp_path / "ws1", "a")
    src2 = _mkbundle(tmp_path / "ws2", "b")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="Pong", description="", kind="game",
        source_path=src1, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    b = await store.create_or_overwrite(
        user_id="u2", workspace_id="", title="Pong", description="", kind="game",
        source_path=src2, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    assert a["id"] != b["id"]
    assert a["version"] == 1 and b["version"] == 1


# ── Title collisions ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_title_collision_abort(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    first = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="Dup", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    with pytest.raises(TitleCollision) as ei:
        await store.create_or_overwrite(
            user_id="u1", workspace_id="", title="Dup", description="", kind="game",
            source_path=src, entry_point="index.html",
            on_collision="abort",
            max_bytes=_BIG, user_budget_bytes=_BIG,
        )
    assert ei.value.existing["id"] == first["id"]


@pytest.mark.asyncio
async def test_overwrite_bumps_version_and_replaces_content(setup, tmp_path: Path):
    store, _, storage = setup
    src = _mkbundle(tmp_path / "ws_v1")
    first = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="Game", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    first_id = first["id"]
    first_size = first["size_bytes"]

    # V2 has a bigger payload so size_bytes change is detectable.
    src2 = tmp_path / "ws_v2"
    src2.mkdir()
    (src2 / "index.html").write_text("v2" * 5000)
    (src2 / "game.js").write_text("v2js" * 5000)

    second = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="Game", description="", kind="game",
        source_path=src2, entry_point="index.html",
        on_collision="overwrite",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    assert second["id"] == first_id     # same row, same disk dir
    assert second["version"] == 2
    assert second["size_bytes"] != first_size
    assert second["_action"] == "overwritten"

    # Old content was replaced — assets dir from V1 is gone.
    pub_dir = storage.publication_dir("u1", first_id)
    assert not (pub_dir / "content" / "assets").exists()
    assert (pub_dir / "content" / "index.html").read_text().startswith("v2")


# ── Size budgets ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_per_publication_cap_rejects(setup, tmp_path: Path):
    store, _, storage = setup
    src = _mkbundle(tmp_path / "ws")

    with pytest.raises(SizeBudgetExceeded) as ei:
        await store.create_or_overwrite(
            user_id="u1", workspace_id="", title="Big", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=_TINY,                # tiny cap, real bundle exceeds it
            user_budget_bytes=_BIG,
        )
    assert ei.value.scope == "per_publication"
    # The rejected save must have rolled back the storage dir.
    pub_dirs = list((tmp_path / "library_published" / "u1").glob("pub_*")) \
        if (tmp_path / "library_published" / "u1").exists() else []
    assert pub_dirs == []
    # And no catalog row.
    assert await store.get_by_title(user_id="u1", title="Big") is None


@pytest.mark.asyncio
async def test_user_budget_cap_rejects(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    # First publication fits.
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="A", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    budget = a["size_bytes"] + 100  # only ~100 bytes of headroom

    src2 = _mkbundle(tmp_path / "ws2", "b")
    with pytest.raises(SizeBudgetExceeded) as ei:
        await store.create_or_overwrite(
            user_id="u1", workspace_id="", title="B", description="", kind="game",
            source_path=src2, entry_point="index.html",
            max_bytes=_BIG, user_budget_bytes=budget,
        )
    assert ei.value.scope == "user_total"


@pytest.mark.asyncio
async def test_overwrite_excludes_old_bytes_from_budget(setup, tmp_path: Path):
    """An overwrite at the same title shouldn't double-count the row's
    old bytes against the user's cumulative cap — it's a replacement."""
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="A", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    # Budget just barely fits one bundle. Overwrite must succeed because
    # the old bytes go away; the new bytes don't get added on top.
    budget = a["size_bytes"] + 200
    a2 = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="A", description="", kind="game",
        source_path=src, entry_point="index.html",
        on_collision="overwrite",
        max_bytes=_BIG, user_budget_bytes=budget,
    )
    assert a2["id"] == a["id"]
    assert a2["version"] == 2


# ── Patch / delete / launch ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_rename_and_description(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="Old", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    patched = await store.patch(
        a["id"], user_id="u1", title="New", description="updated",
    )
    assert patched and patched["title"] == "New"
    assert patched["description"] == "updated"


@pytest.mark.asyncio
async def test_patch_rename_to_collision_raises(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="A", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    src2 = _mkbundle(tmp_path / "ws2", "b")
    await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="B", description="", kind="game",
        source_path=src2, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    with pytest.raises(TitleCollision):
        await store.patch(a["id"], user_id="u1", title="B")


@pytest.mark.asyncio
async def test_delete_removes_row_and_storage(setup, tmp_path: Path):
    store, _, storage = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="Del", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    pub_dir = storage.publication_dir("u1", a["id"])
    assert pub_dir.is_dir()

    ok = await store.delete(a["id"], user_id="u1")
    assert ok is True
    assert await store.get(a["id"], user_id="u1") is None
    assert not pub_dir.exists()

    # Idempotent.
    assert await store.delete(a["id"], user_id="u1") is False


@pytest.mark.asyncio
async def test_record_launch_bumps_count(setup, tmp_path: Path):
    store, _, _ = setup
    src = _mkbundle(tmp_path / "ws")
    a = await store.create_or_overwrite(
        user_id="u1", workspace_id="", title="L", description="", kind="game",
        source_path=src, entry_point="index.html",
        max_bytes=_BIG, user_budget_bytes=_BIG,
    )
    assert a["launch_count"] == 0
    await store.record_launch(a["id"], user_id="u1")
    await store.record_launch(a["id"], user_id="u1")
    refreshed = await store.get(a["id"], user_id="u1")
    assert refreshed["launch_count"] == 2
    assert refreshed["last_launched_at"] is not None


# ── Storage safety ─────────────────────────────────────────────────────


def test_asset_path_blocks_traversal(tmp_path: Path):
    storage = LibraryStorage(tmp_path / "library_published")
    pub_dir = storage.publication_dir("u1", "pub_x")
    content = pub_dir / "content"
    content.mkdir(parents=True)
    (content / "ok.html").write_text("ok")

    assert storage.asset_path(user_id="u1", publication_id="pub_x", rel_path="ok.html") is not None
    assert storage.asset_path(user_id="u1", publication_id="pub_x", rel_path="../../etc/passwd") is None
    assert storage.asset_path(user_id="u1", publication_id="pub_x", rel_path="/etc/passwd") is None
    assert storage.asset_path(user_id="u1", publication_id="pub_x", rel_path="missing.html") is None


def test_write_bundle_rejects_missing_entry_point(tmp_path: Path):
    storage = LibraryStorage(tmp_path / "library_published")
    src = _mkbundle(tmp_path / "ws")
    with pytest.raises(FileNotFoundError):
        storage.write_bundle(
            user_id="u1", publication_id="pub_x",
            source_path=src, entry_point="nope.html",
        )


# ── Preview kind classifier ────────────────────────────────────────────


def test_classify_static_html_no_dynamic_markers():
    body = "<html><body><canvas id='c'></canvas><script src='game.js'></script></body></html>"
    assert classify_response(
        status_code=200, content_type="text/html; charset=utf-8", body=body,
    ) == "static"


def test_classify_dynamic_fetch_call():
    body = "<html><script>fetch('/data').then(r => r.json())</script></html>"
    assert classify_response(
        status_code=200, content_type="text/html", body=body,
    ) == "dynamic"


def test_classify_dynamic_websocket():
    body = "<html><script>const ws = new WebSocket('ws://localhost:8000/socket');</script></html>"
    assert classify_response(
        status_code=200, content_type="text/html", body=body,
    ) == "dynamic"


def test_classify_dynamic_api_path():
    body = '<html><script>const URL = "/api/posts";</script></html>'
    assert classify_response(
        status_code=200, content_type="text/html", body=body,
    ) == "dynamic"


def test_classify_unknown_on_error_status():
    assert classify_response(
        status_code=500, content_type="text/html", body="<html></html>",
    ) == "unknown"


def test_classify_unknown_on_non_html_response():
    assert classify_response(
        status_code=200, content_type="application/json", body='{"k": "v"}',
    ) == "unknown"


def test_classify_unknown_on_empty_body():
    assert classify_response(
        status_code=200, content_type="text/html", body="",
    ) == "unknown"


def test_classify_accepts_bytes_body():
    body = b"<html><body>hi</body></html>"
    assert classify_response(
        status_code=200, content_type="text/html", body=body,
    ) == "static"
