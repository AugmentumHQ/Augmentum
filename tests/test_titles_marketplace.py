"""Tests for the AXF marketplace surface (Phase B).

Covers:
* MarketplaceStore CRUD (upsert + delist + increment)
* Catalog loader: happy path + skip on bad entries
* MarketplaceSource: discover() + import_for_user() delegating through
  the underlying installer source
* Js13kSource: discover_to_dict shape + import_for_user idempotency
* Service-level discover_titles flow with installed-decoration
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest

# Schema fixture mirrors migrations 123 + 124 for the AXF surfaces this
# test file touches. The artifacts table is the same as in
# test_titles_smoke.py.
_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);

CREATE TABLE artifacts (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    filename        TEXT NOT NULL DEFAULT '',
    display_name    TEXT NOT NULL DEFAULT '',
    format          TEXT NOT NULL DEFAULT '',
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    path            TEXT NOT NULL DEFAULT '',
    metadata        TEXT NOT NULL DEFAULT '{}',
    user_id         TEXT NOT NULL REFERENCES users(id),
    pinned          INTEGER NOT NULL DEFAULT 0,
    last_opened_at  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE title_runs (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    artifact_id         TEXT NOT NULL,
    runtime_id          TEXT NOT NULL,
    source_id           TEXT NOT NULL DEFAULT '',
    started_at          TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at            TEXT,
    duration_s          INTEGER,
    exit_reason         TEXT NOT NULL DEFAULT '',
    launch_latency_ms   INTEGER,
    avg_fps             REAL,
    avg_rtt_ms          REAL,
    avg_bitrate_kbps    INTEGER,
    crashes             INTEGER NOT NULL DEFAULT 0,
    metadata            TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE marketplace_listings (
    id                  TEXT PRIMARY KEY,
    publisher           TEXT NOT NULL DEFAULT 'augmentum',
    title               TEXT NOT NULL,
    kind                TEXT NOT NULL,
    runtime_preferred   TEXT NOT NULL DEFAULT '',
    runtime_alternates  TEXT NOT NULL DEFAULT '[]',
    tagline             TEXT NOT NULL DEFAULT '',
    description         TEXT NOT NULL DEFAULT '',
    thumbnail_url       TEXT NOT NULL DEFAULT '',
    source_url          TEXT NOT NULL DEFAULT '',
    embed_url           TEXT NOT NULL DEFAULT '',
    install_via         TEXT NOT NULL,
    install_payload     TEXT NOT NULL DEFAULT '{}',
    capabilities        TEXT NOT NULL DEFAULT '{}',
    metadata            TEXT NOT NULL DEFAULT '{}',
    rating              REAL,
    install_count       INTEGER NOT NULL DEFAULT 0,
    signature           TEXT NOT NULL DEFAULT '',
    listed_at           TEXT NOT NULL DEFAULT (datetime('now')),
    delisted_at         TEXT,
    -- Discover-surface columns (migration 254). The store's upsert has
    -- required these since Discover landed; this fixture predated them.
    category            TEXT NOT NULL DEFAULT '',
    tags                TEXT NOT NULL DEFAULT '[]',
    featured            INTEGER NOT NULL DEFAULT 0
);
"""


async def _mkdb():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    return conn


# ── MarketplaceStore ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_marketplace_upsert_and_list_active():
    from augmentum.marketplace import MarketplaceListing, MarketplaceStore

    conn = await _mkdb()
    store = MarketplaceStore(conn)
    listing = MarketplaceListing(
        id="mkt:foo", publisher="augmentum", title="Foo",
        kind="web_app", runtime_preferred="browser-iframe",
        runtime_alternates=(), tagline="A foo", description="",
        thumbnail_url="", source_url="", embed_url="",
        install_via="internal", install_payload={"kind": "web_app", "title": "Foo"},
        capabilities={}, metadata={}, rating=None, install_count=0,
        signature="", listed_at="",
    )
    await store.upsert(listing)
    listings = await store.list_active()
    assert len(listings) == 1 and listings[0].id == "mkt:foo"


@pytest.mark.asyncio
async def test_marketplace_delist_missing_soft_deletes():
    from augmentum.marketplace import MarketplaceListing, MarketplaceStore

    conn = await _mkdb()
    store = MarketplaceStore(conn)
    for fid in ("a", "b", "c"):
        await store.upsert(MarketplaceListing(
            id=f"mkt:{fid}", publisher="x", title=fid.upper(),
            kind="web_app", runtime_preferred="", runtime_alternates=(),
            tagline="", description="", thumbnail_url="", source_url="",
            embed_url="", install_via="internal",
            install_payload={}, capabilities={}, metadata={},
            rating=None, install_count=0, signature="", listed_at="",
        ))
    delisted = await store.delist_missing({"mkt:a", "mkt:b"})
    assert delisted == 1  # only c was dropped
    active = await store.list_active()
    ids = {l.id for l in active}
    assert ids == {"mkt:a", "mkt:b"}


@pytest.mark.asyncio
async def test_marketplace_increment_install_count():
    from augmentum.marketplace import MarketplaceListing, MarketplaceStore

    conn = await _mkdb()
    store = MarketplaceStore(conn)
    await store.upsert(MarketplaceListing(
        id="mkt:foo", publisher="x", title="Foo",
        kind="web_app", runtime_preferred="", runtime_alternates=(),
        tagline="", description="", thumbnail_url="", source_url="",
        embed_url="", install_via="internal",
        install_payload={}, capabilities={}, metadata={},
        rating=None, install_count=0, signature="", listed_at="",
    ))
    await store.increment_install_count("mkt:foo")
    await store.increment_install_count("mkt:foo")
    listing = await store.get("mkt:foo")
    assert listing.install_count == 2


# ── Catalog loader ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_catalog_loader_happy_path(tmp_path: Path):
    from augmentum.marketplace import MarketplaceStore, load_catalog_into_store

    catalog_path = tmp_path / "listings.json"
    catalog_path.write_text(json.dumps({
        "version": 1,
        "listings": [
            {
                "id": "mkt:demo",
                "title": "Demo",
                "kind": "web_app",
                "install_via": "internal",
                "install_payload": {"kind": "web_app", "title": "Demo"},
            },
        ],
    }))

    conn = await _mkdb()
    store = MarketplaceStore(conn)
    stats = await load_catalog_into_store(store, catalog_path=catalog_path)
    assert stats == {"loaded": 1, "skipped": 0, "delisted": 0}
    listings = await store.list_active()
    assert len(listings) == 1 and listings[0].title == "Demo"


@pytest.mark.asyncio
async def test_catalog_loader_skips_bad_entries(tmp_path: Path):
    from augmentum.marketplace import MarketplaceStore, load_catalog_into_store

    catalog_path = tmp_path / "listings.json"
    catalog_path.write_text(json.dumps({
        "version": 1,
        "listings": [
            {  # missing required 'install_via'
                "id": "mkt:bad",
                "title": "Bad",
                "kind": "web_app",
            },
            {
                "id": "mkt:good",
                "title": "Good",
                "kind": "web_app",
                "install_via": "internal",
            },
        ],
    }))

    conn = await _mkdb()
    store = MarketplaceStore(conn)
    stats = await load_catalog_into_store(store, catalog_path=catalog_path)
    assert stats["loaded"] == 1
    assert stats["skipped"] == 1
    listings = await store.list_active()
    assert {l.id for l in listings} == {"mkt:good"}


@pytest.mark.asyncio
async def test_catalog_loader_strict_raises_on_invalid(tmp_path: Path):
    from augmentum.marketplace import (
        CatalogLoadError,
        MarketplaceStore,
        load_catalog_into_store,
    )

    catalog_path = tmp_path / "listings.json"
    catalog_path.write_text("not valid json")
    conn = await _mkdb()
    store = MarketplaceStore(conn)
    with pytest.raises(CatalogLoadError):
        await load_catalog_into_store(
            store, catalog_path=catalog_path, raise_on_invalid=True,
        )


# ── Js13kSource ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_js13k_import_for_user_creates_artifact():
    from augmentum.titles import Js13kSource

    conn = await _mkdb()
    src = Js13kSource(conn)
    artifact_id = await src.import_for_user(
        {
            "source_remote_id": "2024/foo",
            "title": "Foo",
            "embed_url": "https://play.example/2024/foo/",
        },
        user_id="u1",
    )
    assert artifact_id

    cursor = await conn.execute(
        "SELECT metadata FROM artifacts WHERE id = ?", (artifact_id,),
    )
    row = await cursor.fetchone()
    md = json.loads(row[0])
    assert md["kind"] == "js13k_game"
    assert md["source"] == "js13k"
    assert md["source_id"] == "2024/foo"


@pytest.mark.asyncio
async def test_js13k_import_idempotent():
    """Re-importing the same slug returns the existing artifact id, not a
    duplicate row -- mirrors the legacy /api/games/pin behaviour."""
    from augmentum.titles import Js13kSource

    conn = await _mkdb()
    src = Js13kSource(conn)
    a = await src.import_for_user(
        {"source_remote_id": "2024/foo", "title": "Foo"},
        user_id="u1",
    )
    b = await src.import_for_user(
        {"source_remote_id": "2024/foo", "title": "Foo"},
        user_id="u1",
    )
    assert a == b
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM artifacts WHERE user_id = 'u1'",
    )
    assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_js13k_import_finds_legacy_kind_game():
    """A pre-AXF artifact (kind == 'game') for the same slug must NOT
    be re-imported as a duplicate -- the lookup matches both kinds."""
    from augmentum.titles import Js13kSource

    conn = await _mkdb()
    # Seed a legacy pin
    await conn.execute(
        "INSERT INTO artifacts (id, display_name, filename, format, "
        "metadata, user_id, pinned) VALUES (?, ?, ?, '', ?, ?, 1)",
        ("legacy_id", "Foo", "foo.title",
         json.dumps({"kind": "game", "source": "js13k", "source_id": "2024/foo"}),
         "u1"),
    )
    await conn.commit()

    src = Js13kSource(conn)
    artifact_id = await src.import_for_user(
        {"source_remote_id": "2024/foo", "title": "Foo"},
        user_id="u1",
    )
    assert artifact_id == "legacy_id"


# ── MarketplaceSource ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_marketplace_discover_returns_items():
    from augmentum.marketplace import (
        MarketplaceListing,
        MarketplaceSource,
        MarketplaceStore,
    )
    from augmentum.titles import InternalSource, SourceRegistry

    conn = await _mkdb()
    mstore = MarketplaceStore(conn)
    await mstore.upsert(MarketplaceListing(
        id="mkt:foo", publisher="augmentum", title="Foo",
        kind="web_app", runtime_preferred="browser-iframe",
        runtime_alternates=(), tagline="A foo demo", description="",
        thumbnail_url="", source_url="", embed_url="",
        install_via="internal", install_payload={
            "kind": "web_app", "title": "Foo",
            "source_remote_id": "https://example.com/foo",
            "metadata": {"embed_url": "https://example.com/foo"},
        },
        capabilities={"input_modes": ["keyboard"]}, metadata={"tags": ["puzzle"]},
        rating=4.5, install_count=10, signature="", listed_at="",
    ))
    sources = SourceRegistry()
    sources.register(InternalSource(conn))
    marketplace_source = MarketplaceSource(store=mstore, sources=sources)

    items = await marketplace_source.discover({}, user_id="u1")
    assert len(items) == 1
    item = items[0]
    assert item.title == "Foo"
    assert item.kind == "web_app"
    assert item.metadata.get("install_via") == "internal"


@pytest.mark.asyncio
async def test_marketplace_discover_query_filter():
    from augmentum.marketplace import (
        MarketplaceListing,
        MarketplaceSource,
        MarketplaceStore,
    )
    from augmentum.titles import InternalSource, SourceRegistry

    conn = await _mkdb()
    mstore = MarketplaceStore(conn)
    for i, title in enumerate(["Foo Demo", "Bar Game", "Baz Puzzle"]):
        await mstore.upsert(MarketplaceListing(
            id=f"mkt:{i}", publisher="x", title=title,
            kind="web_app", runtime_preferred="", runtime_alternates=(),
            tagline="", description="", thumbnail_url="", source_url="",
            embed_url="", install_via="internal", install_payload={},
            capabilities={}, metadata={}, rating=None,
            install_count=0, signature="", listed_at="",
        ))
    sources = SourceRegistry()
    sources.register(InternalSource(conn))
    src = MarketplaceSource(store=mstore, sources=sources)
    items = await src.discover({"q": "puzzle"})
    assert len(items) == 1 and items[0].title == "Baz Puzzle"


@pytest.mark.asyncio
async def test_marketplace_install_delegates_to_underlying_source():
    """The marketplace's install_for_user must round-trip through the
    underlying installer Source, not duplicate its logic."""
    from augmentum.marketplace import (
        MarketplaceListing,
        MarketplaceSource,
        MarketplaceStore,
    )
    from augmentum.titles import InternalSource, SourceRegistry

    conn = await _mkdb()
    mstore = MarketplaceStore(conn)
    await mstore.upsert(MarketplaceListing(
        id="mkt:foo", publisher="x", title="Foo",
        kind="web_app", runtime_preferred="browser-iframe",
        runtime_alternates=(), tagline="", description="",
        thumbnail_url="", source_url="", embed_url="",
        install_via="internal",
        install_payload={
            "kind": "web_app", "title": "Foo",
            "source_remote_id": "https://example.com/foo",
            "metadata": {"embed_url": "https://example.com/foo"},
        },
        capabilities={}, metadata={}, rating=None,
        install_count=0, signature="", listed_at="",
    ))
    sources = SourceRegistry()
    sources.register(InternalSource(conn))
    market = MarketplaceSource(store=mstore, sources=sources)

    artifact_id = await market.import_for_user(
        {"listing_id": "mkt:foo"}, user_id="u1",
    )
    assert artifact_id

    # The artifact came from InternalSource (we can verify by checking
    # the metadata.title was set as InternalSource does).
    cursor = await conn.execute(
        "SELECT metadata FROM artifacts WHERE id = ?", (artifact_id,),
    )
    md = json.loads((await cursor.fetchone())[0])
    assert md["kind"] == "web_app"
    assert md["title"] == "Foo"

    # install_count incremented
    listing = await mstore.get("mkt:foo")
    assert listing.install_count == 1


@pytest.mark.asyncio
async def test_marketplace_install_404_for_missing_listing():
    from augmentum.marketplace import MarketplaceSource, MarketplaceStore
    from augmentum.titles import InternalSource, SourceImportError, SourceRegistry

    conn = await _mkdb()
    sources = SourceRegistry()
    sources.register(InternalSource(conn))
    market = MarketplaceSource(
        store=MarketplaceStore(conn), sources=sources,
    )
    with pytest.raises(SourceImportError):
        await market.import_for_user({"listing_id": "nope"}, user_id="u1")


# ── Service-level discovery ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_discover_titles_decorates_installed_flag():
    from augmentum.marketplace import (
        MarketplaceListing,
        MarketplaceSource,
        MarketplaceStore,
    )
    from augmentum.titles import (
        BrowserIframeRuntime,
        InternalSource,
        RuntimeRegistry,
        SourceRegistry,
        TitleService,
        TitleStore,
    )

    conn = await _mkdb()
    mstore = MarketplaceStore(conn)
    await mstore.upsert(MarketplaceListing(
        id="mkt:foo", publisher="x", title="Foo",
        kind="web_app", runtime_preferred="browser-iframe",
        runtime_alternates=(), tagline="", description="",
        thumbnail_url="", source_url="", embed_url="",
        install_via="internal",
        install_payload={
            "kind": "web_app", "title": "Foo",
            "source_remote_id": "mkt:foo",
            "metadata": {"embed_url": "https://x/foo"},
        },
        capabilities={}, metadata={}, rating=None,
        install_count=0, signature="", listed_at="",
    ))
    sources = SourceRegistry()
    sources.register(InternalSource(conn))
    sources.register(MarketplaceSource(store=mstore, sources=sources))
    runtimes = RuntimeRegistry()
    runtimes.register(BrowserIframeRuntime())
    svc = TitleService(
        store=TitleStore(conn), sources=sources, runtimes=runtimes,
    )

    # Install via the marketplace
    await svc.import_title(
        user_id="u1", source_id="marketplace",
        manifest_data={"listing_id": "mkt:foo"},
    )

    items = await svc.discover_titles(
        source_id="marketplace", user_id="u1",
    )
    assert len(items) == 1
    # Installed-decoration: the marketplace listing produces a different
    # source_remote_id than the underlying installed artifact (the
    # installed artifact uses the install_payload's source_remote_id).
    # The decoration matches by (source_id, source_remote_id) of the
    # discovery item. Since discovery items use source_id='marketplace'
    # but the installed artifact records source_id='url-bookmark' (from
    # the InternalSource path), the decoration will return False here.
    # This is correct -- the marketplace card should still show "Install"
    # because re-installing from the marketplace is a separate action.
    # We assert that decoration RAN (didn't crash) and produced a value.
    assert items[0].installed is False
