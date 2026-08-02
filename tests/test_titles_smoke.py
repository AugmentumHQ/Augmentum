"""Smoke tests for the Augmentum Experience Framework (AXF) -- titles
substrate.

Covers: package imports, manifest projection (including legacy js13k
bridge), source/runtime registries, store CRUD + run telemetry. No
route layer here -- that's tests/test_titles_routes.py.
"""

from __future__ import annotations

import json

import aiosqlite
import pytest

# Minimal schema mirroring the parts we read from. Includes the
# existing ``artifacts`` table (relevant columns only) and the new
# ``title_runs`` table (migration 123). Same pattern as the smoke
# tests for game_stream and jobs.
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
"""


async def _mkstore():
    from augmentum.titles import TitleStore

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    return TitleStore(conn), conn


async def _seed_artifact(
    conn,
    *,
    artifact_id: str,
    user_id: str = "u1",
    title: str = "Some Title",
    metadata: dict | None = None,
    pinned: bool = True,
) -> str:
    md = metadata or {}
    await conn.execute(
        """INSERT INTO artifacts
           (id, display_name, filename, format, metadata, user_id, pinned)
           VALUES (?, ?, ?, '', ?, ?, ?)""",
        (
            artifact_id,
            title,
            f"{title}.title",
            json.dumps(md),
            user_id,
            1 if pinned else 0,
        ),
    )
    await conn.commit()
    return artifact_id


# ── Imports ───────────────────────────────────────────────────────


def test_package_surface_imports():
    from augmentum.proxy.titles_routes import router  # noqa: F401
    from augmentum.titles import (  # noqa: F401
        KIND_EMULATOR_ROM,
        KIND_GIT_PROJECT,
        KIND_JS13K_GAME,
        KIND_STREAMED_GAME,
        KIND_WEB_APP,
        TITLE_KINDS,
        BrowserIframeRuntime,
        InternalSource,
        LaunchHandle,
        Runtime,
        RuntimeRegistry,
        Source,
        SourceImportError,
        SourceRegistry,
        TitleManifest,
        TitleService,
        TitleServiceError,
        TitleStore,
        is_title_kind,
        runtime_registry,
        source_registry,
    )


def test_browser_iframe_runtime_registered_eagerly():
    """The browser-iframe runtime has no external deps and is registered
    on import. Subsequent server-side init only adds the AGSP adapter."""
    from augmentum.titles import runtime_registry

    iframe = runtime_registry.get("browser-iframe")
    assert iframe is not None
    assert iframe.label == "Browser (in-app)"


# ── Manifest projection ────────────────────────────────────────────


def test_manifest_projects_modern_kind():
    from augmentum.titles import TitleManifest

    row = {
        "id": "art_1",
        "user_id": "u1",
        "display_name": "Voxel World",
        "filename": "world.title",
        "metadata": json.dumps({
            "kind": "streamed_game",
            "source": "agsp-profile",
            "source_id": "luanti",
            "runtime_preferred": "agsp-streamed",
            "metadata": {"genre": ["sandbox"]},
            "capabilities": {"input_modes": ["keyboard","mouse","gamepad"]},
        }),
        "pinned": 1,
        "last_opened_at": "2026-05-08 12:00:00",
    }
    manifest = TitleManifest.from_artifact_row(row)
    assert manifest is not None
    assert manifest.kind == "streamed_game"
    assert manifest.source_id == "agsp-profile"
    assert manifest.source_remote_id == "luanti"
    assert manifest.runtime_preferred == "agsp-streamed"
    assert manifest.pinned is True


def test_manifest_legacy_js13k_bridge():
    """Existing pinned games use ``kind == 'game'`` -- they must project
    cleanly into KIND_JS13K_GAME so the new surface lists them without
    a data migration."""
    from augmentum.titles import KIND_JS13K_GAME, TitleManifest

    row = {
        "id": "art_2",
        "user_id": "u1",
        "display_name": "Legacy js13k pin",
        "filename": "legacy.title",
        "metadata": json.dumps({
            "kind": "game",
            "source": "js13k",
            "source_id": "2024/foo",
            "embed_url": "https://js13kgames.com/games/foo",
        }),
        "pinned": 1,
        "last_opened_at": None,
    }
    manifest = TitleManifest.from_artifact_row(row)
    assert manifest is not None
    assert manifest.kind == KIND_JS13K_GAME
    assert manifest.source_id == "js13k"
    assert manifest.runtime_preferred == "browser-iframe"
    # Embed URL is preserved in raw_metadata so the BrowserIframeRuntime
    # can read it without re-querying.
    assert manifest.raw_metadata["embed_url"] == "https://js13kgames.com/games/foo"


def test_manifest_returns_none_for_non_title_artifacts():
    from augmentum.titles import TitleManifest

    row = {
        "id": "art_3",
        "user_id": "u1",
        "metadata": json.dumps({"kind": "image"}),
        "pinned": 0,
    }
    assert TitleManifest.from_artifact_row(row) is None


# ── Store: read path ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_returns_only_titles():
    """list_for_user filters out artifacts that aren't titles."""
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_1",
                         metadata={"kind": "streamed_game", "source": "agsp-profile"})
    await _seed_artifact(conn, artifact_id="t_2",
                         metadata={"kind": "image"})  # not a title
    await _seed_artifact(conn, artifact_id="t_3",
                         metadata={"kind": "game", "source": "js13k"})  # legacy js13k

    titles = await store.list_for_user(user_id="u1")
    ids = {m.id for m in titles}
    assert ids == {"t_1", "t_3"}  # legacy js13k bridges through


@pytest.mark.asyncio
async def test_list_user_scoped():
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_a", user_id="u1",
                         metadata={"kind": "web_app"})
    await _seed_artifact(conn, artifact_id="t_b", user_id="u2",
                         metadata={"kind": "web_app"})
    own = await store.list_for_user(user_id="u1")
    assert {m.id for m in own} == {"t_a"}


@pytest.mark.asyncio
async def test_list_pinned_only_filter():
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_p", pinned=True,
                         metadata={"kind": "web_app"})
    await _seed_artifact(conn, artifact_id="t_u", pinned=False,
                         metadata={"kind": "web_app"})
    pinned = await store.list_for_user(user_id="u1", pinned_only=True)
    assert {m.id for m in pinned} == {"t_p"}


# ── Store: writes ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_pinned_round_trip():
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_x",
                         metadata={"kind": "web_app"}, pinned=False)
    assert await store.set_pinned("t_x", user_id="u1", pinned=True) is True
    m = await store.get("t_x", user_id="u1")
    assert m.pinned is True


@pytest.mark.asyncio
async def test_update_metadata_merge():
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_meta",
                         metadata={"kind": "web_app", "title": "First"})
    ok = await store.update_metadata(
        "t_meta", user_id="u1",
        patch={"title": "Second", "tags": ["alpha"]},
    )
    assert ok
    m = await store.get("t_meta", user_id="u1")
    assert m.title == "Second"
    assert m.metadata.get("tags") == ["alpha"]


@pytest.mark.asyncio
async def test_get_scoped_by_user_no_cross_tenant_leak():
    """A title read must be scoped to its owner. An empty or other user_id
    must never return another user's title by id.

    Regression guard: get() historically dropped the WHERE clause on an
    empty user_id, returning any artifact by id (cross-tenant leak).
    Titles are user-owned artifact rows (no bundled NULL rows), so an
    empty scope yields nothing rather than a cross-tenant manifest.
    """
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_scoped", user_id="u1",
                         metadata={"kind": "web_app"})
    # Owner sees it.
    assert await store.get("t_scoped", user_id="u1") is not None
    # A different tenant does NOT.
    assert await store.get("t_scoped", user_id="u2") is None
    # No scope: also None.
    assert await store.get("t_scoped", user_id="") is None


# ── Store: run telemetry ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_lifecycle_records_duration():
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_run",
                         metadata={"kind": "web_app"})
    run_id = await store.create_run(
        user_id="u1", artifact_id="t_run", runtime_id="browser-iframe",
        source_id="js13k", launch_latency_ms=42,
    )
    # Open run is visible in the live-runs list.
    open_runs = await store.list_open_runs(user_id="u1")
    assert len(open_runs) == 1 and open_runs[0]["id"] == run_id

    ok = await store.end_run(
        run_id, user_id="u1", exit_reason="clean",
        avg_fps=58.5, crashes=0,
    )
    assert ok
    runs = await store.list_runs(user_id="u1", artifact_id="t_run")
    assert len(runs) == 1
    assert runs[0]["exit_reason"] == "clean"
    assert runs[0]["avg_fps"] == 58.5
    # duration_s is materialised; value is small but >= 0
    assert runs[0]["duration_s"] is not None and runs[0]["duration_s"] >= 0


@pytest.mark.asyncio
async def test_total_play_time_aggregated_into_manifest():
    store, conn = await _mkstore()
    await _seed_artifact(conn, artifact_id="t_total",
                         metadata={"kind": "web_app"})
    # Seed two completed runs directly to avoid sleeping in the test.
    await conn.execute(
        "INSERT INTO title_runs (id, user_id, artifact_id, runtime_id, "
        " duration_s, ended_at, exit_reason) "
        "VALUES (?, ?, ?, 'browser-iframe', ?, datetime('now'), 'clean')",
        ("r1", "u1", "t_total", 60),
    )
    await conn.execute(
        "INSERT INTO title_runs (id, user_id, artifact_id, runtime_id, "
        " duration_s, ended_at, exit_reason) "
        "VALUES (?, ?, ?, 'browser-iframe', ?, datetime('now'), 'clean')",
        ("r2", "u1", "t_total", 120),
    )
    await conn.commit()
    m = await store.get("t_total", user_id="u1")
    assert m.total_play_time_s == 180


# ── Source registry ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_internal_source_imports_to_artifact():
    from augmentum.titles import InternalSource

    _store, conn = await _mkstore()
    src = InternalSource(conn)
    artifact_id = await src.import_for_user(
        {
            "kind": "web_app",
            "title": "Some Site",
            "source_id": "url-bookmark",
            "source_remote_id": "https://example.com/game",
            "runtime_preferred": "browser-iframe",
            "metadata": {"genre": ["puzzle"]},
        },
        user_id="u1",
    )
    assert artifact_id

    cursor = await conn.execute(
        "SELECT metadata, user_id, pinned FROM artifacts WHERE id = ?",
        (artifact_id,),
    )
    row = await cursor.fetchone()
    md = json.loads(row[0])
    assert md["kind"] == "web_app"
    assert md["source"] == "url-bookmark"
    assert md["source_id"] == "https://example.com/game"
    assert md["genre"] == ["puzzle"]
    assert row[1] == "u1"
    assert row[2] == 1


@pytest.mark.asyncio
async def test_internal_source_rejects_unknown_kind():
    from augmentum.titles import InternalSource, SourceImportError

    _store, conn = await _mkstore()
    src = InternalSource(conn)
    with pytest.raises(SourceImportError):
        await src.import_for_user(
            {"kind": "no-such-kind", "title": "X"},
            user_id="u1",
        )


# ── Runtime registry ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browser_iframe_supports_correct_kinds():
    from augmentum.titles import (
        KIND_EMULATOR_ROM,
        KIND_JS13K_GAME,
        KIND_STREAMED_GAME,
        KIND_WEB_APP,
        BrowserIframeRuntime,
        TitleManifest,
    )

    rt = BrowserIframeRuntime()

    def _mk(kind: str) -> TitleManifest:
        row = {
            "id": "x",
            "user_id": "u1",
            "metadata": json.dumps({"kind": kind, "source": ""}),
        }
        return TitleManifest.from_artifact_row(row)

    assert await rt.supports(_mk(KIND_JS13K_GAME))
    assert await rt.supports(_mk(KIND_WEB_APP))
    assert not await rt.supports(_mk(KIND_STREAMED_GAME))
    assert not await rt.supports(_mk(KIND_EMULATOR_ROM))


@pytest.mark.asyncio
async def test_runtime_resolver_picks_supporting_runtime():
    from augmentum.titles import (
        KIND_WEB_APP,
        BrowserIframeRuntime,
        RuntimeRegistry,
        TitleManifest,
    )

    reg = RuntimeRegistry()
    reg.register(BrowserIframeRuntime())
    row = {
        "id": "y",
        "user_id": "u1",
        "metadata": json.dumps({"kind": KIND_WEB_APP, "source": "url-bookmark"}),
    }
    manifest = TitleManifest.from_artifact_row(row)
    rt = await reg.resolve_for(manifest)
    assert rt is not None and rt.id == "browser-iframe"


# ── Service: launch flow ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_service_launch_records_run_and_returns_handle():
    from augmentum.titles import (
        BrowserIframeRuntime,
        InternalSource,
        RuntimeRegistry,
        SourceRegistry,
        TitleService,
    )

    store, conn = await _mkstore()
    # Seed via the InternalSource so the artifact row matches what a
    # real import would produce.
    src = InternalSource(conn)
    src_reg = SourceRegistry()
    src_reg.register(src)
    rt_reg = RuntimeRegistry()
    rt_reg.register(BrowserIframeRuntime())
    svc = TitleService(store=store, sources=src_reg, runtimes=rt_reg)

    title = await svc.import_title(
        user_id="u1",
        source_id="internal",
        manifest_data={
            "kind": "web_app",
            "title": "Some Game",
            "source_id": "url-bookmark",
            "source_remote_id": "https://example.com/g",
            "metadata": {"embed_url": "https://example.com/g"},
        },
    )
    result = await svc.launch(title.id, user_id="u1")
    assert result["run_id"]
    assert result["handle"]["runtime_id"] == "browser-iframe"
    assert result["handle"]["target"] == "https://example.com/g"

    # last_played_at was touched
    refreshed = await svc.get_title(title.id, user_id="u1")
    assert refreshed.last_played_at is not None

    # Closing the run materialises duration
    ok = await svc.end_run(
        result["run_id"], user_id="u1",
        runtime_id="browser-iframe", exit_reason="clean",
    )
    assert ok


@pytest.mark.asyncio
async def test_service_launch_404_on_missing_title():
    from augmentum.titles import (
        BrowserIframeRuntime,
        InternalSource,
        RuntimeRegistry,
        SourceRegistry,
        TitleNotFound,
        TitleService,
    )

    store, conn = await _mkstore()
    src_reg = SourceRegistry()
    src_reg.register(InternalSource(conn))
    rt_reg = RuntimeRegistry()
    rt_reg.register(BrowserIframeRuntime())
    svc = TitleService(store=store, sources=src_reg, runtimes=rt_reg)
    with pytest.raises(TitleNotFound):
        await svc.launch("nope", user_id="u1")
