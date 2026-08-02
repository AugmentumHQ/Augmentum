"""Behavior tests for /api/titles/* -- the AXF route surface.

Covers:
* Master toggle (titles_enabled) -- 503 when off
* Auth gate -- 401 when no user
* User isolation -- u1 cannot see u2's titles
* Import via InternalSource -- happy path + bad kind
* Launch + run lifecycle -- 201 + run_id, end_run closes
* Registry introspection -- list runtimes/sources
"""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from fastapi.testclient import TestClient

# Reuse the schema fixture from the smoke test rather than duplicating SQL --
# the migration is the source of truth, this is a behavioural test of routes.
from tests.test_titles_smoke import _SCHEMA_SQL  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def titles_client(app, monkeypatch):
    """TestClient with TitleService wired against an in-memory DB."""
    from augmentum.config import settings as config_settings
    from augmentum.titles import (
        BrowserIframeRuntime,
        InternalSource,
        RuntimeRegistry,
        SourceRegistry,
        TitleService,
        TitleStore,
    )

    monkeypatch.setattr(config_settings, "titles_enabled", True)

    conn = _run(aiosqlite.connect(":memory:"))
    _run(conn.executescript(_SCHEMA_SQL))
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_test')"))
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_other')"))
    _run(conn.commit())

    store = TitleStore(conn)
    sources = SourceRegistry()
    sources.register(InternalSource(conn))
    runtimes = RuntimeRegistry()
    runtimes.register(BrowserIframeRuntime())
    svc = TitleService(store=store, sources=sources, runtimes=runtimes)
    app.state.title_service = svc
    app.state.title_store = store

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, svc, conn
    _run(conn.close())


@pytest.fixture
def titles_client_disabled(app, monkeypatch):
    from augmentum.config import settings as config_settings

    monkeypatch.setattr(config_settings, "titles_enabled", False)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc


# ── Master toggle ────────────────────────────────────────────────────


class TestMasterToggle:
    def test_list_503_when_disabled(self, titles_client_disabled):
        r = titles_client_disabled.get("/api/titles/")
        assert r.status_code == 503

    def test_import_503_when_disabled(self, titles_client_disabled):
        r = titles_client_disabled.post(
            "/api/titles/", json={"source_id": "internal", "manifest": {}},
        )
        assert r.status_code == 503

    def test_launch_503_when_disabled(self, titles_client_disabled):
        r = titles_client_disabled.post("/api/titles/abc/launch")
        assert r.status_code == 503


# ── Import ──────────────────────────────────────────────────────────


class TestImport:
    def test_happy_path(self, titles_client):
        client, _svc, _conn = titles_client
        r = client.post(
            "/api/titles/",
            json={
                "source_id": "internal",
                "manifest": {
                    "kind": "web_app",
                    "title": "Acme Demo",
                    "source_id": "url-bookmark",
                    "source_remote_id": "https://example.com/acme",
                    "metadata": {"embed_url": "https://example.com/acme"},
                },
            },
        )
        assert r.status_code == 201
        body = r.json()["title"]
        assert body["title"] == "Acme Demo"
        assert body["kind"] == "web_app"
        assert body["library_state"]["pinned"] is True

    def test_unknown_kind_400(self, titles_client):
        client, *_ = titles_client
        r = client.post(
            "/api/titles/",
            json={
                "source_id": "internal",
                "manifest": {"kind": "no-such-kind", "title": "X"},
            },
        )
        assert r.status_code == 400

    def test_unknown_source_400(self, titles_client):
        client, *_ = titles_client
        r = client.post(
            "/api/titles/",
            json={
                "source_id": "no-such-source",
                "manifest": {"kind": "web_app", "title": "X"},
            },
        )
        assert r.status_code == 400


# ── List + isolation ─────────────────────────────────────────────────


class TestList:
    def test_user_isolation(self, titles_client):
        """Core isolation guarantee: u2's title is invisible to u1."""
        client, svc, _conn = titles_client
        _run(svc.import_title(
            user_id="usr_other",
            source_id="internal",
            manifest_data={
                "kind": "web_app", "title": "Theirs",
                "source_remote_id": "x",
                "metadata": {"embed_url": "https://x/x"},
            },
        ))
        r = client.get("/api/titles/")
        assert r.json()["titles"] == []

    def test_kind_filter(self, titles_client):
        client, svc, _conn = titles_client
        _run(svc.import_title(
            user_id="usr_test", source_id="internal",
            manifest_data={
                "kind": "web_app", "title": "WA",
                "source_remote_id": "wa",
                "metadata": {"embed_url": "https://x/wa"},
            },
        ))
        _run(svc.import_title(
            user_id="usr_test", source_id="internal",
            manifest_data={
                "kind": "git_project", "title": "GP",
                "source_remote_id": "gp",
                "metadata": {"embed_url": "https://x/gp"},
            },
        ))
        r = client.get("/api/titles/?kind=web_app")
        assert {t["title"] for t in r.json()["titles"]} == {"WA"}

    def test_unknown_kind_filter_400(self, titles_client):
        client, *_ = titles_client
        r = client.get("/api/titles/?kind=bogus")
        assert r.status_code == 400


# ── Launch + end_run ─────────────────────────────────────────────────


class TestLaunch:
    def test_launch_returns_handle_and_run_id(self, titles_client):
        client, svc, _conn = titles_client
        title = _run(svc.import_title(
            user_id="usr_test", source_id="internal",
            manifest_data={
                "kind": "web_app", "title": "Game",
                "source_remote_id": "g",
                "metadata": {"embed_url": "https://x/g"},
            },
        ))
        r = client.post(f"/api/titles/{title.id}/launch", json={})
        assert r.status_code == 201
        body = r.json()
        assert body["run_id"]
        assert body["handle"]["runtime_id"] == "browser-iframe"
        assert body["handle"]["target"] == "https://x/g"

    def test_launch_404_for_other_users_title(self, titles_client):
        client, svc, _conn = titles_client
        title = _run(svc.import_title(
            user_id="usr_other", source_id="internal",
            manifest_data={
                "kind": "web_app", "title": "Theirs",
                "source_remote_id": "x",
                "metadata": {"embed_url": "https://x/x"},
            },
        ))
        r = client.post(f"/api/titles/{title.id}/launch", json={})
        assert r.status_code == 404

    def test_end_run_closes(self, titles_client):
        client, svc, _conn = titles_client
        title = _run(svc.import_title(
            user_id="usr_test", source_id="internal",
            manifest_data={
                "kind": "web_app", "title": "Game",
                "source_remote_id": "g",
                "metadata": {"embed_url": "https://x/g"},
            },
        ))
        launched = client.post(f"/api/titles/{title.id}/launch", json={}).json()
        run_id = launched["run_id"]

        r = client.post(
            f"/api/titles/{title.id}/runs/{run_id}/end",
            json={
                "runtime_id": "browser-iframe",
                "exit_reason": "clean",
                "avg_fps": 60.0,
            },
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        runs = client.get(f"/api/titles/{title.id}/runs").json()["runs"]
        assert runs[0]["exit_reason"] == "clean"
        assert runs[0]["avg_fps"] == 60.0


# ── PATCH (pin / metadata) ───────────────────────────────────────────


class TestPatch:
    def test_unpin_round_trip(self, titles_client):
        client, svc, _conn = titles_client
        title = _run(svc.import_title(
            user_id="usr_test", source_id="internal",
            manifest_data={
                "kind": "web_app", "title": "T",
                "source_remote_id": "t",
                "metadata": {"embed_url": "https://x/t"},
            },
        ))
        # Imports default to pinned=True
        assert client.get(f"/api/titles/{title.id}").json()["title"]["library_state"]["pinned"] is True
        client.patch(f"/api/titles/{title.id}", json={"pinned": False})
        assert client.get(f"/api/titles/{title.id}").json()["title"]["library_state"]["pinned"] is False


# ── Registry introspection ───────────────────────────────────────────


class TestRegistry:
    def test_runtimes_includes_browser_iframe(self, titles_client):
        client, *_ = titles_client
        r = client.get("/api/titles/_/runtimes")
        ids = {rt["id"] for rt in r.json()["runtimes"]}
        assert "browser-iframe" in ids

    def test_sources_includes_internal(self, titles_client):
        client, *_ = titles_client
        r = client.get("/api/titles/_/sources")
        ids = {s["id"] for s in r.json()["sources"]}
        assert "internal" in ids
