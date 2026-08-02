"""Tests for discovery_routes.py — signals, history, knowledge search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_discovery_store():
    store = MagicMock()
    store.log_signal = AsyncMock(return_value={"id": "sig_1", "status": "logged"})
    store.has_source = AsyncMock(return_value=False)
    store.upsert_history = AsyncMock()
    store.list_history = AsyncMock(return_value=[
        {"id": "h1", "url": "https://example.com", "title": "Example"},
    ])
    store.delete_history = AsyncMock(return_value=True)
    store.search_library = AsyncMock(return_value=[])
    store.check_visited = AsyncMock(return_value=["https://example.com"])
    store.store_chunk = AsyncMock()
    store.increment_retrieved = AsyncMock()
    return store


class TestSignal:
    def test_signal_disabled(self, app, client, monkeypatch):
        from augmentum.config import settings
        object.__setattr__(settings, "discovery_enabled", False)
        resp = client.post(
            "/api/discovery/signal",
            json={"signal_type": "page_visit", "source_url": "https://example.com"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"
        object.__setattr__(settings, "discovery_enabled", True)

    def test_signal_no_store(self, app, client, monkeypatch):
        from augmentum.config import settings
        object.__setattr__(settings, "discovery_enabled", True)
        resp = client.post(
            "/api/discovery/signal",
            json={"signal_type": "page_visit"},
        )
        assert resp.status_code == 503

    def test_signal_missing_type(self, app, client, monkeypatch):
        from augmentum.config import settings
        object.__setattr__(settings, "discovery_enabled", True)
        app.state.discovery_store = _mock_discovery_store()
        resp = client.post("/api/discovery/signal", json={})
        assert resp.status_code == 400

    def test_signal_success(self, app, client, monkeypatch):
        from augmentum.config import settings
        object.__setattr__(settings, "discovery_enabled", True)
        app.state.discovery_store = _mock_discovery_store()
        resp = client.post(
            "/api/discovery/signal",
            json={
                "signal_type": "page_visit",
                "source_url": "https://example.com",
                "source_title": "Example",
            },
        )
        assert resp.status_code == 200


class TestHistory:
    def test_list_history_no_store(self, client):
        resp = client.get("/api/discovery/history")
        assert resp.status_code == 503

    def test_list_history_success(self, app, client):
        app.state.discovery_store = _mock_discovery_store()
        resp = client.get("/api/discovery/history?page=1")
        assert resp.status_code == 200
        assert "items" in resp.json()

    def test_delete_history_not_found(self, app, client):
        store = _mock_discovery_store()
        store.delete_history = AsyncMock(return_value=False)
        app.state.discovery_store = store
        resp = client.delete("/api/discovery/history/nonexistent")
        assert resp.status_code == 404

    def test_delete_history_success(self, app, client):
        app.state.discovery_store = _mock_discovery_store()
        resp = client.delete("/api/discovery/history/h1")
        assert resp.status_code == 200


class TestCheckVisited:
    def test_check_no_store(self, client):
        resp = client.get("/api/discovery/check-visited?urls=")
        assert resp.status_code == 503

    def test_check_success(self, app, client):
        app.state.discovery_store = _mock_discovery_store()
        resp = client.get(
            "/api/discovery/check-visited?urls=https://example.com,https://other.com"
        )
        assert resp.status_code == 200
        assert "visited" in resp.json()


class TestLibraryZones:
    """`/api/discovery/library` returns up to 4 zones (comics, audiobooks,
    movies, shows) seeded from the file_index. Empty zones are dropped
    from the response so the frontend doesn't render placeholder strips
    for media types the user hasn't installed.
    """

    def _stub_entry(self, *, file_id: str, name: str, kind: str,
                    entity_kind: str, source: str = "test_source",
                    progress_pct: float = 0.0, updated_at: str = "2026-05-07"):
        # Mirror the FileEntry attributes touched by _library_card.
        e = MagicMock()
        e.id = file_id
        e.name = name
        e.kind = kind
        e.source = source
        e.updated_at = updated_at
        e.source_metadata = {
            "entity_kind": entity_kind,
            "progress_pct": progress_pct,
            "author": "Test Author",
            "extra": {"year": 2024, "narrator": "Test Narrator"},
        }
        return e

    def test_library_no_user(self, client):
        # Unauthenticated path — conftest.py:74 autouses an auth bypass,
        # but with no file_index on app.state we expect an empty zones
        # dict (the endpoint short-circuits before calling list_recent).
        resp = client.get("/api/discovery/library")
        # Conftest provides a test user, so we hit the real path; with
        # no file_index attached, get_library_zones returns {"zones": {}}.
        assert resp.status_code == 200
        assert resp.json() == {"zones": {}}

    def test_library_empty_when_no_items(self, app, client):
        idx = MagicMock()
        idx.list_recent = AsyncMock(return_value=[])
        app.state.file_index = idx
        resp = client.get("/api/discovery/library")
        assert resp.status_code == 200
        # All four zone queries return empty → empty zones dict (zones
        # are dropped, not emitted as []).
        assert resp.json() == {"zones": {}}

    def test_library_populated_zones(self, app, client):
        idx = MagicMock()

        # Different return per query — keyed by (kind, entity_kinds).
        # in_progress sort comes first; "newest" sort comes second.
        async def _list_recent(**kwargs):
            kind = kwargs.get("kind")
            sort = kwargs.get("sort")
            entity_kinds = tuple(kwargs.get("entity_kinds") or [])
            if kind == "audio" and entity_kinds == ("book",):
                if sort == "progress":
                    return [self._stub_entry(
                        file_id="ab1", name="Dune",
                        kind="audio", entity_kind="book",
                        source="audiobookshelf", progress_pct=0.42,
                    )]
                return [self._stub_entry(
                    file_id="ab2", name="Foundation",
                    kind="audio", entity_kind="book",
                    source="audiobookshelf",
                )]
            if kind == "video" and entity_kinds == ("movie",):
                if sort == "progress":
                    return []
                return [self._stub_entry(
                    file_id="m1", name="Blade Runner",
                    kind="video", entity_kind="movie", source="jellyfin",
                )]
            return []

        idx.list_recent = _list_recent
        app.state.file_index = idx
        resp = client.get("/api/discovery/library")
        assert resp.status_code == 200
        zones = resp.json()["zones"]
        assert "audiobooks" in zones
        assert "movies" in zones
        # Empty zones (comics, shows) are dropped — keeps the frontend
        # from rendering empty strips for un-installed media types.
        assert "comics" not in zones
        assert "shows" not in zones
        # Audiobook zone: in-progress first (Dune), then recent (Foundation).
        assert [c["file_id"] for c in zones["audiobooks"]] == ["ab1", "ab2"]
        assert zones["audiobooks"][0]["status"] == "in_progress"
        assert zones["audiobooks"][1]["status"] == "recent"
        # Card shape includes everything the renderer needs.
        for c in zones["audiobooks"]:
            for field in ("title", "subtitle", "cover_url", "kind",
                          "entity_kind", "progress_pct", "source"):
                assert field in c, f"missing {field} in {c}"

    def test_library_no_overlap_between_slices(self, app, client):
        """An item that's both in-progress AND recently added must
        appear once (in the in_progress slice), not twice."""
        idx = MagicMock()
        shared = self._stub_entry(
            file_id="overlap", name="Shared",
            kind="audio", entity_kind="book",
            source="audiobookshelf", progress_pct=0.3,
        )

        async def _list_recent(**kwargs):
            if kwargs.get("kind") != "audio":
                return []
            if kwargs.get("sort") == "progress":
                return [shared]
            # "newest" returns the same id — should be filtered out.
            return [shared, self._stub_entry(
                file_id="other", name="Other",
                kind="audio", entity_kind="book",
                source="audiobookshelf",
            )]

        idx.list_recent = _list_recent
        app.state.file_index = idx
        resp = client.get("/api/discovery/library")
        assert resp.status_code == 200
        ids = [c["file_id"] for c in resp.json()["zones"]["audiobooks"]]
        # "overlap" should appear exactly once, "other" appended after.
        assert ids == ["overlap", "other"]


class TestInterests:
    def test_interests_placeholder(self, client):
        resp = client.get("/api/discovery/interests")
        assert resp.status_code == 200
        assert resp.json()["clusters"] == []


class TestDismiss:
    def test_dismiss_placeholder(self, client):
        resp = client.post("/api/discovery/dismiss", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "noted"


class _FakeSettingsStore:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value):
        self.values[key] = value


class TestFeeds:
    """GET/PUT /api/discovery/feeds — the For-You sources editor
    (2026-06-11; the four discovery_feeds_* keys previously had no
    edit path at all)."""

    def test_get_returns_defaults(self, app, client):
        app.state.settings_store = _FakeSettingsStore()
        resp = client.get("/api/discovery/feeds")
        assert resp.status_code == 200
        body = resp.json()
        assert body["hn"] is True          # HN defaults on
        assert body["rss_urls"] == []

    def test_put_normalizes_and_persists(self, app, client):
        store = _FakeSettingsStore()
        app.state.settings_store = store
        resp = client.put("/api/discovery/feeds", json={
            "hn": False,
            "reddit_subs": ["selfhosted", " LocalLLaMA ", "selfhosted"],
            "arxiv_cats": ["cs.AI"],
            "rss_urls": [
                "https://example.com/feed.xml",
                "rsshub://github/release/owner/repo",
                "not a url at all",
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["reddit_subs"] == ["selfhosted", "LocalLLaMA"]  # deduped
        assert body["rss_urls"] == [
            "https://example.com/feed.xml",
            "rsshub://github/release/owner/repo",
        ]
        assert store.values["discovery_feeds_hn"] == "0"
        assert store.values["discovery_feeds_rss"] == (
            "https://example.com/feed.xml,rsshub://github/release/owner/repo"
        )

    def test_put_then_get_round_trips(self, app, client):
        app.state.settings_store = _FakeSettingsStore()
        client.put("/api/discovery/feeds", json={
            "hn": True, "reddit_subs": ["augmentum"],
            "arxiv_cats": [], "rss_urls": ["rsshub://youtube/user/@x"],
        })
        body = client.get("/api/discovery/feeds").json()
        assert body["reddit_subs"] == ["augmentum"]
        assert body["rss_urls"] == ["rsshub://youtube/user/@x"]

    def test_put_invalid_body_400(self, app, client):
        app.state.settings_store = _FakeSettingsStore()
        resp = client.put(
            "/api/discovery/feeds",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
