"""HTTP routes for the notification substrate.

Uses the conftest ``sqlite_client`` fixture so migrations are
applied automatically. Tests flip ``settings.notifications_enabled``
on for the duration of each test — the default is off so each test
controls its own visibility.
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.config import settings
from augmentum.notifications import NotificationAction
from augmentum.notifications.store import publish


@pytest.fixture(autouse=True)
def _enable_notifications():
    """Force the flag on for these tests. Restores prior value after."""
    orig = settings.notifications_enabled
    object.__setattr__(settings, "notifications_enabled", True)
    yield
    object.__setattr__(settings, "notifications_enabled", orig)


def _seed_notification(
    sqlite_client, *, channel_id: str = "connect.message",
    source: str = "connect", title: str = "hi",
    actions: list[NotificationAction] | None = None,
    payload: dict | None = None,
) -> str:
    """Helper: insert a row directly through the store, return its id."""
    app = sqlite_client.app
    conn = app.state.state_manager.backend.conn

    async def _do() -> str:
        return await publish(
            conn,
            user_id="usr_test",  # matches conftest test_user.id
            channel_id=channel_id,
            source=source,
            title=title,
            actions=actions,
            payload=payload,
        )

    return asyncio.get_event_loop().run_until_complete(_do())


class TestFeatureFlag:
    def test_feed_503_when_disabled(self, sqlite_client) -> None:
        object.__setattr__(settings, "notifications_enabled", False)
        resp = sqlite_client.get("/api/notify/feed")
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()

    def test_channels_503_when_disabled(self, sqlite_client) -> None:
        object.__setattr__(settings, "notifications_enabled", False)
        resp = sqlite_client.get("/api/notify/channels")
        assert resp.status_code == 503

    def test_action_503_when_disabled(self, sqlite_client) -> None:
        object.__setattr__(settings, "notifications_enabled", False)
        resp = sqlite_client.post("/api/notify/x/action/y")
        assert resp.status_code == 503


class TestFeed:
    def test_empty_feed(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/notify/feed")
        assert resp.status_code == 200
        assert resp.json() == {"items": []}

    def test_seeded_row_shows_up(self, sqlite_client) -> None:
        nid = _seed_notification(sqlite_client, title="hello")
        resp = sqlite_client.get("/api/notify/feed")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["notification_id"] == nid
        assert items[0]["title"] == "hello"

    def test_unread_only_filter(self, sqlite_client) -> None:
        nid = _seed_notification(sqlite_client)
        # Mark read directly.
        sqlite_client.post(f"/api/notify/{nid}/read")
        unread = sqlite_client.get(
            "/api/notify/feed", params={"include_read": False},
        ).json()
        assert unread["items"] == []

    def test_dismissed_hidden_by_default(self, sqlite_client) -> None:
        nid = _seed_notification(sqlite_client)
        sqlite_client.post(f"/api/notify/{nid}/dismiss")
        feed = sqlite_client.get("/api/notify/feed").json()
        assert feed["items"] == []
        # But include_dismissed=true brings them back.
        archived = sqlite_client.get(
            "/api/notify/feed", params={"include_dismissed": True},
        ).json()
        assert len(archived["items"]) == 1


class TestChannels:
    def test_resolved_channels_include_catalog_defaults(
        self, sqlite_client,
    ) -> None:
        resp = sqlite_client.get("/api/notify/channels")
        assert resp.status_code == 200
        channels = resp.json()["channels"]
        ids = {c["channel_id"] for c in channels}
        # The canonical critical channel must always surface.
        assert "connect.call.incoming" in ids
        critical = next(c for c in channels if c["channel_id"] == "connect.call.incoming")
        assert critical["importance"] == 4

    def test_mute_then_read_back(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/notify/channels/connect.message/mute",
            json={"until": "2030-01-01T00:00:00+00:00"},
        )
        assert resp.status_code == 200

        channels = sqlite_client.get("/api/notify/channels").json()["channels"]
        muted = next(c for c in channels if c["channel_id"] == "connect.message")
        assert muted["user_customized"] is True
        assert muted["muted_until"].startswith("2030-")

    def test_unmute_clears_muted_until(self, sqlite_client) -> None:
        sqlite_client.post(
            "/api/notify/channels/connect.message/mute",
            json={"until": "2030-01-01T00:00:00+00:00"},
        )
        # Empty string = unmute.
        sqlite_client.post(
            "/api/notify/channels/connect.message/mute",
            json={"until": ""},
        )
        channels = sqlite_client.get("/api/notify/channels").json()["channels"]
        unmuted = next(c for c in channels if c["channel_id"] == "connect.message")
        assert unmuted["muted_until"] == ""


class TestReadDismiss:
    def test_mark_read_idempotent(self, sqlite_client) -> None:
        nid = _seed_notification(sqlite_client)
        # First call: 200, read: True
        first = sqlite_client.post(f"/api/notify/{nid}/read")
        assert first.status_code == 200
        # Second call: also 200 (no error). Idempotent semantics.
        second = sqlite_client.post(f"/api/notify/{nid}/read")
        assert second.status_code == 200

    def test_read_404_when_missing(self, sqlite_client) -> None:
        # Don't confirm existence to a hostile caller; nonexistent ids 404.
        resp = sqlite_client.post("/api/notify/never-existed/read")
        assert resp.status_code == 404

    def test_dismiss_404_when_missing(self, sqlite_client) -> None:
        resp = sqlite_client.post("/api/notify/never-existed/dismiss")
        assert resp.status_code == 404


class TestActions:
    def test_action_404_on_missing_notification(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/notify/never-existed/action/accept",
        )
        assert resp.status_code == 404

    def test_action_400_on_unknown_action_id(self, sqlite_client) -> None:
        nid = _seed_notification(
            sqlite_client,
            channel_id="connect.call.incoming",
            actions=[
                NotificationAction(id="accept", label="A"),
                NotificationAction(id="decline", label="D"),
            ],
        )
        resp = sqlite_client.post(f"/api/notify/{nid}/action/teleport")
        assert resp.status_code == 400
        assert "teleport" in resp.json()["detail"]

    def test_action_404_when_channel_has_no_handler(
        self, sqlite_client,
    ) -> None:
        # A channel with no registered handler returns 404 with a
        # clear message rather than 500.
        nid = _seed_notification(
            sqlite_client,
            channel_id="completely.unhandled.channel",
            actions=[NotificationAction(id="anything", label="x")],
        )
        resp = sqlite_client.post(f"/api/notify/{nid}/action/anything")
        assert resp.status_code == 404
        assert "handler" in resp.json()["detail"]
