"""Session C — Web Push substrate tests.

Three slices:

1. VAPID key lifecycle (generation idempotent + b64url shape).
2. Subscription HTTP routes (POST/GET/DELETE round-trip + upsert).
3. Dispatcher: ``_dispatch_webpush`` calls ``send_webpush`` for each
   matching subscription, prunes 410 responses, and is skipped when
   the live WS dispatched to at least one client.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiosqlite
import pytest

from augmentum.config import settings
from augmentum.notifications.hub import (
    NotificationHub,
    _dispatch_webpush,
    publish_and_dispatch,
)
from augmentum.notifications.store import NotificationAction
from augmentum.notifications.webpush import (
    DEFAULT_SUBJECT,
    SETTING_VAPID_PRIVATE,
    SETTING_VAPID_PUBLIC,
    WebPushSendResult,
    ensure_vapid_keys,
    get_public_key,
)

NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text()

# Inline DDL for app_settings — pulling migration 007 in would also
# require schema_version, which isn't relevant to these tests.
APP_SETTINGS_DDL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(APP_SETTINGS_DDL)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        await c.commit()
        yield c


@pytest.fixture(autouse=True)
def _enable_notifications():
    orig = settings.notifications_enabled
    object.__setattr__(settings, "notifications_enabled", True)
    yield
    object.__setattr__(settings, "notifications_enabled", orig)


# ── VAPID lifecycle ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vapid_keys_generated_on_first_call(conn) -> None:
    keys = await ensure_vapid_keys(conn)
    assert keys.public_b64url
    assert keys.private_b64url
    assert keys.subject == DEFAULT_SUBJECT
    # b64url charset only — no '+', '/' or '=' padding.
    for ch in keys.public_b64url + keys.private_b64url:
        assert ch.isalnum() or ch in "-_"


@pytest.mark.asyncio
async def test_vapid_keys_idempotent(conn) -> None:
    a = await ensure_vapid_keys(conn)
    b = await ensure_vapid_keys(conn)
    assert a.public_b64url == b.public_b64url
    assert a.private_b64url == b.private_b64url


@pytest.mark.asyncio
async def test_get_public_key_matches_ensure(conn) -> None:
    keys = await ensure_vapid_keys(conn)
    public = await get_public_key(conn)
    assert public == keys.public_b64url


@pytest.mark.asyncio
async def test_vapid_keys_persist_across_calls(conn) -> None:
    """Once persisted, a fresh ensure call reads from app_settings —
    verify both the public and private survived the round-trip via
    the table rather than living only in module-scope state."""

    keys = await ensure_vapid_keys(conn)
    cur = await conn.execute(
        "SELECT value FROM app_settings WHERE key = ?",
        (SETTING_VAPID_PUBLIC,),
    )
    row = await cur.fetchone()
    assert row[0] == keys.public_b64url
    cur = await conn.execute(
        "SELECT value FROM app_settings WHERE key = ?",
        (SETTING_VAPID_PRIVATE,),
    )
    row = await cur.fetchone()
    assert row[0] == keys.private_b64url


# ── Subscription routes ──────────────────────────────────────────


def _stub_subscription(endpoint: str = "https://push.example/abc") -> dict:
    return {
        "endpoint": endpoint,
        "p256dh": "BAgYz-stub-p256dh-bytes-padded-out-32-chars",
        "auth": "auth-stub-padding",
    }


class TestSubscriptionRoutes:
    def test_vapid_key_endpoint(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/notify/vapid-public-key")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("public_key")

    def test_vapid_key_503_when_disabled(self, sqlite_client) -> None:
        object.__setattr__(settings, "notifications_enabled", False)
        resp = sqlite_client.get("/api/notify/vapid-public-key")
        assert resp.status_code == 503

    def test_subscribe_creates_row(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/notify/subscriptions",
            json={**_stub_subscription(), "channel_pattern": "connect.*"},
        )
        assert resp.status_code == 200, resp.text
        sub_id = resp.json()["subscription_id"]
        assert sub_id.startswith("sub_")

        # List shows it back, with no secrets leaked.
        listed = sqlite_client.get("/api/notify/subscriptions").json()
        subs = listed["subscriptions"]
        assert len(subs) == 1
        s = subs[0]
        assert s["subscription_id"] == sub_id
        assert s["target_kind"] == "webpush"
        assert s["endpoint"] == "https://push.example/abc"
        assert s["channel_pattern"] == "connect.*"
        # Keys must NOT appear in the list response.
        assert "p256dh" not in s
        assert "auth" not in s

    def test_subscribe_is_idempotent_on_endpoint(self, sqlite_client) -> None:
        """Re-posting the same endpoint updates in place instead of
        creating duplicate rows. Browsers may re-subscribe on every
        session restart with the same endpoint string."""

        resp1 = sqlite_client.post(
            "/api/notify/subscriptions",
            json={**_stub_subscription(), "channel_pattern": "*"},
        )
        sub_id_1 = resp1.json()["subscription_id"]

        # Re-post with a different pattern + floor.
        resp2 = sqlite_client.post(
            "/api/notify/subscriptions",
            json={
                **_stub_subscription(),
                "channel_pattern": "connect.message",
                "importance_floor": 3,
            },
        )
        assert resp2.status_code == 200
        sub_id_2 = resp2.json()["subscription_id"]
        assert sub_id_2 == sub_id_1, "expected same row, got a duplicate"

        listed = sqlite_client.get("/api/notify/subscriptions").json()
        assert len(listed["subscriptions"]) == 1
        s = listed["subscriptions"][0]
        assert s["channel_pattern"] == "connect.message"
        assert s["importance_floor"] == 3

    def test_subscribe_rejects_bad_endpoint(self, sqlite_client) -> None:
        bad = {**_stub_subscription(), "endpoint": "not-a-url"}
        resp = sqlite_client.post("/api/notify/subscriptions", json=bad)
        assert resp.status_code == 400

    def test_subscribe_rejects_missing_keys(self, sqlite_client) -> None:
        bad = _stub_subscription()
        bad["p256dh"] = ""
        resp = sqlite_client.post("/api/notify/subscriptions", json=bad)
        assert resp.status_code == 400

    def test_delete_subscription(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/notify/subscriptions",
            json=_stub_subscription(),
        )
        sub_id = resp.json()["subscription_id"]
        delete = sqlite_client.delete(f"/api/notify/subscriptions/{sub_id}")
        assert delete.status_code == 200
        listed = sqlite_client.get("/api/notify/subscriptions").json()
        assert listed["subscriptions"] == []

    def test_delete_unknown_404(self, sqlite_client) -> None:
        resp = sqlite_client.delete("/api/notify/subscriptions/nope")
        assert resp.status_code == 404


# ── Dispatcher ───────────────────────────────────────────────────


async def _seed_subscription(
    conn, *, user_id: str, sub_id: str, endpoint: str = "https://push.example/x",
    pattern: str = "*", floor: int = 0,
) -> None:
    target_address = json.dumps({
        "endpoint": endpoint,
        "p256dh": "p-stub",
        "auth": "a-stub",
    })
    await conn.execute(
        "INSERT INTO notification_subscriptions "
        "(subscription_id, user_id, channel_pattern, target_kind, "
        "target_address, importance_floor) "
        "VALUES (?, ?, ?, 'webpush', ?, ?)",
        (sub_id, user_id, pattern, target_address, floor),
    )
    await conn.commit()


def _fake_notification(user_id="alice"):
    """Build a Notification dataclass with the minimum the dispatcher needs."""

    from augmentum.notifications.store import Notification

    return Notification(
        notification_id="n1",
        user_id=user_id,
        channel_id="connect.message",
        source="connect",
        title="Alice",
        body="hi from offline test",
        importance=2,
        thread_id="t1",
        actions=(NotificationAction(id="open_thread", label="Open"),),
        payload={"thread_id": "t1"},
    )


@pytest.mark.asyncio
async def test_dispatcher_sends_to_matching_subscription(conn) -> None:
    await _seed_subscription(conn, user_id="alice", sub_id="sub1", pattern="*")
    notif = _fake_notification("alice")

    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=201)),
    ) as mocked:
        delivered = await _dispatch_webpush(conn, notification=notif)

    assert delivered == 1
    assert mocked.call_count == 1
    kwargs = mocked.call_args.kwargs
    assert kwargs["endpoint"] == "https://push.example/x"
    assert kwargs["payload"]["notification_id"] == "n1"
    assert kwargs["payload"]["title"] == "Alice"
    # Actions get rewritten to the SW shape ({action, title}).
    actions = kwargs["payload"]["actions"]
    assert actions == [{"action": "open_thread", "title": "Open"}]


@pytest.mark.asyncio
async def test_dispatcher_filters_by_pattern(conn) -> None:
    await _seed_subscription(
        conn, user_id="alice", sub_id="sub1",
        pattern="coder.*",  # won't match connect.message
    )
    notif = _fake_notification("alice")
    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=201)),
    ) as mocked:
        delivered = await _dispatch_webpush(conn, notification=notif)
    assert delivered == 0
    assert mocked.call_count == 0


@pytest.mark.asyncio
async def test_dispatcher_filters_by_importance_floor(conn) -> None:
    await _seed_subscription(
        conn, user_id="alice", sub_id="sub1", floor=4,
    )
    notif = _fake_notification("alice")  # importance=2 < floor=4
    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=201)),
    ) as mocked:
        delivered = await _dispatch_webpush(conn, notification=notif)
    assert delivered == 0
    assert mocked.call_count == 0


@pytest.mark.asyncio
async def test_dispatcher_prunes_410_responses(conn) -> None:
    await _seed_subscription(conn, user_id="alice", sub_id="dead")
    notif = _fake_notification("alice")

    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=410, expired=True)),
    ):
        await _dispatch_webpush(conn, notification=notif)

    # Row should be deleted after the expired send.
    cur = await conn.execute(
        "SELECT COUNT(*) FROM notification_subscriptions",
    )
    row = await cur.fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_publish_skips_webpush_when_ws_delivered(conn) -> None:
    """When the in-process hub dispatches to at least one live WS,
    we don't also fan out via Web Push (avoid double-buzzing the
    user's already-open desktop)."""

    await _seed_subscription(conn, user_id="alice", sub_id="sub1")

    hub = NotificationHub()
    # Stub a fake attachment so hub.dispatch returns >0.
    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.sent.append(payload)

    fake_ws = _FakeWS()
    await hub.attach(ws=fake_ws, user_id="alice", channel_pattern="*")

    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=201)),
    ) as mocked:
        await publish_and_dispatch(
            conn,
            hub=hub,
            user_id="alice",
            channel_id="connect.message",
            source="connect",
            title="t", body="b",
        )

    assert len(fake_ws.sent) == 1  # WS got it
    assert mocked.call_count == 0  # webpush did NOT fire


@pytest.mark.asyncio
async def test_publish_uses_webpush_when_no_live_ws(conn) -> None:
    await _seed_subscription(conn, user_id="alice", sub_id="sub1")

    hub = NotificationHub()  # nothing attached
    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=201)),
    ) as mocked:
        await publish_and_dispatch(
            conn,
            hub=hub,
            user_id="alice",
            channel_id="connect.message",
            source="connect",
            title="offline test", body="hello",
        )

    assert mocked.call_count == 1


@pytest.mark.asyncio
async def test_publish_skips_webpush_for_transient_notifications(conn) -> None:
    """Transient = ephemeral UI flash; should never wake up a
    sleeping browser via Web Push."""

    await _seed_subscription(conn, user_id="alice", sub_id="sub1")
    hub = NotificationHub()  # offline
    with patch(
        "augmentum.notifications.webpush.send_webpush",
        new=MagicMock(return_value=WebPushSendResult(status=201)),
    ) as mocked:
        await publish_and_dispatch(
            conn,
            hub=hub,
            user_id="alice",
            channel_id="connect.message",
            source="connect",
            title="t", body="b",
            transient=True,
        )
    assert mocked.call_count == 0
