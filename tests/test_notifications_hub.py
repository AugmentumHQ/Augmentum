"""NotificationHub — attach/detach/dispatch with filter semantics."""

from __future__ import annotations

import json

import pytest

from augmentum.notifications import (
    IMPORTANCE_CRITICAL,
    IMPORTANCE_DEFAULT,
    IMPORTANCE_LOW,
    NotificationAction,
)
from augmentum.notifications.hub import NotificationHub
from augmentum.notifications.store import Notification


class FakeWS:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    async def send_text(self, payload: str) -> None:
        if self.fail:
            raise RuntimeError("simulated failure")
        self.sent.append(payload)


def _notif(
    *, user_id: str = "u1",
    channel_id: str = "connect.message",
    importance: int = IMPORTANCE_DEFAULT,
    title: str = "hi",
) -> Notification:
    return Notification(
        notification_id="nid-1",
        user_id=user_id,
        channel_id=channel_id,
        source="connect",
        title=title,
        importance=importance,
    )


@pytest.mark.asyncio
class TestAttachDetach:
    async def test_attach_records_user(self) -> None:
        hub = NotificationHub()
        await hub.attach(ws=FakeWS(), user_id="u1")
        assert hub.online_user_ids() == ["u1"]

    async def test_attach_requires_user_id(self) -> None:
        hub = NotificationHub()
        with pytest.raises(ValueError, match="user_id"):
            await hub.attach(ws=FakeWS(), user_id="")

    async def test_detach_removes_when_last_connection_gone(self) -> None:
        hub = NotificationHub()
        att = await hub.attach(ws=FakeWS(), user_id="u1")
        await hub.detach(att.connection_id)
        assert hub.online_user_ids() == []

    async def test_multiple_attachments_per_user(self) -> None:
        # Same user on desktop + phone — both should receive dispatches.
        hub = NotificationHub()
        a = FakeWS()
        b = FakeWS()
        await hub.attach(ws=a, user_id="u1")
        await hub.attach(ws=b, user_id="u1")

        delivered = await hub.dispatch(notification=_notif())
        assert delivered == 2

    async def test_detach_unknown_is_noop(self) -> None:
        hub = NotificationHub()
        await hub.detach("conn-does-not-exist")  # no crash


@pytest.mark.asyncio
class TestDispatch:
    async def test_per_user_isolation(self) -> None:
        # The single most important security property: u1's
        # notifications never reach u2's WS.
        hub = NotificationHub()
        u1 = FakeWS()
        u2 = FakeWS()
        await hub.attach(ws=u1, user_id="u1")
        await hub.attach(ws=u2, user_id="u2")

        delivered = await hub.dispatch(notification=_notif(user_id="u1"))
        assert delivered == 1
        assert len(u1.sent) == 1
        assert len(u2.sent) == 0

    async def test_offline_user_returns_zero(self) -> None:
        hub = NotificationHub()
        assert await hub.dispatch(notification=_notif(user_id="nobody")) == 0

    async def test_channel_pattern_filter_matches(self) -> None:
        # Subscriber asks only for "connect.call.*" — connect.message
        # must not reach them.
        hub = NotificationHub()
        ws = FakeWS()
        await hub.attach(ws=ws, user_id="u1", channel_pattern="connect.call.*")

        # Non-matching channel: skipped.
        assert await hub.dispatch(
            notification=_notif(channel_id="connect.message"),
        ) == 0
        # Matching channel: delivered.
        assert await hub.dispatch(
            notification=_notif(channel_id="connect.call.incoming"),
        ) == 1

    async def test_wildcard_pattern_matches_anything(self) -> None:
        hub = NotificationHub()
        await hub.attach(ws=FakeWS(), user_id="u1", channel_pattern="*")
        assert await hub.dispatch(notification=_notif(channel_id="x.y.z")) == 1

    async def test_importance_floor_drops_below(self) -> None:
        # Subscriber only wants HIGH+; LOW must be dropped.
        hub = NotificationHub()
        await hub.attach(
            ws=FakeWS(), user_id="u1", importance_floor=IMPORTANCE_CRITICAL,
        )
        assert await hub.dispatch(
            notification=_notif(importance=IMPORTANCE_LOW),
        ) == 0
        assert await hub.dispatch(
            notification=_notif(importance=IMPORTANCE_CRITICAL),
        ) == 1

    async def test_failed_send_does_not_break_others(self) -> None:
        hub = NotificationHub()
        bad = FakeWS(fail=True)
        good = FakeWS()
        await hub.attach(ws=bad, user_id="u1")
        await hub.attach(ws=good, user_id="u1")

        delivered = await hub.dispatch(notification=_notif())
        assert delivered == 1  # only the good one succeeded
        assert len(good.sent) == 1

    async def test_dispatch_payload_shape(self) -> None:
        # The wire-shape of a push frame is contract — UIs parse on it.
        hub = NotificationHub()
        ws = FakeWS()
        await hub.attach(ws=ws, user_id="u1")

        notif = Notification(
            notification_id="nid-X",
            user_id="u1",
            channel_id="connect.message",
            source="connect",
            title="hi",
            body="from bob",
            importance=IMPORTANCE_DEFAULT,
            actions=(NotificationAction(id="reply", label="Reply"),),
            payload={"call_id": "c-1"},
        )
        await hub.dispatch(notification=notif)

        parsed = json.loads(ws.sent[0])
        assert parsed["type"] == "notification"
        n = parsed["notification"]
        assert n["notification_id"] == "nid-X"
        assert n["title"] == "hi"
        assert n["actions"] == [{"id": "reply", "label": "Reply"}]
        assert n["payload"] == {"call_id": "c-1"}


@pytest.mark.asyncio
class TestPublishAndDispatch:
    """The wrapper that bridges store.publish + hub.dispatch."""

    async def test_persists_then_pushes(self, tmp_path) -> None:
        from pathlib import Path

        import aiosqlite

        from augmentum.notifications.hub import publish_and_dispatch
        from augmentum.notifications.store import list_for_user

        migration = Path(
            "augmentum/state/migrations/221_notification_substrate.sql"
        ).read_text()

        async with aiosqlite.connect(":memory:") as conn:
            await conn.executescript(migration)
            await conn.commit()

            hub = NotificationHub()
            ws = FakeWS()
            await hub.attach(ws=ws, user_id="u1")

            nid = await publish_and_dispatch(
                conn,
                hub=hub,
                user_id="u1",
                channel_id="connect.message",
                source="connect",
                title="hi",
            )
            # Persisted...
            feed = await list_for_user(conn, user_id="u1")
            assert len(feed) == 1
            assert feed[0].notification_id == nid
            # ...and pushed.
            assert len(ws.sent) == 1


@pytest.mark.asyncio
class TestImportanceFanout:
    """2026-06-11 policy: importance >= HIGH web-pushes ALL devices even
    when a live WS client already rendered it. The old offline-only
    gate left a locked phone silent for a tornado warning whenever a
    desktop tab was open at home. Below HIGH keeps offline-only."""

    async def _publish(self, monkeypatch, *, importance, attach_ws,
                       transient=False):
        from pathlib import Path

        import aiosqlite

        from augmentum.notifications import hub as hub_mod

        pushed = []

        async def _spy_webpush(conn, *, notification):
            pushed.append(notification.importance)
            return 1
        monkeypatch.setattr(hub_mod, "_dispatch_webpush", _spy_webpush)

        migration = Path(
            "augmentum/state/migrations/221_notification_substrate.sql"
        ).read_text()
        async with aiosqlite.connect(":memory:") as conn:
            await conn.executescript(migration)
            await conn.commit()
            hub = NotificationHub()
            if attach_ws:
                ws = FakeWS()
                await hub.attach(ws=ws, user_id="u1")
            await hub_mod.publish_and_dispatch(
                conn, hub=hub, user_id="u1",
                channel_id="alerts.home", source="t",
                title="x", importance=importance, transient=transient,
            )
        return pushed

    async def test_high_pushes_even_with_live_client(self, monkeypatch):
        from augmentum.notifications.catalog import IMPORTANCE_HIGH
        pushed = await self._publish(
            monkeypatch, importance=IMPORTANCE_HIGH, attach_ws=True,
        )
        assert pushed == [IMPORTANCE_HIGH]

    async def test_critical_pushes_even_with_live_client(self, monkeypatch):
        from augmentum.notifications.catalog import IMPORTANCE_CRITICAL
        pushed = await self._publish(
            monkeypatch, importance=IMPORTANCE_CRITICAL, attach_ws=True,
        )
        assert pushed == [IMPORTANCE_CRITICAL]

    async def test_default_skips_push_when_live_client(self, monkeypatch):
        from augmentum.notifications.catalog import IMPORTANCE_DEFAULT
        pushed = await self._publish(
            monkeypatch, importance=IMPORTANCE_DEFAULT, attach_ws=True,
        )
        assert pushed == []

    async def test_default_pushes_when_offline(self, monkeypatch):
        from augmentum.notifications.catalog import IMPORTANCE_DEFAULT
        pushed = await self._publish(
            monkeypatch, importance=IMPORTANCE_DEFAULT, attach_ws=False,
        )
        assert pushed == [IMPORTANCE_DEFAULT]

    async def test_transient_never_pushes(self, monkeypatch):
        from augmentum.notifications.catalog import IMPORTANCE_CRITICAL
        pushed = await self._publish(
            monkeypatch, importance=IMPORTANCE_CRITICAL, attach_ws=True,
            transient=True,
        )
        assert pushed == []
