"""Notification store — publish/read/dedup/mute/expire contract.

Tests apply migration 221 directly to an in-memory SQLite so the
schema-under-test is the actual schema the production DB will run.
This pins both the migration AND the store layer together — a
breaking edit to either surfaces here.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest

from augmentum.notifications import (
    DEFAULT_CHANNELS,
    IMPORTANCE_CRITICAL,
    IMPORTANCE_DEFAULT,
    IMPORTANCE_HIGH,
    IMPORTANCE_LOW,
    Notification,
    NotificationAction,
    NotificationStore,
    catalog_channel,
)
from augmentum.notifications.store import (
    dismiss,
    expire_transient,
    get_notification,
    list_for_user,
    mark_delivered,
    mark_read,
    mute_channel,
    publish,
    resolved_channels,
)


MIGRATION = (
    Path("augmentum/state/migrations/221_notification_substrate.sql").read_text()
)


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(MIGRATION)
        await c.commit()
        yield c


U1 = "user-alpha"
U2 = "user-beta"


# ── Publish ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPublish:
    async def test_minimal_publish_inserts_row(self, conn) -> None:
        notification_id = await publish(
            conn,
            user_id=U1,
            channel_id="connect.message",
            source="connect",
            title="New message",
            body="Bob sent: hi",
        )

        assert notification_id  # 16-char id
        assert len(notification_id) == 16

        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert notif is not None
        assert notif.title == "New message"
        assert notif.body == "Bob sent: hi"
        assert notif.channel_id == "connect.message"
        # Catalog default for connect.message is DEFAULT.
        assert notif.importance == IMPORTANCE_DEFAULT

    async def test_importance_falls_back_to_catalog(self, conn) -> None:
        # connect.call.incoming is CRITICAL in the catalog.
        notification_id = await publish(
            conn,
            user_id=U1,
            channel_id="connect.call.incoming",
            source="connect",
            title="Ringing",
        )
        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert notif.importance == IMPORTANCE_CRITICAL

    async def test_publisher_can_override_importance(self, conn) -> None:
        # Publisher escalates a normally-low channel for one event.
        notification_id = await publish(
            conn,
            user_id=U1,
            channel_id="companion.initiative",
            source="companion",
            title="!!",
            importance=IMPORTANCE_HIGH,
        )
        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert notif.importance == IMPORTANCE_HIGH

    async def test_unknown_channel_defaults_to_importance_2(self, conn) -> None:
        # Forward-compat: a future publisher may pick a channel id
        # we don't yet ship. We accept it (no foreign-key constraint)
        # and fall back to DEFAULT importance.
        notification_id = await publish(
            conn,
            user_id=U1,
            channel_id="future.subsystem.event",
            source="future",
            title="?",
        )
        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert notif.importance == IMPORTANCE_DEFAULT

    async def test_publish_requires_user_id(self, conn) -> None:
        # Per-user isolation must be enforced at publish — never
        # quietly accept "" as the anon row.
        with pytest.raises(ValueError, match="user_id"):
            await publish(
                conn, user_id="", channel_id="x", source="y", title="z",
            )

    async def test_publish_requires_title(self, conn) -> None:
        # A title-less notification has nothing to render.
        with pytest.raises(ValueError, match="title"):
            await publish(
                conn, user_id=U1, channel_id="x", source="y", title="",
            )

    async def test_actions_round_trip(self, conn) -> None:
        notification_id = await publish(
            conn,
            user_id=U1,
            channel_id="connect.call.incoming",
            source="connect",
            title="Ringing",
            actions=[
                NotificationAction(id="accept", label="Accept", style="primary"),
                NotificationAction(id="decline", label="Decline", style="danger"),
            ],
        )
        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert len(notif.actions) == 2
        assert notif.actions[0].id == "accept"
        assert notif.actions[0].style == "primary"
        assert notif.actions[1].style == "danger"

    async def test_payload_round_trips(self, conn) -> None:
        # Opaque per-source state. The action handler reads this to
        # know what to do — e.g. which call_id to accept.
        payload = {"call_id": "c-1", "party_id": "abcd1234", "modalities": "audio"}
        notification_id = await publish(
            conn, user_id=U1, channel_id="connect.call.incoming",
            source="connect", title="Ringing", payload=payload,
        )
        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert notif.payload == payload


# ── Dedupe ───────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDedupe:
    async def test_same_dedupe_key_updates_in_place(self, conn) -> None:
        # The single most important guarantee: repost = update, not
        # new row. This is what makes "ringing → connected → ended"
        # one entry instead of three.
        first_id = await publish(
            conn, user_id=U1, channel_id="connect.call.incoming",
            source="connect", title="Ringing", dedupe_key="call-7",
        )
        second_id = await publish(
            conn, user_id=U1, channel_id="connect.call.incoming",
            source="connect", title="Connected", dedupe_key="call-7",
        )
        assert first_id == second_id

        notif = await get_notification(
            conn, user_id=U1, notification_id=first_id,
        )
        assert notif.title == "Connected"

    async def test_repost_preserves_created_at(self, conn) -> None:
        # Feed ordering by created_at must stay stable across
        # in-place updates so notifications don't jump to the top
        # of the list when their state changes.
        first_id = await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="v1", dedupe_key="k",
        )
        first = await get_notification(
            conn, user_id=U1, notification_id=first_id,
        )
        created_at_first = first.created_at

        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="v2", dedupe_key="k",
        )
        second = await get_notification(
            conn, user_id=U1, notification_id=first_id,
        )
        assert second.created_at == created_at_first
        assert second.updated_at >= created_at_first

    async def test_repost_clears_read_and_dismissed(self, conn) -> None:
        # If the user already read or dismissed a row and the source
        # reposts (state moved on), the row should re-surface.
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="v1", dedupe_key="k",
        )
        await mark_read(conn, user_id=U1, notification_id=notification_id)
        await dismiss(conn, user_id=U1, notification_id=notification_id)

        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="v2", dedupe_key="k",
        )
        notif = await get_notification(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert notif.read_at == ""
        assert notif.dismissed_at == ""

    async def test_empty_dedupe_key_does_not_collide(self, conn) -> None:
        # The partial index excludes empty keys — many ad-hoc toasts
        # with no dedupe_key must NOT collapse into one row.
        a = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        b = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="b",
        )
        assert a != b
        feed = await list_for_user(conn, user_id=U1)
        assert len(feed) == 2

    async def test_dedupe_scoped_per_user(self, conn) -> None:
        # Two users may legitimately share the same source + dedupe_key
        # (e.g. both have a "call-7" against the same fabric peer);
        # rows must stay distinct.
        a = await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="alpha", dedupe_key="call-7",
        )
        b = await publish(
            conn, user_id=U2, channel_id="x", source="s",
            title="beta", dedupe_key="call-7",
        )
        assert a != b

    async def test_dedupe_scoped_per_source(self, conn) -> None:
        # Coder and Connect can share dedupe_key="42" without colliding.
        a = await publish(
            conn, user_id=U1, channel_id="x", source="coder",
            title="A", dedupe_key="42",
        )
        b = await publish(
            conn, user_id=U1, channel_id="y", source="connect",
            title="B", dedupe_key="42",
        )
        assert a != b


# ── List / filter ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestListForUser:
    async def test_per_user_isolation(self, conn) -> None:
        # The single most important security property: no cross-
        # user read path.
        await publish(
            conn, user_id=U1, channel_id="x", source="s", title="alpha",
        )
        await publish(
            conn, user_id=U2, channel_id="x", source="s", title="beta",
        )
        feed_u1 = await list_for_user(conn, user_id=U1)
        feed_u2 = await list_for_user(conn, user_id=U2)
        assert len(feed_u1) == 1
        assert len(feed_u2) == 1
        assert feed_u1[0].title == "alpha"
        assert feed_u2[0].title == "beta"

    async def test_newest_first_ordering(self, conn) -> None:
        import asyncio
        a = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="first",
        )
        # Windows datetime.now() has 15.6ms resolution; sleep > one
        # tick so the two rows get distinct created_at values and
        # the test exercises real time-ordering, not the
        # notification_id tiebreaker.
        await asyncio.sleep(0.02)
        b = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="second",
        )
        feed = await list_for_user(conn, user_id=U1)
        assert feed[0].notification_id == b  # newest first
        assert feed[1].notification_id == a

    async def test_include_read_default_true(self, conn) -> None:
        # Default: feed includes already-read items, only dismissed
        # are filtered. Mirrors what "all undismissed notifications"
        # means in Slack/Discord.
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        await mark_read(conn, user_id=U1, notification_id=notification_id)
        feed = await list_for_user(conn, user_id=U1)
        assert len(feed) == 1  # still there

    async def test_include_read_false_filters_read(self, conn) -> None:
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        await mark_read(conn, user_id=U1, notification_id=notification_id)
        feed = await list_for_user(conn, user_id=U1, include_read=False)
        assert feed == []

    async def test_dismissed_hidden_by_default(self, conn) -> None:
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        await dismiss(conn, user_id=U1, notification_id=notification_id)
        feed = await list_for_user(conn, user_id=U1)
        assert feed == []

    async def test_include_dismissed_true_includes_them(self, conn) -> None:
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        await dismiss(conn, user_id=U1, notification_id=notification_id)
        feed = await list_for_user(
            conn, user_id=U1, include_dismissed=True,
        )
        assert len(feed) == 1

    async def test_thread_filter(self, conn) -> None:
        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="A", thread_id="t1",
        )
        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="B", thread_id="t1",
        )
        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="C", thread_id="t2",
        )
        scoped = await list_for_user(conn, user_id=U1, thread_id="t1")
        assert {n.title for n in scoped} == {"A", "B"}

    async def test_limit_caps_returned_rows(self, conn) -> None:
        for i in range(5):
            await publish(
                conn, user_id=U1, channel_id="x", source="s", title=str(i),
            )
        feed = await list_for_user(conn, user_id=U1, limit=2)
        assert len(feed) == 2

    async def test_empty_user_returns_empty(self, conn) -> None:
        # Belt-and-suspenders for the per-user filter.
        feed = await list_for_user(conn, user_id="")
        assert feed == []


# ── Lifecycle (delivered / read / dismiss) ───────────────────────


@pytest.mark.asyncio
class TestLifecycle:
    async def test_mark_read_is_idempotent(self, conn) -> None:
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        assert await mark_read(
            conn, user_id=U1, notification_id=notification_id,
        )
        # Second call shouldn't error and shouldn't claim a change.
        assert not await mark_read(
            conn, user_id=U1, notification_id=notification_id,
        )

    async def test_dismiss_is_idempotent(self, conn) -> None:
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        assert await dismiss(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert not await dismiss(
            conn, user_id=U1, notification_id=notification_id,
        )

    async def test_mark_read_wrong_user_is_noop(self, conn) -> None:
        # Cross-user mark-read must not work — defense in depth even
        # though the route layer should already prevent the call.
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        assert not await mark_read(
            conn, user_id=U2, notification_id=notification_id,
        )

    async def test_mark_delivered_stamps_first_only(self, conn) -> None:
        notification_id = await publish(
            conn, user_id=U1, channel_id="x", source="s", title="a",
        )
        assert await mark_delivered(
            conn, user_id=U1, notification_id=notification_id,
        )
        assert not await mark_delivered(
            conn, user_id=U1, notification_id=notification_id,
        )


# ── Channels (catalog + mute) ────────────────────────────────────


@pytest.mark.asyncio
class TestChannels:
    async def test_default_channels_resolve_from_catalog(self, conn) -> None:
        # No overrides exist yet — every default channel should
        # show up with its catalog defaults and user_customized=False.
        channels = await resolved_channels(conn, user_id=U1)
        assert len(channels) == len(DEFAULT_CHANNELS)
        by_id = {c.channel_id: c for c in channels}
        assert by_id["connect.call.incoming"].importance == IMPORTANCE_CRITICAL
        assert not by_id["connect.call.incoming"].user_customized

    async def test_mute_materialises_row_and_shows_in_resolved(self, conn) -> None:
        # Muting a default-catalog channel lazily creates the row.
        await mute_channel(
            conn, user_id=U1,
            channel_id="companion.initiative",
            until_iso="2030-01-01T00:00:00+00:00",
        )
        channels = await resolved_channels(conn, user_id=U1)
        by_id = {c.channel_id: c for c in channels}
        assert by_id["companion.initiative"].user_customized
        assert by_id["companion.initiative"].muted_until.startswith("2030-")

    async def test_mute_then_unmute_clears(self, conn) -> None:
        # Unmute = pass None.
        await mute_channel(
            conn, user_id=U1, channel_id="connect.message",
            until_iso="2030-01-01T00:00:00+00:00",
        )
        await mute_channel(
            conn, user_id=U1, channel_id="connect.message", until_iso=None,
        )
        channels = await resolved_channels(conn, user_id=U1)
        by_id = {c.channel_id: c for c in channels}
        assert by_id["connect.message"].muted_until == ""

    async def test_per_user_mute_isolation(self, conn) -> None:
        # U1 mutes their own channel; U2's view is unaffected.
        await mute_channel(
            conn, user_id=U1, channel_id="connect.message",
            until_iso="2030-01-01T00:00:00+00:00",
        )
        channels_u2 = await resolved_channels(conn, user_id=U2)
        by_id_u2 = {c.channel_id: c for c in channels_u2}
        assert not by_id_u2["connect.message"].user_customized

    async def test_mute_unknown_channel_does_not_crash(self, conn) -> None:
        # Forward-compat: muting a channel id we don't have a
        # template for must still work (with safe defaults).
        await mute_channel(
            conn, user_id=U1, channel_id="future.channel.id",
            until_iso="2030-01-01T00:00:00+00:00",
        )
        channels = await resolved_channels(conn, user_id=U1)
        ids = {c.channel_id for c in channels}
        assert "future.channel.id" in ids


# ── Expiration ───────────────────────────────────────────────────


@pytest.mark.asyncio
class TestExpiration:
    async def test_expire_removes_past_transient(self, conn) -> None:
        # transient=True + expires_at past now → swept entirely.
        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="a", transient=True,
            expires_at="2000-01-01T00:00:00+00:00",
        )
        deleted = await expire_transient(
            conn, now_iso="2025-01-01T00:00:00+00:00",
        )
        assert deleted == 1
        assert await list_for_user(conn, user_id=U1) == []

    async def test_expire_leaves_persistent_alone(self, conn) -> None:
        # Persistent (non-transient) past-expiry rows stay in the
        # feed — UIs filter them out by expires_at if they want.
        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="a", transient=False,
            expires_at="2000-01-01T00:00:00+00:00",
        )
        deleted = await expire_transient(
            conn, now_iso="2025-01-01T00:00:00+00:00",
        )
        assert deleted == 0
        assert len(await list_for_user(conn, user_id=U1)) == 1

    async def test_expire_leaves_future_alone(self, conn) -> None:
        await publish(
            conn, user_id=U1, channel_id="x", source="s",
            title="a", transient=True,
            expires_at="2099-01-01T00:00:00+00:00",
        )
        deleted = await expire_transient(
            conn, now_iso="2025-01-01T00:00:00+00:00",
        )
        assert deleted == 0


# ── Class wrapper (smoke) ────────────────────────────────────────


@pytest.mark.asyncio
class TestStoreClassWrapper:
    async def test_class_forwards_to_module_funcs(self, conn) -> None:
        # The NotificationStore class is the injectable surface for
        # the future route layer. It must produce identical results
        # to the free functions.
        store = NotificationStore(conn)
        nid = await store.publish(
            user_id=U1, channel_id="connect.message",
            source="connect", title="hi",
        )
        notif = await store.get(user_id=U1, notification_id=nid)
        assert notif is not None
        assert notif.title == "hi"

        await store.mark_read(user_id=U1, notification_id=nid)
        unread = await store.list_for_user(user_id=U1, include_read=False)
        assert unread == []


# ── Catalog smoke ────────────────────────────────────────────────


class TestCatalog:
    def test_critical_channel_is_incoming_call(self) -> None:
        # The catalog ladder is a documented contract for UI authors
        # — a regression that drops the call channel out of CRITICAL
        # silently breaks "ringing pierces DND" UX.
        tmpl = catalog_channel("connect.call.incoming")
        assert tmpl is not None
        assert tmpl.importance == IMPORTANCE_CRITICAL

    def test_companion_initiative_is_low(self) -> None:
        # The companion shouldn't sound + toast on every initiative.
        tmpl = catalog_channel("companion.initiative")
        assert tmpl is not None
        assert tmpl.importance == IMPORTANCE_LOW

    def test_unknown_channel_returns_none(self) -> None:
        assert catalog_channel("totally.not.a.channel") is None
