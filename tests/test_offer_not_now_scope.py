""""Not now" means not now — it must not suppress anything.

Regression cover for the failure documented in migration 326. The offer chip's
middle button wrote a 30-day suppression row, so one tap on one device muted
that capability for the whole account, in every chat, for a month. The
passthrough gated-capability path then got ``ok=False`` back from
``propose_offer`` and answered "I could write that up if you'd like — just say
the word", inviting a retry that could never succeed and leaving no log line to
explain it.

Two properties are pinned here:

1. ``snooze`` (the "Not now" action) dismisses one notification and writes NO
   suppression row, so the next request for the same capability still works.
2. ``never`` still writes a permanent row — that one the user asked for.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.notifications.store import list_for_user, publish
from augmentum.offers.handlers.system_offer import handle_offer_action
from augmentum.offers.store import is_suppressed, list_suppressions

MIG_221 = Path("augmentum/state/migrations/221_notification_substrate.sql").read_text()
MIG_224 = Path("augmentum/state/migrations/224_offer_suppressions.sql").read_text()
MIG_326 = Path(
    "augmentum/state/migrations/326_drop_offer_snooze_rows.sql",
).read_text()

USER = "usr_test"


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(MIG_221)
        await c.executescript(MIG_224)
        await c.commit()
        yield c


@pytest.fixture
def request_stub(conn):
    """Minimal ``Request`` stand-in: the handler only needs the SQLite conn
    off ``app.state.state_manager.backend`` plus ``scope['user']``."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = MagicMock(spec=SQLiteBackend)
    backend.conn = conn
    req = MagicMock()
    req.app.state.state_manager.backend = backend
    req.scope = {"user": None}
    return req


async def _publish_offer(conn, *, target_id: str = "create_document") -> object:
    notification_id = await publish(
        conn,
        user_id=USER,
        channel_id="system.offer",
        source="chat.propose_offer",
        title="Write that up?",
        body="reason",
        payload={"kind": "gated_tool", "target_id": target_id, "scope": "user"},
    )
    rows = await list_for_user(conn, user_id=USER, include_dismissed=True)
    return next(n for n in rows if n.notification_id == notification_id)


class TestNotNowWritesNoSuppression:
    @pytest.mark.asyncio
    async def test_snooze_leaves_capability_available(self, conn, request_stub):
        notif = await _publish_offer(conn)

        result = await handle_offer_action(notif, "snooze", request_stub)

        assert result["ok"] is True
        assert result.get("suppressed") is False
        # The whole point: the next request for this capability still works.
        assert not await is_suppressed(
            conn, user_id=USER, kind="gated_tool", target_id="create_document",
        )
        assert await list_suppressions(conn, user_id=USER) == []

    @pytest.mark.asyncio
    async def test_snooze_still_dismisses_the_chip(self, conn, request_stub):
        """Declining has to remove the chip, or it just sits there."""
        notif = await _publish_offer(conn)

        await handle_offer_action(notif, "snooze", request_stub)

        pending = await list_for_user(conn, user_id=USER, include_dismissed=False)
        assert [n.notification_id for n in pending] == []

    @pytest.mark.asyncio
    async def test_snooze_does_not_leak_an_until_date(self, conn, request_stub):
        """The old result carried ``until``; a caller or UI keying off it would
        render "snoozed until X" for something that isn't snoozed at all."""
        notif = await _publish_offer(conn)

        result = await handle_offer_action(notif, "snooze", request_stub)

        assert "until" not in result

    @pytest.mark.asyncio
    async def test_declining_twice_never_accumulates_suppression(
        self, conn, request_stub,
    ):
        """Repeated declines are the common case (the offer re-surfaces because
        it's genuinely relevant); none of them may add up to a block."""
        for _ in range(3):
            notif = await _publish_offer(conn)
            await handle_offer_action(notif, "snooze", request_stub)

        assert not await is_suppressed(
            conn, user_id=USER, kind="gated_tool", target_id="create_document",
        )


class TestNeverStillSuppresses:
    @pytest.mark.asyncio
    async def test_never_writes_a_permanent_row(self, conn, request_stub):
        notif = await _publish_offer(conn)

        result = await handle_offer_action(notif, "never", request_stub)

        assert result["ok"] is True
        assert await is_suppressed(
            conn, user_id=USER, kind="gated_tool", target_id="create_document",
        )
        rows = await list_suppressions(conn, user_id=USER)
        assert len(rows) == 1
        assert rows[0].reason == "never"
        assert rows[0].is_permanent is True


class TestMigration326:
    @pytest.mark.asyncio
    async def test_clears_legacy_snooze_rows_but_keeps_never(self, conn):
        """Installs carry rows the retired button already wrote — including the
        ``create_document`` block that motivated this change. The migration
        must clear those while leaving explicit Never choices intact."""
        from augmentum.offers.store import never as _never
        from augmentum.offers.store import snooze as _snooze

        await _snooze(
            conn, user_id=USER, kind="gated_tool", target_id="create_document",
        )
        await _never(
            conn, user_id=USER, kind="gated_tool", target_id="build_application",
        )

        await conn.executescript(MIG_326)
        await conn.commit()

        remaining = await list_suppressions(conn, user_id=USER)
        assert [(r.target_id, r.reason) for r in remaining] == [
            ("build_application", "never"),
        ]
        assert not await is_suppressed(
            conn, user_id=USER, kind="gated_tool", target_id="create_document",
        )
