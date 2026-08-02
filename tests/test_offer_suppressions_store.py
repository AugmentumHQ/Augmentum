"""Offer suppressions store — snooze / never / list / undo / sweep.

Applies migration 224 directly to an in-memory SQLite so the schema
and store stay pinned together. Mirrors the pattern in
test_notifications_store.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from augmentum.offers.store import (
    SUPPRESSION_NEVER,
    delete_suppression,
    is_suppressed,
    list_suppressions,
    never,
    set_suppression,
    snooze,
    sweep_expired_suppressions,
)

MIGRATION = (
    Path("augmentum/state/migrations/224_offer_suppressions.sql").read_text()
)


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(MIGRATION)
        await c.commit()
        yield c


U1 = "user-alpha"
U2 = "user-beta"


@pytest.mark.asyncio
class TestSetSuppression:
    async def test_inserts_row(self, conn) -> None:
        await set_suppression(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
            suppressed_until=SUPPRESSION_NEVER, reason="never",
        )
        rows = await list_suppressions(conn, user_id=U1)
        assert len(rows) == 1
        assert rows[0].kind == "mcp_server"
        assert rows[0].target_id == "gmail"
        assert rows[0].reason == "never"
        assert rows[0].is_permanent

    async def test_replaces_existing_row(self, conn) -> None:
        # Snooze first.
        until_iso = await snooze(
            conn, user_id=U1, kind="mcp_server", target_id="gmail", days=30,
        )
        # Then promote to Never. The PK collision must overwrite, not
        # raise — otherwise the user can't escalate.
        await never(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        )
        rows = await list_suppressions(conn, user_id=U1)
        assert len(rows) == 1
        assert rows[0].reason == "never"
        assert rows[0].suppressed_until == SUPPRESSION_NEVER
        assert rows[0].suppressed_until != until_iso

    async def test_user_id_required(self, conn) -> None:
        with pytest.raises(ValueError):
            await set_suppression(
                conn, user_id="", kind="x", target_id="y",
                suppressed_until=SUPPRESSION_NEVER,
            )


@pytest.mark.asyncio
class TestIsSuppressed:
    async def test_returns_false_when_no_row(self, conn) -> None:
        assert await is_suppressed(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        ) is False

    async def test_returns_true_after_never(self, conn) -> None:
        await never(conn, user_id=U1, kind="mcp_server", target_id="gmail")
        assert await is_suppressed(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        ) is True

    async def test_returns_true_during_snooze(self, conn) -> None:
        await snooze(
            conn, user_id=U1, kind="mcp_server", target_id="gmail", days=30,
        )
        assert await is_suppressed(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        ) is True

    async def test_expired_snooze_does_not_suppress(self, conn) -> None:
        # Write a row whose suppressed_until is in the past directly via
        # the store helper — snooze() can't take negative days, and we
        # want to exercise the "expired but still in table" branch.
        past_iso = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        await set_suppression(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
            suppressed_until=past_iso, reason="snooze",
        )
        assert await is_suppressed(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        ) is False

    async def test_isolated_per_user(self, conn) -> None:
        await never(conn, user_id=U1, kind="mcp_server", target_id="gmail")
        assert await is_suppressed(
            conn, user_id=U2, kind="mcp_server", target_id="gmail",
        ) is False


@pytest.mark.asyncio
class TestDeleteSuppression:
    async def test_undo_removes_row(self, conn) -> None:
        await never(conn, user_id=U1, kind="mcp_server", target_id="gmail")
        removed = await delete_suppression(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        )
        assert removed is True
        # After undo, the offer can surface again.
        assert await is_suppressed(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        ) is False

    async def test_undo_missing_row_returns_false(self, conn) -> None:
        removed = await delete_suppression(
            conn, user_id=U1, kind="mcp_server", target_id="gmail",
        )
        assert removed is False


@pytest.mark.asyncio
class TestSweep:
    async def test_prunes_expired_snoozes_only(self, conn) -> None:
        # One active snooze, one expired snooze, one Never.
        past_iso = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        await set_suppression(
            conn, user_id=U1, kind="mcp_server", target_id="active",
            suppressed_until=(datetime.now(UTC) + timedelta(days=10)).isoformat(),
            reason="snooze",
        )
        await set_suppression(
            conn, user_id=U1, kind="mcp_server", target_id="expired",
            suppressed_until=past_iso, reason="snooze",
        )
        await never(
            conn, user_id=U1, kind="mcp_server", target_id="forever",
        )

        removed = await sweep_expired_suppressions(conn)
        assert removed == 1
        remaining = {r.target_id for r in await list_suppressions(conn, user_id=U1)}
        assert remaining == {"active", "forever"}
