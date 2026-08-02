"""Offer dispatcher — catalog lookup, suppression, rate-limit, publish.

These tests pin the dispatcher in isolation: the catalog is patched
with a synthetic entry, suppressions are written directly via the
store, and the published notifications are read back via the
notification store. No app stack, no HTTP.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.notifications.store import list_for_user
from augmentum.offers.catalog.base import (
    CATALOG,
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.offers.dispatcher import (
    OFFER_CHANNEL_ID,
    propose_offer,
    reset_turn_counts,
)
from augmentum.offers.store import never, snooze

# Apply both migrations — 221 owns the notifications table, 224 owns
# the suppressions table. The dispatcher touches both.
MIG_221 = Path("augmentum/state/migrations/221_notification_substrate.sql").read_text()
MIG_224 = Path("augmentum/state/migrations/224_offer_suppressions.sql").read_text()


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(MIG_221)
        await c.executescript(MIG_224)
        await c.commit()
        yield c


@pytest.fixture
def synthetic_catalog():
    """Register a synthetic catalog kind for the duration of one test.

    Avoids depending on the real mcp_servers entries — the dispatcher
    contract should be testable without coupling to a specific kind.
    """

    saved = {k: dict(v) for k, v in CATALOG.items()}
    reset_turn_counts()

    async def _preview(target_id: str, user_id: str) -> OfferPreview | None:
        if target_id == "skip":
            return None
        return OfferPreview(label="syn-label", hint="syn-hint")

    async def _accept(payload, request):  # type: ignore[no-untyped-def]
        return {"ok": True, "synthetic": True}

    entries = [
        CatalogEntry(
            kind="synthetic",
            target_id="default",
            title="Try the synthetic offer?",
            scope="user",
            build_preview=_preview,
            accept=_accept,
        ),
        CatalogEntry(
            kind="synthetic",
            target_id="skip",
            title="Skip target",
            scope="user",
            build_preview=_preview,
            accept=_accept,
        ),
        CatalogEntry(
            kind="synthetic",
            target_id="admin_only",
            title="Admin offer",
            scope="admin",
            build_preview=_preview,
            accept=_accept,
        ),
        CatalogEntry(
            kind="synthetic",
            target_id="coder_only",
            title="Coder-only offer",
            scope="user",
            build_preview=_preview,
            accept=_accept,
            allowed_modes=("coder",),
        ),
    ]
    register_kind("synthetic", entries)
    yield
    CATALOG.clear()
    CATALOG.update(saved)
    reset_turn_counts()


U1 = "user-alpha"


@pytest.mark.asyncio
class TestCatalogLookup:
    async def test_unknown_kind_returns_unknown_target(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="nope", target_id="x",
            reason="why",
        )
        assert r.ok is False
        assert r.reason == "unknown_target"

    async def test_unknown_target_returns_unknown_target(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="missing",
            reason="why",
        )
        assert r.ok is False
        assert r.reason == "unknown_target"

    async def test_missing_user_returns_missing_user(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id="", kind="synthetic", target_id="default",
            reason="why",
        )
        assert r.ok is False
        assert r.reason == "missing_user"


@pytest.mark.asyncio
class TestSuppression:
    async def test_snoozed_offer_does_not_publish(
        self, conn, synthetic_catalog,
    ) -> None:
        await snooze(
            conn, user_id=U1, kind="synthetic", target_id="default", days=30,
        )
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="why",
        )
        assert r.ok is False
        assert r.suppressed is True
        assert r.reason == "suppressed"
        rows = await list_for_user(conn, user_id=U1)
        assert len(rows) == 0

    async def test_never_offer_does_not_publish(
        self, conn, synthetic_catalog,
    ) -> None:
        await never(
            conn, user_id=U1, kind="synthetic", target_id="default",
        )
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="why",
        )
        assert r.ok is False
        assert r.suppressed is True


@pytest.mark.asyncio
class TestModeGate:
    """allowed_modes restricts where an entry can be proposed from."""

    async def test_coder_only_blocked_in_passthrough(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic",
            target_id="coder_only", reason="why", mode="passthrough",
        )
        assert r.ok is False
        assert r.reason == "mode_mismatch"

    async def test_coder_only_allowed_in_coder(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic",
            target_id="coder_only", reason="why", mode="coder",
        )
        assert r.ok is True

    async def test_empty_mode_falls_open(
        self, conn, synthetic_catalog,
    ) -> None:
        # Non-handler callers (tests, scripts) may not stamp a mode;
        # the gate falls open rather than dropping every offer.
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic",
            target_id="coder_only", reason="why", mode="",
        )
        assert r.ok is True

    async def test_unrestricted_entry_ignores_mode(
        self, conn, synthetic_catalog,
    ) -> None:
        # ``default`` has no allowed_modes — any mode value passes.
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic",
            target_id="default", reason="why", mode="narrative",
        )
        assert r.ok is True


@pytest.mark.asyncio
class TestRateLimits:
    async def test_per_turn_cap(self, conn, synthetic_catalog) -> None:
        # Different targets so per-(kind,target) dedupe doesn't paper
        # over the per-turn cap.
        for entry in [
            CatalogEntry(
                kind="synthetic", target_id=f"t{i}",
                title=f"Offer {i}", scope="user",
                build_preview=lambda t, u: _preview_ok(),
                accept=_accept_ok,
            )
            for i in range(5)
        ]:
            CATALOG["synthetic"][entry.target_id] = entry

        turn_id = "turn-xyz"
        ok_count = 0
        rate_limited = 0
        for i in range(5):
            r = await propose_offer(
                conn, hub=None, user_id=U1, kind="synthetic",
                target_id=f"t{i}", reason=f"r{i}",
                turn_id=turn_id, max_per_turn=2,
                max_pending_per_session=100, max_per_day=100,
            )
            if r.ok:
                ok_count += 1
            elif r.reason == "rate_limit:turn":
                rate_limited += 1
        assert ok_count == 2
        assert rate_limited == 3

    async def test_per_day_cap(self, conn, synthetic_catalog) -> None:
        # Pre-populate three notification rows directly to simulate
        # prior-in-day publishes, then assert the next one hits cap.
        from augmentum.notifications.store import publish as _publish

        for i in range(3):
            await _publish(
                conn, user_id=U1, channel_id=OFFER_CHANNEL_ID,
                source="chat.propose_offer",
                title=f"prev-{i}", body="",
                dedupe_key=f"synthetic:prev-{i}",
            )

        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="why",
            max_per_turn=10, max_pending_per_session=100, max_per_day=3,
        )
        assert r.ok is False
        assert r.reason == "rate_limit:day"

    async def test_per_session_pending_cap(
        self, conn, synthetic_catalog,
    ) -> None:
        from augmentum.notifications.store import publish as _publish

        thread = "sess-1"
        for i in range(3):
            await _publish(
                conn, user_id=U1, channel_id=OFFER_CHANNEL_ID,
                source="chat.propose_offer",
                title=f"prev-{i}", body="",
                dedupe_key=f"synthetic:prev-{i}",
                thread_id=thread,
            )
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="why",
            thread_id=thread,
            max_per_turn=10, max_pending_per_session=3, max_per_day=100,
        )
        assert r.ok is False
        assert r.reason == "rate_limit:pending"


@pytest.mark.asyncio
class TestPublish:
    async def test_happy_path_publishes_notification(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="because you said so",
        )
        assert r.ok is True
        assert r.offer_id

        rows = await list_for_user(conn, user_id=U1)
        assert len(rows) == 1
        n = rows[0]
        assert n.channel_id == OFFER_CHANNEL_ID
        assert n.title == "Try the synthetic offer?"
        assert n.body == "because you said so"
        assert n.dedupe_key == "synthetic:default"
        assert n.payload["kind"] == "synthetic"
        assert n.payload["target_id"] == "default"
        assert n.payload["scope"] == "user"
        assert n.payload["preview"]["label"] == "syn-label"

    async def test_dedupe_key_replaces_in_place(
        self, conn, synthetic_catalog,
    ) -> None:
        await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="first",
        )
        await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="default",
            reason="second",
        )
        rows = await list_for_user(conn, user_id=U1)
        assert len(rows) == 1
        assert rows[0].body == "second"  # replaced in place

    async def test_not_relevant_skips_publish(
        self, conn, synthetic_catalog,
    ) -> None:
        # The "skip" target returns None from its preview builder.
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="skip",
            reason="why",
        )
        assert r.ok is False
        assert r.reason == "not_relevant"
        rows = await list_for_user(conn, user_id=U1)
        assert len(rows) == 0

    async def test_admin_scope_recorded_in_payload(
        self, conn, synthetic_catalog,
    ) -> None:
        r = await propose_offer(
            conn, hub=None, user_id=U1, kind="synthetic", target_id="admin_only",
            reason="why",
        )
        assert r.ok is True
        rows = await list_for_user(conn, user_id=U1)
        assert rows[0].payload["scope"] == "admin"


# Helpers used by the per-turn cap test (defined at module level so
# the lambda closures pick them up without per-iteration capture).


async def _preview_ok():  # type: ignore[no-untyped-def]
    return OfferPreview(label="ok")


async def _accept_ok(payload, request):  # type: ignore[no-untyped-def]
    return {"ok": True}
