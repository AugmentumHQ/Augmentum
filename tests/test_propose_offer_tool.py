"""``propose_offer`` Tool — end-to-end LLM-tool-call shape.

Exercises the Tool subclass + dispatcher together with a real
in-memory SQLite + the synthetic catalog. Mocks ``app.state`` and
``state_manager`` so the tool can resolve the connection without
spinning up the full app.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    reset_turn_counts,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.tools.propose_offer import ProposeOfferTool


MIG_221 = Path("augmentum/state/migrations/221_notification_substrate.sql").read_text()
MIG_224 = Path("augmentum/state/migrations/224_offer_suppressions.sql").read_text()


U1 = "user-alpha"


async def _preview(target_id: str, user_id: str) -> OfferPreview | None:
    return OfferPreview(label="syn-label")


async def _accept(payload, request):  # type: ignore[no-untyped-def]
    return {"ok": True}


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(MIG_221)
        await c.executescript(MIG_224)
        await c.commit()
        yield c


@pytest.fixture
def synthetic_catalog():
    saved = {k: dict(v) for k, v in CATALOG.items()}
    reset_turn_counts()
    register_kind("synthetic", [
        CatalogEntry(
            kind="synthetic",
            target_id="default",
            title="Try the synthetic offer?",
            scope="user",
            build_preview=_preview,
            accept=_accept,
        ),
    ])
    yield
    CATALOG.clear()
    CATALOG.update(saved)
    reset_turn_counts()


def _make_app_state(conn) -> SimpleNamespace:
    # The tool checks ``isinstance(backend, SQLiteBackend)`` — build
    # one without invoking ``__init__`` (which would wire vacuum
    # scheduling, migrations, etc.) and stuff the conn into the
    # private slot the @property reads.
    backend = SQLiteBackend.__new__(SQLiteBackend)
    backend._conn = conn  # noqa: SLF001 -- intentional test double
    sm = SimpleNamespace(backend=backend)
    return SimpleNamespace(state_manager=sm, notification_hub=None)


@pytest.mark.asyncio
class TestProposeOfferTool:
    async def test_publishes_when_inputs_valid(
        self, conn, synthetic_catalog,
    ) -> None:
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="default", reason="why",
            _user_id=U1,
        )
        assert result.success is True
        assert result.metadata["ok"] is True
        assert result.metadata["offer_id"]
        rows = await list_for_user(conn, user_id=U1)
        assert len(rows) == 1
        assert rows[0].channel_id == OFFER_CHANNEL_ID

    async def test_missing_user_returns_validation_failure(
        self, conn, synthetic_catalog,
    ) -> None:
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="default", reason="why",
        )
        assert result.success is False
        assert result.metadata["reason"] == "missing_user"

    async def test_missing_kind_or_target_returns_validation_error(
        self, conn, synthetic_catalog,
    ) -> None:
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="", target_id="default", reason="why", _user_id=U1,
        )
        assert result.success is False
        assert result.validation_error is True

    async def test_unknown_target_returns_unknown_target(
        self, conn, synthetic_catalog,
    ) -> None:
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="ghost", reason="why", _user_id=U1,
        )
        # Tool wraps dispatcher result — success=True (tool ran ok),
        # but metadata.ok=False and reason="unknown_target".
        assert result.success is True
        assert result.metadata["ok"] is False
        assert result.metadata["reason"] == "unknown_target"

    async def test_suppressed_offer_signals_suppressed(
        self, conn, synthetic_catalog,
    ) -> None:
        from augmentum.offers.store import never

        await never(
            conn, user_id=U1, kind="synthetic", target_id="default",
        )
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="default", reason="why", _user_id=U1,
        )
        assert result.success is True
        assert result.metadata["ok"] is False
        assert result.metadata["suppressed"] is True

    async def test_context_passes_session_and_turn_ids(
        self, conn, synthetic_catalog,
    ) -> None:
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="default", reason="why",
            _user_id=U1,
            _context={"session_id": "sess-1", "turn_id": "turn-1"},
        )
        assert result.success is True
        rows = await list_for_user(conn, user_id=U1)
        # thread_id should be the session_id from context.
        assert rows[0].thread_id == "sess-1"

    async def test_workspace_id_stashed_into_extra(
        self, conn, synthetic_catalog,
    ) -> None:
        # When _context carries workspace_id (coder mode), the tool
        # stashes it into extra._workspace_id so the catalog entry's
        # accept handler can target the workspace at activation time
        # without re-reading session state.
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="default", reason="why",
            _user_id=U1,
            _context={"mode": "coder", "workspace_id": "ws-99"},
        )
        assert result.success is True
        rows = await list_for_user(conn, user_id=U1)
        # Notification.payload is the parsed dict directly — no JSON
        # to re-parse from the store.
        extra = (rows[0].payload or {}).get("extra") or {}
        assert extra.get("_workspace_id") == "ws-99"

    async def test_workspace_id_omitted_when_empty(
        self, conn, synthetic_catalog,
    ) -> None:
        # No workspace_id in context (e.g. passthrough mode) →
        # extra._workspace_id NOT set, leaving the payload clean.
        tool = ProposeOfferTool(_make_app_state(conn))
        result = await tool.execute(
            kind="synthetic", target_id="default", reason="why",
            _user_id=U1,
            _context={"mode": "passthrough"},
        )
        assert result.success is True
        rows = await list_for_user(conn, user_id=U1)
        extra = (rows[0].payload or {}).get("extra") or {}
        assert "_workspace_id" not in extra
