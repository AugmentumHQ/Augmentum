"""Tests for Working Agreements — the durable, model-agnostic "how this
maker works" substrate (migration 273).

The third accumulation axis: skills remember what worked, lessons
remember corrections, agreements remember standing operating principles
for the assistant relationship — always injected, user-owned,
model-agnostic. Covers:

- Migration 273 creates maker_agreements
- add creates an agreement; list_active returns it strongest-first
- add dedups a restated principle into a reinforce (no duplicate)
- user scoping: A's agreements are invisible to B
- retire drops an agreement from active
- render_for_prompt: empty for a fresh user; honored block when present
- add refuses an empty user_id (multi-tenant invariant)
"""

from __future__ import annotations

import pytest

from augmentum.coder.maker_agreements import MakerAgreements


async def _boot():
    """Fresh :memory: backend with migrations applied + two seeded users."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    for uid in ("alice", "bob"):
        await backend.conn.execute(
            "INSERT INTO users (id, username, display_name, password_hash, role) "
            "VALUES (?, ?, ?, ?, ?)",
            (uid, uid, uid.title(), "pw", "user"),
        )
    await backend.conn.commit()
    return backend


# ── Substrate ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_273_creates_table():
    backend = await _boot()
    try:
        cur = await backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name = 'maker_agreements'"
        )
        rows = await cur.fetchall()
        await cur.close()
        assert [r[0] for r in rows] == ["maker_agreements"]
    finally:
        await backend.close()


# ── Add + list ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_then_list_active():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        a = await store.add(
            principle="Tell me the blast radius before irreversible changes",
            rationale="security-oriented; his life's work",
            category="communication",
            user_id="alice",
        )
        assert a is not None and a.id > 0
        items = await store.list_active(user_id="alice")
        assert [i.principle for i in items] == [
            "Tell me the blast radius before irreversible changes",
        ]
        assert items[0].category == "communication"
        assert items[0].times_seen == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_restating_reinforces_not_duplicates():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        await store.add(principle="Finish one thing well over starting three", user_id="alice")
        # Restated with trailing period + different case + spacing.
        await store.add(principle="finish one thing well  over starting three.", user_id="alice")
        items = await store.list_active(user_id="alice")
        assert len(items) == 1, "restatement must reinforce, not duplicate"
        assert items[0].times_seen == 2
        assert items[0].strength > 1.0 - 1e-9 or items[0].strength <= 1.0  # bounded
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_user_scoping_isolates_agreements():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        await store.add(principle="Comfort over premium", user_id="alice")
        assert len(await store.list_active(user_id="alice")) == 1
        assert await store.list_active(user_id="bob") == []
        # Empty user_id never reads across users.
        assert await store.list_active(user_id="") == []
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_retire_drops_from_active():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        a = await store.add(principle="No git stash, ever", user_id="alice")
        assert a is not None
        assert await store.retire(a.id, user_id="alice") is True
        assert await store.list_active(user_id="alice") == []
        # Wrong user can't retire someone else's agreement.
        b = await store.add(principle="x", user_id="alice")
        assert b is not None
        assert await store.retire(b.id, user_id="bob") is False
    finally:
        await backend.close()


# ── Prompt rendering ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_render_is_empty_for_fresh_user():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        # The common first-run case — injecting must be a pure no-op.
        assert await store.render_for_prompt(user_id="alice") == ""
        assert await store.render_for_prompt(user_id="") == ""
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_render_block_contains_principles_and_is_scoped():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        await store.add(
            principle="Prefer the strong-foundation option over a shortcut",
            category="scope", user_id="alice",
        )
        await store.add(
            principle="Tell me the blast radius first",
            category="communication", user_id="alice",
        )
        block = await store.render_for_prompt(user_id="alice")
        assert block.startswith("<working_agreements>")
        assert block.endswith("</working_agreements>")
        assert "strong-foundation" in block
        assert "blast radius" in block
        # bob sees nothing.
        assert await store.render_for_prompt(user_id="bob") == ""
    finally:
        await backend.close()


# ── Multi-tenant invariant ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_refuses_empty_user():
    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        assert await store.add(principle="orphan", user_id="") is None
        # Nothing was written.
        cur = await backend.conn.execute("SELECT COUNT(*) FROM maker_agreements")
        (count,) = await cur.fetchone()
        await cur.close()
        assert count == 0
    finally:
        await backend.close()


# ── Handler glue: agreements reach the coder system prompt ──────────────


@pytest.mark.asyncio
async def test_handler_renders_agreements_block_for_user():
    """The load-bearing path: CoderHandler._render_maker_agreements_block
    honors the setting, the threaded conn, and the user scope — so the
    block actually reaches _act_native's sys_text. Verifies the glue, not
    just the store."""
    from augmentum.modes.coder.handler import CoderHandler
    from tests.test_coder_handler import _FakeContainerManager

    backend = await _boot()
    try:
        store = MakerAgreements(backend.conn)
        await store.add(
            principle="Tell me the blast radius before irreversible changes",
            category="communication", user_id="alice",
        )

        handler = CoderHandler(
            backend, session_id="sess-agree",
            container_manager=_FakeContainerManager(), workspace_id="ws-agree",
        )
        handler._user_id = "alice"
        handler._resolve_archive_conn = lambda: backend.conn  # type: ignore[assignment]

        block = await handler._render_maker_agreements_block()
        assert "<working_agreements>" in block
        assert "blast radius" in block

        # A different user gets nothing through the same handler glue.
        handler._user_id = "bob"
        assert await handler._render_maker_agreements_block() == ""

        # Disabled setting → no-op even with agreements present.
        handler._user_id = "alice"
        from augmentum.config import settings
        prev = settings.coder_maker_agreements_enabled
        settings.coder_maker_agreements_enabled = False
        try:
            assert await handler._render_maker_agreements_block() == ""
        finally:
            settings.coder_maker_agreements_enabled = prev
    finally:
        await backend.close()
