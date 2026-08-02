"""End-to-end dream system multi-tenant isolation test.

Exercises the full real stack — migrated SQLite, DreamJournal, PortraitManager,
DreamEngine with a mock LLM backend, DreamScheduler — and asserts user A's
dream data is invisible to user B at every layer.

Why this test exists: the dream system pre-dates multi-tenancy (migration 058
landed before 071/072). The 089 migration backfills user_id, and code
changes thread user_id through every store/manager. This test pins those
guarantees in place so a future refactor can't silently regress to
cross-user leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest

from augmentum.dream.engine import DreamEngine
from augmentum.dream.journal import DreamJournal
from augmentum.dream.portrait import PortraitManager
from augmentum.dream.scheduler import DreamScheduler

_FAKE_LLM_RESPONSE = json.dumps({
    "reflections": [
        {"type": "reflection", "content": "What we discussed stayed with me."},
        {"type": "voice_note", "content": "I notice we trade short turns naturally."},
    ],
})


class _FakeBackend:
    """Mock LLM backend that returns a canned dream JSON."""

    async def chat(self, request):
        from augmentum.models.base import InternalChatResponse, Message, Usage
        return InternalChatResponse(
            message=Message(role="assistant", content=_FAKE_LLM_RESPONSE),
            model=request.model or "fake",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


class _FakeRegistry:
    async def resolve_model_for_role(self, role, override=None, settings=None):
        return _FakeBackend(), "fake-model"


class _FakeMemoryStore:
    """Wraps a connection so DreamEngine.memory_store has a `_conn` attribute."""

    def __init__(self, conn):
        self._conn = conn


class _FakeStateManager:
    settings_store = None  # unused; engine reads settings_store directly now


class _FakeSettingsStore:
    """In-memory settings store for the scheduler's counter persistence."""

    def __init__(self):
        self._data: dict[str, str] = {}

    async def get(self, key, default=None):
        return self._data.get(key, default)

    async def set(self, key, value):
        self._data[key] = value


async def _apply_dream_schema(db_path: str) -> None:
    """Apply migrations 058 + 089 to a fresh sqlite db."""
    repo = Path(__file__).resolve().parent.parent
    sql_058 = (repo / "augmentum" / "state" / "migrations" / "058_dream_system.sql").read_text()
    sql_089 = (repo / "augmentum" / "state" / "migrations" / "089_dream_user_id.sql").read_text()
    async with aiosqlite.connect(db_path) as db:
        # schema_version table needed before 089's INSERT OR IGNORE
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, description TEXT, "
            "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        # memories table for the engine's _get_dream_eligible_memories query.
        # Migration 058 adds source_message_id + user_approved itself, so we
        # leave those out of the base CREATE.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                evidence TEXT,
                tier TEXT NOT NULL DEFAULT 'active',
                session_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                valid_until TEXT,
                user_id TEXT
            )
        """)
        await db.executescript(sql_058)
        await db.executescript(sql_089)
        await db.commit()


async def _seed_memory(conn, *, mid: str, user_id: str, content: str) -> None:
    await conn.execute(
        "INSERT INTO memories (id, content, evidence, tier, session_id, "
        "source_message_id, user_approved, user_id) "
        "VALUES (?, ?, ?, 'active', 'sess1', 'msg1', 1, ?)",
        (mid, content, content, user_id),
    )
    await conn.commit()


@pytest.fixture
async def stack(tmp_path):
    """Build the full dream stack against a real (migrated) sqlite db."""
    db_path = str(tmp_path / "dream.db")
    await _apply_dream_schema(db_path)

    journal = DreamJournal(db_path)
    await journal.initialize()

    settings_store = _FakeSettingsStore()

    portrait_mgr = PortraitManager(
        journal, settings_store,
        model_backend=_FakeBackend(),
    )

    engine = DreamEngine(
        journal=journal,
        memory_store=_FakeMemoryStore(journal._db),
        state_manager=_FakeStateManager(),
        embedding_service=None,
        portrait_manager=portrait_mgr,
        settings={"aiName": "Aurora"},
        provider_registry=_FakeRegistry(),
    )

    scheduler = DreamScheduler(
        engine=engine,
        settings_store=settings_store,
        enabled=True,
        message_threshold=2,
        idle_minutes=0,
        cooldown_minutes=0,
    )
    await scheduler.initialize()

    yield {
        "db_path": db_path,
        "journal": journal,
        "portrait_mgr": portrait_mgr,
        "engine": engine,
        "scheduler": scheduler,
        "settings_store": settings_store,
    }

    await journal.close()


@pytest.mark.asyncio
async def test_run_cycle_scopes_entries_to_user(stack):
    journal = stack["journal"]
    engine = stack["engine"]

    # Seed two users with different approved memories
    await _seed_memory(journal._db, mid="mA", user_id="userA", content="A had coffee.")
    await _seed_memory(journal._db, mid="mB", user_id="userB", content="B saw a film.")

    # Run a cycle for user A
    cycle_a = await engine.run_cycle("default", "manual", user_id="userA")
    assert cycle_a.status == "completed", cycle_a.error
    assert cycle_a.memories_count == 1, "userA cycle should only see userA's memory"
    assert cycle_a.entries_count >= 1, "fake LLM returns 2 reflections, expected >=1 after filtering"

    # Run a cycle for user B
    cycle_b = await engine.run_cycle("default", "manual", user_id="userB")
    assert cycle_b.status == "completed", cycle_b.error
    assert cycle_b.memories_count == 1, "userB cycle should only see userB's memory"

    # Cross-user read isolation via the journal
    a_entries, a_total = await journal.list_entries("default", user_id="userA")
    b_entries, b_total = await journal.list_entries("default", user_id="userB")
    assert a_total >= 1
    assert b_total >= 1
    a_ids = {e.id for e in a_entries}
    b_ids = {e.id for e in b_entries}
    assert a_ids.isdisjoint(b_ids), "users must not see each other's entries"


@pytest.mark.asyncio
async def test_dreamed_memory_log_is_user_scoped(stack):
    """User A having dreamed mem 'shared' must not block user B from dreaming theirs."""
    journal = stack["journal"]
    engine = stack["engine"]
    # Both users have a memory with the same id (extreme adversarial case)
    await _seed_memory(journal._db, mid="shared-id", user_id="userA", content="A view.")
    await _seed_memory(journal._db, mid="shared-id-b", user_id="userB", content="B view.")

    cycle_a = await engine.run_cycle("default", "manual", user_id="userA")
    assert cycle_a.memories_count == 1

    # User B's run is independent — A's dreamed_memory_log entries are scoped
    cycle_b = await engine.run_cycle("default", "manual", user_id="userB")
    assert cycle_b.memories_count == 1, "userB should still see their own undreamed memory"


@pytest.mark.asyncio
async def test_portrait_get_current_is_scoped(stack):
    journal = stack["journal"]
    engine = stack["engine"]
    portrait_mgr = stack["portrait_mgr"]

    await _seed_memory(journal._db, mid="mA", user_id="userA", content="A reflection.")
    cycle_a = await engine.run_cycle("default", "manual", user_id="userA")
    assert cycle_a.entries_count >= 1

    p_a = await portrait_mgr.get_current("default", user_id="userA")
    p_b = await portrait_mgr.get_current("default", user_id="userB")
    assert p_a is not None, "user A should have a portrait after running a cycle"
    assert p_b is None, "user B must not see user A's portrait"


@pytest.mark.asyncio
async def test_dream_cycle_persisted_with_user_id(stack):
    """The dream_cycles table was previously never written. Confirm it now is."""
    journal = stack["journal"]
    engine = stack["engine"]

    await _seed_memory(journal._db, mid="m1", user_id="userA", content="x")
    cycle = await engine.run_cycle("default", "manual", user_id="userA")
    assert cycle.status == "completed"

    cursor = await journal._db.execute(
        "SELECT id, user_id, status, entries_count FROM dream_cycles WHERE id = ?",
        (cycle.id,),
    )
    row = await cursor.fetchone()
    assert row is not None, "dream_cycles row must be persisted"
    assert row[1] == "userA", "cycle must carry user_id"
    assert row[2] == "completed"


@pytest.mark.asyncio
async def test_persona_id_no_longer_hardcoded(stack):
    """Regression: engine used to pass literal 'default' to _parse_dream_response."""
    journal = stack["journal"]
    engine = stack["engine"]
    await _seed_memory(journal._db, mid="m1", user_id="userA", content="x")

    cycle = await engine.run_cycle("personaX", "manual", user_id="userA")
    assert cycle.entries_count >= 1
    entries, _ = await journal.list_entries("personaX", user_id="userA")
    assert all(e.persona_id == "personaX" for e in entries), \
        "entries must carry the persona_id passed to run_cycle"


@pytest.mark.asyncio
async def test_scheduler_per_user_counters(stack):
    sched = stack["scheduler"]
    sched.notify_message(user_id="userA")
    sched.notify_message(user_id="userA")
    sched.notify_approval("memX", user_id="userA")

    sched.notify_message(user_id="userB")
    sched.notify_approval("memY", user_id="userB")

    sa = await sched.get_status(user_id="userA")
    sb = await sched.get_status(user_id="userB")
    assert sa["messages_since_dream"] == 2
    assert sa["approved_memories_since_dream"] == 1
    assert sb["messages_since_dream"] == 1
    assert sb["approved_memories_since_dream"] == 1

    # Counters are independent — flushing/loading round-trips per-user state
    await sched._flush_counters()
    sched2 = DreamScheduler(
        engine=sched._engine, settings_store=sched._settings_store,
        message_threshold=10, idle_minutes=0, cooldown_minutes=0,
    )
    await sched2.initialize()
    assert (await sched2.get_status(user_id="userA"))["messages_since_dream"] == 2
    assert (await sched2.get_status(user_id="userB"))["messages_since_dream"] == 1


@pytest.mark.asyncio
async def test_reset_to_foundation_is_user_scoped(stack):
    """Resetting user A's dream data must not delete user B's."""
    journal = stack["journal"]
    engine = stack["engine"]
    portrait_mgr = stack["portrait_mgr"]

    await _seed_memory(journal._db, mid="mA", user_id="userA", content="A.")
    await _seed_memory(journal._db, mid="mB", user_id="userB", content="B.")
    await engine.run_cycle("default", "manual", user_id="userA")
    await engine.run_cycle("default", "manual", user_id="userB")

    await portrait_mgr.reset_to_foundation("default", user_id="userA")

    a_entries, _ = await journal.list_entries("default", user_id="userA")
    b_entries, _ = await journal.list_entries("default", user_id="userB")
    assert len(a_entries) == 0, "user A's entries should be wiped"
    assert len(b_entries) >= 1, "user B's entries must survive"
