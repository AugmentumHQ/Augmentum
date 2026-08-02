"""Tests for DreamEngine persona resolution.

Verifies the caller's per-user ``ui.aiName`` / ``ui.aiInstructions`` /
``ui.responseStyle`` actually reach the dream prompt. The legacy code
read settings via ``state_manager.settings_store`` — an attribute
StateManager never had — so every user's dreams silently used the
install-wide defaults.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import aiosqlite

from augmentum.dream.engine import DreamEngine
from augmentum.state.settings_store import SettingsStore


async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        "CREATE TABLE user_settings ("
        "  user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')),"
        "  PRIMARY KEY (user_id, key))"
    )
    await conn.commit()
    return conn, SettingsStore(conn)


def _build_engine(store: SettingsStore | None) -> DreamEngine:
    """Minimal engine for persona-resolution tests. Nothing else matters here."""
    journal = AsyncMock()
    journal._conn = None
    journal._db = None
    memory_store = AsyncMock()
    return DreamEngine(
        journal=journal,
        memory_store=memory_store,
        state_manager=None,
        embedding_service=None,
        portrait_manager=None,
        settings=object(),  # must not be read for persona
        provider_registry=None,
        settings_store=store,
    )


class TestLoadPersona:

    async def test_defaults_without_store(self):
        engine = _build_engine(None)
        name, foundation = await engine._load_persona(user_id="alice")
        assert name == "Assistant"
        assert foundation == "A helpful assistant."

    async def test_defaults_when_user_has_no_overrides(self):
        """No per-user values and no global values → safe defaults."""
        conn, store = await _make_store()
        engine = _build_engine(store)
        name, foundation = await engine._load_persona(user_id="alice")
        assert name == "Assistant"
        assert foundation == "Your name is Assistant."
        await conn.close()

    async def test_user_ai_name_reaches_foundation(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.aiName", "Iris")
        engine = _build_engine(store)
        name, foundation = await engine._load_persona(user_id="alice")
        assert name == "Iris"
        assert "Your name is Iris." in foundation
        await conn.close()

    async def test_user_instructions_and_style_appended(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.aiName", "Iris")
        await store.set_user("alice", "ui.aiInstructions", "You love metaphor.")
        await store.set_user("alice", "ui.responseStyle", "warm and curious")
        engine = _build_engine(store)
        name, foundation = await engine._load_persona(user_id="alice")
        assert name == "Iris"
        assert "Your name is Iris." in foundation
        assert "You love metaphor." in foundation
        assert "Your communication style is warm and curious." in foundation
        await conn.close()

    async def test_two_users_get_independent_personas(self):
        """The whole point of the refactor — each tenant's dreams
        speak in their own voice."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.aiName", "Iris")
        await store.set_user("alice", "ui.responseStyle", "direct")
        await store.set_user("bob", "ui.aiName", "Nova")
        await store.set_user("bob", "ui.responseStyle", "playful")
        engine = _build_engine(store)

        alice_name, alice_found = await engine._load_persona(user_id="alice")
        bob_name, bob_found = await engine._load_persona(user_id="bob")

        assert alice_name == "Iris"
        assert bob_name == "Nova"
        assert "Iris" in alice_found and "direct" in alice_found
        assert "Nova" in bob_found and "playful" in bob_found
        assert "Nova" not in alice_found
        assert "Iris" not in bob_found
        await conn.close()

    async def test_global_fallback(self):
        """Install-wide default persona applies when user hasn't customised."""
        conn, store = await _make_store()
        await store.set("ui.aiName", "HouseAssistant")
        engine = _build_engine(store)
        name, _ = await engine._load_persona(user_id="alice")
        assert name == "HouseAssistant"
        await conn.close()

    async def test_store_error_falls_back_safely(self):
        """A broken settings store must not crash the dream cycle."""

        class _ExplodingStore:
            async def get_user_or_global(self, *_a, **_kw):
                raise RuntimeError("db unavailable")

        engine = _build_engine(_ExplodingStore())
        name, foundation = await engine._load_persona(user_id="alice")
        assert name == "Assistant"
        assert foundation == "A helpful assistant."
