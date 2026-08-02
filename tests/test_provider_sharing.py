"""Tests for per-user provider ownership + sharing (migration 305).

Locks in the two security-critical behaviors:
  1. ``ProviderStore.list_providers(visible_to=...)`` filters to
     shared + global + the user's own private providers.
  2. ``ProviderRegistry.provider_visible_to`` — the predicate both model
     LIST surfaces (/v1/models, /api/tags) and the RESOLVE gate use — hides
     another user's private provider from everyone but its owner.
"""

from __future__ import annotations

import aiosqlite
import httpx
import pytest

from augmentum.models.provider_registry import ProviderRegistry
from augmentum.state.provider_store import ProviderConfig, ProviderStore


@pytest.fixture
async def store_conn():
    """In-memory providers schema mirroring migrations 003 + 305."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            api_key TEXT,
            provider_type TEXT NOT NULL DEFAULT 'openai',
            is_enabled INTEGER NOT NULL DEFAULT 1,
            is_default INTEGER NOT NULL DEFAULT 0,
            profile_id TEXT NOT NULL DEFAULT '',
            owner_user_id TEXT DEFAULT '',
            shared INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    await conn.commit()
    yield conn
    await conn.close()


def _cfg(pid: str, *, owner: str = "", shared: bool = False) -> ProviderConfig:
    return ProviderConfig(
        id=pid,
        name=pid,
        base_url=f"https://{pid}.example.com/v1",
        owner_user_id=owner,
        shared=shared,
    )


class TestStoreVisibilityFilter:
    async def test_list_visible_to_scopes_to_owner_shared_and_global(self, store_conn):
        store = ProviderStore(store_conn)
        # Global (admin/pre-305 backfill) providers are shared=1 with owner=''.
        await store.create_provider(_cfg("globalG", owner="", shared=True))
        await store.create_provider(_cfg("sharedX", owner="userB", shared=True))
        await store.create_provider(_cfg("privA", owner="userA", shared=False))
        await store.create_provider(_cfg("privB", owner="userB", shared=False))

        # userA sees: global + shared + own private (privA) — NOT userB's private.
        visible = {p.id for p in await store.list_providers(visible_to="userA")}
        assert visible == {"globalG", "sharedX", "privA"}
        assert "privB" not in visible

        # visible_to=None returns everything (startup load / admin management).
        assert len(await store.list_providers()) == 4

    async def test_unshared_ownerless_row_hidden_from_users(self, store_conn):
        """Regression: the 2026-07-01 leak. An admin Unshare on a pre-305
        provider left ``shared=0, owner_user_id=''`` — and the original
        empty-owner escape hatch kept it visible to every user. It must be
        hidden from non-owners (the share route now stamps ownership, but
        the filter must fail safe on the raw state regardless)."""
        store = ProviderStore(store_conn)
        await store.create_provider(_cfg("unshared_legacy", owner="", shared=False))
        visible = {p.id for p in await store.list_providers(visible_to="benchUser")}
        assert "unshared_legacy" not in visible

    async def test_roundtrip_preserves_owner_and_shared(self, store_conn):
        store = ProviderStore(store_conn)
        await store.create_provider(_cfg("privA", owner="userA", shared=False))
        got = await store.get_provider("privA")
        assert got is not None
        assert got.owner_user_id == "userA"
        assert got.shared is False

    async def test_share_toggle_updates_row(self, store_conn):
        store = ProviderStore(store_conn)
        await store.create_provider(_cfg("privA", owner="userA", shared=False))
        updated = await store.update_provider("privA", shared=True)
        assert updated is not None and updated.shared is True
        # Now visible to a different user through the filter.
        visible = {p.id for p in await store.list_providers(visible_to="userZ")}
        assert "privA" in visible


class TestRegistryVisibilityPredicate:
    def _registry(self) -> ProviderRegistry:
        return ProviderRegistry(httpx.AsyncClient())

    def test_builtin_backend_visible_to_everyone(self):
        r = self._registry()
        # No meta entry => builtin/global infrastructure => always visible.
        assert r.provider_visible_to("ollama", "anyone") is True
        assert r.provider_visible_to("ollama", "") is True

    def test_private_provider_only_visible_to_owner(self):
        r = self._registry()
        r.set_provider_meta("privA", "userA", False)
        assert r.provider_visible_to("privA", "userA") is True
        assert r.provider_visible_to("privA", "userB") is False
        assert r.provider_visible_to("privA", "") is False

    def test_shared_and_global_visible_to_all(self):
        r = self._registry()
        r.set_provider_meta("sharedX", "userB", True)   # shared
        r.set_provider_meta("globalG", "", True)         # admin/global = shared
        assert r.provider_visible_to("sharedX", "userZ") is True
        assert r.provider_visible_to("globalG", "userZ") is True

    def test_unshared_ownerless_hidden_from_everyone(self):
        """Regression: the 2026-07-01 leak — ``shared=False, owner=''`` must
        NOT fall through to 'global infrastructure, visible to all'."""
        r = self._registry()
        r.set_provider_meta("unshared_legacy", "", False)
        assert r.provider_visible_to("unshared_legacy", "userZ") is False
        assert r.provider_visible_to("unshared_legacy", "") is False

    def test_clear_meta_reverts_to_global(self):
        r = self._registry()
        r.set_provider_meta("privA", "userA", False)
        assert r.provider_visible_to("privA", "userB") is False
        r.clear_provider_meta("privA")
        # Once meta is gone it's treated as builtin/global again.
        assert r.provider_visible_to("privA", "userB") is True
