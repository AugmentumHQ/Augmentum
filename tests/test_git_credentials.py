"""Tests for git credential token storage."""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from augmentum.coder.git_credentials import GitTokenStore


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def token_store(event_loop):
    async def _make():
        conn = await aiosqlite.connect(":memory:")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS user_settings "
            "(user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, updated_at TEXT, "
            "PRIMARY KEY (user_id, key))"
        )
        await conn.commit()
        return GitTokenStore(conn), conn

    store, conn = event_loop.run_until_complete(_make())
    yield store
    event_loop.run_until_complete(conn.close())


def test_set_and_get_token(token_store, event_loop):
    async def _run():
        await token_store.set_token("github.com", "ghp_abc123", username="oauth2")
        token = await token_store.get_token("github.com")
        assert token is not None
        assert token["token"] == "ghp_abc123"
        assert token["username"] == "oauth2"
        assert token["host"] == "github.com"
    event_loop.run_until_complete(_run())


def test_get_missing_token(token_store, event_loop):
    async def _run():
        token = await token_store.get_token("gitlab.com")
        assert token is None
    event_loop.run_until_complete(_run())


def test_list_tokens_redacted(token_store, event_loop):
    async def _run():
        await token_store.set_token("github.com", "ghp_abc123")
        await token_store.set_token("gitlab.com", "glpat-xyz789")
        tokens = await token_store.list_tokens()
        assert len(tokens) == 2
        for t in tokens:
            assert "token" not in t
            assert t["host"] in ("github.com", "gitlab.com")
    event_loop.run_until_complete(_run())


def test_delete_token(token_store, event_loop):
    async def _run():
        await token_store.set_token("github.com", "ghp_abc123")
        await token_store.delete_token("github.com")
        token = await token_store.get_token("github.com")
        assert token is None
    event_loop.run_until_complete(_run())


def test_overwrite_token(token_store, event_loop):
    async def _run():
        await token_store.set_token("github.com", "ghp_old")
        await token_store.set_token("github.com", "ghp_new")
        token = await token_store.get_token("github.com")
        assert token["token"] == "ghp_new"
    event_loop.run_until_complete(_run())


def test_user_scoped_tokens_are_isolated(token_store, event_loop):
    async def _run():
        await token_store.set_token("github.com", "alice-token", user_id="alice")
        await token_store.set_token("github.com", "bob-token", user_id="bob")

        alice = await token_store.get_token("github.com", user_id="alice")
        bob = await token_store.get_token("github.com", user_id="bob")
        assert alice is not None and alice["token"] == "alice-token"
        assert bob is not None and bob["token"] == "bob-token"

        alice_hosts = await token_store.list_tokens(user_id="alice")
        bob_hosts = await token_store.list_tokens(user_id="bob")
        assert [t["host"] for t in alice_hosts] == ["github.com"]
        assert [t["host"] for t in bob_hosts] == ["github.com"]
    event_loop.run_until_complete(_run())


def test_user_scoped_get_falls_back_to_legacy_global(token_store, event_loop):
    async def _run():
        await token_store.set_token("gitlab.com", "legacy-global")

        token = await token_store.get_token("gitlab.com", user_id="alice")

        assert token is not None
        assert token["token"] == "legacy-global"
        assert await token_store.list_tokens(user_id="alice") == []
    event_loop.run_until_complete(_run())
