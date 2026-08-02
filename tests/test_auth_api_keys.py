"""Tests for inbound API keys.

Covers the module-level helpers, the ``ApiKeyManager`` round-trip
against in-memory SQLite, and the middleware acceptance of
``Authorization: Bearer sk-aug-...`` so external OpenAI clients can
hit user-scoped endpoints without a browser session.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from augmentum.auth.api_keys import (
    KEY_PREFIX,
    ApiKeyManager,
    _generate_raw,
    _hash,
    is_api_key,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def test_is_api_key_recognizes_prefix():
    assert is_api_key("sk-aug-abc123")
    assert not is_api_key("sk-openai-abc")
    assert not is_api_key("session-token-xyz")
    assert not is_api_key("")


def test_generate_raw_starts_with_prefix_and_is_unique():
    a = _generate_raw()
    b = _generate_raw()
    assert a.startswith(KEY_PREFIX)
    assert b.startswith(KEY_PREFIX)
    assert a != b
    # 7 chars prefix + 32 url-safe body = 39 chars (URL-safe base64 of 24 bytes)
    assert len(a) >= 35


def test_hash_is_deterministic_and_distinct_per_input():
    assert _hash("a") == _hash("a")
    assert _hash("a") != _hash("b")


# ---------------------------------------------------------------------------
# ApiKeyManager round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_db():
    """Real in-memory SQLite with migrations applied + a seeded user."""
    from augmentum.auth.session_manager import SessionManager
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    sm = SessionManager(backend._conn)
    user = _run(sm.create_user("alice", "supersecretpw", role="user"))
    yield backend, sm, user
    _run(backend.close())


def test_create_returns_raw_key_and_persists_only_hash(auth_db):
    _, sm, user = auth_db
    akm = ApiKeyManager(sm._db)

    raw, meta = _run(akm.create(user.id, name="laptop"))

    assert raw.startswith(KEY_PREFIX)
    assert meta["name"] == "laptop"
    assert meta["scope"] == "chat"
    assert meta["prefix"].startswith(KEY_PREFIX)
    # Raw key never appears in the listed metadata.
    keys = _run(akm.list_for_user(user.id))
    assert len(keys) == 1
    listed = keys[0]
    for value in listed.values():
        if not isinstance(value, str):
            continue
        assert raw not in value, "raw key leaked into listing"


def test_validate_returns_user_for_known_key(auth_db):
    _, sm, user = auth_db
    akm = ApiKeyManager(sm._db)
    raw, _ = _run(akm.create(user.id))

    resolved = _run(akm.validate(raw))
    assert resolved is not None
    assert resolved.id == user.id
    assert resolved.username == "alice"


def test_validate_rejects_unknown_and_malformed(auth_db):
    _, sm, _ = auth_db
    akm = ApiKeyManager(sm._db)

    assert _run(akm.validate("sk-aug-bogus-not-issued")) is None
    assert _run(akm.validate("not-an-api-key")) is None
    assert _run(akm.validate("")) is None


def test_validate_rejects_inactive_user(auth_db):
    backend, sm, user = auth_db
    akm = ApiKeyManager(sm._db)
    raw, _ = _run(akm.create(user.id))

    # Deactivate the user.
    _run(sm._db.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user.id,)))
    _run(sm._db.commit())
    akm.invalidate_cache()  # clear the previous lookup

    assert _run(akm.validate(raw)) is None


def test_revoke_deletes_and_invalidates_cache(auth_db):
    _, sm, user = auth_db
    akm = ApiKeyManager(sm._db)
    raw, meta = _run(akm.create(user.id))

    # Prime the cache.
    assert _run(akm.validate(raw)) is not None

    deleted = _run(akm.revoke(meta["id"], user.id))
    assert deleted is True

    # Subsequent validate must miss — cache invalidation is the test.
    assert _run(akm.validate(raw)) is None
    keys = _run(akm.list_for_user(user.id))
    assert keys == []


def test_revoke_other_users_key_returns_false(auth_db):
    _, sm, user = auth_db
    other = _run(sm.create_user("bob", "anotherpassw", role="user"))
    akm = ApiKeyManager(sm._db)
    _, meta = _run(akm.create(other.id))

    # Alice tries to revoke Bob's key — must fail without affecting it.
    assert _run(akm.revoke(meta["id"], user.id)) is False
    assert len(_run(akm.list_for_user(other.id))) == 1


def test_invalidate_user_cache_drops_only_that_users_entries(auth_db):
    """Per-user cache invalidation — used to keep ApiKeyManager
    coherent with SessionManager when a user's ``is_active`` flips."""
    _, sm, alice = auth_db
    bob = _run(sm.create_user("bob_iuc", "anotherpassw", role="user"))
    akm = ApiKeyManager(sm._db)
    raw_a, _ = _run(akm.create(alice.id))
    raw_b, _ = _run(akm.create(bob.id))

    # Prime both caches
    assert _run(akm.validate(raw_a)) is not None
    assert _run(akm.validate(raw_b)) is not None
    assert len(akm._cache) == 2

    akm.invalidate_user_cache(alice.id)
    # Alice's cache entry gone; Bob's still there
    assert len(akm._cache) == 1
    assert any(u.id == bob.id for u, _ in akm._cache.values())


def test_session_manager_update_user_chains_api_key_cache_invalidation(auth_db):
    """The bug this regression locks: previously, deactivating a user
    via SessionManager.update_user only invalidated SessionManager's
    token cache. ApiKeyManager kept its cached User (with is_active=True)
    until its own TTL, leaving the API key valid for up to 60s after
    deactivation. The chained invalidator fixes that."""
    _, sm, user = auth_db
    akm = ApiKeyManager(sm._db)
    sm.register_user_cache_invalidator(akm.invalidate_user_cache)

    raw, _ = _run(akm.create(user.id))
    # Prime ApiKeyManager's cache
    assert _run(akm.validate(raw)) is not None
    assert len(akm._cache) == 1

    # Admin deactivates user via SessionManager
    _run(sm.update_user(user.id, is_active=False))

    # Without the chained invalidation, akm._cache would still have
    # the (User, ts) pair with is_active=True and validate() would
    # return the stale User. With the chain, the cache is empty and
    # validate() re-queries the DB, sees is_active=0, returns None.
    assert len(akm._cache) == 0
    assert _run(akm.validate(raw)) is None


# ---------------------------------------------------------------------------
# Middleware acceptance
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_api_keys():
    """Fresh app with both SessionManager and ApiKeyManager wired."""
    from augmentum.auth.session_manager import SessionManager
    from augmentum.proxy.server import create_app
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())

    app = create_app()
    app.state.session_manager = SessionManager(backend._conn)
    app.state.api_key_manager = ApiKeyManager(backend._conn)
    app.state.state_manager = StateManager(backend)
    app.state.setup_token = "test-setup-token"

    yield app, backend

    _run(backend.close())


def test_middleware_accepts_api_key_bearer(app_with_api_keys):
    """A Bearer sk-aug-... header reaches /api/auth/me as the right user."""
    app, _ = app_with_api_keys
    sm = app.state.session_manager
    user = _run(sm.create_user("alice", "supersecret", role="user"))
    raw, _ = _run(app.state.api_key_manager.create(user.id, name="ext"))

    tc = TestClient(app)
    resp = tc.get("/api/auth/me", headers={"Authorization": f"Bearer {raw}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice"


def test_middleware_rejects_revoked_api_key(app_with_api_keys):
    """After revoke, the same key must stop authenticating."""
    app, _ = app_with_api_keys
    sm = app.state.session_manager
    user = _run(sm.create_user("alice", "supersecret", role="user"))
    raw, meta = _run(app.state.api_key_manager.create(user.id, name="ext"))

    tc = TestClient(app)
    headers = {"Authorization": f"Bearer {raw}"}
    assert tc.get("/api/auth/me", headers=headers).status_code == 200

    _run(app.state.api_key_manager.revoke(meta["id"], user.id))
    assert tc.get("/api/auth/me", headers=headers).status_code == 401


def test_create_route_returns_raw_key_once(app_with_api_keys):
    """POST /api/auth/keys returns the raw key; subsequent GET hides it."""
    app, _ = app_with_api_keys
    sm = app.state.session_manager
    user = _run(sm.create_user("alice", "supersecret", role="user"))
    session_token = _run(sm.create_session(user.id))

    tc = TestClient(app)
    tc.headers.update({"Authorization": f"Bearer {session_token}"})

    create = tc.post("/api/auth/keys", json={"name": "my laptop"})
    assert create.status_code == 200
    body = create.json()
    assert body["key"].startswith(KEY_PREFIX)
    assert body["name"] == "my laptop"

    listing = tc.get("/api/auth/keys").json()
    assert len(listing["keys"]) == 1
    listed = listing["keys"][0]
    # Only the prefix is exposed in the listing.
    assert listed["prefix"].startswith(KEY_PREFIX)
    assert "key" not in listed
    assert "key_hash" not in listed
