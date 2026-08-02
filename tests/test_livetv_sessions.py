"""Unit tests for the Live TV in-memory session store."""

from __future__ import annotations

import time

import pytest

from augmentum.media.livetv_sessions import LiveTvSessionStore


def _create(store, **overrides):
    kwargs = dict(
        user_id="usr_a",
        server_id="ms_1",
        provider="emby",
        base_url="http://emby.local:8096",
        access_token="t",
        channel_id="ch_1",
        play_session_id="ps_1",
        media_source_id="src_1",
        title="CNN",
        device_id="dev_1",
    )
    kwargs.update(overrides)
    return store.create(**kwargs)


def test_create_returns_session_with_url_safe_token():
    store = LiveTvSessionStore()
    s = _create(store)
    assert s.token  # non-empty
    # token_urlsafe is base64-url-safe alphanumerics + '-' + '_'
    assert all(c.isalnum() or c in "-_" for c in s.token)


def test_get_returns_session_for_owning_user():
    store = LiveTvSessionStore()
    s = _create(store)
    got = store.get(s.token, user_id="usr_a")
    assert got is not None
    assert got.channel_id == "ch_1"


def test_get_returns_none_for_wrong_user():
    """Cross-user lookup must look identical to "not found" — never
    confirm token existence to a wrong user."""
    store = LiveTvSessionStore()
    s = _create(store, user_id="usr_a")
    assert store.get(s.token, user_id="usr_b") is None


def test_get_returns_none_for_unknown_token():
    store = LiveTvSessionStore()
    assert store.get("garbage-token", user_id="usr_a") is None


def test_get_bumps_last_activity():
    store = LiveTvSessionStore()
    s = _create(store)
    original = s.last_activity
    time.sleep(0.01)
    store.get(s.token, user_id="usr_a")
    assert s.last_activity > original


def test_idle_expiry_drops_session_from_get():
    store = LiveTvSessionStore(idle_ttl_s=0.05)
    s = _create(store)
    time.sleep(0.1)
    assert store.get(s.token, user_id="usr_a") is None


def test_remove_drops_session_and_returns_it():
    store = LiveTvSessionStore()
    s = _create(store)
    removed = store.remove(s.token, user_id="usr_a")
    assert removed is not None
    assert removed.token == s.token
    assert store.get(s.token, user_id="usr_a") is None


def test_remove_with_wrong_user_does_nothing():
    store = LiveTvSessionStore()
    s = _create(store, user_id="usr_a")
    assert store.remove(s.token, user_id="usr_b") is None
    # Original owner can still see it
    assert store.get(s.token, user_id="usr_a") is not None


def test_remove_is_idempotent():
    store = LiveTvSessionStore()
    s = _create(store)
    assert store.remove(s.token, user_id="usr_a") is not None
    assert store.remove(s.token, user_id="usr_a") is None  # second call


def test_sweep_expired_drops_only_stale_sessions():
    store = LiveTvSessionStore(idle_ttl_s=0.05)
    fresh = _create(store, user_id="usr_a", channel_id="ch_fresh")
    stale = _create(store, user_id="usr_b", channel_id="ch_stale")
    # Backdate stale
    stale.last_activity = time.time() - 1.0
    swept = store.sweep_expired()
    assert swept == 1
    assert store.get(fresh.token, user_id="usr_a") is not None
    assert store.get(stale.token, user_id="usr_b") is None


def test_count_reflects_active_sessions():
    store = LiveTvSessionStore()
    assert store.count() == 0
    _create(store, user_id="usr_a")
    _create(store, user_id="usr_b")
    assert store.count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
