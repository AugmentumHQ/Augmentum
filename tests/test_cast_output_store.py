"""Tests for the RenderOutputStore.

Pins:

  - store() returns a RenderOutput with token + caller's bytes
  - fetch(token) returns the same bytes
  - fetch(expired_token) returns None and drops the entry
  - fetch(single_use_token) returns once then None
  - revoke() drops the entry
  - eviction kicks in at max_active capacity
"""
from __future__ import annotations

import time

from augmentum.cast.output_store import RenderOutputStore


def test_store_and_fetch_roundtrip():
    store = RenderOutputStore()
    out = store.store(body=b"\x89PNG\r\n\x1a\nfake", content_type="image/png", user_id="u")
    assert out.token.startswith("ro_")
    assert out.content_type == "image/png"

    fetched = store.fetch(out.token)
    assert fetched is not None
    assert fetched.body == b"\x89PNG\r\n\x1a\nfake"
    assert fetched.user_id == "u"


def test_fetch_unknown_token_returns_none():
    store = RenderOutputStore()
    assert store.fetch("ro_nonexistent") is None


def test_fetch_expired_token_returns_none_and_drops():
    store = RenderOutputStore()
    # Cap TTL effectively to zero — clamped to 10s minimum, but we
    # set expires_at directly to force expiration.
    out = store.store(body=b"x", content_type="text/plain")
    out.expires_at = time.time() - 1.0  # forced expired

    assert store.fetch(out.token) is None
    # Re-fetching confirms it was dropped.
    assert store.fetch(out.token) is None


def test_single_use_fetches_once_then_gone():
    """Single-use tokens delete on successful fetch — receiver that
    re-requests gets a 404, which is the intended UX for one-shot casts."""
    store = RenderOutputStore()
    out = store.store(body=b"once", content_type="image/png", single_use=True)
    first = store.fetch(out.token)
    assert first is not None
    assert first.body == b"once"

    second = store.fetch(out.token)
    assert second is None


def test_multi_use_fetches_repeatedly():
    """Default tokens stay valid until TTL — receiver can re-fetch."""
    store = RenderOutputStore()
    out = store.store(body=b"keep", content_type="image/png")
    for _ in range(3):
        result = store.fetch(out.token)
        assert result is not None
        assert result.body == b"keep"


def test_revoke_drops_token():
    store = RenderOutputStore()
    out = store.store(body=b"x", content_type="text/plain")
    assert store.revoke(out.token) is True
    assert store.fetch(out.token) is None
    # Idempotent — second revoke is a no-op.
    assert store.revoke(out.token) is False


def test_eviction_kicks_in_at_capacity():
    """Storing N+1 outputs at capacity N evicts the oldest by
    expiration — first-in stays in memory only until the store is
    full + a new entry comes in."""
    store = RenderOutputStore(max_active=3)
    a = store.store(body=b"a", content_type="text/plain")
    b = store.store(body=b"b", content_type="text/plain")
    c = store.store(body=b"c", content_type="text/plain")

    # The fourth store triggers eviction of the oldest (a).
    d = store.store(body=b"d", content_type="text/plain")

    # b, c, d remain reachable; a is gone.
    assert store.fetch(b.token) is not None
    assert store.fetch(c.token) is not None
    assert store.fetch(d.token) is not None
    assert store.fetch(a.token) is None
