"""Tests for augmentum/proxy/session.py helpers."""

from __future__ import annotations

from augmentum.proxy.session import derive_kv_session_key


def test_derive_kv_session_key_is_opaque_hash() -> None:
    """Returned key carries no readable user_id or session_id substring."""
    key = derive_kv_session_key("alice@example.com", "ses_deadbeef1234")
    assert key.startswith("kv_")
    assert "alice" not in key
    assert "ses_deadbeef" not in key
    # 3-char prefix + 24-char hex = 27 chars
    assert len(key) == 27


def test_derive_kv_session_key_scopes_per_user() -> None:
    """Same session_id under different users yields distinct keys."""
    key_a = derive_kv_session_key("user-a", "ses_shared")
    key_b = derive_kv_session_key("user-b", "ses_shared")
    assert key_a != key_b


def test_derive_kv_session_key_is_deterministic() -> None:
    """Same inputs always produce the same key (manifest lookup invariant)."""
    key1 = derive_kv_session_key("user-x", "ses_42")
    key2 = derive_kv_session_key("user-x", "ses_42")
    assert key1 == key2


def test_derive_kv_session_key_anon_bucket_when_user_empty() -> None:
    """Empty user_id (auth disabled) collapses to a stable shared bucket.

    This is intentional: without auth there is no tenancy to enforce, so
    sharing across requests is fine. We just need consistency so a single
    no-auth deployment gets cache hits across restarts.
    """
    a = derive_kv_session_key("", "ses_42")
    b = derive_kv_session_key("", "ses_42")
    assert a == b
    # And distinct from any explicit user.
    assert a != derive_kv_session_key("alice", "ses_42")


def test_derive_kv_session_key_returns_empty_for_empty_session() -> None:
    """No session_id => no KV scoping (caller treats falsy as 'no slot')."""
    assert derive_kv_session_key("alice", "") == ""
