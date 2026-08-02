"""Tests for the cast pair-token store.

Pins:
  - start() returns a pending record with a unique 8-char code
  - approve() ties a record to a user_id + issues a single-use ws_token
  - approve rejects: unknown code, expired, already-approved, empty user_id
  - poll() returns current state; expired pending records flip to expired
  - consume_token() validates + marks consumed (single-use)
  - revoke() drops record + token index
  - eviction kicks in at capacity
"""
from __future__ import annotations

import time

from augmentum.cast.pair_store import (
    STATE_APPROVED,
    STATE_CONSUMED,
    STATE_EXPIRED,
    STATE_PENDING,
    PairStore,
)


# ── start ─────────────────────────────────────────────────────────


def test_start_returns_pending_record_with_unique_code():
    store = PairStore()
    a = store.start()
    b = store.start()
    assert a.state == STATE_PENDING
    assert b.state == STATE_PENDING
    assert a.pair_code != b.pair_code
    assert len(a.pair_code) == 8
    # Codes use only readable charset (no I/O/0/1).
    assert all(c in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" for c in a.pair_code)


# ── approve ───────────────────────────────────────────────────────


def test_approve_pending_record_issues_token():
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice")
    assert approved is not None
    assert approved.state == STATE_APPROVED
    assert approved.user_id == "alice"
    assert approved.ws_token.startswith("wsp_")


def test_approve_defaults_lifetime_to_home():
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice")
    assert approved.lifetime == "home"


def test_approve_records_away_lifetime():
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice", lifetime="away")
    assert approved.lifetime == "away"


def test_approve_unknown_lifetime_falls_back_to_away():
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice", lifetime="bogus")
    assert approved.lifetime == "away"


def test_approve_rejects_unknown_code():
    store = PairStore()
    assert store.approve("NOPE9999", user_id="alice") is None


def test_approve_rejects_expired_record():
    store = PairStore()
    r = store.start()
    r.expires_at = time.time() - 1.0  # forced expired
    assert store.approve(r.pair_code, user_id="alice") is None
    # State flipped to expired so subsequent poll reports it.
    polled = store.poll(r.pair_code)
    assert polled is not None and polled.state == STATE_EXPIRED


def test_approve_rejects_double_approval():
    """A second approve attempt — even from the same user — must fail
    so a leaked-then-recovered code can't be silently re-bound."""
    store = PairStore()
    r = store.start()
    first = store.approve(r.pair_code, user_id="alice")
    assert first is not None
    second = store.approve(r.pair_code, user_id="bob")
    assert second is None


def test_approve_rejects_empty_user_id():
    """Defensive: an empty user_id reaching approve() is a bug
    upstream — refuse to bind."""
    store = PairStore()
    r = store.start()
    assert store.approve(r.pair_code, user_id="") is None
    polled = store.poll(r.pair_code)
    assert polled is not None and polled.state == STATE_PENDING


# ── poll ──────────────────────────────────────────────────────────


def test_poll_shows_state_transitions():
    store = PairStore()
    r = store.start()
    pending = store.poll(r.pair_code)
    assert pending is not None and pending.state == STATE_PENDING
    store.approve(r.pair_code, user_id="alice")
    approved = store.poll(r.pair_code)
    assert approved is not None and approved.state == STATE_APPROVED


def test_poll_unknown_code_returns_none():
    store = PairStore()
    assert store.poll("NOPE9999") is None


# ── consume_token ─────────────────────────────────────────────────


def test_consume_token_validates_and_marks_consumed():
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice")
    record = store.consume_token(approved.ws_token)
    assert record is not None
    assert record.user_id == "alice"
    assert record.state == STATE_CONSUMED


def test_consume_token_single_use():
    """Replaying a token must fail — single-use is the load-bearing
    security property of this flow."""
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice")
    first = store.consume_token(approved.ws_token)
    assert first is not None
    replay = store.consume_token(approved.ws_token)
    assert replay is None


def test_consume_token_rejects_unknown():
    store = PairStore()
    assert store.consume_token("wsp_doesnotexist") is None
    assert store.consume_token("") is None


def test_consume_token_rejects_pending_record_token():
    """A token shouldn't even exist for a pending record, but defensive:
    if one's faked, it must not auth anything."""
    store = PairStore()
    r = store.start()
    r.ws_token = "wsp_fake"  # injected without going through approve
    assert store.consume_token("wsp_fake") is None


# ── revoke ────────────────────────────────────────────────────────


def test_revoke_drops_record_and_token_index():
    store = PairStore()
    r = store.start()
    approved = store.approve(r.pair_code, user_id="alice")
    assert store.revoke(r.pair_code) is True
    # Token can't be consumed after revocation.
    assert store.consume_token(approved.ws_token) is None
    # Idempotent.
    assert store.revoke(r.pair_code) is False


# ── eviction ──────────────────────────────────────────────────────


def test_eviction_at_capacity():
    store = PairStore(max_active=3)
    a = store.start()
    b = store.start()
    c = store.start()
    d = store.start()
    # Oldest (a) should be evicted to make room for d.
    assert store.poll(a.pair_code) is None
    assert store.poll(b.pair_code) is not None
    assert store.poll(c.pair_code) is not None
    assert store.poll(d.pair_code) is not None
