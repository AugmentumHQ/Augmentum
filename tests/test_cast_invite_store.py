"""Tests for the cast couch co-op InviteStore.

Pins the substrate that grants short-TTL "join this session" auth to
guest phones. Counter-decrement behaviour, TTL expiry semantics, the
host-scoped revoke guard, and the session-sweep API are all
load-bearing for the auth boundary, so each gets explicit coverage.

See spec at ``docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md``.
"""

from __future__ import annotations

import time

import pytest

from augmentum.cast.invite_store import InviteStore

# ── mint ─────────────────────────────────────────────────────────────


class TestMint:
    def test_returns_wsi_prefixed_token(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1")
        assert rec.token.startswith("wsi_")
        assert len(rec.token) > len("wsi_") + 16  # at least 12 bytes hex

    def test_defaults_to_3_slots(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1")
        assert rec.slots_remaining == 3

    def test_max_slots_caps_at_caller_value(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=1)
        assert rec.slots_remaining == 1

    def test_max_slots_lower_bound_is_one(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=0)
        # 0 is nonsense; store clamps to 1 rather than minting a dead token.
        assert rec.slots_remaining == 1

    def test_ttl_override(self):
        store = InviteStore()
        rec = store.mint(
            session_id="s1", host_user_id="u1", ttl_s=600,
        )
        assert rec.expires_at - time.time() == pytest.approx(600, abs=2)

    def test_empty_session_id_raises(self):
        store = InviteStore()
        with pytest.raises(ValueError):
            store.mint(session_id="", host_user_id="u1")

    def test_empty_user_id_raises(self):
        store = InviteStore()
        with pytest.raises(ValueError):
            store.mint(session_id="s1", host_user_id="")


# ── claim ────────────────────────────────────────────────────────────


class TestClaim:
    def test_decrements_slots_remaining(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=3)
        claimed = store.claim(rec.token)
        assert claimed is not None
        assert claimed.slots_remaining == 2

    def test_unknown_token_returns_none(self):
        store = InviteStore()
        assert store.claim("wsi_unknown") is None

    def test_returns_none_when_slots_exhausted(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=1)
        first = store.claim(rec.token)
        assert first is not None
        assert first.slots_remaining == 0
        # Second claim fails — slots_remaining hit zero, token now dead.
        second = store.claim(rec.token)
        assert second is None

    def test_returns_none_after_expiry(self):
        store = InviteStore()
        rec = store.mint(
            session_id="s1", host_user_id="u1", ttl_s=30,
        )
        # Force expiry without sleeping.
        rec.expires_at = time.time() - 1
        assert store.claim(rec.token) is None

    def test_returns_none_after_revoke(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1")
        store.revoke(rec.token)
        assert store.claim(rec.token) is None

    def test_records_guest_profile_id_when_provided(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=3)
        store.claim(rec.token, guest_profile_id="gp_alice")
        store.claim(rec.token, guest_profile_id="gp_bob")
        assert rec.claimed_by == ["gp_alice", "gp_bob"]

    def test_same_profile_cannot_double_claim(self):
        """Re-claim by the same profile must not consume an extra slot.

        Otherwise a flapping guest connection could drain the slot pool
        with a single profile.
        """
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=3)
        first = store.claim(rec.token, guest_profile_id="gp_alice")
        assert first is not None
        again = store.claim(rec.token, guest_profile_id="gp_alice")
        assert again is None
        assert rec.slots_remaining == 2  # only one slot spent

    def test_anonymous_claim_does_not_block_others(self):
        """Phase 1 anonymous joins pass guest_profile_id='' — those
        must not collide via the dedup path.
        """
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1", max_slots=3)
        first = store.claim(rec.token, guest_profile_id="")
        assert first is not None
        again = store.claim(rec.token, guest_profile_id="")
        assert again is not None
        assert rec.slots_remaining == 1


# ── revoke ───────────────────────────────────────────────────────────


class TestRevoke:
    def test_returns_true_when_removed(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1")
        assert store.revoke(rec.token) is True

    def test_unknown_token_returns_false(self):
        store = InviteStore()
        assert store.revoke("wsi_unknown") is False

    def test_cross_user_revoke_rejected(self):
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1")
        assert store.revoke(rec.token, host_user_id="u2") is False
        # Token still active for the rightful owner.
        assert store.get(rec.token) is not None

    def test_system_revoke_bypasses_user_check(self):
        """Passing host_user_id='' explicitly skips the ownership guard
        so session-end sweeps can clear invites regardless of owner.
        """
        store = InviteStore()
        rec = store.mint(session_id="s1", host_user_id="u1")
        assert store.revoke(rec.token, host_user_id="") is True


# ── session sweep ────────────────────────────────────────────────────


class TestSessionSweep:
    def test_revoke_for_session_clears_all_matching(self):
        store = InviteStore()
        a = store.mint(session_id="s1", host_user_id="u1")
        b = store.mint(session_id="s1", host_user_id="u1")
        c = store.mint(session_id="s2", host_user_id="u1")
        count = store.revoke_for_session("s1")
        assert count == 2
        assert store.get(a.token) is None
        assert store.get(b.token) is None
        # Other session untouched.
        assert store.get(c.token) is not None

    def test_revoke_for_unknown_session_returns_zero(self):
        store = InviteStore()
        assert store.revoke_for_session("ghost") == 0


# ── listing / inspection ─────────────────────────────────────────────


class TestListings:
    def test_list_for_session_returns_only_active(self):
        store = InviteStore()
        a = store.mint(session_id="s1", host_user_id="u1")
        b = store.mint(session_id="s1", host_user_id="u1", max_slots=1)
        # Drain b — should drop out of active list.
        store.claim(b.token)
        active = store.list_for_session("s1")
        assert len(active) == 1
        assert active[0].token == a.token

    def test_list_for_host_filters_by_owner(self):
        store = InviteStore()
        store.mint(session_id="s1", host_user_id="alice")
        store.mint(session_id="s2", host_user_id="alice")
        store.mint(session_id="s3", host_user_id="bob")
        assert len(store.list_for_host("alice")) == 2
        assert len(store.list_for_host("bob")) == 1
