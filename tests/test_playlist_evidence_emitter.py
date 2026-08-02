"""Earned Understanding P2 activation: the playlist evidence emitter.

A named playlist is a deliberate signal. The emitter CONSERVATIVELY matches it
against existing beliefs and corroborates on a confident match (convergence) —
or, with no match, seeds a PROVISIONAL candidate + a review-card offer rather
than silently creating durable memory.

See docs/superpowers/specs/2026-06-20-earned-understanding-design.md (P2).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from augmentum.memory.evidence import EvidenceStore
from augmentum.memory.evidence_emitters import (
    _has_standalone_token,
    _subject_terms,
    emit_playlist_evidence,
)
from augmentum.memory.models import MemoryTier, MemoryType, SourceType
from augmentum.state.backends.sqlite import SQLiteBackend

# --------------------------------------------------------------------------- #
# Subject extraction + matching guards (pure)
# --------------------------------------------------------------------------- #

def test_subject_terms_strips_generic_words():
    assert _subject_terms("Study Jazz") == ["study", "jazz"]
    assert _subject_terms("My Favorites Mix") == []   # all generic
    assert _subject_terms("Jazz") == ["jazz"]
    assert _subject_terms("") == []


def test_standalone_token_rejects_substrings():
    assert _has_standalone_token("User likes jazz", "jazz")
    assert not _has_standalone_token("User put it in the cart", "art")
    assert not _has_standalone_token("User likes painting", "paint")  # only whole words


# --------------------------------------------------------------------------- #
# The two emitter paths (real in-memory store)
# --------------------------------------------------------------------------- #

def _fake_embed(text: str) -> list[float]:
    import hashlib
    vec: list[float] = []
    seed = text.encode()
    while len(vec) < 768:
        seed = hashlib.sha256(seed).digest()
        vec.extend(b / 255.0 for b in seed)
    return vec[:768]


def _patch_embed():
    p = patch("augmentum.memory.store.EmbeddingService")
    svc = p.start()
    svc.embed_one = MagicMock(side_effect=_fake_embed)
    svc.to_blob = MagicMock(side_effect=lambda v: __import__("struct").pack(f"<{len(v)}f", *v))
    return p


def _tier(mem) -> str:
    return mem.tier if isinstance(mem.tier, str) else mem.tier.value


async def _make():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    from augmentum.memory.store import MemoryStore
    return backend, MemoryStore(backend), EvidenceStore(backend)


@pytest.mark.asyncio
async def test_playlist_corroborates_and_promotes_matching_belief(monkeypatch):
    """A jazz belief mentioned once in chat (PROVISIONAL) + a Jazz playlist
    (a second independent source) → converges to ACTIVE."""
    monkeypatch.setattr("augmentum.config.settings.memory_earned_permanence", True, raising=False)
    backend, store, ev = await _make()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User likes jazz", memory_type=MemoryType.FACT,
            user_id="u1", source_type=SourceType.EXTRACTED)
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.PROVISIONAL.value
        a0 = (await store.get(mid, user_id="u1")).access_count
        monkeypatch.setattr(
            "augmentum.config.settings.memory_corroboration_promote_access",
            a0 + 1, raising=False)  # one independent corroboration promotes

        res = await emit_playlist_evidence(
            store, ev, user_id="u1", playlist_name="Study Jazz")

        assert res.matched_memory_id == mid
        assert res.corroborated and res.promoted_checked
        assert res.offered is False
        # The playlist was a second independent source → promoted.
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.ACTIVE.value
        # Evidence trail recorded the playlist.
        assert await ev.distinct_sources(user_id="u1", memory_id=mid) == 1
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_playlist_with_no_match_offers_candidate(monkeypatch):
    """No existing belief → PROVISIONAL candidate + a review-card offer, not a
    silent durable memory."""
    monkeypatch.setattr("augmentum.config.settings.memory_earned_permanence", True, raising=False)
    backend, store, ev = await _make()
    p = _patch_embed()
    try:
        res = await emit_playlist_evidence(
            store, ev, user_id="u1", playlist_name="Salsa")
        assert res.matched_memory_id == ""
        assert res.offered and res.candidate_memory_id
        cand = await store.get(res.candidate_memory_id, user_id="u1")
        assert "salsa" in cand.content.lower()
        assert _tier(cand) == MemoryTier.PROVISIONAL.value  # quarantined until approved
        # A review notification was queued for it.
        from augmentum.memory.notifications import get_pending
        pend = await get_pending(backend.conn, user_id="u1")
        assert any(n["id"] == res.candidate_memory_id for n in pend)
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_generic_playlist_name_is_ignored(monkeypatch):
    backend, store, ev = await _make()
    p = _patch_embed()
    try:
        res = await emit_playlist_evidence(
            store, ev, user_id="u1", playlist_name="My Favorites")
        assert res.terms == [] and not res.offered and not res.corroborated
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_missing_stores_is_safe(monkeypatch):
    res = await emit_playlist_evidence(None, None, user_id="u1", playlist_name="Jazz")
    assert not res.corroborated and not res.offered
