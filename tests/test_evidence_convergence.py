"""Earned Understanding P2: the Evidence Bus + convergence.

Belief is earned by INDEPENDENT sources converging (triangulation), not by one
channel repeating itself. These tests pin both the pure convergence math and
the ladder bridge: two distinct sources promote a PROVISIONAL belief; a
same-source repeat in between does NOT advance it.

See docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from augmentum.memory.evidence import (
    EvidenceStore,
    convergence_score,
    source_trust,
)
from augmentum.memory.models import MemoryTier, MemoryType, SourceType
from augmentum.state.backends.sqlite import SQLiteBackend

# --------------------------------------------------------------------------- #
# Pure convergence math
# --------------------------------------------------------------------------- #

def test_triangulation_beats_repetition():
    """Two distinct channels outweigh one channel mentioned twice."""
    repetition = convergence_score({"chat": 2})
    triangulation = convergence_score({"chat": 1, "playlist": 1})
    assert triangulation > repetition


def test_single_source_saturates():
    """One channel can't earn unbounded confidence by repeating."""
    once = convergence_score({"chat": 1})
    hundred = convergence_score({"chat": 100})
    ceiling = source_trust("chat") / (1 - 0.5)  # trust/(1-decay)
    assert hundred > once
    assert hundred <= ceiling + 1e-6
    # The killer property: ONE new independent channel (a single chat + a
    # single playlist) outweighs a hundred repeats of the same channel.
    assert convergence_score({"chat": 1, "playlist": 1}) > hundred


def test_deliberate_sources_outweigh_ambient():
    assert convergence_score({"chat_explicit": 1}) > convergence_score({"browse": 1})


def test_empty_is_zero():
    assert convergence_score({}) == 0.0
    assert convergence_score({"chat": 0}) == 0.0


# --------------------------------------------------------------------------- #
# The ladder bridge — corroborate_belief
# --------------------------------------------------------------------------- #

def _fake_embed(text: str) -> list[float]:
    import hashlib
    vec: list[float] = []
    seed = text.encode()
    while len(vec) < 768:
        seed = hashlib.sha256(seed).digest()
        vec.extend(b / 255.0 for b in seed)
    return vec[:768]


def _fake_to_blob(vec: list[float]) -> bytes:
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)


def _patch_embed():
    p = patch("augmentum.memory.store.EmbeddingService")
    svc = p.start()
    svc.embed_one = MagicMock(side_effect=_fake_embed)
    svc.to_blob = MagicMock(side_effect=_fake_to_blob)
    return p


def _tier(mem) -> str:
    return mem.tier if isinstance(mem.tier, str) else mem.tier.value


async def _make():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    from augmentum.memory.store import MemoryStore
    return backend, MemoryStore(backend), EvidenceStore(backend)


@pytest.mark.asyncio
async def test_independent_sources_promote_repeats_do_not(monkeypatch):
    monkeypatch.setattr("augmentum.config.settings.memory_earned_permanence", True, raising=False)
    backend, store, ev = await _make()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User likes jazz",
            memory_type=MemoryType.FACT, user_id="u1",
            source_type=SourceType.EXTRACTED,
        )
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.PROVISIONAL.value
        a0 = (await store.get(mid, user_id="u1")).access_count
        # Bar = two independent sources from here.
        monkeypatch.setattr(
            "augmentum.config.settings.memory_corroboration_promote_access",
            a0 + 2, raising=False)

        # Source A (chat) — first independent channel.
        r1 = await ev.corroborate_belief(
            store, user_id="u1", memory_id=mid, source="chat", claim="said they like jazz")
        assert r1.new_source and r1.distinct_sources == 1
        assert (await store.get(mid, user_id="u1")).access_count == a0 + 1
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.PROVISIONAL.value

        # Source A AGAIN — same channel. Recorded, but must NOT advance the ladder.
        r2 = await ev.corroborate_belief(
            store, user_id="u1", memory_id=mid, source="chat", claim="mentioned jazz again")
        assert r2.new_source is False and r2.distinct_sources == 1
        assert (await store.get(mid, user_id="u1")).access_count == a0 + 1  # unchanged
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.PROVISIONAL.value

        # Source B (playlist) — a SECOND independent channel converges → promote.
        r3 = await ev.corroborate_belief(
            store, user_id="u1", memory_id=mid, source="playlist", claim="made a Jazz playlist")
        assert r3.new_source and r3.distinct_sources == 2
        assert (await store.get(mid, user_id="u1")).access_count == a0 + 2
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.ACTIVE.value

        # Score reflects two channels (one repeated once).
        assert r3.score == convergence_score({"chat": 2, "playlist": 1})
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_evidence_trail_is_recorded(monkeypatch):
    """The Mirror: every belief traces to why she believes it."""
    backend, store, ev = await _make()
    try:
        await ev.record(user_id="u1", memory_id="m1", source="chat", claim="said it")
        await ev.record(user_id="u1", memory_id="m1", source="playlist", claim="curated it")
        trail = await ev.evidence_for(user_id="u1", memory_id="m1")
        assert len(trail) == 2
        assert {t["source"] for t in trail} == {"chat", "playlist"}
        assert await ev.distinct_sources(user_id="u1", memory_id="m1") == 2
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_evidence_is_user_scoped(monkeypatch):
    backend, store, ev = await _make()
    try:
        await ev.record(user_id="u1", memory_id="m1", source="chat")
        await ev.record(user_id="u2", memory_id="m1", source="chat")
        assert await ev.distinct_sources(user_id="u1", memory_id="m1") == 1
        assert (await ev.source_counts(user_id="u2", memory_id="m1")) == {"chat": 1}
    finally:
        await backend.close()
