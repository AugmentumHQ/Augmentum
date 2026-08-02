"""Subtractive-memory Slice 2: the write bar (earned permanence).

Passively EXTRACTED facts land in PROVISIONAL (never injected) and only climb
to durable ACTIVE on CORROBORATION (re-mention). Deliberate EXPLICIT writes
bypass and land ACTIVE immediately.

See docs/superpowers/specs/2026-06-20-memory-subtractive-design.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from augmentum.memory.models import MemoryTier, MemoryType, SourceType
from augmentum.state.backends.sqlite import SQLiteBackend


async def _make_store():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    from augmentum.memory.store import MemoryStore
    return backend, MemoryStore(backend)


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


def _tier(mem) -> str:
    return mem.tier if isinstance(mem.tier, str) else mem.tier.value


def _patch_embed():
    p = patch("augmentum.memory.store.EmbeddingService")
    svc = p.start()
    svc.embed_one = MagicMock(side_effect=_fake_embed)
    svc.to_blob = MagicMock(side_effect=_fake_to_blob)
    return p


@pytest.mark.asyncio
async def test_extracted_fact_quarantined_to_provisional(monkeypatch):
    monkeypatch.setattr(
        "augmentum.config.settings.memory_earned_permanence", True, raising=False)
    backend, store = await _make_store()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User mentioned reading Isekai manga",
            memory_type=MemoryType.FACT, user_id="u1",
            source_type=SourceType.EXTRACTED,
        )
        mem = await store.get(mid, user_id="u1")
        assert _tier(mem) == MemoryTier.PROVISIONAL.value
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_explicit_fact_bypasses_to_active(monkeypatch):
    """Deliberate 'remember that…' lands ACTIVE immediately."""
    monkeypatch.setattr(
        "augmentum.config.settings.memory_earned_permanence", True, raising=False)
    backend, store = await _make_store()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User's dentist appointment is Friday",
            memory_type=MemoryType.FACT, user_id="u1",
            source_type=SourceType.EXPLICIT,
        )
        mem = await store.get(mid, user_id="u1")
        assert _tier(mem) == MemoryTier.ACTIVE.value
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_corroboration_promotes_to_active(monkeypatch):
    """Re-mention bumps access and promotes PROVISIONAL → ACTIVE at threshold."""
    monkeypatch.setattr(
        "augmentum.config.settings.memory_earned_permanence", True, raising=False)
    monkeypatch.setattr(
        "augmentum.config.settings.memory_corroboration_promote_access", 2, raising=False)
    backend, store = await _make_store()
    p = _patch_embed()
    try:
        content = "User has a cat named Moo"
        mid = await store.store(
            content=content, memory_type=MemoryType.FACT, user_id="u1",
            source_type=SourceType.EXTRACTED,
        )
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.PROVISIONAL.value

        # Two more exact re-mentions (corroboration) → access 1, then 2 → promote.
        for _ in range(2):
            again = await store.store(
                content=content, memory_type=MemoryType.FACT, user_id="u1",
                source_type=SourceType.EXTRACTED,
            )
            assert again == mid  # deduped onto the same row

        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.ACTIVE.value
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_legacy_mode_keeps_extracted_active(monkeypatch):
    """earned_permanence off → EXTRACTED lands ACTIVE (revert path)."""
    monkeypatch.setattr(
        "augmentum.config.settings.memory_earned_permanence", False, raising=False)
    backend, store = await _make_store()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User said the weather was nice",
            memory_type=MemoryType.FACT, user_id="u1",
            source_type=SourceType.EXTRACTED,
        )
        assert _tier(await store.get(mid, user_id="u1")) == MemoryTier.ACTIVE.value
    finally:
        p.stop()
        await backend.close()
