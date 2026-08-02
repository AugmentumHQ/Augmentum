"""Tests for augmentum/memory/store.py — MemoryStore CRUD (without embeddings)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.memory.models import ExtractedFact, Memory, MemoryTier, MemoryType, SourceType
from augmentum.state.backends.sqlite import SQLiteBackend


async def _make_store():
    """Create a MemoryStore backed by :memory: SQLite with migrations."""
    backend = SQLiteBackend(":memory:")
    await backend.connect()

    # MemoryStore requires embedding — we'll mock it for direct DB tests
    from augmentum.memory.store import MemoryStore
    store = MemoryStore(backend)
    return backend, store


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding based on hash — unique per text."""
    import hashlib
    # Use multiple rounds of hashing to fill 768 dims with unique values
    vec = []
    seed = text.encode()
    while len(vec) < 768:
        seed = hashlib.sha256(seed).digest()
        vec.extend(b / 255.0 for b in seed)
    return vec[:768]


def _fake_to_blob(vec: list[float]) -> bytes:
    """Pack float vector into blob, same as real EmbeddingService.to_blob."""
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)


class TestMemoryStoreCRUD:
    """Direct DB operations on the memory store."""

    async def test_store_and_get(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                mem_id = await store.store(
                    content="User likes Python",
                    memory_type=MemoryType.PREFERENCE,
                    user_id="default",
                    importance=0.8,
                )
                assert mem_id is not None
                assert len(mem_id) > 0

                mem = await store.get(mem_id, user_id="default")
                assert mem is not None
                assert mem.content == "User likes Python"
                assert mem.memory_type == MemoryType.PREFERENCE
        finally:
            await backend.close()

    async def test_forget_soft_deletes(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                mem_id = await store.store(
                    content="Temp fact",
                    memory_type=MemoryType.FACT,
                )
                result = await store.forget(mem_id, user_id="default")
                assert result is True

                # Soft-deleted: still exists but has valid_until set
                mem = await store.get(mem_id, user_id="default")
                assert mem is not None
                assert mem.valid_until is not None
        finally:
            await backend.close()

    async def test_forget_nonexistent(self):
        backend, store = await _make_store()
        try:
            result = await store.forget("nonexistent-id", user_id="default")
            assert result is False
        finally:
            await backend.close()

    async def test_list_all(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                await store.store(content="Fact 1", memory_type=MemoryType.FACT)
                await store.store(content="Fact 2 something different entirely", memory_type=MemoryType.FACT)
                await store.store(content="Pref 1 unique preference about colors", memory_type=MemoryType.PREFERENCE)

                all_mems = await store.list_all(user_id="default", limit=50)
                assert len(all_mems) == 3
        finally:
            await backend.close()

    async def test_list_all_by_type(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                await store.store(content="Fact", memory_type=MemoryType.FACT)
                await store.store(content="Pref", memory_type=MemoryType.PREFERENCE)

                facts = await store.list_all(
                    user_id="default", memory_type=MemoryType.FACT,
                )
                assert len(facts) == 1
                assert facts[0].memory_type == MemoryType.FACT
        finally:
            await backend.close()

    async def test_count(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                await store.store(content="F1", memory_type=MemoryType.FACT)
                await store.store(content="F2", memory_type=MemoryType.FACT)
                await store.store(content="P1", memory_type=MemoryType.PREFERENCE)

                counts = await store.count(user_id="default")
                assert counts["total"] == 3
                assert counts.get("fact", 0) == 2
                assert counts.get("preference", 0) == 1
        finally:
            await backend.close()

    async def test_store_fact_convenience(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                fact = ExtractedFact(
                    content="User is a software engineer",
                    type=MemoryType.FACT,
                    importance=0.9,
                    confidence=0.95,
                )
                mem_id = await store.store_fact(fact)
                assert mem_id is not None

                mem = await store.get(mem_id, user_id="default")
                assert mem is not None
                assert mem.content == "User is a software engineer"
        finally:
            await backend.close()

    async def test_provisional_tier_for_low_confidence(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)

                mem_id = await store.store(
                    content="Maybe likes cats",
                    memory_type=MemoryType.FACT,
                    confidence=0.5,  # Below 0.7 threshold
                    source_type=SourceType.EXTRACTED,
                )
                mem = await store.get(mem_id, user_id="default")
                assert mem is not None
                tier_val = mem.tier if isinstance(mem.tier, str) else mem.tier.value
                assert tier_val == MemoryTier.PROVISIONAL
        finally:
            await backend.close()

    async def test_memory_model_fields(self):
        """Verify Memory dataclass has expected fields."""
        m = Memory(
            id="test",
            user_id="default",
            content="hello",
            memory_type=MemoryType.FACT,
        )
        assert m.access_count == 0
        assert m.tier == MemoryTier.ACTIVE
        assert m.evidence == ""


class TestIsolatedScopeMirror:
    """An isolated scope (``harness``) must never surface in the GENERAL pool.

    Mirror of the C1 fix: C1 stopped harness recall from reading other scopes;
    this guards the reverse — the general memory store / management UI (which
    read with ``scope=None``) must NOT return harness rows, or the agent's
    coding conventions leak into the user's personal memory store. Reported by
    Matt 2026-06-29 ("memories from using it were leaking through to my general
    memory store ... like it was input in chat/passthrough").
    """

    async def _seed(self, store):
        # One harness-scoped row, one general (unscoped) row, one mode-scoped row.
        await store.store(
            content="Fix the root cause, not the symptom.",
            memory_type=MemoryType.PROCEDURAL, user_id="u1",
            scope="harness", scope_strict=True, importance=0.9,
        )
        await store.store(
            content="User's favorite color is teal.",
            memory_type=MemoryType.FACT, user_id="u1", importance=0.9,
        )
        await store.store(
            content="User wants every grain of it.",
            memory_type=MemoryType.PREFERENCE, user_id="u1",
            scope="passthrough", importance=0.8,
        )

    async def test_general_list_excludes_harness(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)
                await self._seed(store)

                general = await store.list_all(user_id="u1", scope=None, limit=50)
                texts = [m.content for m in general]
                assert not any("root cause" in t for t in texts), (
                    "harness row leaked into the general store"
                )
                # The user's own (general + mode-scoped) memories still show.
                assert any("teal" in t for t in texts)
                assert any("every grain" in t for t in texts)
        finally:
            await backend.close()

    async def test_explicit_harness_scope_still_reads_it(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)
                await self._seed(store)

                harness = await store.list_all(
                    user_id="u1", scope="harness", limit=50,
                )
                texts = [m.content for m in harness]
                assert any("root cause" in t for t in texts), (
                    "harness's own scoped read must still see harness rows"
                )
        finally:
            await backend.close()

    async def test_count_excludes_harness(self):
        backend, store = await _make_store()
        try:
            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_fake_embed)
                mock_svc.to_blob = MagicMock(side_effect=_fake_to_blob)
                await self._seed(store)

                counts = await store.count(user_id="u1")
                # 2 general/mode rows counted, harness excluded.
                assert counts["total"] == 2
                assert counts.get(MemoryType.PROCEDURAL.value, 0) == 0
        finally:
            await backend.close()
