"""Integration test — full memory pipeline: store -> embed -> retrieve."""

from __future__ import annotations

import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.memory.models import ExtractedFact, Memory, MemoryTier, MemoryType, SourceType
from augmentum.state.backends.sqlite import SQLiteBackend


def _deterministic_embed(text: str) -> list[float]:
    """Deterministic embedding based on text hash — same text = same vector."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    vec = [b / 255.0 for b in h]
    # Pad/truncate to 768 dims
    if len(vec) < 768:
        vec = vec + [0.0] * (768 - len(vec))
    return vec[:768]


def _deterministic_embed_batch(texts: list[str]) -> list[list[float]]:
    return [_deterministic_embed(t) for t in texts]


def _blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


class TestIntegrationMemoryRAG:
    """End-to-end memory pipeline with real SQLite."""

    async def test_store_and_retrieve_fact(self):
        """Full pipeline: create fact -> embed -> store -> query -> retrieve."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.embed_query = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)
                mock_svc.from_blob = MagicMock(
                    side_effect=lambda b: list(struct.unpack(f"<{len(b)//4}f", b))
                )

                # Store a fact
                mem_id = await store.store(
                    content="User is a Python developer who builds AI tools",
                    memory_type=MemoryType.FACT,
                    user_id="default",
                    importance=0.9,
                    confidence=0.95,
                    source_type=SourceType.EXTRACTED,
                )
                assert mem_id is not None

                # Verify it was stored
                mem = await store.get(mem_id)
                assert mem is not None
                assert "Python developer" in mem.content
                assert mem.importance == 0.9

                # Verify count
                counts = await store.count(user_id="default")
                assert counts["total"] == 1

                # Verify listing
                all_mems = await store.list_all(user_id="default")
                assert len(all_mems) == 1
                assert all_mems[0].id == mem_id
        finally:
            await backend.close()

    async def test_store_and_forget(self):
        """Store a fact, then soft-delete it."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                mem_id = await store.store(
                    content="Temporary fact to delete",
                    memory_type=MemoryType.FACT,
                )

                # Forget it
                result = await store.forget(mem_id)
                assert result is True

                # Should not appear in active listing
                active = await store.list_all(user_id="default", include_expired=False)
                assert len(active) == 0

                # Should still exist in DB
                mem = await store.get(mem_id)
                assert mem is not None
                assert mem.valid_until is not None
        finally:
            await backend.close()

    async def test_multiple_facts_different_types(self):
        """Store facts of different types and verify filtering."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                await store.store(content="Likes dark mode", memory_type=MemoryType.PREFERENCE)
                await store.store(content="Works at Acme Corp", memory_type=MemoryType.FACT)
                await store.store(content="Knows Python well", memory_type=MemoryType.SKILL)

                # Filter by type
                prefs = await store.list_all(memory_type=MemoryType.PREFERENCE)
                assert len(prefs) == 1
                assert prefs[0].memory_type == MemoryType.PREFERENCE

                facts = await store.list_all(memory_type=MemoryType.FACT)
                assert len(facts) == 1

                # Total
                all_mems = await store.list_all()
                assert len(all_mems) == 3
        finally:
            await backend.close()

    async def test_provisional_tier_low_confidence(self):
        """Low-confidence extracted facts go to PROVISIONAL tier."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                mem_id = await store.store(
                    content="Might enjoy hiking",
                    memory_type=MemoryType.FACT,
                    confidence=0.4,
                    source_type=SourceType.EXTRACTED,
                )
                mem = await store.get(mem_id)
                tier_val = mem.tier if isinstance(mem.tier, str) else mem.tier.value
                assert tier_val == MemoryTier.PROVISIONAL
        finally:
            await backend.close()

    async def test_explicit_source_stays_active(self):
        """Explicit user statements stay ACTIVE even at low confidence."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                mem_id = await store.store(
                    content="I definitely like hiking",
                    memory_type=MemoryType.FACT,
                    confidence=0.5,
                    source_type=SourceType.EXPLICIT,
                )
                mem = await store.get(mem_id)
                tier_val = mem.tier if isinstance(mem.tier, str) else mem.tier.value
                assert tier_val == MemoryTier.ACTIVE
        finally:
            await backend.close()

    async def test_store_fact_with_evidence(self):
        """ExtractedFact.evidence should be persisted to the evidence column."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                fact = ExtractedFact(
                    content="User uses VS Code",
                    importance=0.7,
                    confidence=0.9,
                    evidence="I always use VS Code for development",
                )
                mem_id = await store.store_fact(fact)
                mem = await store.get(mem_id)
                assert mem is not None
                assert mem.evidence == "I always use VS Code for development"
        finally:
            await backend.close()

    async def test_embedding_blob_correct_dimensions(self):
        """Verify the stored blob has the right byte count for 768 dims."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                mem_id = await store.store(
                    content="Test embedding dimensions",
                    memory_type=MemoryType.FACT,
                )

                # Read the raw blob from DB
                cursor = await backend.conn.execute(
                    "SELECT embedding FROM memories WHERE id = ?", (mem_id,)
                )
                row = await cursor.fetchone()
                assert row is not None
                blob_data = row[0]
                assert len(blob_data) == 768 * 4
        finally:
            await backend.close()

    async def test_store_and_edit(self):
        """Edit a stored memory's content."""
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            from augmentum.memory.store import MemoryStore

            store = MemoryStore(backend)

            with patch("augmentum.memory.store.EmbeddingService") as mock_svc:
                mock_svc.embed_one = MagicMock(side_effect=_deterministic_embed)
                mock_svc.to_blob = MagicMock(side_effect=_blob)

                mem_id = await store.store(
                    content="Old content",
                    memory_type=MemoryType.FACT,
                )
                success = await store.edit(mem_id, "New updated content")
                assert success is True

                mem = await store.get(mem_id)
                assert mem.content == "New updated content"
        finally:
            await backend.close()
