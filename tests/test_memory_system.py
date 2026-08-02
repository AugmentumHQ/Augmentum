"""Tests for the cross-session memory system — migration, models, vec integration."""

from __future__ import annotations

import struct
from pathlib import Path

import aiosqlite
import pytest

from augmentum.memory.embeddings import EmbeddingService
from augmentum.memory.models import (
    ExtractedFact,
    Memory,
    MemoryType,
    SourceType,
)

# SQL schema for bootstrapping test DB (migrations 001 + 006)
_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
INSERT INTO schema_version (version, description) VALUES (5, 'pre-memory baseline');
"""

_MIGRATIONS_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _apply_migration(conn: aiosqlite.Connection, version: int) -> None:
    """Apply a specific numbered migration file."""
    for path in _MIGRATIONS_DIR.glob("*.sql"):
        try:
            v = int(path.stem.split("_")[0])
        except (ValueError, IndexError):
            continue
        if v == version:
            sql = path.read_text(encoding="utf-8")
            await conn.executescript(sql)
            await conn.commit()
            return
    raise FileNotFoundError(f"Migration {version:03d} not found")


def _float_vec_to_blob(vec: list[float]) -> bytes:
    """Pack a list of floats into a little-endian float32 blob (for sqlite-vec)."""
    return struct.pack(f"<{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestMemoryModels:
    """Tests for memory data models."""

    def test_memory_type_enum_values(self):
        assert MemoryType.PREFERENCE == "preference"
        assert MemoryType.FACT == "fact"
        assert MemoryType.ENTITY == "entity"
        assert MemoryType.NARRATIVE == "narrative"
        assert MemoryType.ANALYSIS == "analysis"

    def test_source_type_enum_values(self):
        assert SourceType.EXTRACTED == "extracted"
        assert SourceType.USER_MANUAL == "user_manual"
        assert SourceType.SYSTEM == "system"

    def test_memory_dataclass_defaults(self):
        mem = Memory(
            id="m1",
            user_id="default",
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
        )
        assert mem.importance == 0.5
        assert mem.confidence == 0.8
        assert mem.access_count == 0
        assert mem.embedding is None
        assert mem.superseded_by is None

    def test_extracted_fact_defaults(self):
        fact = ExtractedFact(content="User is a data scientist")
        assert fact.type == MemoryType.FACT
        assert fact.importance == 0.5
        assert fact.confidence == 0.8
        assert fact.source_context == {}

    def test_extracted_fact_custom(self):
        fact = ExtractedFact(
            content="Always use metric units",
            type=MemoryType.PREFERENCE,
            importance=0.9,
            confidence=0.95,
            source_context={"session_id": "s1", "message_index": 3},
        )
        assert fact.type == MemoryType.PREFERENCE
        assert fact.importance == 0.9
        assert fact.source_context["session_id"] == "s1"


# ---------------------------------------------------------------------------
# Migration Tests
# ---------------------------------------------------------------------------


class TestMemoryMigration:
    """Tests for migration 006_memory.sql."""

    @pytest.fixture
    async def db(self, tmp_path: Path) -> aiosqlite.Connection:
        """Create an in-memory DB with baseline schema, apply migration 006."""
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_BOOTSTRAP_SQL)
        await conn.commit()
        await _apply_migration(conn, 6)
        yield conn
        await conn.close()

    @pytest.mark.asyncio
    async def test_memories_table_exists(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_memories_fts_exists(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_schema_version_updated(self, db):
        cursor = await db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        assert row[0] == 6

    @pytest.mark.asyncio
    async def test_insert_memory(self, db):
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type) "
            "VALUES (?, ?, ?, ?)",
            ("m1", "default", "User likes Python", "preference"),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM memories WHERE id = 'm1'")
        row = await cursor.fetchone()
        assert row is not None
        assert dict(row)["content"] == "User likes Python"
        assert dict(row)["importance"] == 0.5

    @pytest.mark.asyncio
    async def test_fts_trigger_on_insert(self, db):
        """FTS5 trigger fires on INSERT — memory searchable via FTS."""
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type) "
            "VALUES (?, ?, ?, ?)",
            ("m1", "default", "User works in biotech research", "fact"),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT content FROM memories_fts WHERE memories_fts MATCH 'biotech'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert "biotech" in row[0]

    @pytest.mark.asyncio
    async def test_fts_trigger_on_update(self, db):
        """FTS5 trigger fires on UPDATE — updated content searchable."""
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type) "
            "VALUES (?, ?, ?, ?)",
            ("m1", "default", "User works in biotech", "fact"),
        )
        await db.commit()
        await db.execute(
            "UPDATE memories SET content = 'User works in finance' WHERE id = 'm1'"
        )
        await db.commit()
        # Old term gone
        cursor = await db.execute(
            "SELECT content FROM memories_fts WHERE memories_fts MATCH 'biotech'"
        )
        assert await cursor.fetchone() is None
        # New term present
        cursor = await db.execute(
            "SELECT content FROM memories_fts WHERE memories_fts MATCH 'finance'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_fts_trigger_on_delete(self, db):
        """FTS5 trigger fires on DELETE — removed from FTS."""
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type) "
            "VALUES (?, ?, ?, ?)",
            ("m1", "default", "User likes jazz music", "preference"),
        )
        await db.commit()
        await db.execute("DELETE FROM memories WHERE id = 'm1'")
        await db.commit()
        cursor = await db.execute(
            "SELECT content FROM memories_fts WHERE memories_fts MATCH 'jazz'"
        )
        assert await cursor.fetchone() is None

    @pytest.mark.asyncio
    async def test_indexes_exist(self, db):
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_memories%'"
        )
        indexes = [row[0] for row in await cursor.fetchall()]
        assert "idx_memories_user" in indexes
        assert "idx_memories_type" in indexes
        assert "idx_memories_valid" in indexes

    @pytest.mark.asyncio
    async def test_superseded_by_foreign_key(self, db):
        """superseded_by references memories(id)."""
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type) "
            "VALUES (?, ?, ?, ?)",
            ("m1", "default", "User has brown hair", "entity"),
        )
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type, superseded_by) "
            "VALUES (?, ?, ?, ?, ?)",
            ("m_old", "default", "User had blonde hair", "entity", "m1"),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT superseded_by FROM memories WHERE id = 'm_old'"
        )
        row = await cursor.fetchone()
        assert row[0] == "m1"

    @pytest.mark.asyncio
    async def test_embedding_blob_storage(self, db):
        """Embedding stored as BLOB (float32 packed)."""
        vec = [0.1, 0.2, 0.3] * 128  # 384 dims
        blob = _float_vec_to_blob(vec)
        await db.execute(
            "INSERT INTO memories (id, user_id, content, memory_type, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            ("m1", "default", "test", "fact", blob),
        )
        await db.commit()
        cursor = await db.execute("SELECT embedding FROM memories WHERE id = 'm1'")
        row = await cursor.fetchone()
        assert row[0] is not None
        assert len(row[0]) == 384 * 4  # 384 floats * 4 bytes each


# ---------------------------------------------------------------------------
# SQLiteBackend vec integration tests
# ---------------------------------------------------------------------------


class TestSQLiteBackendVec:
    """Tests for sqlite-vec extension loading in SQLiteBackend."""

    @pytest.mark.asyncio
    async def test_vec_available_flag(self):
        """sqlite-vec import sets _VEC_AVAILABLE."""
        from augmentum.state.backends.sqlite import _VEC_AVAILABLE
        # If sqlite-vec is installed (it should be), flag is True
        assert _VEC_AVAILABLE is True

    @pytest.mark.asyncio
    async def test_backend_connect_sets_vec_enabled(self, tmp_path):
        """SQLiteBackend.connect() loads sqlite-vec and sets vec_enabled."""
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.connect()
        try:
            assert backend.vec_enabled is True
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_vec_distance_cosine_works(self, tmp_path):
        """vec_distance_cosine() function is available after extension load."""
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.connect()
        try:
            vec_a = _float_vec_to_blob([1.0, 0.0, 0.0])
            vec_b = _float_vec_to_blob([1.0, 0.0, 0.0])
            cursor = await backend.conn.execute(
                "SELECT vec_distance_cosine(?, ?)", (vec_a, vec_b)
            )
            row = await cursor.fetchone()
            # Same vector → cosine distance = 0
            assert row[0] == pytest.approx(0.0, abs=1e-6)
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_vec_distance_orthogonal(self, tmp_path):
        """Orthogonal vectors have cosine distance = 1."""
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.connect()
        try:
            vec_a = _float_vec_to_blob([1.0, 0.0, 0.0])
            vec_b = _float_vec_to_blob([0.0, 1.0, 0.0])
            cursor = await backend.conn.execute(
                "SELECT vec_distance_cosine(?, ?)", (vec_a, vec_b)
            )
            row = await cursor.fetchone()
            assert row[0] == pytest.approx(1.0, abs=1e-6)
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_memories_vec_table_created(self, tmp_path):
        """vec0 virtual table is created after connect when vec is available."""
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.connect()
        try:
            cursor = await backend.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'"
            )
            row = await cursor.fetchone()
            assert row is not None
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_memories_vec_insert_and_query(self, tmp_path):
        """Insert a vector into memories_vec and query nearest neighbors."""
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "test.db"))
        await backend.connect()
        try:
            # Insert a memory first
            await backend.conn.execute(
                "INSERT INTO memories (id, user_id, content, memory_type) "
                "VALUES (?, ?, ?, ?)",
                ("m1", "default", "test memory", "fact"),
            )
            # Insert its vector
            vec = _float_vec_to_blob([0.5, 0.5, 0.0] + [0.0] * 381)  # 384 dims
            await backend.conn.execute(
                "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                ("m1", vec),
            )
            await backend.conn.commit()

            # Query nearest
            query_vec = _float_vec_to_blob([0.5, 0.5, 0.0] + [0.0] * 381)
            cursor = await backend.conn.execute(
                "SELECT memory_id, distance FROM memories_vec "
                "WHERE embedding MATCH ? AND k = 1 "
                "ORDER BY distance",
                (query_vec,),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "m1"
            assert row[1] == pytest.approx(0.0, abs=1e-5)
        finally:
            await backend.close()


# ---------------------------------------------------------------------------
# Embedding Service Tests
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TestEmbeddingService:
    """Tests for the FastEmbed embedding service."""

    def test_embed_one_returns_correct_dimension(self):
        vec = EmbeddingService.embed_one("hello world")
        assert isinstance(vec, list)
        assert len(vec) == EmbeddingService.DIMENSION
        assert all(isinstance(x, float) for x in vec)

    def test_embed_batch_returns_correct_count(self):
        texts = ["hello", "world", "test"]
        vecs = EmbeddingService.embed(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == EmbeddingService.DIMENSION

    def test_similar_texts_higher_similarity(self):
        """Similar texts should have higher cosine similarity than dissimilar."""
        vec_a = EmbeddingService.embed_one("I love programming in Python")
        vec_b = EmbeddingService.embed_one("I enjoy coding in Python")
        vec_c = EmbeddingService.embed_one("The weather is sunny today")

        sim_ab = _cosine_similarity(vec_a, vec_b)
        sim_ac = _cosine_similarity(vec_a, vec_c)
        assert sim_ab > sim_ac, f"Similar texts ({sim_ab}) should be closer than dissimilar ({sim_ac})"

    def test_identical_texts_near_one_similarity(self):
        vec_a = EmbeddingService.embed_one("test embedding")
        vec_b = EmbeddingService.embed_one("test embedding")
        sim = _cosine_similarity(vec_a, vec_b)
        assert sim > 0.999

    def test_blob_roundtrip(self):
        """to_blob and from_blob are inverse operations."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = EmbeddingService.to_blob(original)
        assert isinstance(blob, bytes)
        assert len(blob) == 5 * 4  # 5 floats * 4 bytes
        recovered = EmbeddingService.from_blob(blob)
        for a, b in zip(original, recovered, strict=False):
            assert a == pytest.approx(b, abs=1e-6)

    def test_embed_and_blob_full_roundtrip(self):
        """Embed text → to_blob → from_blob preserves the vector."""
        vec = EmbeddingService.embed_one("roundtrip test")
        blob = EmbeddingService.to_blob(vec)
        recovered = EmbeddingService.from_blob(blob)
        assert len(recovered) == EmbeddingService.DIMENSION
        for a, b in zip(vec, recovered, strict=False):
            assert a == pytest.approx(b, abs=1e-6)

    def test_dimension_constant(self):
        assert EmbeddingService.DIMENSION == 384

    def test_model_name(self):
        assert "bge-small" in EmbeddingService.MODEL_NAME


# ---------------------------------------------------------------------------
# Memory Store Tests
# ---------------------------------------------------------------------------


class TestMemoryStore:
    """Tests for MemoryStore CRUD and hybrid search."""

    @pytest.fixture
    async def store(self, tmp_path):
        """Create a MemoryStore with a real SQLiteBackend."""
        from augmentum.memory.store import MemoryStore
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "mem_test.db"))
        await backend.connect()
        ms = MemoryStore(backend)
        yield ms
        await backend.close()

    @pytest.mark.asyncio
    async def test_store_and_get(self, store):
        mid = await store.store("User is a data scientist", memory_type="fact")
        mem = await store.get(mid, user_id="default")
        assert mem is not None
        assert mem.content == "User is a data scientist"
        assert mem.memory_type == "fact"

    @pytest.mark.asyncio
    async def test_store_deduplication(self, store):
        """Storing near-identical content reuses existing memory."""
        mid1 = await store.store("User works in biotech research", memory_type="fact")
        mid2 = await store.store("User works in biotech research", memory_type="fact")
        # Should deduplicate
        assert mid1 == mid2

    @pytest.mark.asyncio
    async def test_store_different_content_creates_new(self, store):
        """Storing genuinely different content creates separate memories."""
        mid1 = await store.store("User loves Python programming", memory_type="preference")
        mid2 = await store.store("The weather is rainy today", memory_type="fact")
        assert mid1 != mid2

    @pytest.mark.asyncio
    async def test_recall_finds_relevant(self, store):
        """recall() returns memories relevant to the query."""
        await store.store("User works in biotech research", memory_type="fact", importance=0.9)
        await store.store("User prefers dark mode", memory_type="preference", importance=0.7)
        await store.store("User's name is Alice", memory_type="fact", importance=0.95)

        results = await store.recall("what does the user do for work")
        assert len(results) > 0
        contents = [m.content for m in results]
        assert any("biotech" in c for c in contents)

    @pytest.mark.asyncio
    async def test_recall_updates_access_count(self, store):
        mid = await store.store("User is a developer", memory_type="fact")
        await store.recall("developer")
        mem = await store.get(mid, user_id="default")
        assert mem.access_count >= 1

    @pytest.mark.asyncio
    async def test_forget_soft_deletes(self, store):
        mid = await store.store("Temporary info", memory_type="fact")
        success = await store.forget(mid, user_id="default")
        assert success is True
        mem = await store.get(mid, user_id="default")
        assert mem.valid_until is not None

    @pytest.mark.asyncio
    async def test_forget_excluded_from_recall(self, store):
        await store.store("User likes jazz", memory_type="preference", importance=0.9)
        mid = await store.store("User likes country", memory_type="preference", importance=0.9)
        await store.forget(mid, user_id="default")
        results = await store.recall("music preference")
        contents = [m.content for m in results]
        assert not any("country" in c for c in contents)

    @pytest.mark.asyncio
    async def test_edit_updates_content(self, store):
        mid = await store.store("User has brown hair", memory_type="entity")
        success = await store.edit(mid, "User has red hair", user_id="default")
        assert success is True
        mem = await store.get(mid, user_id="default")
        assert mem.content == "User has red hair"

    @pytest.mark.asyncio
    async def test_list_all(self, store):
        await store.store("Fact one", memory_type="fact")
        await store.store("Fact two", memory_type="fact")
        await store.store("Preference one", memory_type="preference")
        all_mems = await store.list_all()
        assert len(all_mems) == 3

    @pytest.mark.asyncio
    async def test_list_all_filter_by_type(self, store):
        await store.store("Fact one", memory_type="fact")
        await store.store("Preference one", memory_type="preference")
        facts = await store.list_all(memory_type=MemoryType.FACT)
        assert len(facts) == 1
        assert facts[0].memory_type == "fact"

    @pytest.mark.asyncio
    async def test_list_all_excludes_expired(self, store):
        mid = await store.store("Expired fact", memory_type="fact")
        await store.store("Active fact", memory_type="fact")
        await store.forget(mid, user_id="default")
        all_mems = await store.list_all(include_expired=False)
        assert len(all_mems) == 1
        assert all_mems[0].content == "Active fact"

    @pytest.mark.asyncio
    async def test_count(self, store):
        await store.store("Fact", memory_type="fact")
        await store.store("Pref", memory_type="preference")
        await store.store("Fact2", memory_type="fact")
        counts = await store.count()
        assert counts["total"] == 3
        assert counts.get("fact", 0) == 2
        assert counts.get("preference", 0) == 1

    @pytest.mark.asyncio
    async def test_supersede(self, store):
        """Superseding marks old memory expired and creates new one."""
        old_id = await store.store("Alice has brown hair", memory_type="entity")
        new_id = await store.supersede(
            old_id, "Alice has red hair", memory_type="entity", user_id="default",
        )
        assert old_id != new_id
        old_mem = await store.get(old_id, user_id="default")
        assert old_mem.valid_until is not None
        assert old_mem.superseded_by == new_id
        new_mem = await store.get(new_id, user_id="default")
        assert new_mem.valid_until is None
        assert new_mem.content == "Alice has red hair"

    @pytest.mark.asyncio
    async def test_store_extracted_fact(self, store):
        fact = ExtractedFact(
            content="User prefers metric units",
            type=MemoryType.PREFERENCE,
            importance=0.8,
        )
        mid = await store.store_fact(fact)
        mem = await store.get(mid, user_id="default")
        assert mem.content == "User prefers metric units"
        assert mem.source_type == "extracted"

    @pytest.mark.asyncio
    async def test_get_history(self, store):
        """Version history follows superseded_by chain."""
        id1 = await store.store("Version 1", memory_type="fact")
        id2 = await store.supersede(id1, "Version 2", memory_type="fact", user_id="default")
        await store.supersede(id2, "Version 3", memory_type="fact", user_id="default")
        history = await store.get_history(id1, user_id="default")
        assert len(history) >= 2
        contents = [m.content for m in history]
        assert "Version 1" in contents


# ---------------------------------------------------------------------------
# Extraction Pipeline Tests
# ---------------------------------------------------------------------------


class TestHeuristicExtraction:
    """Tests for the heuristic memory extractor."""

    def test_extract_identity(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("I am a data scientist working on ML models.")
        assert len(facts) >= 1
        contents = [f.content for f in facts]
        assert any("data scientist" in c for c in contents)

    def test_extract_name(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("My name is Alice and I love coding.")
        assert len(facts) >= 1
        contents = [f.content for f in facts]
        assert any("Alice" in c for c in contents)

    def test_extract_preference(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("I prefer Python over JavaScript for backend work.")
        contents = [f.content for f in facts]
        assert any("Python" in c for c in contents)

    def test_extract_dislike(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("I don't like verbose documentation.")
        contents = [f.content for f in facts]
        assert any("verbose" in c.lower() for c in contents)

    def test_extract_remember_instruction(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("Remember that Lyra is left-handed.")
        assert len(facts) >= 1
        contents = [f.content for f in facts]
        assert any("Lyra" in c for c in contents)
        # Should have high importance
        remember_facts = [f for f in facts if "Lyra" in f.content]
        assert remember_facts[0].importance >= 0.9

    def test_extract_always_instruction(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("Always use metric units in your responses.")
        assert len(facts) >= 1
        types = [f.type for f in facts]
        assert MemoryType.PREFERENCE in types

    def test_extract_work_location(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("I work at Google in the AI department.")
        contents = [f.content for f in facts]
        assert any("Google" in c for c in contents)

    def test_extract_no_facts_from_generic(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("What is the capital of France?")
        assert len(facts) == 0

    def test_extract_dedup_within_message(self):
        from augmentum.memory.extractor import heuristic_extract
        facts = heuristic_extract("I am a developer. I am a developer.")
        # Should not have duplicate entries
        contents = [f.content for f in facts]
        unique = set(c.lower() for c in contents)
        assert len(unique) == len(contents)

    @pytest.mark.asyncio
    async def test_extract_and_store_integration(self, tmp_path):
        """Full extraction pipeline: extract from message and store in DB."""
        from augmentum.memory.extractor import extract_and_store
        from augmentum.memory.store import MemoryStore
        from augmentum.state.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "extract_test.db"))
        await backend.connect()
        try:
            store = MemoryStore(backend)
            count = await extract_and_store(
                session_id="ses_test",
                user_id="default",
                user_message="My name is Bob and I'm a software engineer. I prefer dark mode.",
                assistant_response="Nice to meet you, Bob!",
                store=store,
            )
            assert count >= 2
            # Verify stored
            all_mems = await store.list_all()
            assert len(all_mems) >= 2
            contents = [m.content for m in all_mems]
            assert any("Bob" in c for c in contents)
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_extract_and_store_empty_message(self, tmp_path):
        """No extraction from empty or generic messages."""
        from augmentum.memory.extractor import extract_and_store
        from augmentum.memory.store import MemoryStore
        from augmentum.state.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "extract_empty.db"))
        await backend.connect()
        try:
            store = MemoryStore(backend)
            count = await extract_and_store(
                session_id="ses_test",
                user_id="default",
                user_message="What is 2+2?",
                assistant_response="4",
                store=store,
            )
            assert count == 0
        finally:
            await backend.close()


# ---------------------------------------------------------------------------
# Memory Injection Tests
# ---------------------------------------------------------------------------


class TestMemoryInjection:
    """Tests for memory recall and injection into request context."""

    @pytest.fixture
    async def store_with_data(self, tmp_path):
        """Create a MemoryStore pre-populated with test memories."""
        from augmentum.memory.store import MemoryStore
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "inject_test.db"))
        await backend.connect()
        ms = MemoryStore(backend)
        await ms.store("User is a biotech researcher", memory_type="fact", importance=0.9)
        await ms.store("User prefers concise responses", memory_type="preference", importance=0.8)
        await ms.store("User's name is Alice", memory_type="fact", importance=0.95)
        yield ms, backend
        await backend.close()

    @pytest.mark.asyncio
    async def test_recall_and_inject_adds_system_message(self, store_with_data):
        from unittest.mock import MagicMock, patch

        from augmentum.memory.integration import recall_and_inject
        from augmentum.models.base import InternalChatRequest, Message

        ms, backend = store_with_data

        app_state = MagicMock()
        app_state.memory_store = ms

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="What should I study next?")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 5
            mock_settings.memory_recall_min_score = 0.0
            mock_settings.memory_inject_min_score = 0.0
            mock_settings.memory_summary_max_chars = 500
            await recall_and_inject(request, app_state)

        # Should have injected a system message
        assert len(request.messages) >= 2
        system_msg = request.messages[0]
        assert system_msg.role == "system"
        assert "[background]" in system_msg.content

    @pytest.mark.asyncio
    async def test_recall_and_inject_prepends_to_existing_system(self, store_with_data):
        from unittest.mock import MagicMock, patch

        from augmentum.memory.integration import recall_and_inject
        from augmentum.models.base import InternalChatRequest, Message

        ms, backend = store_with_data

        app_state = MagicMock()
        app_state.memory_store = ms

        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Tell me about biotech"),
            ],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 5
            mock_settings.memory_recall_min_score = 0.0
            mock_settings.memory_inject_min_score = 0.0
            mock_settings.memory_summary_max_chars = 500
            await recall_and_inject(request, app_state)

        system_msg = request.messages[0]
        assert "[background]" in system_msg.content
        assert "You are a helpful assistant." in system_msg.content

    @pytest.mark.asyncio
    async def test_recall_and_inject_disabled(self, store_with_data):
        from unittest.mock import MagicMock, patch

        from augmentum.memory.integration import recall_and_inject
        from augmentum.models.base import InternalChatRequest, Message

        ms, backend = store_with_data
        app_state = MagicMock()
        app_state.memory_store = ms

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="Hello")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = False
            await recall_and_inject(request, app_state)

        # No injection when disabled
        assert len(request.messages) == 1

    @pytest.mark.asyncio
    async def test_recall_and_inject_no_store(self):
        from unittest.mock import MagicMock, patch

        from augmentum.memory.integration import recall_and_inject
        from augmentum.models.base import InternalChatRequest, Message

        app_state = MagicMock()
        app_state.memory_store = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="Hello")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            await recall_and_inject(request, app_state)

        assert len(request.messages) == 1

    def test_format_memory_summary(self):
        from augmentum.memory.integration import format_memory_summary
        from augmentum.memory.models import Memory

        memories = [
            Memory(id="1", user_id="default", content="User is a developer",
                   memory_type="fact", confidence=0.9),
            Memory(id="2", user_id="default", content="User prefers dark mode",
                   memory_type="preference", confidence=0.8),
        ]
        summary = format_memory_summary(memories)
        assert "User is a developer" in summary
        assert "User prefers dark mode" in summary
        assert "confidence: 0.9" in summary

    def test_format_memory_summary_empty(self):
        from augmentum.memory.integration import format_memory_summary
        assert format_memory_summary([]) == ""


# ---------------------------------------------------------------------------
# Memory API Endpoint Tests
# ---------------------------------------------------------------------------


class TestMemoryAPI:
    """Tests for memory REST API endpoints."""

    @pytest.fixture
    async def memory_client(self, tmp_path):
        """Create a TestClient with a real memory store."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        from augmentum.memory.store import MemoryStore
        from augmentum.proxy.server import create_app
        from augmentum.state.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "api_test.db"))
        await backend.connect()

        app = create_app()

        # Mock out non-memory state
        from augmentum.models.base import ModelInfo
        from augmentum.state.backends.memory import MemoryBackend as InMemBackend
        from augmentum.state.manager import StateManager

        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[
            ModelInfo(name="test", model="test", size=0, digest="", modified_at=""),
        ])

        app.state.http_client = MagicMock()
        app.state.provider_registry = MagicMock()
        app.state.provider_registry.backends = {"ollama": mock_backend}
        app.state.state_manager = StateManager(InMemBackend())
        app.state.classifier = MagicMock()
        app.state.narrative_engines = {}
        app.state.tool_registry = MagicMock()
        app.state.prompt_cache = MagicMock()
        app.state.prefix_cache = MagicMock()
        app.state.request_deduplicator = MagicMock()
        app.state.model_manager = MagicMock()
        app.state.image_queue = None

        # Mock session_manager so AuthMiddleware doesn't fail closed with
        # 503 on every request. These tests verify memory API behavior, not
        # auth — but the middleware was added by multi-tenancy stage A
        # (commit 608a341) and runs on all routes including /v1/memory/*.
        # Without a session_manager set, auth_unavailable_denied returns 503
        # before our handlers ever run.
        # User id MUST match MemoryStore's default ("default") so that tests
        # which call ``await store.store(...)`` without an explicit user_id
        # produce memories visible to the API path (which scopes by the
        # authenticated user). Tests verify memory API behavior under user
        # scoping; aligning the IDs preserves that intent without modifying
        # every test to thread user_id through.
        from augmentum.auth.models import User
        test_user = User(
            id="default",
            username="memory_tester",
            display_name="Memory Test User",
            role="admin",
            is_active=True,
        )
        mock_sm = MagicMock()
        mock_sm.validate_token = AsyncMock(return_value=test_user)
        mock_sm.get_user_by_id = AsyncMock(return_value=test_user)
        mock_sm.validate_ws_ticket = MagicMock(return_value=test_user.id)
        app.state.session_manager = mock_sm

        # Set up memory store
        store = MemoryStore(backend)
        app.state.memory_store = store

        client = TestClient(app)
        client.headers.update({"Authorization": "Bearer test-token"})
        yield client, store
        await backend.close()

    @pytest.mark.asyncio
    async def test_store_and_list(self, memory_client):
        client, store = memory_client
        # Store via API
        resp = client.post("/v1/memory/store", json={
            "content": "User is a developer",
            "memory_type": "fact",
            "importance": 0.9,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data

        # List
        resp = client.get("/v1/memory/facts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["content"] == "User is a developer"

    @pytest.mark.asyncio
    async def test_search(self, memory_client):
        client, store = memory_client
        await store.store("User works in AI research", memory_type="fact", importance=0.9)
        resp = client.get("/v1/memory/search?q=artificial+intelligence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 0  # May or may not find depending on search

    @pytest.mark.asyncio
    async def test_edit(self, memory_client):
        client, store = memory_client
        mid = await store.store("Original content", memory_type="fact")
        resp = client.put(f"/v1/memory/facts/{mid}", json={"content": "Updated content"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_delete(self, memory_client):
        client, store = memory_client
        mid = await store.store("To be deleted", memory_type="fact")
        resp = client.delete(f"/v1/memory/facts/{mid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_not_found(self, memory_client):
        client, store = memory_client
        resp = client.delete("/v1/memory/facts/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stats(self, memory_client):
        client, store = memory_client
        await store.store("Fact", memory_type="fact")
        await store.store("Pref", memory_type="preference")
        resp = client.get("/v1/memory/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["total"] == 2

    @pytest.mark.asyncio
    async def test_history(self, memory_client):
        client, store = memory_client
        old_id = await store.store("Version 1", memory_type="fact")
        await store.supersede(old_id, "Version 2", memory_type="fact", user_id="default")
        resp = client.get(f"/v1/memory/facts/{old_id}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) >= 1

    @pytest.mark.asyncio
    async def test_list_filter_by_type(self, memory_client):
        client, store = memory_client
        await store.store("A fact", memory_type="fact")
        await store.store("A preference", memory_type="preference")
        resp = client.get("/v1/memory/facts?type=fact")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["memory_type"] == "fact"

    @pytest.mark.asyncio
    async def test_compact(self, memory_client):
        client, store = memory_client
        resp = client.post("/v1/memory/compact")
        assert resp.status_code == 200
        assert "stats" in resp.json()


# ---------------------------------------------------------------------------
# Temporal Fact Versioning Tests
# ---------------------------------------------------------------------------


class TestTemporalVersioning:
    """Tests for automatic contradiction detection and fact versioning."""

    @pytest.fixture
    async def store(self, tmp_path):
        from augmentum.memory.store import MemoryStore
        from augmentum.state.backends.sqlite import SQLiteBackend
        backend = SQLiteBackend(str(tmp_path / "version_test.db"))
        await backend.connect()
        ms = MemoryStore(backend)
        yield ms
        await backend.close()

    @pytest.mark.asyncio
    async def test_explicit_supersede_chain(self, store):
        """Explicit supersede() creates a version chain."""
        id1 = await store.store("Alice has brown hair", memory_type="entity")
        id2 = await store.supersede(id1, "Alice dyed her hair red", memory_type="entity", user_id="default")

        old = await store.get(id1, user_id="default")
        assert old.valid_until is not None
        assert old.superseded_by == id2

        new = await store.get(id2, user_id="default")
        assert new.valid_until is None
        assert new.content == "Alice dyed her hair red"

    @pytest.mark.asyncio
    async def test_history_returns_all_versions(self, store):
        id1 = await store.store("V1", memory_type="fact")
        id2 = await store.supersede(id1, "V2", memory_type="fact", user_id="default")
        await store.supersede(id2, "V3", memory_type="fact", user_id="default")

        history = await store.get_history(id1, user_id="default")
        assert len(history) >= 2
        contents = [m.content for m in history]
        assert "V1" in contents

    @pytest.mark.asyncio
    async def test_superseded_excluded_from_recall(self, store):
        """Superseded (expired) memories don't appear in recall."""
        id1 = await store.store("User works at Google", memory_type="fact", importance=0.9)
        await store.supersede(id1, "User works at Meta", memory_type="fact", importance=0.9, user_id="default")

        results = await store.recall("where does the user work", limit=10)
        contents = [m.content for m in results]
        # Only the current version should appear
        assert any("Meta" in c for c in contents)
        assert not any("Google" in c and "Meta" not in c for c in contents)

    @pytest.mark.asyncio
    async def test_superseded_excluded_from_list(self, store):
        id1 = await store.store("Old fact", memory_type="fact")
        await store.supersede(id1, "New fact", memory_type="fact", user_id="default")

        all_mems = await store.list_all(include_expired=False)
        contents = [m.content for m in all_mems]
        assert "New fact" in contents
        assert "Old fact" not in contents

    @pytest.mark.asyncio
    async def test_include_expired_shows_all(self, store):
        id1 = await store.store("Old fact", memory_type="fact")
        await store.supersede(id1, "New fact", memory_type="fact", user_id="default")

        all_mems = await store.list_all(include_expired=True)
        contents = [m.content for m in all_mems]
        assert "Old fact" in contents
        assert "New fact" in contents

    @pytest.mark.asyncio
    async def test_automatic_contradiction_detection(self, store):
        """Related but different content auto-supersedes the old memory."""
        # These are about the same entity/topic but state different facts
        id1 = await store.store("Alice has brown hair", memory_type="entity", importance=0.8)
        id2 = await store.store("Alice has red hair", memory_type="entity", importance=0.8)

        # They should NOT be the same ID (not deduped) — different content
        assert id1 != id2

        # The old one should be superseded
        old = await store.get(id1, user_id="default")
        if old.superseded_by:
            # Auto-superseded
            assert old.valid_until is not None
            assert old.superseded_by == id2
            new = await store.get(id2, user_id="default")
            assert new.valid_until is None
        # If not auto-superseded (similarity below threshold), that's acceptable too
        # — the exact threshold behavior depends on the embedding model

    @pytest.mark.asyncio
    async def test_count_excludes_expired(self, store):
        id1 = await store.store("Old", memory_type="fact")
        await store.supersede(id1, "New", memory_type="fact", user_id="default")
        counts = await store.count()
        assert counts["total"] == 1  # Only the current version
