"""Tests for state stores — notes, balancer, discovery, provider."""

from __future__ import annotations

import aiosqlite

from augmentum.state.balancer_store import BalancerConfig, BalancerStore
from augmentum.state.discovery_store import DiscoveryStore
from augmentum.state.notes_store import NotesStore


async def _make_notes_db() -> tuple[aiosqlite.Connection, NotesStore]:
    """Create in-memory DB with browse_notes table mirroring the live
    migration shape (including user_id for tenant scoping)."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("""
        CREATE TABLE browse_notes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            source_url TEXT DEFAULT '',
            source_title TEXT DEFAULT '',
            format TEXT DEFAULT 'note',
            word_count INTEGER DEFAULT 0,
            reading_time_min INTEGER DEFAULT 0,
            ai_blocks TEXT DEFAULT '[]',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT '',
            user_id TEXT NOT NULL DEFAULT ''
        )
    """)
    await conn.commit()
    return conn, NotesStore(conn)


async def _make_balancer_db() -> tuple[aiosqlite.Connection, BalancerStore]:
    """Create in-memory DB with load_balancers and related tables."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("""
        CREATE TABLE load_balancers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            strategy TEXT DEFAULT 'round_robin',
            fallback_enabled INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await conn.execute("""
        CREATE TABLE load_balancer_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balancer_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            backend_key TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            priority INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            last_used_at TEXT
        )
    """)
    await conn.execute("""
        CREATE TABLE ab_test_votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balancer_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            backend_key TEXT NOT NULL,
            vote TEXT NOT NULL,
            session_id TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await conn.commit()
    return conn, BalancerStore(conn)


async def _make_discovery_db() -> tuple[aiosqlite.Connection, DiscoveryStore]:
    """Create in-memory DB with discovery tables."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("""
        CREATE TABLE interaction_signals (
            id TEXT PRIMARY KEY,
            signal_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_title TEXT DEFAULT '',
            source_domain TEXT DEFAULT '',
            content_type TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            weight REAL DEFAULT 1.0,
            cluster_id TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await conn.execute("""
        CREATE TABLE browse_history (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT DEFAULT '',
            domain TEXT DEFAULT '',
            content_type TEXT DEFAULT '',
            thumbnail TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            cluster_id TEXT,
            visit_count INTEGER DEFAULT 0,
            first_visited TEXT DEFAULT '',
            last_visited TEXT DEFAULT ''
        )
    """)
    await conn.execute("""
        CREATE TABLE content_library (
            chunk_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            source_title TEXT DEFAULT '',
            source_type TEXT DEFAULT '',
            content TEXT NOT NULL,
            embedding BLOB,
            cluster_id TEXT,
            retrieved_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await conn.commit()
    return conn, DiscoveryStore(conn)


class TestNotesStore:
    """CRUD for browse notes — every method requires ``user_id``."""

    UID = "u_alice"

    async def test_create_and_get(self):
        conn, store = await _make_notes_db()
        note = {"id": "n1", "title": "Test Note", "content": "Hello", "tags": ["a", "b"]}
        await store.create(note, user_id=self.UID)
        fetched = await store.get("n1", user_id=self.UID)
        assert fetched is not None
        assert fetched["title"] == "Test Note"
        assert fetched["tags"] == ["a", "b"]
        await conn.close()

    async def test_get_nonexistent(self):
        conn, store = await _make_notes_db()
        assert await store.get("nope", user_id=self.UID) is None
        await conn.close()

    async def test_update(self):
        conn, store = await _make_notes_db()
        await store.create(
            {"id": "n1", "title": "Old", "content": "old text"},
            user_id=self.UID,
        )
        updated = await store.update(
            "n1", {"title": "New", "content": "new text"}, user_id=self.UID,
        )
        assert updated is not None
        assert updated["title"] == "New"
        assert updated["content"] == "new text"
        await conn.close()

    async def test_update_nonexistent(self):
        conn, store = await _make_notes_db()
        assert await store.update("nope", {"title": "x"}, user_id=self.UID) is None
        await conn.close()

    async def test_delete(self):
        conn, store = await _make_notes_db()
        await store.create({"id": "n1", "title": "Del Me"}, user_id=self.UID)
        assert await store.delete("n1", user_id=self.UID) is True
        assert await store.get("n1", user_id=self.UID) is None
        await conn.close()

    async def test_delete_nonexistent(self):
        conn, store = await _make_notes_db()
        assert await store.delete("nope", user_id=self.UID) is False
        await conn.close()

    async def test_list_stubs(self):
        conn, store = await _make_notes_db()
        await store.create({"id": "n1", "title": "A"}, user_id=self.UID)
        await store.create({"id": "n2", "title": "B"}, user_id=self.UID)
        stubs = await store.list_stubs(user_id=self.UID)
        assert len(stubs) == 2
        await conn.close()

    async def test_get_rejects_empty_user_id(self):
        """Defense-in-depth: empty user_id raises rather than silently
        bypassing scoping (regression-locks the audit-driven hardening)."""
        conn, store = await _make_notes_db()
        try:
            await store.get("anything", user_id="")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "user_id" in str(exc).lower()
        finally:
            await conn.close()

    async def test_list_stubs_rejects_empty_user_id(self):
        conn, store = await _make_notes_db()
        try:
            await store.list_stubs(user_id="")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "user_id" in str(exc).lower()
        finally:
            await conn.close()

    async def test_cross_user_isolation(self):
        """The original audit finding: scoping must be enforced — alice
        can't read bob's notes."""
        conn, store = await _make_notes_db()
        await store.create({"id": "n_a", "title": "alice's"}, user_id="u_alice")
        await store.create({"id": "n_b", "title": "bob's"},   user_id="u_bob")
        # Alice sees only her own
        alice_stubs = await store.list_stubs(user_id="u_alice")
        assert [s["id"] for s in alice_stubs] == ["n_a"]
        # Alice cannot fetch bob's note even with the right id
        assert await store.get("n_b", user_id="u_alice") is None
        # Alice cannot update or delete bob's note
        assert await store.update("n_b", {"title": "pwned"}, user_id="u_alice") is None
        assert await store.delete("n_b", user_id="u_alice") is False
        # Bob's note is intact
        assert (await store.get("n_b", user_id="u_bob"))["title"] == "bob's"
        await conn.close()


class TestBalancerStore:
    """CRUD for load balancers."""

    async def test_create_and_get_balancer(self):
        conn, store = await _make_balancer_db()
        cfg = BalancerConfig(id="b1", name="Test Balancer")
        result = await store.create_balancer(cfg)
        assert result.id == "b1"
        assert result.name == "Test Balancer"
        await conn.close()

    async def test_list_balancers(self):
        conn, store = await _make_balancer_db()
        await store.create_balancer(BalancerConfig(id="b1", name="B1"))
        await store.create_balancer(BalancerConfig(id="b2", name="B2"))
        balancers = await store.list_balancers()
        assert len(balancers) == 2
        await conn.close()

    async def test_update_balancer(self):
        conn, store = await _make_balancer_db()
        await store.create_balancer(BalancerConfig(id="b1", name="Old"))
        updated = await store.update_balancer("b1", name="New")
        assert updated is not None
        assert updated.name == "New"
        await conn.close()

    async def test_delete_balancer(self):
        conn, store = await _make_balancer_db()
        await store.create_balancer(BalancerConfig(id="b1", name="Del"))
        assert await store.delete_balancer("b1") is True
        assert await store.get_balancer("b1") is None
        await conn.close()

    async def test_add_and_list_members(self):
        conn, store = await _make_balancer_db()
        await store.create_balancer(BalancerConfig(id="b1", name="B1"))
        member = await store.add_member("b1", "llama3:8b", "ollama", weight=2.0)
        assert member.model_name == "llama3:8b"
        members = await store.list_members("b1")
        assert len(members) == 1
        assert members[0].weight == 2.0
        await conn.close()

    async def test_remove_member(self):
        conn, store = await _make_balancer_db()
        await store.create_balancer(BalancerConfig(id="b1", name="B1"))
        member = await store.add_member("b1", "m1", "ollama")
        assert await store.remove_member(member.id) is True
        assert len(await store.list_members("b1")) == 0
        await conn.close()

    async def test_record_and_get_votes(self):
        conn, store = await _make_balancer_db()
        await store.create_balancer(BalancerConfig(id="b1", name="B1"))
        await store.record_vote("b1", "m1", "ollama", "up")
        await store.record_vote("b1", "m1", "ollama", "up")
        await store.record_vote("b1", "m1", "ollama", "down")
        stats = await store.get_vote_stats("b1")
        assert len(stats) == 1
        assert stats[0].up == 2
        assert stats[0].down == 1
        await conn.close()


class TestDiscoveryStore:
    """CRUD for discovery engine state."""

    async def test_log_signal(self):
        conn, store = await _make_discovery_db()
        sig = await store.log_signal(
            signal_type="click",
            source_url="https://example.com",
            source_title="Example",
            content_type="article",
            weight=1.0,
            metadata={"tag": "test"},
        )
        assert sig["signal_type"] == "click"
        assert sig["deduplicated"] is False
        await conn.close()

    async def test_list_signals(self):
        conn, store = await _make_discovery_db()
        await store.log_signal(
            signal_type="click", source_url="https://a.com",
            source_title="A", content_type="article",
            weight=1.0, metadata={},
        )
        signals = await store.list_signals()
        assert len(signals) == 1
        await conn.close()

    async def test_upsert_history_new(self):
        conn, store = await _make_discovery_db()
        entry = await store.upsert_history(
            url="https://example.com", title="Example",
            domain="example.com", content_type="article",
            thumbnail="", metadata={},
        )
        assert entry["visit_count"] == 1
        await conn.close()

    async def test_upsert_history_revisit(self):
        conn, store = await _make_discovery_db()
        await store.upsert_history(
            url="https://example.com", title="Example",
            domain="example.com", content_type="article",
            thumbnail="", metadata={},
        )
        entry = await store.upsert_history(
            url="https://example.com", title="Example Updated",
            domain="example.com", content_type="article",
            thumbnail="", metadata={"extra": True},
        )
        assert entry["visit_count"] == 2
        await conn.close()

    async def test_delete_history(self):
        conn, store = await _make_discovery_db()
        entry = await store.upsert_history(
            url="https://del.com", title="Delete Me",
            domain="del.com", content_type="article",
            thumbnail="", metadata={},
        )
        assert await store.delete_history(entry["id"]) is True
        await conn.close()

    async def test_check_visited(self):
        conn, store = await _make_discovery_db()
        await store.upsert_history(
            url="https://visited.com", title="V",
            domain="visited.com", content_type="article",
            thumbnail="", metadata={},
        )
        result = await store.check_visited(["https://visited.com", "https://not.com"])
        assert "https://visited.com" in result
        assert "https://not.com" not in result
        await conn.close()

    async def test_store_and_get_chunk(self):
        conn, store = await _make_discovery_db()
        chunk = await store.store_chunk(
            source_url="https://src.com", source_title="Src",
            source_type="article", content="Some text content",
            embedding=None, cluster_id=None,
        )
        fetched = await store.get_chunk(chunk["chunk_id"])
        assert fetched is not None
        assert fetched["content"] == "Some text content"
        await conn.close()
