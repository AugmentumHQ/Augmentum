"""Tests for the Observation Substrate L0 + seeder + exporter.

Covers the Phase A pipeline end-to-end against in-memory aiosqlite,
with the llama-lookup-create subprocess stubbed out so the test
doesn't depend on the bundled binary being present on the dev box.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from augmentum.observation.fingerprint import (
    fingerprint_prefix,
    normalize_prefix,
)
from augmentum.observation.store import ObservationStore


# ---------------------------------------------------------------------------
# Schema bootstrap — apply the real migration so drift is caught here.
# ---------------------------------------------------------------------------


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "augmentum" / "state" / "migrations"
    / "234_observation_substrate_l0.sql"
)


_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);
"""

_USERS_TABLE_MIN = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL DEFAULT ''
);
"""

_UI_SESSIONS_TABLE_MIN = """
CREATE TABLE IF NOT EXISTS ui_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'passthrough',
    data TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT
);
"""


async def _mkdb() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_VERSION_TABLE)
    await conn.executescript(_USERS_TABLE_MIN)
    await conn.executescript(_UI_SESSIONS_TABLE_MIN)
    await conn.executescript(_MIGRATION_PATH.read_text(encoding="utf-8"))
    await conn.execute(
        "INSERT INTO users (id, username) VALUES (?, ?)",
        ("u_test", "tester"),
    )
    await conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_normalize_lowercases_and_collapses(self):
        assert normalize_prefix("  Hello\tWORLD  ") == "hello world"

    def test_normalize_handles_empty(self):
        assert normalize_prefix("") == ""
        assert normalize_prefix("   ") == ""

    def test_normalize_truncates_long_prefix(self):
        result = normalize_prefix("x " * 200)
        assert len(result) <= 200

    def test_fingerprint_deterministic(self):
        assert fingerprint_prefix("hello world") == fingerprint_prefix("hello world")

    def test_fingerprint_case_insensitive(self):
        # normalize_prefix lowercases, so the fingerprints match.
        assert fingerprint_prefix("Hello World") == fingerprint_prefix("hello world")

    def test_fingerprint_surface_changes_result(self):
        a = fingerprint_prefix("hello world", surface="chat")
        b = fingerprint_prefix("hello world", surface="notes")
        assert a != b

    def test_fingerprint_mode_changes_result(self):
        a = fingerprint_prefix("x", mode="passthrough")
        b = fingerprint_prefix("x", mode="narrative")
        assert a != b

    def test_fingerprint_is_hex_sha1(self):
        fp = fingerprint_prefix("anything")
        assert len(fp) == 40
        # All hex characters.
        int(fp, 16)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestObservationStore:
    async def test_observe_round_trips(self):
        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(
            user_id="u_test",
            prefix_text="the cat sat on",
            continuation="the mat watching",
        )
        await conn.commit()

        rows = await store.top_k(user_id="u_test")
        assert len(rows) == 1
        assert rows[0].prefix_text == "the cat sat on"
        assert rows[0].continuation == "the mat watching"
        assert rows[0].observation_count == 1
        await conn.close()

    async def test_observe_increments_count_on_dup(self):
        """Same (prefix, continuation, surface, mode) → upsert, not duplicate."""
        conn = await _mkdb()
        store = ObservationStore(conn)
        for _ in range(3):
            await store.observe(
                user_id="u_test",
                prefix_text="hello world",
                continuation="how are you",
            )
        await conn.commit()
        rows = await store.top_k(user_id="u_test")
        assert len(rows) == 1
        assert rows[0].observation_count == 3
        await conn.close()

    async def test_observe_drops_empty_inputs(self):
        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(user_id="", prefix_text="x", continuation="y")
        await store.observe(user_id="u_test", prefix_text="", continuation="y")
        await store.observe(user_id="u_test", prefix_text="x", continuation="")
        await conn.commit()
        assert await store.count(user_id="u_test") == 0
        await conn.close()

    async def test_top_k_ranks_by_weighted_count(self):
        """observation_count × decay_weight desc."""
        conn = await _mkdb()
        store = ObservationStore(conn)
        # high-count entry
        for _ in range(5):
            await store.observe(
                user_id="u_test", prefix_text="alpha", continuation="beta",
            )
        # low-count entry
        await store.observe(
            user_id="u_test", prefix_text="gamma", continuation="delta",
        )
        await conn.commit()
        rows = await store.top_k(user_id="u_test", k=10)
        assert rows[0].prefix_text == "alpha"
        assert rows[1].prefix_text == "gamma"
        await conn.close()

    async def test_top_k_filters_by_surface(self):
        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(
            user_id="u_test", prefix_text="a", continuation="b",
            surface="chat",
        )
        await store.observe(
            user_id="u_test", prefix_text="c", continuation="d",
            surface="notes",
        )
        await conn.commit()
        chat_only = await store.top_k(user_id="u_test", surface="chat")
        notes_only = await store.top_k(user_id="u_test", surface="notes")
        assert len(chat_only) == 1 and chat_only[0].prefix_text == "a"
        assert len(notes_only) == 1 and notes_only[0].prefix_text == "c"
        await conn.close()

    async def test_purge_clears_user(self):
        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(user_id="u_test", prefix_text="x", continuation="y")
        await conn.commit()
        deleted = await store.purge(user_id="u_test")
        assert deleted == 1
        assert await store.count(user_id="u_test") == 0
        await conn.close()

    async def test_count_user_isolation(self):
        """One user's count doesn't include another user's rows."""
        conn = await _mkdb()
        await conn.execute("INSERT INTO users (id) VALUES (?)", ("u_other",))
        await conn.commit()
        store = ObservationStore(conn)
        await store.observe(user_id="u_test", prefix_text="mine", continuation="x")
        await store.observe(user_id="u_other", prefix_text="theirs", continuation="y")
        await conn.commit()
        assert await store.count(user_id="u_test") == 1
        assert await store.count(user_id="u_other") == 1
        await conn.close()


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSeeder:
    async def test_seeds_from_assistant_messages(self):
        from augmentum.observation.seeder import seed_from_chat_history

        conn = await _mkdb()
        store = ObservationStore(conn)
        # Long-enough assistant message — at least 12 words.
        session_blob = {
            "tree": {
                "n1": {
                    "role": "assistant",
                    "content": (
                        "this is a longer sample response with enough "
                        "words to clear the floor for sliding windows"
                    ),
                },
            },
        }
        await conn.execute(
            "INSERT INTO ui_sessions (id, mode, data, user_id) "
            "VALUES (?, ?, ?, ?)",
            ("s1", "chat", json.dumps(session_blob), "u_test"),
        )
        await conn.commit()

        counters = await seed_from_chat_history(
            store, user_id="u_test", conn=conn,
        )
        await conn.commit()

        assert counters["sessions_scanned"] == 1
        assert counters["messages_processed"] == 1
        assert counters["windows_written"] > 0
        assert await store.count(user_id="u_test") > 0
        await conn.close()

    async def test_seeder_skips_short_messages(self):
        from augmentum.observation.seeder import seed_from_chat_history

        conn = await _mkdb()
        store = ObservationStore(conn)
        await conn.execute(
            "INSERT INTO ui_sessions (id, mode, data, user_id) "
            "VALUES (?, ?, ?, ?)",
            ("s1", "chat",
             json.dumps({"tree": {"n1": {"role": "assistant", "content": "ok"}}}),
             "u_test"),
        )
        await conn.commit()

        counters = await seed_from_chat_history(
            store, user_id="u_test", conn=conn,
        )
        assert counters["messages_processed"] == 0
        assert counters["windows_written"] == 0
        await conn.close()

    async def test_seeder_skips_user_messages(self):
        """User-typed text isn't useful for the decoding cache."""
        from augmentum.observation.seeder import seed_from_chat_history

        conn = await _mkdb()
        store = ObservationStore(conn)
        await conn.execute(
            "INSERT INTO ui_sessions (id, mode, data, user_id) "
            "VALUES (?, ?, ?, ?)",
            ("s1", "chat",
             json.dumps({"tree": {"n1": {
                 "role": "user",
                 "content": "this is plenty long enough but a user message and should be skipped",
             }}}),
             "u_test"),
        )
        await conn.commit()
        counters = await seed_from_chat_history(
            store, user_id="u_test", conn=conn,
        )
        assert counters["messages_processed"] == 0
        await conn.close()

    async def test_seeder_idempotent(self):
        """Re-seeding bumps counts rather than duplicating rows."""
        from augmentum.observation.seeder import seed_from_chat_history

        conn = await _mkdb()
        store = ObservationStore(conn)
        blob = {"tree": {"n1": {
            "role": "assistant",
            "content": (
                "alpha beta gamma delta epsilon zeta eta theta iota "
                "kappa lambda mu nu xi omicron pi rho sigma tau"
            ),
        }}}
        await conn.execute(
            "INSERT INTO ui_sessions (id, mode, data, user_id) "
            "VALUES (?, ?, ?, ?)",
            ("s1", "chat", json.dumps(blob), "u_test"),
        )
        await conn.commit()

        await seed_from_chat_history(store, user_id="u_test", conn=conn)
        await conn.commit()
        count_first = await store.count(user_id="u_test")

        await seed_from_chat_history(store, user_id="u_test", conn=conn)
        await conn.commit()
        count_second = await store.count(user_id="u_test")

        assert count_first == count_second  # same rows, bumped counts
        # And the counts should have grown.
        rows = await store.top_k(user_id="u_test", k=1)
        assert rows[0].observation_count >= 2
        await conn.close()


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExporter:
    async def test_export_refuses_empty_store(self, tmp_path):
        from augmentum.observation.exporter import export_lookup_cache

        conn = await _mkdb()
        store = ObservationStore(conn)
        # Need a model file too so the FileNotFoundError check passes
        # and we hit the empty-store branch.
        model_path = tmp_path / "fake-model.gguf"
        model_path.write_bytes(b"not a real model")

        with pytest.raises(RuntimeError, match="observation store empty"):
            await export_lookup_cache(
                store,
                user_id="u_test",
                model_path=str(model_path),
                cache_root=tmp_path / "cache",
            )
        await conn.close()

    async def test_export_refuses_missing_model(self, tmp_path):
        from augmentum.observation.exporter import export_lookup_cache

        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(
            user_id="u_test", prefix_text="a b c d e f g h",
            continuation="i j k l",
        )
        await conn.commit()

        with pytest.raises(FileNotFoundError):
            await export_lookup_cache(
                store,
                user_id="u_test",
                model_path="/does/not/exist.gguf",
                cache_root=tmp_path / "cache",
            )
        await conn.close()

    async def test_export_invokes_subprocess_and_renames_atomically(self, tmp_path):
        """Full happy path with the subprocess stubbed.

        The stub creates a tiny file at the requested output path,
        mimicking what the real binary would produce. Lets us verify
        the atomic-rename + return-shape contract without needing the
        real binary on the dev box.
        """
        from augmentum.observation import exporter as exporter_mod

        conn = await _mkdb()
        store = ObservationStore(conn)
        for i in range(5):
            await store.observe(
                user_id="u_test",
                prefix_text=f"prefix number {i} alpha beta gamma",
                continuation=f"delta epsilon zeta {i}",
            )
        await conn.commit()

        model_path = tmp_path / "fake-model.gguf"
        model_path.write_bytes(b"fake")

        async def _fake_run(binary, *, model_path, corpus_path, output_path):
            # Verify the corpus file actually got written before we
            # "build" the cache from it.
            assert Path(corpus_path).exists()
            assert Path(corpus_path).stat().st_size > 0
            # Mimic llama-lookup-create's output — a binary file at the
            # requested path. Real format is opaque to us; tests only
            # check that something landed and got atomically renamed.
            Path(output_path).write_bytes(b"\x00fakecache\x00")

        with patch.object(
            exporter_mod, "_run_llama_lookup_create", side_effect=_fake_run,
        ):
            result = await exporter_mod.export_lookup_cache(
                store,
                user_id="u_test",
                model_path=str(model_path),
                cache_root=tmp_path / "cache",
                llama_lookup_create_bin="/usr/bin/true",  # arbitrary
            )

        assert result.observations_used == 5
        assert result.cache_path.exists()
        assert result.cache_bytes > 0
        # Atomic-rename leaves no .partial file behind.
        partial = result.cache_path.with_suffix(".bin.partial")
        assert not partial.exists()
        # Corpus file is cleaned up after success.
        assert not result.corpus_path.exists()
        await conn.close()

    async def test_cache_path_for_is_deterministic(self, tmp_path):
        from augmentum.observation.exporter import cache_path_for

        a = cache_path_for("u1", "/models/foo.gguf", cache_root=tmp_path)
        b = cache_path_for("u1", "/models/foo.gguf", cache_root=tmp_path)
        assert a == b

    async def test_cache_path_for_separates_users(self, tmp_path):
        from augmentum.observation.exporter import cache_path_for

        a = cache_path_for("u1", "/models/foo.gguf", cache_root=tmp_path)
        b = cache_path_for("u2", "/models/foo.gguf", cache_root=tmp_path)
        assert a != b
        assert "u1" in str(a) and "u2" in str(b)

    async def test_cache_path_for_separates_models(self, tmp_path):
        from augmentum.observation.exporter import cache_path_for

        a = cache_path_for("u1", "/models/foo.gguf", cache_root=tmp_path)
        b = cache_path_for("u1", "/models/bar.gguf", cache_root=tmp_path)
        assert a != b

    async def test_cache_path_for_sanitizes_user_id(self, tmp_path):
        from augmentum.observation.exporter import cache_path_for

        # Path-traversal-ish user id must not escape the cache_root.
        bad = cache_path_for(
            "../escape", "/models/foo.gguf", cache_root=tmp_path,
        )
        # The sanitized component should be a child of tmp_path/...
        assert str(tmp_path) in str(bad)
        assert ".." not in bad.parent.name


# ---------------------------------------------------------------------------
# Autocomplete (Phase D — chat composer ghost-text consumer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestComplete:
    async def test_complete_returns_top_match_on_exact_tail(self):
        """When the user's tail fingerprint matches a stored prefix,
        the continuation comes back ranked by count."""
        conn = await _mkdb()
        store = ObservationStore(conn)
        # Two continuations for the same 3-word tail; high-count wins.
        for _ in range(5):
            await store.observe(
                user_id="u_test", prefix_text="i want to",
                continuation="ship",
            )
        await store.observe(
            user_id="u_test", prefix_text="i want to",
            continuation="rest",
        )
        await conn.commit()
        matched, hits = await store.complete(
            user_id="u_test", current_text="i want to",
        )
        assert matched == "i want to"
        assert hits[0] == ("ship", 5)
        await conn.close()

    async def test_complete_falls_back_through_tail_lengths(self):
        """No 8-word match but a 3-word match → returns the short one."""
        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(
            user_id="u_test", prefix_text="the cat sat",
            continuation="quietly",
        )
        await conn.commit()
        # User has typed a longer prefix; the 8-word tail won't match
        # but the 3-word tail "the cat sat" will.
        matched, hits = await store.complete(
            user_id="u_test",
            current_text="this morning the cat sat",
        )
        assert matched == "the cat sat"
        assert hits[0][0] == "quietly"
        await conn.close()

    async def test_complete_empty_store_returns_empty(self):
        conn = await _mkdb()
        store = ObservationStore(conn)
        matched, hits = await store.complete(
            user_id="u_test", current_text="anything",
        )
        assert matched == ""
        assert hits == []
        await conn.close()

    async def test_complete_user_isolation(self):
        """User A's observations never surface for user B."""
        conn = await _mkdb()
        await conn.execute("INSERT INTO users (id) VALUES (?)", ("u_other",))
        await conn.commit()
        store = ObservationStore(conn)
        await store.observe(
            user_id="u_test", prefix_text="hello world",
            continuation="friend",
        )
        await conn.commit()
        matched, hits = await store.complete(
            user_id="u_other", current_text="hello world",
        )
        assert matched == ""
        assert hits == []
        await conn.close()

    async def test_complete_skips_immediate_dup(self):
        """The "the the" guard: if the continuation's first word
        equals the user's last typed word, drop it.

        Uses a 3-word seeded prefix to match complete()'s smallest
        tail length (8, 5, 3).
        """
        conn = await _mkdb()
        store = ObservationStore(conn)
        # User just typed "want to to". Top continuation "to fix"
        # would produce "want to to to fix" — filter it.
        await store.observe(
            user_id="u_test", prefix_text="want to to",
            continuation="to fix",
        )
        await store.observe(
            user_id="u_test", prefix_text="want to to",
            continuation="address",
        )
        await conn.commit()
        matched, hits = await store.complete(
            user_id="u_test",
            current_text="i want to to",
        )
        # "to fix" filtered (would dup 'to'); "address" survives.
        assert matched == "want to to"
        assert hits == [("address", 1)]
        await conn.close()

    async def test_complete_empty_inputs(self):
        conn = await _mkdb()
        store = ObservationStore(conn)
        m1, h1 = await store.complete(user_id="", current_text="hi")
        m2, h2 = await store.complete(user_id="u_test", current_text="")
        assert m1 == "" and h1 == []
        assert m2 == "" and h2 == []
        await conn.close()

    async def test_complete_surface_mode_scoping(self):
        """Same prefix_text but different surface → different fingerprint
        → complete() in surface A does not surface surface B's hits.

        Uses a 3-word prefix because complete()'s default tail_lengths
        floor is 3 (matches the seeder's smallest emitted window).
        """
        conn = await _mkdb()
        store = ObservationStore(conn)
        await store.observe(
            user_id="u_test", prefix_text="hello dear world",
            continuation="friend", surface="chat",
        )
        await store.observe(
            user_id="u_test", prefix_text="hello dear world",
            continuation="reader", surface="notes",
        )
        await conn.commit()
        _, chat_hits = await store.complete(
            user_id="u_test",
            current_text="hello dear world", surface="chat",
        )
        _, notes_hits = await store.complete(
            user_id="u_test",
            current_text="hello dear world", surface="notes",
        )
        assert chat_hits[0][0] == "friend"
        assert notes_hits[0][0] == "reader"
        await conn.close()
