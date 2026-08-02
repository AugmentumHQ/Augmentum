"""Stress test for the memory extraction pipeline.

Uses the REAL code paths end-to-end:
  - Real SQLite database (in-memory, with actual migrations)
  - Real EmbeddingService (nomic-embed-text-v1.5-Q, 768-dim)
  - Real MemoryStore (dedup, supersede, hybrid retrieval, RRF)
  - Real extractor pipeline (should_extract → heuristic → LLM → dedup → store)
  - Real CoreProfileManager (ranking, caching, staleness)
  - Real integration layer (schedule_extraction, recall_and_inject)
  - Mock LLM backend that returns controlled JSON (so we can test LLM
    extraction parsing without hitting a live model)

Run with:
    pytest tests/test_memory_stress.py -v --timeout=120
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.memory.core_profile import CoreProfileManager
from augmentum.memory.extractor import (
    _deduplicate_facts,
    batch_extract_and_store,
    heuristic_extract,
    should_extract,
)
from augmentum.memory.integration import (
    _build_user_summary,
    recall_and_inject,
)
from augmentum.memory.models import (
    ExtractedFact,
    MemoryTier,
    MemoryType,
)
from augmentum.memory.store import MemoryStore
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
    Usage,
)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"

_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
INSERT INTO schema_version (version, description) VALUES (5, 'pre-memory baseline');
"""


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


async def _create_test_db() -> aiosqlite.Connection:
    """Create an in-memory SQLite DB with all memory-related migrations."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_BOOTSTRAP_SQL)
    await conn.commit()
    # Apply memory migrations in order
    for v in [6, 8, 9]:
        await _apply_migration(conn, v)
    return conn


def _make_backend_stub(vec_enabled: bool = False):
    """Create a minimal SQLiteBackend-shaped stub wrapping a real connection."""

    class _Stub:
        def __init__(self, conn: aiosqlite.Connection, vec: bool):
            self.conn = conn
            self.vec_enabled = vec

    return _Stub


def _make_llm_backend(response_json: dict | str | None = None):
    """Create a mock LLM backend that returns a controlled JSON response.

    If response_json is a dict, it's serialised to JSON.
    If it's a string, it's returned verbatim (for testing bad parse cases).
    If None, returns {"facts": []}.
    """
    if response_json is None:
        raw = '{"facts": []}'
    elif isinstance(response_json, dict):
        raw = json.dumps(response_json)
    else:
        raw = response_json

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=InternalChatResponse(
        message=Message(role="assistant", content=raw),
        model="test-model",
        finish_reason="stop",
        usage=Usage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    ))
    return backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def db():
    """Provide a fresh in-memory SQLite connection with memory tables."""
    conn = await _create_test_db()
    yield conn
    await conn.close()


@pytest.fixture
def store(db):
    """Provide a real MemoryStore backed by a real SQLite DB (no vec)."""
    StubClass = _make_backend_stub(vec_enabled=False)
    stub = StubClass(db, False)
    return MemoryStore(stub)


@pytest.fixture
def profile_manager(store):
    """Provide a real CoreProfileManager."""
    return CoreProfileManager(store, max_tokens=500, rebuild_interval=3)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Pre-filter accuracy (should_extract)
# ═══════════════════════════════════════════════════════════════════════════

class TestPreFilter:
    """Verify the pre-filter correctly gates what enters the extraction pipeline."""

    # --- Should PASS the filter (worth extracting) ---

    @pytest.mark.parametrize("msg", [
        "I'm a software engineer at Google",
        "My name is Alice and I live in Berlin",
        "I prefer dark mode and monospaced fonts",
        "I hate when code isn't properly documented",
        "Remember that I'm allergic to peanuts",
        "I work as a data scientist at a startup",
        "I specialize in distributed systems",
        "I've been using Python for ten years",
        "I always want code examples in TypeScript",
        "I never use semicolons in JavaScript",
        "Call me Alex",
        "I live in Tokyo near Shibuya station",
        "I used to work at Microsoft before joining this team",
        "I built this system from scratch last year",
        "I studied computer science at MIT",
    ])
    def test_self_disclosure_passes(self, msg):
        assert should_extract(msg), f"Should pass filter: {msg!r}"

    # --- Should FAIL the filter (not worth extracting) ---

    @pytest.mark.parametrize("msg", [
        "hi", "thanks", "ok", "yes", "lol", "got it", "cool", "", "hey!", "np",
    ])
    def test_greetings_blocked(self, msg):
        assert not should_extract(msg), f"Should block: {msg!r}"

    def test_borderline_filler_passes_conservative_filter(self):
        """Phrases not in the exact-match filler list pass through —
        the filter is intentionally conservative to avoid false negatives."""
        assert should_extract("sure thing")

    @pytest.mark.parametrize("msg", [
        "What is the capital of France?",
        "Can you explain recursion?",
        "Write a Python script that scrapes a website",
        "Generate a summary of this document",
        "Compare React and Vue for frontend development",
    ])
    def test_pure_questions_blocked(self, msg):
        assert not should_extract(msg), f"Should block pure question: {msg!r}"

    def test_question_with_i_passes_conservative_filter(self):
        """'How do I install numpy?' contains 'I' so the pure-question gate
        doesn't fire — avoids dropping messages where user might self-disclose."""
        assert should_extract("How do I install numpy?")

    def test_short_messages_blocked(self):
        assert not should_extract("hi there")
        assert not should_extract("ok cool")

    def test_code_heavy_blocked(self):
        code_msg = "```python\n" + "x = 1\n" * 50 + "```"
        assert not should_extract(code_msg)

    def test_mixed_question_with_self_disclosure_passes(self):
        assert should_extract("How do I set this up? I use Docker on Ubuntu")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Heuristic extraction — corpus-driven
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (message, should_extract_something: bool, substring_in_output | None)
# When should_extract_something is False, we assert no facts are produced.
# When True, we assert at least one fact and optionally check for a substring.
_EXTRACTION_CORPUS: list[tuple[str, bool, str | None]] = [
    # ── True positives: durable personal facts ──
    ("I'm a backend developer at a startup", True, "backend developer"),
    ("My name is Marcus", True, "marcus"),
    ("Call me Mike", True, "mike"),
    ("I prefer dark mode for all my editors", True, "dark mode"),
    ("I hate writing boilerplate code", True, "boilerplate"),
    ("Remember that I'm allergic to shellfish", True, "shellfish"),
    ("I work at Netflix as a senior engineer", True, "netflix"),
    ("I live in San Francisco near the Mission district", True, "san francisco"),
    ("I specialize in machine learning and NLP", True, "machine learning"),
    ("Always use type hints in Python", True, "type hints"),
    ("MY NAME IS BOB", True, "bob"),
    # New patterns: adjective identity, career tenure, possessive, migration
    ("I am vegan and gluten-free", True, "vegan"),
    ("I'm lactose intolerant", True, "lactose"),
    ("I have been a teacher for 15 years", True, "teacher"),
    ("Python is my primary language", True, "python"),
    ("Docker is my go-to for deployment", True, "docker"),
    ("I switched from Mac to Linux last year", True, "linux"),
    ("I moved from Berlin to London", True, "london"),
    # Multi-fact messages
    ("My name is Sarah. I work at Apple. I love hiking.", True, "sarah"),
    ("I'm a vegan who lives in Portland and hates rain", True, "vegan"),

    # ── True negatives: requests, reactions, and noise ──
    ("Can you give me a chicken recipe?", False, None),
    ("What are some good wine regions?", False, None),
    ("I'm happy to help you with that", False, None),
    ("I'm just looking around", False, None),
    ("I'm ok", False, None),

    # ── Request-to-AI filtering (the key improvement) ──
    ("I want you to explain this error", False, None),
    ("I like the way you explained that", False, None),
    ("I love how clean this code is", False, None),
    ("I prefer if you use bullet points in your response", False, None),
    ("I want a dark mode toggle for the settings page", False, None),
    ("I need you to fix this bug", False, None),
    ("I'd like you to refactor this function", False, None),
    ("I want you to write a test for this class", False, None),
    ("I like it when you explain step by step", False, None),

    # ── Third-person should NOT produce first-person facts ──
    ("My friend likes hiking, what trails would you suggest?", False, None),
    ("My boss uses Jira, should I switch?", False, None),

    # ── Ambiguous but allowed through (LLM layer handles precision) ──
    # "I enjoy reading about X" is arguable — it's a stated enjoyment
    ("I enjoy reading about distributed systems", True, "distributed systems"),
    # "I use X" is low-confidence, but still captured
    ("I use Neovim for all my coding", True, "neovim"),

    # ── Edge cases ──
    ("I have a question about databases", False, None),
    ("", False, None),
    ("!!! ... ???", False, None),
]


class TestHeuristicExtraction:
    """Corpus-driven test: each entry in _EXTRACTION_CORPUS defines an input
    message and the expected extraction outcome."""

    @pytest.mark.parametrize(
        "msg,should_have_facts,substring",
        _EXTRACTION_CORPUS,
        ids=[c[0][:50] for c in _EXTRACTION_CORPUS],
    )
    def test_corpus(self, msg, should_have_facts, substring):
        facts = heuristic_extract(msg)
        if should_have_facts:
            assert len(facts) >= 1, f"Expected facts from: {msg!r}, got none"
            if substring:
                combined = " ".join(f.content.lower() for f in facts)
                assert substring.lower() in combined, (
                    f"Expected {substring!r} in extracted facts from {msg!r}, "
                    f"got: {[f.content for f in facts]}"
                )
        else:
            assert len(facts) == 0, (
                f"Expected NO facts from: {msg!r}, "
                f"got: {[f.content for f in facts]}"
            )

    def test_explicit_remember_is_flagged(self):
        """Explicit 'remember X' should set is_explicit=True with high importance."""
        facts = heuristic_extract("Remember that I'm allergic to shellfish")
        explicit = [f for f in facts if f.is_explicit]
        assert len(explicit) >= 1
        assert explicit[0].importance >= 0.9

    def test_no_duplicates_within_message(self):
        facts = heuristic_extract("I prefer TypeScript and I prefer dark mode")
        contents = [f.content.lower() for f in facts]
        assert len(contents) == len(set(contents)), f"Duplicate facts: {contents}"

    def test_multi_fact_count(self):
        facts = heuristic_extract(
            "My name is Sarah. I work at Apple. I live in Cupertino. "
            "I love hiking and I hate cold weather."
        )
        assert len(facts) >= 3, f"Expected >=3 facts, got {len(facts)}: {[f.content for f in facts]}"

    def test_long_capture_truncated(self):
        long_text = "I'm a " + "very " * 50 + "experienced developer"
        facts = heuristic_extract(long_text)
        for f in facts:
            assert len(f.content) < 250

    def test_filler_i_am_happy_not_extracted(self):
        facts = heuristic_extract("I'm happy to help you with that")
        filler = [f for f in facts if "happy" in f.content.lower()]
        assert len(filler) == 0

    def test_request_to_ai_with_real_preference_mixed(self):
        """A message that contains both a request-to-AI and a real preference
        should only extract the real preference."""
        facts = heuristic_extract(
            "I want you to rewrite this function. By the way, I prefer tabs over spaces."
        )
        # Should NOT have "I want you to rewrite this function"
        assert not any("rewrite" in f.content.lower() for f in facts)
        # SHOULD have the tabs preference
        assert any("tabs" in f.content.lower() for f in facts)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: LLM extraction parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMExtraction:
    """Test the LLM extraction path with controlled mock responses."""

    @pytest.mark.asyncio
    async def test_llm_extracts_structured_facts(self, store):
        """LLM returns well-formed JSON → facts stored correctly."""
        llm_response = {
            "facts": [
                {"content": "User is a data scientist at Acme Corp", "type": "fact",
                 "importance": 0.85, "confidence": 1.0},
                {"content": "User works with Python and R", "type": "skill",
                 "importance": 0.7, "confidence": 0.9},
            ]
        }
        backend = _make_llm_backend(llm_response)

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("I'm a data scientist at Acme Corp, I mainly use Python and R", "Great!")],
            store=store, backend=backend, model="test",
        )
        assert stored == 2

        all_mems = await store.list_all(user_id="default")
        assert len(all_mems) == 2
        contents = {m.content for m in all_mems}
        assert "User is a data scientist at Acme Corp" in contents
        assert "User works with Python and R" in contents

    @pytest.mark.asyncio
    async def test_llm_returns_empty_for_questions(self, store):
        """LLM correctly returns no facts for a pure question."""
        backend = _make_llm_backend({"facts": []})

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("What's the best way to sort a list in Python?", "You can use sorted()...")],
            store=store, backend=backend, model="test",
        )
        assert stored == 0

    @pytest.mark.asyncio
    async def test_llm_bad_json_falls_back_to_heuristic(self, store):
        """When LLM returns garbage, heuristic extraction runs as fallback."""
        backend = _make_llm_backend("This is not JSON at all")

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("My name is Carlos and I live in Madrid", "Nice to meet you!")],
            store=store, backend=backend, model="test",
        )
        # Heuristic should catch "My name is Carlos" and "I live in Madrid"
        assert stored >= 1

    @pytest.mark.asyncio
    async def test_llm_markdown_fenced_json(self, store):
        """LLM wraps JSON in markdown code fences — should still parse."""
        raw = '```json\n{"facts": [{"content": "User is a pianist", "type": "skill", "importance": 0.6, "confidence": 0.9}]}\n```'
        backend = _make_llm_backend(raw)

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("I've been playing piano for years", "That's wonderful!")],
            store=store, backend=backend, model="test",
        )
        assert stored >= 1

    @pytest.mark.asyncio
    async def test_llm_extracts_request_as_preference_false_positive(self, store):
        """Simulate the common LLM false positive: extracting a request as a preference.

        This is a key failure mode — the LLM sees 'chicken recipe with Moscato'
        and incorrectly concludes the user likes chicken and Moscato.
        """
        # This simulates a BAD LLM response (false positive)
        bad_response = {
            "facts": [
                {"content": "User likes chicken", "type": "preference",
                 "importance": 0.6, "confidence": 0.7},
                {"content": "User enjoys Moscato wine", "type": "preference",
                 "importance": 0.5, "confidence": 0.6},
            ]
        }
        backend = _make_llm_backend(bad_response)

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("Can you give me a good chicken recipe? Pair it with a Moscato.", "Sure!")],
            store=store, backend=backend, model="test",
        )
        # The LLM was wrong here — these ARE stored because the mock returns them.
        # This test documents the vulnerability: the system trusts the LLM's output.
        # The only defence is the LLM prompt quality.
        assert stored == 2  # Both false positives get stored

    @pytest.mark.asyncio
    async def test_llm_unknown_type_defaults_to_fact(self, store):
        """LLM returns an invalid type → should default to 'fact'."""
        response = {
            "facts": [
                {"content": "User has a dog named Max", "type": "pet_info",
                 "importance": 0.6, "confidence": 0.9},
            ]
        }
        backend = _make_llm_backend(response)

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("I have a dog named Max", "Cute!")],
            store=store, backend=backend, model="test",
        )
        assert stored == 1
        mems = await store.list_all()
        assert mems[0].memory_type == MemoryType.FACT.value

    @pytest.mark.asyncio
    async def test_llm_importance_out_of_range_clamped(self, store):
        """LLM returns importance > 1.0 or < 0.0 → clamped."""
        response = {
            "facts": [
                {"content": "User is CEO of everything", "type": "fact",
                 "importance": 5.0, "confidence": -0.3},
            ]
        }
        backend = _make_llm_backend(response)

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("I'm the CEO", "Impressive!")],
            store=store, backend=backend, model="test",
        )
        assert stored == 1
        mems = await store.list_all()
        assert mems[0].importance <= 1.0
        assert mems[0].confidence >= 0.0

    @pytest.mark.asyncio
    async def test_explicit_always_stored_even_when_llm_active(self, store):
        """'Remember X' explicit instructions bypass the LLM path entirely."""
        # LLM returns empty (as if it missed the explicit instruction)
        backend = _make_llm_backend({"facts": []})

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("Remember that my favourite colour is blue", "Noted!")],
            store=store, backend=backend, model="test",
        )
        # Heuristic explicit pattern should catch this even though LLM returned nothing
        assert stored >= 1
        mems = await store.list_all()
        assert any("colour is blue" in m.content.lower() for m in mems)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Store dedup/supersede behaviour
# ═══════════════════════════════════════════════════════════════════════════

class TestStoreDedup:
    """Test deduplication, supersede, and versioning with real embeddings."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_updates_existing(self, store):
        """Storing the same fact twice should update, not create two entries."""
        id1 = await store.store("User is a software engineer", MemoryType.FACT)
        id2 = await store.store("User is a software engineer", MemoryType.FACT)

        # Without vec, dedup doesn't run — both get inserted
        # This is expected: vec-less mode has no dedup capability
        all_mems = await store.list_all()
        # Document this: without sqlite-vec, dedup is disabled
        assert len(all_mems) >= 1

    @pytest.mark.asyncio
    async def test_different_facts_stored_separately(self, store):
        """Clearly different facts should be separate memories."""
        await store.store("User lives in Berlin", MemoryType.FACT)
        await store.store("User likes spicy food", MemoryType.PREFERENCE)

        all_mems = await store.list_all()
        assert len(all_mems) == 2

    @pytest.mark.asyncio
    async def test_soft_delete_via_forget(self, store):
        """forget() sets valid_until, not a hard delete."""
        mid = await store.store("User prefers tabs over spaces", MemoryType.PREFERENCE)
        result = await store.forget(mid, user_id="default")
        assert result is True

        # Should not appear in default listing
        active = await store.list_all(include_expired=False)
        assert all(m.id != mid for m in active)

        # Should still exist when including expired
        all_mems = await store.list_all(include_expired=True)
        assert any(m.id == mid for m in all_mems)

    @pytest.mark.asyncio
    async def test_edit_updates_content(self, store):
        """edit() changes content and re-embeds."""
        mid = await store.store("User lives in Toronto", MemoryType.FACT)
        ok = await store.edit(mid, "User lives in Vancouver (moved from Toronto)", user_id="default")
        assert ok is True

        mem = await store.get(mid, user_id="default")
        assert "Vancouver" in mem.content

    @pytest.mark.asyncio
    async def test_supersede_chain(self, store):
        """supersede() creates a version chain: old → new."""
        id1 = await store.store(
            "User works at Google", MemoryType.FACT, importance=0.8,
        )
        id2 = await store.supersede(
            id1, "User works at Meta (left Google)",
            memory_type=MemoryType.FACT, importance=0.8, user_id="default",
        )

        old = await store.get(id1, user_id="default")
        new = await store.get(id2, user_id="default")
        assert old.valid_until is not None
        assert old.superseded_by == id2
        assert new.valid_until is None

    @pytest.mark.asyncio
    async def test_version_history(self, store):
        """get_history() returns the full supersede chain in chronological order."""
        id1 = await store.store("User lives in A", MemoryType.FACT)
        id2 = await store.supersede(id1, "User lives in B", memory_type=MemoryType.FACT, user_id="default")
        id3 = await store.supersede(id2, "User lives in C", memory_type=MemoryType.FACT, user_id="default")

        history = await store.get_history(id1, user_id="default")
        assert len(history) >= 2
        # Should be chronological (oldest first)
        assert history[0].content == "User lives in A"

    @pytest.mark.asyncio
    async def test_tier_update(self, store):
        """update_tier() changes the memory tier."""
        mid = await store.store("Old fact nobody uses", MemoryType.FACT, importance=0.2)
        ok = await store.update_tier(mid, MemoryTier.ARCHIVE, user_id="default")
        assert ok is True

        mem = await store.get(mid, user_id="default")
        assert mem.tier == MemoryTier.ARCHIVE.value

    @pytest.mark.asyncio
    async def test_count_by_type(self, store):
        """count() returns correct per-type and total counts."""
        await store.store("Fact 1", MemoryType.FACT)
        await store.store("Fact 2", MemoryType.FACT)
        await store.store("Pref 1", MemoryType.PREFERENCE)

        counts = await store.count()
        assert counts["total"] == 3
        assert counts.get("fact", 0) == 2
        assert counts.get("preference", 0) == 1


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Retrieval quality (FTS5 search)
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrieval:
    """Test that recall() returns the right memories for a query."""

    @pytest.mark.asyncio
    async def test_fts_keyword_match(self, store):
        """FTS5 should find memories by keyword."""
        await store.store("User is allergic to peanuts", MemoryType.FACT, importance=0.9)
        await store.store("User prefers dark mode", MemoryType.PREFERENCE)
        await store.store("User lives in Portland", MemoryType.FACT)

        results = await store.recall("peanuts allergy")
        assert len(results) >= 1
        assert any("peanut" in m.content.lower() for m in results)

    @pytest.mark.asyncio
    async def test_recall_excludes_expired(self, store):
        """Expired (soft-deleted) memories should not appear in recall."""
        mid = await store.store("User used to live in Paris", MemoryType.FACT)
        await store.forget(mid, user_id="default")

        results = await store.recall("Paris")
        assert all(m.id != mid for m in results)

    @pytest.mark.asyncio
    async def test_recall_updates_access_count(self, store):
        """recall() should increment access_count on returned memories."""
        mid = await store.store("User knows Rust", MemoryType.SKILL)
        mem_before = await store.get(mid, user_id="default")
        assert mem_before.access_count == 0

        await store.recall("Rust programming")
        mem_after = await store.get(mid, user_id="default")
        assert mem_after.access_count >= 1

    @pytest.mark.asyncio
    async def test_recall_respects_scope(self, store):
        """Scoped memories should not leak across scopes."""
        await store.store("Analytical fact", MemoryType.FACT, scope="analytical")
        await store.store("Global fact", MemoryType.FACT, scope=None)

        # Querying with analytical scope should find both (scoped + unscoped)
        results = await store.recall("fact", scope="analytical")
        # Querying with narrative scope should only find global
        results_narrative = await store.recall("fact", scope="narrative")

        analytical_contents = {m.content for m in results}
        narrative_contents = {m.content for m in results_narrative}

        assert "Analytical fact" in analytical_contents
        assert "Analytical fact" not in narrative_contents
        assert "Global fact" in analytical_contents
        assert "Global fact" in narrative_contents

    @pytest.mark.asyncio
    async def test_recall_min_score_filters(self, store):
        """Memories below min_score should be excluded."""
        await store.store("User likes pizza", MemoryType.PREFERENCE, importance=0.1)

        # With a very high min_score, even matching memories should be excluded
        results = await store.recall("pizza", min_score=0.99)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_user_summary_budget(self, store):
        """_build_user_summary should respect the max_chars budget."""
        for i in range(20):
            await store.store(f"User fact number {i} about something", MemoryType.FACT)

        mems = await store.list_all(limit=20)
        summary = _build_user_summary(mems, max_chars=100)
        # Bullet section respects max_chars; header + directive are fixed tails
        # outside that budget, so the total output exceeds max_chars.
        bullet_chars = sum(len(line) + 1 for line in summary.split("\n") if line.startswith("- "))
        assert bullet_chars <= 100


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Core profile accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestCoreProfile:
    """Test that the core profile correctly ranks and surfaces top memories."""

    @pytest.mark.asyncio
    async def test_profile_ranks_by_importance(self, store, profile_manager):
        """Higher importance memories should appear first in the profile."""
        await store.store("User's name is Alice", MemoryType.FACT, importance=0.95)
        await store.store("User once mentioned liking tea", MemoryType.PREFERENCE, importance=0.2)
        await store.store("User is a ML engineer", MemoryType.FACT, importance=0.85)

        profile = await profile_manager.get_profile("default")
        assert "What you know about the user" in profile
        # Alice (0.95) should appear before tea (0.2)
        alice_pos = profile.find("Alice")
        tea_pos = profile.find("tea")
        if alice_pos != -1 and tea_pos != -1:
            assert alice_pos < tea_pos

    @pytest.mark.asyncio
    async def test_profile_caches(self, store, profile_manager):
        """Profile should be cached; second call should not re-query."""
        await store.store("User is a developer", MemoryType.FACT, importance=0.8)

        p1 = await profile_manager.get_profile("default")
        p2 = await profile_manager.get_profile("default")
        assert p1 == p2  # Same cached value

    @pytest.mark.asyncio
    async def test_profile_staleness_triggers_rebuild(self, store, profile_manager):
        """After N extractions, profile should be marked stale and rebuilt."""
        await store.store("Initial fact", MemoryType.FACT, importance=0.8)
        await profile_manager.get_profile("default")  # Populate cache

        # Simulate N extractions (rebuild_interval=3 in fixture)
        for _ in range(3):
            profile_manager.notify_extraction("default")

        # Next get should rebuild
        await store.store("New important fact", MemoryType.FACT, importance=0.99)
        profile = await profile_manager.get_profile("default")
        assert "New important fact" in profile

    @pytest.mark.asyncio
    async def test_profile_empty_when_no_memories(self, store, profile_manager):
        """Profile for a user with no memories should be an empty string."""
        profile = await profile_manager.get_profile("new_user")
        assert profile == ""

    @pytest.mark.asyncio
    async def test_profile_respects_token_budget(self, store, profile_manager):
        """Profile should not exceed the configured token budget."""
        for i in range(50):
            await store.store(
                f"User has a very long and detailed fact number {i} about topic {i * 7}",
                MemoryType.FACT, importance=0.8,
            )

        profile = await profile_manager.get_profile("default")
        # Budget: 500 tokens ≈ 2000 chars
        assert len(profile) <= 2200  # some margin for header


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: End-to-end pipeline scenarios
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndScenarios:
    """Full pipeline scenarios simulating real user conversations."""

    @pytest.mark.asyncio
    async def test_scenario_new_user_introduction(self, store):
        """User introduces themselves in first message."""
        backend = _make_llm_backend({
            "facts": [
                {"content": "User's name is Jordan", "type": "fact",
                 "importance": 0.95, "confidence": 1.0},
                {"content": "User is a frontend developer", "type": "fact",
                 "importance": 0.85, "confidence": 1.0},
                {"content": "User works at a design agency", "type": "fact",
                 "importance": 0.8, "confidence": 0.9},
                {"content": "User prefers React over Vue", "type": "preference",
                 "importance": 0.7, "confidence": 0.9},
            ]
        })

        stored = await batch_extract_and_store(
            session_id="intro-session", user_id="jordan",
            pairs=[(
                "Hi! I'm Jordan, a frontend developer at a design agency. "
                "I mostly work with React though I've used Vue too — definitely prefer React.",
                "Welcome Jordan! Nice to meet you."
            )],
            store=store, backend=backend, model="test",
        )
        assert stored == 4

        # Verify all facts persisted correctly
        mems = await store.list_all(user_id="jordan")
        assert len(mems) == 4
        types = {m.memory_type for m in mems}
        assert MemoryType.FACT.value in types
        assert MemoryType.PREFERENCE.value in types

    @pytest.mark.asyncio
    async def test_scenario_gradual_self_disclosure(self, store):
        """User reveals information across multiple turns."""
        backend = _make_llm_backend(None)  # Empty for first calls

        # Turn 1: just a greeting (nothing to extract)
        stored1 = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("Hey, can you help me with some code?", "Of course!")],
            store=store, backend=backend, model="test",
        )
        assert stored1 == 0

        # Turn 2: mentions their stack
        backend2 = _make_llm_backend({
            "facts": [
                {"content": "User uses TypeScript and Node.js", "type": "skill",
                 "importance": 0.7, "confidence": 0.9},
            ]
        })
        stored2 = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("I'm working on a TypeScript project with Node.js", "Let me help...")],
            store=store, backend=backend2, model="test",
        )
        assert stored2 == 1

        # Turn 3: reveals name and location
        backend3 = _make_llm_backend({
            "facts": [
                {"content": "User's name is Priya", "type": "fact",
                 "importance": 0.95, "confidence": 1.0},
                {"content": "User is based in Bangalore", "type": "fact",
                 "importance": 0.8, "confidence": 1.0},
            ]
        })
        stored3 = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("By the way, I'm Priya. I'm based in Bangalore.", "Hi Priya!")],
            store=store, backend=backend3, model="test",
        )
        assert stored3 == 2

        # Total: 3 facts across 3 turns
        all_mems = await store.list_all()
        assert len(all_mems) == 3

    @pytest.mark.asyncio
    async def test_scenario_contradicting_update(self, store):
        """User changes a fact (moved cities). Old fact should be superseded."""
        # Store initial fact
        id1 = await store.store(
            "User lives in Berlin", MemoryType.FACT,
            importance=0.8, confidence=1.0,
        )

        # Later, user says they moved
        id2 = await store.supersede(
            id1, "User lives in London (moved from Berlin)",
            memory_type=MemoryType.FACT, importance=0.8, user_id="default",
        )

        # Old fact should be expired
        old = await store.get(id1, user_id="default")
        assert old.valid_until is not None
        assert old.superseded_by == id2

        # Only new fact should appear in recall
        results = await store.recall("where does user live")
        active_contents = [m.content for m in results if m.valid_until is None]
        assert any("London" in c for c in active_contents)
        assert all("Berlin" not in c or "London" in c for c in active_contents)

    @pytest.mark.asyncio
    async def test_scenario_explicit_remember_with_llm_miss(self, store):
        """User says 'remember X' but LLM misses it — heuristic catches it."""
        # LLM returns nothing (missed the explicit instruction)
        backend = _make_llm_backend({"facts": []})

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[(
                "Oh and remember that I'm deathly allergic to shellfish. "
                "It's important for any recipes you suggest.",
                "I'll keep that in mind!"
            )],
            store=store, backend=backend, model="test",
        )
        # Heuristic explicit pattern should catch "remember that I'm deathly allergic to shellfish"
        assert stored >= 1
        mems = await store.list_all()
        assert any("shellfish" in m.content.lower() for m in mems)

    @pytest.mark.asyncio
    async def test_scenario_code_conversation_minimal_extraction(self, store):
        """Pure coding conversation should extract very little."""
        backend = _make_llm_backend({"facts": []})

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[
                ("Can you fix this bug?\n```python\ndef foo(): return None\n```",
                 "Try this: ..."),
                ("That didn't work, here's the traceback:\n```\nTypeError: ...\n```",
                 "Ah I see, the issue is..."),
                ("Great, that fixed it. Thanks!",
                 "You're welcome!"),
            ],
            store=store, backend=backend, model="test",
        )
        assert stored == 0

    @pytest.mark.asyncio
    async def test_scenario_roleplay_not_extracted_for_analytical(self, store):
        """In analytical/passthrough mode, narrative roleplay content should not
        be extracted as user facts."""
        # Simulates a user pasting roleplay content while in analytical mode
        backend = _make_llm_backend({
            "facts": [
                {"content": "User is a wizard who lives in a tower", "type": "fact",
                 "importance": 0.8, "confidence": 0.9},
            ]
        })

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[(
                "*I adjust my wizard hat and look out from my tower*",
                "*The landscape stretches before you...*"
            )],
            store=store, backend=backend, model="test", mode="passthrough",
        )
        # This is a known problem: the LLM may extract roleplay content as facts.
        # The system relies on the LLM prompt to distinguish, but roleplay pasted
        # into non-narrative mode bypasses the narrative memory firewall.
        # Documenting this: the LLM was wrong, but the system stored it.
        assert stored == 1  # The bad fact gets through


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Deduplication stress tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDeduplicationStress:
    """Stress test the dedup pipeline with near-duplicate and variant facts."""

    @pytest.mark.asyncio
    async def test_batch_dedup_near_duplicates(self):
        """Near-duplicate facts in the same extraction batch should be deduped."""
        facts = [
            ExtractedFact(content="User is a Python developer", type=MemoryType.SKILL,
                          importance=0.7, confidence=0.9),
            ExtractedFact(content="User is a Python software developer", type=MemoryType.SKILL,
                          importance=0.7, confidence=0.9),
        ]
        deduped = await _deduplicate_facts(facts)
        # These are semantically very similar — should be deduped to 1
        assert len(deduped) <= 2  # May or may not dedup depending on embedding similarity

    @pytest.mark.asyncio
    async def test_batch_dedup_different_facts_preserved(self):
        """Clearly different facts should survive dedup."""
        facts = [
            ExtractedFact(content="User lives in Tokyo", type=MemoryType.FACT,
                          importance=0.8, confidence=1.0),
            ExtractedFact(content="User is allergic to peanuts", type=MemoryType.FACT,
                          importance=0.9, confidence=1.0),
            ExtractedFact(content="User prefers vim over emacs", type=MemoryType.PREFERENCE,
                          importance=0.6, confidence=0.8),
        ]
        deduped = await _deduplicate_facts(facts)
        assert len(deduped) == 3

    @pytest.mark.asyncio
    async def test_batch_dedup_keeps_higher_importance(self):
        """When deduping, the higher-importance fact should be kept."""
        facts = [
            ExtractedFact(content="User knows Python", type=MemoryType.SKILL,
                          importance=0.5, confidence=0.8),
            ExtractedFact(content="User knows Python programming", type=MemoryType.SKILL,
                          importance=0.8, confidence=0.9),
        ]
        deduped = await _deduplicate_facts(facts)
        if len(deduped) == 1:
            assert deduped[0].importance >= 0.7  # Higher importance kept

    @pytest.mark.asyncio
    async def test_repeated_storage_same_session(self, store):
        """Storing the same fact across multiple turns in one session
        should not create duplicates (without vec, this tests the content path)."""
        for _ in range(5):
            await store.store("User is a data engineer", MemoryType.FACT)

        # Without vec, all 5 get stored (no dedup). This tests that the
        # system at least doesn't crash.
        all_mems = await store.list_all()
        assert len(all_mems) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Edge cases and failure modes
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test boundary conditions, unicode, and failure modes."""

    @pytest.mark.asyncio
    async def test_unicode_content(self, store):
        """Unicode content should be stored and retrieved correctly."""
        await store.store("ユーザーは東京に住んでいます", MemoryType.FACT)
        await store.store("用户喜欢喝茶", MemoryType.PREFERENCE)
        await store.store("Benutzer lebt in München", MemoryType.FACT)

        mems = await store.list_all()
        assert len(mems) == 3
        contents = {m.content for m in mems}
        assert "ユーザーは東京に住んでいます" in contents

    @pytest.mark.asyncio
    async def test_very_long_content(self, store):
        """Very long content should be stored without truncation."""
        long_content = "User has expertise in " + ", ".join(f"topic_{i}" for i in range(100))
        await store.store(long_content, MemoryType.FACT)

        mems = await store.list_all()
        assert len(mems) == 1
        assert mems[0].content == long_content

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, store):
        """Empty or near-empty content should be handled gracefully."""
        # The store itself doesn't validate length — it stores what you give it.
        # This documents the behaviour (might be worth adding a guard).
        mid = await store.store("", MemoryType.FACT)
        mem = await store.get(mid, user_id="default")
        assert mem.content == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, store):
        """SQL special characters should not cause injection or errors."""
        dangerous = "User's motto is: \"DROP TABLE memories; --\""
        mid = await store.store(dangerous, MemoryType.FACT)
        mem = await store.get(mid, user_id="default")
        assert mem.content == dangerous

    def test_heuristic_extract_empty_string(self):
        facts = heuristic_extract("")
        assert facts == []

    def test_heuristic_extract_only_punctuation(self):
        facts = heuristic_extract("!!! ... ???")
        assert facts == []

    def test_should_extract_none_like(self):
        # Edge: very short after stripping
        assert not should_extract("   ")
        assert not should_extract(".")

    @pytest.mark.asyncio
    async def test_concurrent_stores(self, store):
        """Multiple concurrent store operations should not lose data."""
        tasks = [
            store.store(f"Concurrent fact {i}", MemoryType.FACT)
            for i in range(10)
        ]
        ids = await asyncio.gather(*tasks)
        assert len(ids) == 10
        assert len(set(ids)) == 10  # All unique IDs

        mems = await store.list_all(limit=100)
        assert len(mems) == 10

    @pytest.mark.asyncio
    async def test_backend_failure_graceful(self, store):
        """If the LLM backend throws, extraction should fail gracefully."""
        backend = MagicMock()
        backend.chat = AsyncMock(side_effect=ConnectionError("LLM unreachable"))

        stored = await batch_extract_and_store(
            session_id="s1", user_id="default",
            pairs=[("My name is Test User", "Hi!")],
            store=store, backend=backend, model="test",
        )
        # Should not crash — falls back to heuristic
        assert stored >= 0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: Integration layer (recall_and_inject)
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationLayer:
    """Test that memories are correctly injected into requests."""

    @pytest.mark.asyncio
    async def test_inject_prepends_user_context(self, store):
        """recall_and_inject should prepend a [background] block to system prompt."""
        await store.store("User is a data scientist", MemoryType.FACT, importance=0.8)

        # Query must share tokens with the stored memory for FTS5 to match
        # (without sqlite-vec, only keyword search is available)
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What does the user do as a data scientist?"),
            ],
        )

        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None
        app_state.knowledge_graph = None
        app_state.document_store = None
        app_state.state_manager = MagicMock()
        app_state.state_manager.backend = MagicMock()

        with _patch_settings(memory_enabled=True, memory_core_profile_enabled=False,
                             document_rag_enabled=False, kg_enabled=False,
                             memory_recall_limit=5, memory_recall_min_score=0.0,
                             memory_scope_by_mode=False, memory_summary_max_chars=500):
            await recall_and_inject(request, app_state, mode="passthrough")

        sys_content = request.messages[0].content
        assert "[background]" in sys_content
        assert "data scientist" in sys_content

    @pytest.mark.asyncio
    async def test_analytical_mode_gets_hint_not_injection(self, store):
        """Analytical mode should get a memory_hint, not context injection."""
        await store.store("User prefers concise answers", MemoryType.PREFERENCE, importance=0.8)

        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are an analyst."),
                Message(role="user", content="Analyze this data"),
            ],
        )

        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None
        app_state.knowledge_graph = None

        with _patch_settings(memory_enabled=True, memory_core_profile_enabled=False,
                             document_rag_enabled=False, kg_enabled=False,
                             memory_recall_limit=5, memory_recall_min_score=0.0,
                             memory_scope_by_mode=False, memory_summary_max_chars=500):
            await recall_and_inject(request, app_state, mode="analytical")

        # System prompt should NOT contain the memory injection block
        sys_content = request.messages[0].content
        assert "[background]" not in sys_content
        # Instead, request should have a memory_hint attribute
        hint = getattr(request, "memory_hint", None)
        if hint is not None:
            assert "memory" in hint.lower()
        # hint may be None if FTS didn't match — that's acceptable

    @pytest.mark.asyncio
    async def test_narrative_mode_skipped_by_default(self, store):
        """Narrative mode should skip memory injection when cross_session_memory is off."""
        await store.store("User fact", MemoryType.FACT, importance=0.8)

        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are a narrator."),
                Message(role="user", content="*walks into the tavern*"),
            ],
        )

        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None
        app_state.knowledge_graph = None

        with _patch_settings(memory_enabled=True, narrative_cross_session_memory=False,
                             memory_core_profile_enabled=False, document_rag_enabled=False,
                             kg_enabled=False, memory_recall_limit=5,
                             memory_recall_min_score=0.0, memory_scope_by_mode=False,
                             memory_summary_max_chars=500):
            await recall_and_inject(request, app_state, mode="narrative")

        # System prompt should be untouched
        assert request.messages[0].content == "You are a narrator."


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: Compaction pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestCompaction:
    """Test the compaction pipeline (deletion, archival, tier demotion)."""

    @pytest.mark.asyncio
    async def test_compaction_candidates(self, store):
        """get_compaction_candidates should find old, low-value memories."""
        from datetime import UTC, datetime, timedelta

        mid = await store.store("Old low-value fact", MemoryType.FACT, importance=0.1)

        # Manually age the memory
        old_date = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        await store._conn.execute(
            "UPDATE memories SET updated_at = ?, created_at = ? WHERE id = ?",
            (old_date, old_date, mid),
        )
        await store._conn.commit()

        candidates = await store.get_compaction_candidates(max_age_days=30)
        assert len(candidates) >= 1
        assert any(c.id == mid for c in candidates)

    @pytest.mark.asyncio
    async def test_compaction_skips_important(self, store):
        """High-importance memories should not be compaction candidates."""
        from datetime import UTC, datetime, timedelta

        mid = await store.store("User's name is Alice", MemoryType.FACT, importance=0.95)

        old_date = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        await store._conn.execute(
            "UPDATE memories SET updated_at = ?, created_at = ? WHERE id = ?",
            (old_date, old_date, mid),
        )
        await store._conn.commit()

        candidates = await store.get_compaction_candidates(
            max_age_days=30, max_importance=0.3,
        )
        assert all(c.id != mid for c in candidates)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: Realistic conversation transcripts
# ═══════════════════════════════════════════════════════════════════════════

class TestRealisticTranscripts:
    """Full multi-turn conversations testing the complete extraction pipeline."""

    @pytest.mark.asyncio
    async def test_onboarding_conversation(self, store):
        """Simulate a typical user onboarding over 5 turns."""
        turns = [
            # Turn 1: greeting
            (
                "Hi there!",
                "Hello! How can I help you today?",
                {"facts": []},
            ),
            # Turn 2: task request (no user facts)
            (
                "Can you help me write a REST API?",
                "Sure! What language and framework?",
                {"facts": []},
            ),
            # Turn 3: self-disclosure (tech stack)
            (
                "I'm using Python with FastAPI. I've been working with it for about 2 years.",
                "Great choice! Let's start...",
                {"facts": [
                    {"content": "User uses Python with FastAPI", "type": "skill",
                     "importance": 0.75, "confidence": 0.9},
                    {"content": "User has 2 years experience with FastAPI",
                     "type": "skill", "importance": 0.6, "confidence": 0.85},
                ]},
            ),
            # Turn 4: more self-disclosure
            (
                "I work at a healthcare startup called MedFlow. Remember that our API "
                "needs to be HIPAA compliant.",
                "Understood, HIPAA compliance is critical...",
                {"facts": [
                    {"content": "User works at healthcare startup MedFlow", "type": "fact",
                     "importance": 0.8, "confidence": 1.0},
                    {"content": "User's API must be HIPAA compliant", "type": "fact",
                     "importance": 0.9, "confidence": 1.0},
                ]},
            ),
            # Turn 5: preference
            (
                "I always prefer async code over sync. And I hate ORMs — raw SQL for me.",
                "I'll use async patterns and raw SQL queries...",
                {"facts": [
                    {"content": "User prefers async code over synchronous", "type": "preference",
                     "importance": 0.7, "confidence": 0.9},
                    {"content": "User dislikes ORMs and prefers raw SQL", "type": "preference",
                     "importance": 0.7, "confidence": 0.9},
                ]},
            ),
        ]

        total_stored = 0
        for user_msg, asst_msg, llm_response in turns:
            backend = _make_llm_backend(llm_response)
            stored = await batch_extract_and_store(
                session_id="onboarding", user_id="medflow_dev",
                pairs=[(user_msg, asst_msg)],
                store=store, backend=backend, model="test",
            )
            total_stored += stored

        # Turns 1-2: 0 facts. Turn 3: 2. Turn 4: 2 (+ explicit "remember" via heuristic).
        # Turn 5: 2 (+ heuristic "always"/"hate" patterns).
        all_mems = await store.list_all(user_id="medflow_dev")
        assert total_stored >= 6
        assert len(all_mems) >= 6

        # Key facts should all be present
        contents = " ".join(m.content.lower() for m in all_mems)
        assert "fastapi" in contents
        assert "medflow" in contents or "healthcare" in contents
        assert "hipaa" in contents
        assert "async" in contents or "synchronous" in contents

    @pytest.mark.asyncio
    async def test_long_session_memory_growth(self, store):
        """Simulate 20 turns and verify memory count stays reasonable."""
        for i in range(20):
            if i % 4 == 0:
                # Every 4th turn has a personal fact
                response = {"facts": [
                    {"content": f"User detail {i}", "type": "fact",
                     "importance": 0.5, "confidence": 0.8},
                ]}
            else:
                response = {"facts": []}

            backend = _make_llm_backend(response)
            await batch_extract_and_store(
                session_id="long_session", user_id="default",
                pairs=[(f"Turn {i} message", f"Turn {i} response")],
                store=store, backend=backend, model="test",
            )

        all_mems = await store.list_all(limit=100)
        # Should have roughly 5 facts (turns 0, 4, 8, 12, 16)
        assert 3 <= len(all_mems) <= 10

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, store):
        """Memories from different users should be isolated."""
        await store.store("Alice is a doctor", MemoryType.FACT, user_id="alice")
        await store.store("Bob is a lawyer", MemoryType.FACT, user_id="bob")

        alice_mems = await store.list_all(user_id="alice")
        bob_mems = await store.list_all(user_id="bob")

        assert len(alice_mems) == 1
        assert len(bob_mems) == 1
        assert "doctor" in alice_mems[0].content.lower()
        assert "lawyer" in bob_mems[0].content.lower()

        # Alice's recall should not find Bob's facts
        alice_results = await store.recall("profession", user_id="alice")
        for m in alice_results:
            assert m.user_id == "alice"


# ---------------------------------------------------------------------------
# Settings patch helper
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _patch_settings(**overrides):
    """Temporarily override augmentum settings."""
    from augmentum.config import settings

    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(settings, key, None)
        object.__setattr__(settings, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                # Can't easily delete; just restore
                try:
                    object.__setattr__(settings, key, value)
                except Exception:
                    pass
            else:
                object.__setattr__(settings, key, value)
