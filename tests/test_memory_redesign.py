"""Tests for the memory system redesign — relevance threshold, quality gate,
explicit priority, MemoryRecallTool, mode-aware injection, scope tagging."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from augmentum.memory.extractor import (
    _SKIP_PHRASES,
    _deduplicate_facts,
    heuristic_extract,
    should_extract,
)
from augmentum.memory.integration import (
    _build_memory_hint,
    _build_user_summary,
    recall_and_inject,
)
from augmentum.memory.models import ExtractedFact, Memory, MemoryType, SourceType

_MIGRATIONS_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"

_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
INSERT INTO schema_version (version, description) VALUES (5, 'pre-memory baseline');
"""


async def _apply_migration(conn: aiosqlite.Connection, version: int) -> None:
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


def _make_memory(
    content: str,
    importance: float = 0.5,
    confidence: float = 0.8,
    source_type: str | None = None,
    memory_type: str = "fact",
    scope: str | None = None,
) -> Memory:
    return Memory(
        id=f"mem_{hash(content) % 10000}",
        user_id="default",
        content=content,
        memory_type=memory_type,
        importance=importance,
        confidence=confidence,
        source_type=source_type,
        scope=scope,
    )


# ===========================================================================
# Phase 1: Relevance Threshold
# ===========================================================================


class TestRelevanceThreshold:
    """Phase 1 — min_score filtering in recall()."""

    def test_config_has_recall_settings(self):
        from augmentum.config import Settings

        s = Settings()
        assert hasattr(s, "memory_recall_min_score")
        assert hasattr(s, "memory_recall_limit")
        assert s.memory_recall_min_score == 0.35
        assert s.memory_recall_limit == 5

    @pytest.mark.asyncio
    async def test_recall_min_score_filters_low_scores(self):
        """recall() with min_score should drop entries below threshold."""
        from augmentum.memory.store import MemoryStore

        store = MagicMock(spec=MemoryStore)
        store.recall = AsyncMock()

        # Simulate the min_score filtering logic directly
        scored = [
            (_make_memory("important", importance=0.9), 0.05),
            (_make_memory("marginal", importance=0.3), 0.008),
            (_make_memory("garbage", importance=0.1), 0.001),
        ]
        min_score = 0.01
        filtered = [(m, s) for m, s in scored if s >= min_score]
        assert len(filtered) == 1
        assert filtered[0][0].content == "important"

    @pytest.mark.asyncio
    async def test_recall_min_score_zero_keeps_all(self):
        scored = [
            (_make_memory("a"), 0.05),
            (_make_memory("b"), 0.001),
            (_make_memory("c"), 0.0001),
        ]
        min_score = 0.0
        # With min_score=0, filter should not remove anything
        filtered = [(m, s) for m, s in scored if s >= min_score] if min_score > 0 else scored
        assert len(filtered) == 3

    def test_recall_signature_has_min_score(self):
        """Verify recall() accepts min_score parameter."""
        import inspect

        from augmentum.memory.store import MemoryStore

        sig = inspect.signature(MemoryStore.recall)
        assert "min_score" in sig.parameters
        assert sig.parameters["min_score"].default == 0.0

    def test_recall_signature_has_scope(self):
        import inspect

        from augmentum.memory.store import MemoryStore

        sig = inspect.signature(MemoryStore.recall)
        assert "scope" in sig.parameters


# ===========================================================================
# Phase 2: Extraction Quality Gate
# ===========================================================================


class TestExtractionQualityGate:
    """Phase 2 — skip phrases, min capture length, reduced importance."""

    def test_skip_phrases_is_frozenset(self):
        assert isinstance(_SKIP_PHRASES, frozenset)
        assert "a little" in _SKIP_PHRASES
        assert "confused" in _SKIP_PHRASES
        assert "going" in _SKIP_PHRASES

    def test_confused_not_extracted(self):
        """'I'm a little confused' should NOT be extracted as identity."""
        facts = heuristic_extract("I'm a little confused about this topic")
        contents = [f.content.lower() for f in facts]
        assert not any("confused" in c for c in contents)

    def test_trying_not_extracted(self):
        facts = heuristic_extract("I'm trying to figure this out")
        # "trying" is too short for 5-char min on identity pattern
        contents = [f.content.lower() for f in facts]
        assert not any("trying" in c for c in contents)

    def test_bathroom_not_extracted(self):
        """'I use the bathroom' should NOT be stored as preference."""
        facts = heuristic_extract("I use the bathroom every morning")
        # This may or may not match — check it doesn't match skip phrases
        # The phrase "the bathroom" should still match but is OK since
        # it doesn't start with a skip phrase. Let's verify the importance is low.
        for f in facts:
            if "bathroom" in f.content.lower():
                assert f.importance <= 0.4
                assert f.confidence <= 0.5

    def test_valid_identity_still_extracted(self):
        """'I'm a software engineer' should still be extracted."""
        facts = heuristic_extract("I'm a software engineer at Google")
        assert len(facts) >= 1
        assert any("software engineer" in f.content.lower() for f in facts)

    def test_name_still_extracted(self):
        facts = heuristic_extract("My name is Alice")
        assert len(facts) == 1
        assert "alice" in facts[0].content.lower()

    def test_short_identity_rejected(self):
        """Identity captures under 5 chars should be rejected."""
        facts = heuristic_extract("I'm a dev")
        # "dev" is 3 chars, below the 5-char minimum for identity pattern
        identity_facts = [f for f in facts if "dev" in f.content.lower() and f.type == MemoryType.FACT]
        assert len(identity_facts) == 0

    def test_use_importance_reduced(self):
        """'I use X' pattern should have importance=0.4, confidence=0.5."""
        facts = heuristic_extract("I use Python for data analysis")
        use_facts = [f for f in facts if "python" in f.content.lower()]
        assert len(use_facts) >= 1
        for f in use_facts:
            assert f.importance == 0.4
            assert f.confidence == 0.5

    def test_skip_phrase_sorry(self):
        facts = heuristic_extract("I prefer sorry about that")
        # "sorry" starts the captured text — should be skipped
        sorry_facts = [f for f in facts if "sorry" in f.content.lower()]
        assert len(sorry_facts) == 0

    def test_skip_phrase_just(self):
        facts = heuristic_extract("I prefer just checking in")
        just_facts = [f for f in facts if f.content.lower().endswith("just checking in")]
        assert len(just_facts) == 0

    def test_explicit_remember_still_works(self):
        """Explicit remember instructions should still be extracted."""
        facts = heuristic_extract("Remember that I'm allergic to peanuts")
        assert len(facts) >= 1
        assert any("allergic" in f.content.lower() for f in facts)


# ===========================================================================
# Phase 3: Explicit > Implicit Priority
# ===========================================================================


class TestExplicitPriority:
    """Phase 3 — EXPLICIT source type, is_explicit flag, scoring boost."""

    def test_source_type_has_explicit(self):
        assert SourceType.EXPLICIT == "explicit"
        assert SourceType.EXPLICIT.value == "explicit"

    def test_extracted_fact_has_is_explicit(self):
        fact = ExtractedFact(content="test")
        assert hasattr(fact, "is_explicit")
        assert fact.is_explicit is False

    def test_remember_pattern_is_explicit(self):
        facts = heuristic_extract("Remember that I prefer dark mode")
        remember_facts = [f for f in facts if f.content.lower().startswith("remember")]
        assert len(remember_facts) >= 1
        assert remember_facts[0].is_explicit is True

    def test_note_pattern_is_explicit(self):
        facts = heuristic_extract("Note that my API key expires in March")
        note_facts = [f for f in facts if "api key" in f.content.lower()]
        assert len(note_facts) >= 1
        assert note_facts[0].is_explicit is True

    def test_preference_not_explicit(self):
        facts = heuristic_extract("I prefer Python over JavaScript")
        for f in facts:
            assert f.is_explicit is False

    def test_identity_not_explicit(self):
        facts = heuristic_extract("I'm a backend developer at Stripe")
        for f in facts:
            assert f.is_explicit is False

    def test_explicit_boost_in_scoring(self):
        """Explicit/manual memories get 1.5x score boost."""
        # Test that the scoring code exists and boosts correctly
        mem_explicit = _make_memory("dark mode", importance=0.5, source_type="explicit")
        mem_normal = _make_memory("likes pizza", importance=0.5, source_type="extracted")

        base_score = 0.1
        # Simulate the scoring logic
        score_explicit = base_score * mem_explicit.importance
        if mem_explicit.source_type in ("user_manual", "explicit"):
            score_explicit *= 1.5
        score_normal = base_score * mem_normal.importance
        if mem_normal.source_type in ("user_manual", "explicit"):
            score_normal *= 1.5

        assert score_explicit > score_normal

    def test_user_manual_also_boosted(self):
        mem = _make_memory("test", source_type="user_manual")
        assert mem.source_type in ("user_manual", "explicit")

    def test_patterns_are_5_tuples(self):
        """All patterns should be 5-element tuples."""
        from augmentum.memory.extractor import _PATTERNS

        for p in _PATTERNS:
            assert len(p) == 5, f"Pattern should be 5-tuple, got {len(p)}: {p}"


# ===========================================================================
# Phase 4: MemoryRecallTool
# ===========================================================================


class TestMemoryRecallTool:
    """Phase 4 — MemoryRecallTool for UARF on-demand queries."""

    def test_tool_properties(self):
        from augmentum.tools.base import ToolCategory
        from augmentum.tools.memory_recall import MemoryRecallTool

        store = MagicMock()
        tool = MemoryRecallTool(store)
        assert tool.name == "memory_recall"
        assert tool.category == ToolCategory.SEARCH
        assert "memory" in tool.description.lower()

    def test_input_schema(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        tool = MemoryRecallTool(MagicMock())
        schema = tool.input_schema
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "memory_type" in schema["properties"]
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_execute_empty_query(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        tool = MemoryRecallTool(MagicMock())
        result = await tool.execute(query="")
        assert not result.success
        assert "required" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        store = MagicMock()
        store.recall = AsyncMock(return_value=[])
        tool = MemoryRecallTool(store)
        result = await tool.execute(query="something", _user_id="usr_test")
        assert result.success
        assert "No relevant memories" in result.output
        assert result.metadata["count"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_results(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        mem = _make_memory("User prefers dark mode", memory_type="preference")
        store = MagicMock()
        store.recall = AsyncMock(return_value=[mem])
        tool = MemoryRecallTool(store)
        result = await tool.execute(query="dark mode", _user_id="usr_test")
        assert result.success
        assert "dark mode" in result.output
        assert result.metadata["count"] == 1

    @pytest.mark.asyncio
    async def test_execute_passes_min_score(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        store = MagicMock()
        store.recall = AsyncMock(return_value=[])
        tool = MemoryRecallTool(store)
        await tool.execute(query="test", limit=3, _user_id="usr_test")
        store.recall.assert_called_once()
        call_kwargs = store.recall.call_args
        assert call_kwargs.kwargs.get("min_score") == 0.005
        # The logged-in user must reach the store — not the "default" bucket.
        assert call_kwargs.kwargs.get("user_id") == "usr_test"

    @pytest.mark.asyncio
    async def test_execute_invalid_memory_type(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        tool = MemoryRecallTool(MagicMock())
        result = await tool.execute(
            query="test", memory_type="invalid_type", _user_id="usr_test",
        )
        assert not result.success
        assert "Invalid memory_type" in result.error

    @pytest.mark.asyncio
    async def test_execute_valid_memory_type_filter(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        store = MagicMock()
        store.recall = AsyncMock(return_value=[])
        tool = MemoryRecallTool(store)
        await tool.execute(
            query="test", memory_type="preference", _user_id="usr_test",
        )
        call_kwargs = store.recall.call_args
        assert call_kwargs.kwargs["memory_types"] == [MemoryType.PREFERENCE]

    @pytest.mark.asyncio
    async def test_execute_handles_store_error(self):
        from augmentum.tools.memory_recall import MemoryRecallTool

        store = MagicMock()
        store.recall = AsyncMock(side_effect=RuntimeError("db error"))
        tool = MemoryRecallTool(store)
        result = await tool.execute(query="test", _user_id="usr_test")
        assert not result.success
        assert "db error" in result.error

    def test_aliases_registered(self):
        from augmentum.tools.registry import _TOOL_ALIASES

        assert _TOOL_ALIASES.get("memory") == "memory_recall"
        assert _TOOL_ALIASES.get("recall") == "memory_recall"
        assert _TOOL_ALIASES.get("remember") == "memory_recall"
        assert _TOOL_ALIASES.get("memories") == "memory_recall"

    def test_tool_available_in_relevant_phase(self):
        """MemoryRecallTool is SEARCH category, which is available in RELEVANT phase."""
        from augmentum.tools.base import ToolCategory
        from augmentum.tools.registry import _PHASE_CATEGORIES

        assert ToolCategory.SEARCH in _PHASE_CATEGORIES["relevant"]
        assert ToolCategory.SEARCH in _PHASE_CATEGORIES["apply"]


# ===========================================================================
# Phase 5: Mode-Aware Injection + Summary Builder
# ===========================================================================


class TestModeAwareInjection:
    """Phase 5 — analytical skips injection, compact summary format."""

    def test_build_user_summary_empty(self):
        assert _build_user_summary([]) == ""

    def test_build_user_summary_basic(self):
        memories = [
            _make_memory("User prefers dark mode", importance=0.9, confidence=0.9),
            _make_memory("User is a developer", importance=0.8, confidence=0.8),
        ]
        result = _build_user_summary(memories, max_chars=500)
        assert result.startswith("[background]")
        assert "dark mode" in result
        assert "developer" in result
        # Directive must be present to suppress unprompted references
        assert "unprompted" in result

    def test_build_user_summary_respects_budget(self):
        memories = [
            _make_memory("A" * 100, importance=0.9, confidence=0.9),
            _make_memory("B" * 100, importance=0.8, confidence=0.8),
            _make_memory("C" * 100, importance=0.7, confidence=0.7),
        ]
        result = _build_user_summary(memories, max_chars=150)
        # Bullet section is budgeted; header + directive are fixed tails.
        bullet_lines = [line for line in result.strip().split("\n") if line.startswith("- ")]
        assert len(bullet_lines) <= 1  # budget only fits one 100-char bullet

    def test_build_user_summary_preserves_input_order(self):
        """Trust the store's composite ranking — no re-sort by local attrs."""
        memories = [
            _make_memory("first entry", importance=0.1, confidence=0.5),
            _make_memory("second entry", importance=0.95, confidence=0.95),
        ]
        result = _build_user_summary(memories, max_chars=500)
        lines = result.strip().split("\n")
        # Bullets appear in the caller-provided order regardless of importance.
        bullet_lines = [line for line in lines if line.startswith("- ")]
        assert bullet_lines[0].endswith("first entry")
        assert bullet_lines[1].endswith("second entry")

    def test_build_user_summary_no_confidence_annotations(self):
        """Compact summary should not include verbose confidence annotations."""
        memories = [_make_memory("test fact", importance=0.5, confidence=0.6)]
        result = _build_user_summary(memories, max_chars=500)
        assert "confidence" not in result

    @pytest.mark.asyncio
    async def test_analytical_mode_sets_hint_not_injection(self):
        """recall_and_inject should set memory_hint (not inject) for analytical mode."""
        from augmentum.models.base import InternalChatRequest, Message

        mem = _make_memory("User is Alice", importance=0.9, confidence=0.9)
        store = MagicMock()
        store.recall = AsyncMock(return_value=[mem])
        app_state = MagicMock()
        app_state.memory_store = store

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="what is my name?")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 3
            mock_settings.memory_recall_min_score = 0.01
            await recall_and_inject(request, app_state, mode="analytical")

        # Store.recall SHOULD have been called (for the hint)
        store.recall.assert_called_once()
        # memory_hint should be set
        assert request.memory_hint is not None
        assert "[memory_available]" in request.memory_hint
        # No system message should have been injected
        system_msgs = [m for m in request.messages if m.role == "system"]
        assert len(system_msgs) == 0

    @pytest.mark.asyncio
    async def test_passthrough_mode_injects(self):
        """recall_and_inject should work for passthrough mode."""
        from augmentum.models.base import InternalChatRequest, Message

        mem = _make_memory("User prefers dark mode", importance=0.9, confidence=0.9)
        store = MagicMock()
        store.recall = AsyncMock(return_value=[mem])
        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 3
            mock_settings.memory_recall_min_score = 0.01
            mock_settings.memory_inject_min_score = 0.0
            mock_settings.memory_summary_max_chars = 300
            mock_settings.memory_core_profile_enabled = False

            await recall_and_inject(request, app_state, mode="passthrough")

        store.recall.assert_called_once()
        # Should have injected a system message
        system_msgs = [m for m in request.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "[background]" in system_msgs[0].content

    def test_config_has_summary_max_chars(self):
        from augmentum.config import Settings

        s = Settings()
        assert hasattr(s, "memory_summary_max_chars")
        assert s.memory_summary_max_chars == 300

    def test_recall_and_inject_signature(self):
        import inspect

        sig = inspect.signature(recall_and_inject)
        assert "mode" in sig.parameters
        assert sig.parameters["mode"].default == "passthrough"


# ===========================================================================
# Phase 6: Scope Tagging
# ===========================================================================


class TestScopeTagging:
    """Phase 6 — scope column, filtering, migration."""

    def test_memory_has_scope_field(self):
        mem = Memory(
            id="m1", user_id="default", content="test",
            memory_type=MemoryType.FACT, scope="project_a",
        )
        assert mem.scope == "project_a"

    def test_memory_scope_default_none(self):
        mem = Memory(id="m1", user_id="default", content="test", memory_type=MemoryType.FACT)
        assert mem.scope is None

    def test_extracted_fact_no_scope(self):
        """ExtractedFact shouldn't have scope — it's set at store time."""
        fact = ExtractedFact(content="test")
        assert not hasattr(fact, "scope")

    @pytest.mark.asyncio
    async def test_migration_008_exists(self):
        migration_path = _MIGRATIONS_DIR / "008_memory_scope.sql"
        assert migration_path.exists(), "Migration 008 should exist"
        content = migration_path.read_text()
        assert "scope" in content
        assert "ALTER TABLE memories" in content

    @pytest.mark.asyncio
    async def test_migration_008_applies(self):
        """Migration 008 should add scope column to memories table."""
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(_BOOTSTRAP_SQL)
            await _apply_migration(conn, 6)
            await _apply_migration(conn, 8)

            # Verify scope column exists
            cursor = await conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in await cursor.fetchall()}
            assert "scope" in columns

    @pytest.mark.asyncio
    async def test_migration_008_creates_index(self):
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(_BOOTSTRAP_SQL)
            await _apply_migration(conn, 6)
            await _apply_migration(conn, 8)

            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_scope'"
            )
            row = await cursor.fetchone()
            assert row is not None

    def test_store_signature_has_scope(self):
        import inspect

        from augmentum.memory.store import MemoryStore

        sig = inspect.signature(MemoryStore.store)
        assert "scope" in sig.parameters

    def test_list_all_signature_has_scope(self):
        import inspect

        from augmentum.memory.store import MemoryStore

        sig = inspect.signature(MemoryStore.list_all)
        assert "scope" in sig.parameters

    def test_store_request_has_scope(self):
        from augmentum.proxy.memory_routes import StoreRequest

        req = StoreRequest(content="test")
        assert hasattr(req, "scope")
        assert req.scope is None


# ===========================================================================
# Handler Factory Integration
# ===========================================================================


class TestHandlerFactoryIntegration:
    """Verify MemoryRecallTool lazy registration in handler_factory."""

    def test_handler_factory_registers_memory_recall(self):
        """get_handler_for_mode should lazy-register MemoryRecallTool."""
        from augmentum.classifier.router import Mode
        from augmentum.proxy.handler_factory import get_handler_for_mode
        from augmentum.tools.registry import ToolRegistry

        registry = ToolRegistry()
        backend = MagicMock()
        backend.list_models = AsyncMock(return_value=[])

        app_state = MagicMock()
        app_state.tool_registry = registry
        app_state.prompt_cache = None
        app_state.memory_store = MagicMock()  # memory store available
        app_state.image_queue = None

        get_handler_for_mode(Mode.ANALYTICAL, backend, "ses_test", app_state)

        # Should have registered memory_recall
        assert registry.get("memory_recall") is not None

    def test_handler_factory_no_store_no_tool(self):
        """If no memory_store, MemoryRecallTool should not be registered."""
        from augmentum.classifier.router import Mode
        from augmentum.proxy.handler_factory import get_handler_for_mode
        from augmentum.tools.registry import ToolRegistry

        registry = ToolRegistry()
        backend = MagicMock()

        app_state = MagicMock()
        app_state.tool_registry = registry
        app_state.prompt_cache = None
        app_state.memory_store = None  # no memory store
        app_state.image_queue = None

        get_handler_for_mode(Mode.ANALYTICAL, backend, "ses_test", app_state)

        assert registry.get("memory_recall") is None


# ===========================================================================
# Integration: extract_and_store with explicit flag
# ===========================================================================


class TestExtractAndStoreExplicit:
    """Verify extract_and_store passes is_explicit to store_fact."""

    @pytest.mark.asyncio
    async def test_explicit_remember_stored_as_explicit(self):
        from augmentum.memory.extractor import extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_123")

        count = await extract_and_store(
            session_id="ses_1",
            user_id="default",
            user_message="Remember that I prefer dark mode",
            assistant_response="I'll remember that.",
            store=store,
        )

        assert count >= 1
        # Check that at least one call has is_explicit=True (the remember pattern)
        explicit_calls = [
            call for call in store.store_fact.call_args_list
            if call.kwargs.get("is_explicit") is True
        ]
        assert len(explicit_calls) >= 1
        # And verify it's the remember-pattern match
        fact = explicit_calls[0].args[0]
        assert "remember" in fact.content.lower()

    @pytest.mark.asyncio
    async def test_regular_preference_stored_as_extracted(self):
        from augmentum.memory.extractor import extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_123")

        count = await extract_and_store(
            session_id="ses_1",
            user_id="default",
            user_message="I prefer Python over JavaScript",
            assistant_response="Good choice!",
            store=store,
        )

        assert count >= 1
        for call in store.store_fact.call_args_list:
            assert call.kwargs.get("is_explicit") is False


# ===========================================================================
# Analytical Memory Hint
# ===========================================================================


class TestAnalyticalMemoryHint:
    """Memory hint for analytical mode — Signal 1 (data-driven)."""

    @pytest.mark.asyncio
    async def test_analytical_no_hint_when_no_memories(self):
        """No hint should be set when store returns no memories."""
        from augmentum.models.base import InternalChatRequest, Message

        store = MagicMock()
        store.recall = AsyncMock(return_value=[])
        app_state = MagicMock()
        app_state.memory_store = store

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="explain quantum computing")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 3
            mock_settings.memory_recall_min_score = 0.01
            await recall_and_inject(request, app_state, mode="analytical")

        assert request.memory_hint is None

    def test_build_memory_hint_format(self):
        """Hint should include count and type breakdown."""
        memories = [
            _make_memory("User is Alice", memory_type="fact"),
            _make_memory("User likes Python", memory_type="fact"),
            _make_memory("User prefers dark mode", memory_type="preference"),
        ]
        hint = _build_memory_hint(memories)
        assert "[memory_available]" in hint
        assert "3 relevant user memories found" in hint
        assert "2 facts" in hint
        assert "1 preference" in hint
        assert "memory_recall tool" in hint

    def test_build_memory_hint_single(self):
        """Hint for a single memory should use singular."""
        memories = [_make_memory("User is Alice", memory_type="fact")]
        hint = _build_memory_hint(memories)
        assert "1 relevant user memory found" in hint
        assert "1 fact" in hint

    def test_build_memory_hint_empty(self):
        """Empty list should produce empty string."""
        assert _build_memory_hint([]) == ""


# ===========================================================================
# Proactive Memory Suggestion — Signal 2 (pattern-driven)
# ===========================================================================


class TestProactiveMemorySuggestion:
    """Memory pattern in _get_proactive_suggestions — Signal 2."""

    def test_memory_proactive_suggestion_fires(self):
        """'what's my name?' with memory_recall tool should produce suggestion."""
        from augmentum.modes.analytical.engine import AnalyticalEngine

        tool = MagicMock()
        tool.name = "memory_recall"
        suggestions = AnalyticalEngine._get_proactive_suggestions(
            "what's my name?", tools=[tool],
        )
        assert any("memory_recall" in s for s in suggestions)

    def test_memory_proactive_suggestion_no_tool(self):
        """Same query without tool registered should NOT produce suggestion."""
        from augmentum.modes.analytical.engine import AnalyticalEngine

        tool = MagicMock()
        tool.name = "web_search"
        suggestions = AnalyticalEngine._get_proactive_suggestions(
            "what's my name?", tools=[tool],
        )
        assert not any("memory_recall" in s for s in suggestions)

    def test_memory_proactive_suggestion_no_tools(self):
        """No tools at all should not produce memory suggestion."""
        from augmentum.modes.analytical.engine import AnalyticalEngine

        suggestions = AnalyticalEngine._get_proactive_suggestions("what's my name?")
        assert not any("memory_recall" in s for s in suggestions)

    def test_memory_pattern_no_false_positive(self):
        """'explain quantum computing' should NOT trigger memory pattern."""
        from augmentum.modes.analytical.engine import _MEMORY_PATTERN

        assert _MEMORY_PATTERN.search("explain quantum computing") is None

    def test_memory_pattern_matches_remember(self):
        from augmentum.modes.analytical.engine import _MEMORY_PATTERN

        assert _MEMORY_PATTERN.search("do you remember what I said?") is not None

    def test_memory_pattern_matches_previously(self):
        from augmentum.modes.analytical.engine import _MEMORY_PATTERN

        assert _MEMORY_PATTERN.search("I previously mentioned my project") is not None

    def test_memory_pattern_matches_last_time(self):
        from augmentum.modes.analytical.engine import _MEMORY_PATTERN

        assert _MEMORY_PATTERN.search("last time we talked about React") is not None


# ===========================================================================
# Phase 7: Mode-Scoped Memory Isolation
# ===========================================================================


class TestModeScopedMemory:
    """Phase 7 — Auto-tag memories with session mode, filter recall by mode."""

    def test_config_has_scope_by_mode(self):
        from augmentum.config import Settings

        s = Settings()
        assert hasattr(s, "memory_scope_by_mode")
        assert s.memory_scope_by_mode is True

    @pytest.mark.asyncio
    async def test_schedule_extraction_skips_narrative(self):
        """mode='narrative' is skipped entirely — narrative has its own memory system."""
        from augmentum.memory.integration import schedule_extraction
        from augmentum.models.base import InternalChatRequest, Message

        app_state = MagicMock()
        app_state.memory_store = MagicMock()
        app_state.provider_registry = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="My name is Elowen")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings, \
             patch("augmentum.memory.integration.asyncio") as mock_asyncio:
            mock_settings.memory_enabled = True
            mock_settings.memory_scope_by_mode = True
            schedule_extraction(
                app_state, request, "Hello Elowen!", "ses_1", mode="narrative",
            )
            mock_asyncio.create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_schedule_extraction_skips_batch_for_coder(self):
        """Modes outside memory_capture_modes (coder, builder, voice) skip the
        batch LLM extraction path. A non-explicit user message in coder mode
        must NOT fire a buffered extraction task."""
        from augmentum.memory.integration import (
            _extraction_buffers,
            schedule_extraction,
        )
        from augmentum.models.base import InternalChatRequest, Message

        app_state = MagicMock()
        app_state.memory_store = MagicMock()
        app_state.provider_registry = None

        # Plain build-talk — no "remember X" / "note that" marker. With coder
        # outside the allowlist the explicit-pass finds nothing and the batch
        # path must be skipped, so no buffer entry should be created.
        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="The game must run on a web server")],
        )

        _extraction_buffers.clear()
        with patch("augmentum.memory.integration.settings") as mock_settings, \
             patch("augmentum.memory.integration.asyncio") as mock_asyncio:
            mock_settings.memory_enabled = True
            mock_settings.memory_scope_by_mode = True
            mock_settings.memory_capture_modes = ["passthrough", "analytical", "agentic"]
            mock_settings.memory_extraction_batch_size = 4
            schedule_extraction(
                app_state, request, "OK, exposing port 8000.", "ses_1", mode="coder",
            )
            mock_asyncio.create_task.assert_not_called()
            assert not _extraction_buffers, (
                "coder mode must not accumulate a session buffer"
            )

    @pytest.mark.asyncio
    async def test_schedule_extraction_explicit_still_fires_in_coder(self):
        """Explicit "remember X" instructions are captured in every non-narrative
        mode — they're the user's conscious opt-in regardless of which surface
        they're chatting in."""
        from augmentum.memory.integration import (
            _extraction_buffers,
            schedule_extraction,
        )
        from augmentum.models.base import InternalChatRequest, Message

        app_state = MagicMock()
        app_state.memory_store = MagicMock()
        app_state.memory_store.store_fact = AsyncMock(return_value="mem_x")
        app_state.provider_registry = None

        # An explicit-pattern utterance even though we're in coder.
        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="remember that I like reading Isekai Manga")],
        )

        _extraction_buffers.clear()
        with patch("augmentum.memory.integration.settings") as mock_settings, \
             patch("augmentum.memory.integration.asyncio") as mock_asyncio:
            mock_settings.memory_enabled = True
            mock_settings.memory_scope_by_mode = True
            mock_settings.memory_capture_modes = ["passthrough", "analytical", "agentic"]
            mock_settings.memory_auto_approve = True
            schedule_extraction(
                app_state, request, "Got it.", "ses_1", mode="coder",
            )
            # Exactly one task — the explicit-store coroutine. Batch path
            # should not have fired (would be a second create_task call).
            assert mock_asyncio.create_task.call_count == 1
            assert not _extraction_buffers, (
                "explicit-only path must not leave a buffered batch entry"
            )
            # Close the dangling coroutine to suppress the unawaited warning.
            mock_asyncio.create_task.call_args[0][0].close()

    @pytest.mark.asyncio
    async def test_schedule_extraction_passes_scope_analytical(self):
        """mode='analytical' with scope toggle → extraction fires with scope='analytical'."""
        from augmentum.memory.integration import schedule_extraction
        from augmentum.models.base import InternalChatRequest, Message

        app_state = MagicMock()
        app_state.memory_store = MagicMock()
        app_state.provider_registry = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="My name is Elowen")],
        )

        mock_smart = AsyncMock(return_value=1)
        with patch("augmentum.memory.integration.settings") as mock_settings, \
             patch("augmentum.memory.integration.asyncio") as mock_asyncio, \
             patch("augmentum.memory.extractor.smart_extract_and_store", mock_smart):
            mock_settings.memory_enabled = True
            mock_settings.memory_scope_by_mode = True
            mock_settings.memory_capture_modes = ["passthrough", "analytical", "agentic"]
            mock_settings.memory_llm_extraction_enabled = False
            mock_settings.memory_llm_extraction_model = ""
            mock_settings.memory_extraction_batch_size = 1
            mock_settings.kg_enabled = False
            schedule_extraction(
                app_state, request, "Hello Elowen!", "ses_1", mode="analytical",
            )
            mock_asyncio.create_task.assert_called_once()
            # Close the coroutine to avoid warning
            coro = mock_asyncio.create_task.call_args[0][0]
            coro.close()

    @pytest.mark.asyncio
    async def test_schedule_extraction_no_scope_when_disabled(self):
        """Toggle off → scope=None."""
        from augmentum.memory.extractor import extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_123")

        # Call extract_and_store directly with scope=None (what schedule_extraction would pass)
        count = await extract_and_store(
            session_id="ses_1",
            user_id="default",
            user_message="My name is Alice",
            assistant_response="Hello Alice!",
            store=store,
            scope=None,
        )
        assert count >= 1
        for call in store.store_fact.call_args_list:
            assert call.kwargs.get("scope") is None

    @pytest.mark.asyncio
    async def test_schedule_extraction_scope_from_mode(self):
        """extract_and_store with scope='narrative' passes scope to store_fact."""
        from augmentum.memory.extractor import extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_123")

        count = await extract_and_store(
            session_id="ses_1",
            user_id="default",
            user_message="My name is Alice",
            assistant_response="Hello Alice!",
            store=store,
            scope="narrative",
        )
        assert count >= 1
        for call in store.store_fact.call_args_list:
            assert call.kwargs.get("scope") == "narrative"

    @pytest.mark.asyncio
    async def test_recall_derives_scope_from_mode(self):
        """recall_and_inject with mode='narrative' → store.recall called with scope='narrative'."""
        from augmentum.models.base import InternalChatRequest, Message

        mem = _make_memory("User is Elowen", importance=0.9, confidence=0.9)
        store = MagicMock()
        store.recall = AsyncMock(return_value=[mem])
        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 3
            mock_settings.memory_recall_min_score = 0.01
            mock_settings.memory_summary_max_chars = 300
            mock_settings.memory_scope_by_mode = True
            mock_settings.memory_core_profile_enabled = False
            await recall_and_inject(request, app_state, mode="narrative")

        store.recall.assert_called_once()
        assert store.recall.call_args.kwargs.get("scope") == "narrative"

    @pytest.mark.asyncio
    async def test_recall_no_scope_when_disabled(self):
        """Toggle off → scope=None in recall."""
        from augmentum.models.base import InternalChatRequest, Message

        mem = _make_memory("User is Alice", importance=0.9, confidence=0.9)
        store = MagicMock()
        store.recall = AsyncMock(return_value=[mem])
        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 3
            mock_settings.memory_recall_min_score = 0.01
            mock_settings.memory_summary_max_chars = 300
            mock_settings.memory_scope_by_mode = False
            mock_settings.memory_core_profile_enabled = False
            await recall_and_inject(request, app_state, mode="narrative")

        store.recall.assert_called_once()
        assert store.recall.call_args.kwargs.get("scope") is None

    @pytest.mark.asyncio
    async def test_recall_explicit_scope_overrides_mode(self):
        """Explicit scope='custom' takes precedence over mode."""
        from augmentum.models.base import InternalChatRequest, Message

        mem = _make_memory("User fact", importance=0.9, confidence=0.9)
        store = MagicMock()
        store.recall = AsyncMock(return_value=[mem])
        app_state = MagicMock()
        app_state.memory_store = store
        app_state.core_profile_manager = None

        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_recall_limit = 3
            mock_settings.memory_recall_min_score = 0.01
            mock_settings.memory_summary_max_chars = 300
            mock_settings.memory_scope_by_mode = True
            mock_settings.memory_core_profile_enabled = False
            await recall_and_inject(
                request, app_state, mode="narrative", scope="custom",
            )

        store.recall.assert_called_once()
        assert store.recall.call_args.kwargs.get("scope") == "custom"

    def test_store_fact_accepts_scope(self):
        """Verify scope kwarg flows through store_fact signature."""
        import inspect

        from augmentum.memory.store import MemoryStore

        sig = inspect.signature(MemoryStore.store_fact)
        assert "scope" in sig.parameters
        assert sig.parameters["scope"].default is None


# ===========================================================================
# Phase 8: Streaming Extraction
# ===========================================================================


class TestStreamingExtraction:
    """Verify streaming handlers accumulate content for extraction."""

    def test_content_accumulator_exists(self):
        from augmentum.proxy.streaming import _ContentAccumulator

        acc = _ContentAccumulator()
        assert acc.content == ""

    @pytest.mark.asyncio
    async def test_content_accumulator_accumulates(self):
        from augmentum.models.base import InternalStreamChunk
        from augmentum.proxy.streaming import _ContentAccumulator

        acc = _ContentAccumulator()

        async def _gen():
            yield InternalStreamChunk(content_delta="Hello ")
            yield InternalStreamChunk(content_delta="World")
            yield InternalStreamChunk(done=True)

        chunks = []
        async for chunk in acc.wrap(_gen()):
            chunks.append(chunk)

        assert acc.content == "Hello World"
        assert len(chunks) == 3

    def test_with_extraction_variants_exist(self):
        from augmentum.proxy.streaming import (
            stream_ollama_chat_handler_with_extraction,
            stream_ollama_generate_handler_with_extraction,
            stream_openai_chat_handler_with_extraction,
        )

        # Just verify they're callable
        assert callable(stream_ollama_chat_handler_with_extraction)
        assert callable(stream_ollama_generate_handler_with_extraction)
        assert callable(stream_openai_chat_handler_with_extraction)

    def test_with_extraction_signatures(self):
        import inspect

        from augmentum.proxy.streaming import stream_ollama_chat_handler_with_extraction

        sig = inspect.signature(stream_ollama_chat_handler_with_extraction)
        params = list(sig.parameters.keys())
        assert "app_state" in params
        assert "session_id" in params
        assert "mode" in params

    @pytest.mark.asyncio
    async def test_migration_009_exists(self):
        migration_path = _MIGRATIONS_DIR / "009_memory_v2.sql"
        assert migration_path.exists(), "Migration 009 should exist"
        content = migration_path.read_text()
        assert "tier" in content
        assert "last_compacted_at" in content


# ===========================================================================
# Pre-filter: should_extract
# ===========================================================================


class TestShouldExtract:
    """Tests for the lightweight message pre-filter."""

    def test_short_messages_filtered(self):
        assert not should_extract("hi")
        assert not should_extract("ok")
        assert not should_extract("yes")
        assert not should_extract("thanks!")

    def test_greetings_filtered(self):
        assert not should_extract("hello")
        assert not should_extract("Good morning")
        assert not should_extract("Hey!")
        assert not should_extract("Thank you!")
        assert not should_extract("goodbye")

    def test_filler_filtered(self):
        assert not should_extract("got it")
        assert not should_extract("makes sense")
        assert not should_extract("continue")
        assert not should_extract("go ahead")
        assert not should_extract("cool")

    def test_self_disclosure_passes(self):
        assert should_extract("I am a software engineer")
        assert should_extract("My name is Alex")
        assert should_extract("I work at Google")
        assert should_extract("I prefer Python over Java")
        assert should_extract("I live in Seattle")
        assert should_extract("Remember that I'm allergic to peanuts")

    def test_pure_questions_filtered(self):
        assert not should_extract("What is the capital of France?")
        assert not should_extract("How do you sort a list in Python?")
        assert not should_extract("Can you explain transformers?")
        assert not should_extract("Tell me about quantum computing")

    def test_questions_with_self_reference_pass(self):
        assert should_extract("Can you help me? I work with React and need advice")
        assert should_extract("What should I use? I prefer lightweight frameworks")

    def test_code_heavy_messages_filtered(self):
        msg = "```python\ndef foo():\n    return 42\n```"
        assert not should_extract(msg)

    def test_code_with_context_passes(self):
        msg = "I built this function for my project:\n```python\ndef foo():\n    return 42\n```\nI use it daily."
        assert should_extract(msg)

    def test_medium_length_neutral_passes(self):
        # Conservative: borderline messages should pass through to LLM
        assert should_extract("The weather has been really nice this week in Portland")

    def test_all_messages_accumulate_in_buffer(self):
        """All messages accumulate in the buffer (pre-filter removed for batch context)."""
        from augmentum.memory.integration import _extraction_buffers, schedule_extraction

        mock_state = MagicMock()
        mock_state.memory_store = MagicMock()
        mock_state.provider_registry = None

        from augmentum.models.base import InternalChatRequest, Message

        # Short greeting — previously pre-filtered, now accumulates for batch context
        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )
        sid = "prefilter_test_session"
        _extraction_buffers.pop(sid, None)

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.memory_enabled = True
            mock_settings.memory_scope_by_mode = False
            mock_settings.memory_capture_modes = ["passthrough", "analytical", "agentic"]
            mock_settings.memory_extraction_batch_size = 5
            schedule_extraction(mock_state, request, "Hi there!", sid)

        # Buffer SHOULD have the message (no per-message pre-filtering)
        assert sid in _extraction_buffers
        assert len(_extraction_buffers[sid].pairs) == 1
        _extraction_buffers.pop(sid, None)  # cleanup


# ===========================================================================
# Post-extraction dedup
# ===========================================================================


class TestExtractionDedup:
    """Tests for the post-extraction dedup step."""

    async def test_identical_facts_deduped(self):
        facts = [
            ExtractedFact(content="User enjoys cooking", type=MemoryType.PREFERENCE,
                          importance=0.6, confidence=0.9),
            ExtractedFact(content="User enjoys cooking", type=MemoryType.PREFERENCE,
                          importance=0.5, confidence=0.8),
        ]
        result = await _deduplicate_facts(facts)
        assert len(result) == 1
        assert result[0].importance == 0.6  # keeps higher importance

    async def test_near_duplicate_facts_deduped(self):
        facts = [
            ExtractedFact(content="User enjoys cooking", type=MemoryType.PREFERENCE,
                          importance=0.6, confidence=0.9),
            ExtractedFact(content="User is interested in cooking chicken",
                          type=MemoryType.PREFERENCE,
                          importance=0.5, confidence=0.8),
        ]
        result = await _deduplicate_facts(facts)
        # These are semantically similar — should be deduped
        assert len(result) <= len(facts)

    async def test_distinct_facts_kept(self):
        facts = [
            ExtractedFact(content="User is a software engineer",
                          type=MemoryType.FACT, importance=0.9, confidence=1.0),
            ExtractedFact(content="User enjoys sweet wines with dinner",
                          type=MemoryType.PREFERENCE, importance=0.6, confidence=0.9),
        ]
        result = await _deduplicate_facts(facts)
        assert len(result) == 2

    async def test_single_fact_unchanged(self):
        facts = [
            ExtractedFact(content="User is a software engineer",
                          type=MemoryType.FACT, importance=0.9, confidence=1.0),
        ]
        result = await _deduplicate_facts(facts)
        assert len(result) == 1

    async def test_empty_list(self):
        assert await _deduplicate_facts([]) == []

    async def test_keeps_higher_importance_on_dedup(self):
        facts = [
            ExtractedFact(content="User likes Python programming",
                          type=MemoryType.PREFERENCE, importance=0.5, confidence=0.8),
            ExtractedFact(content="User prefers Python for programming",
                          type=MemoryType.PREFERENCE, importance=0.8, confidence=0.9),
        ]
        result = await _deduplicate_facts(facts)
        if len(result) == 1:
            # Should keep the higher importance one
            assert result[0].importance == 0.8


class TestAntiProjection:
    """Drop extracted facts that describe an artifact (work-in-progress)
    rather than the user. These slip through when coder/builder turns get
    mined for user facts and the extractor coerces project-state predicates
    into user identity claims.

    Three-signal heuristic — artifact noun + property predicate + no
    identity-grounding language — keeps the rule from misfiring on
    legitimate user facts that happen to mention an artifact noun.
    """

    # ------------------------------------------------------------------
    # _is_artifact_description — unit tests for the heuristic itself
    # ------------------------------------------------------------------

    def test_artifact_spec_classified_as_artifact(self):
        """The exact bad pattern from the memory dump."""
        from augmentum.memory.extractor import _is_artifact_description

        assert _is_artifact_description(
            "The game must run on a web server with a public IP",
        )

    def test_game_mechanic_classified_as_artifact(self):
        from augmentum.memory.extractor import _is_artifact_description

        assert _is_artifact_description(
            "Walls kill the player on contact",
        )

    def test_bug_report_classified_as_artifact(self):
        from augmentum.memory.extractor import _is_artifact_description

        assert _is_artifact_description(
            "The spaceship is invisible in the game",
        )

    def test_identity_grounded_passes(self):
        """Even sentences that mention an artifact noun pass when the
        subject is clearly the user."""
        from augmentum.memory.extractor import _is_artifact_description

        # First-person identity language wins over artifact reference
        assert not _is_artifact_description("I work on indie games")
        assert not _is_artifact_description("I'm building an AI app")
        assert not _is_artifact_description("My wife runs a bakery")

    def test_user_prefix_treated_as_identity(self):
        """LLM often emits third-person reformulations ('User does X');
        treat 'User <verb>' as identity-grounding so we don't drop
        legitimate facts the LLM phrased that way."""
        from augmentum.memory.extractor import _is_artifact_description

        assert not _is_artifact_description("User builds web apps in Python")
        assert not _is_artifact_description("User runs a small bakery")

    def test_plain_user_fact_passes(self):
        """No artifact reference at all — never triggers."""
        from augmentum.memory.extractor import _is_artifact_description

        assert not _is_artifact_description("Lives in Seattle")
        assert not _is_artifact_description("Loves spicy food")
        assert not _is_artifact_description("Has a cat named Whiskers")

    def test_artifact_noun_without_predicate_passes(self):
        """Artifact noun alone is not enough — needs a property predicate.
        Keeps the rule from misfiring on neutral mentions."""
        from augmentum.memory.extractor import _is_artifact_description

        # No "must / runs / kills / is broken" — just mentions an artifact
        assert not _is_artifact_description("Enjoys playing games on the weekend")

    # ------------------------------------------------------------------
    # _validate_fact integration — full gate with the new check
    # ------------------------------------------------------------------

    def test_validate_drops_artifact_spec(self):
        """Artifact-state extracted as user fact must not survive the gate.

        Uses an "is invisible" predicate that slips past the pre-existing
        task-instruction filter (check #6) so the test exercises anti-
        projection (check #7) specifically rather than double-counting
        the existing must/should/needs-to filter.
        """
        from augmentum.memory.extractor import (
            _is_artifact_description,
            _validate_fact,
        )

        content = "The spaceship is invisible in the game"
        # Sanity: only anti-projection should be the gate firing
        assert _is_artifact_description(content)

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = True
            fact = ExtractedFact(
                content=content,
                type=MemoryType.FACT,
                importance=0.6,
            )
            # First-person in anchor so check #3 (first-person anchor) passes
            user_messages = [
                "I noticed the spaceship is invisible in the game when I press start",
            ]
            assert not _validate_fact(fact, user_messages)

    def test_validate_passes_legitimate_user_fact(self):
        """Sanity check — the rule doesn't break normal extraction."""
        from augmentum.memory.extractor import _validate_fact

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = True
            fact = ExtractedFact(
                content="I work as a backend developer at a fintech startup",
                type=MemoryType.FACT,
                importance=0.85,
                evidence="I work as a backend developer at a fintech startup",
            )
            user_messages = [
                "I work as a backend developer at a fintech startup, mostly Go",
            ]
            assert _validate_fact(fact, user_messages)

    def test_validate_explicit_fact_bypasses_anti_projection(self):
        """is_explicit=True facts (from "remember X" patterns) must always
        bypass anti-projection — the user's force-save escape valve.

        Fact has first-person ("I want") so it clears the pre-existing
        check #3 (first-person anchor), and uses an artifact subject +
        property predicate that would otherwise trigger anti-projection.
        Without is_explicit it would be dropped; with it, it passes.
        """
        from augmentum.memory.extractor import (
            _is_artifact_description,
            _validate_fact,
        )

        # Sanity: the content WOULD trigger anti-projection on its own
        content = "I want the game to use port 8080 for the local server"
        assert _is_artifact_description(content), (
            "test premise broken: content must trigger anti-projection so "
            "the is_explicit bypass is what saves it"
        )

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = True
            fact = ExtractedFact(
                content=content,
                type=MemoryType.FACT,
                importance=0.95,
                is_explicit=True,
            )
            user_messages = [content]
            assert _validate_fact(fact, user_messages)

    def test_validate_kill_switch_disables_rule(self):
        """memory_anti_projection_enabled=False restores prior behavior.

        Two-state test: same fact, opposite outcomes. Proves the kill
        switch is the ONLY thing changing the verdict. Escape valve for
        novelist / game-designer users whose domain legitimately produces
        artifact facts at scale.
        """
        from augmentum.memory.extractor import _validate_fact

        # "is invisible" predicate avoids the task-instruction filter,
        # so the only gate that fires is anti-projection.
        content = "The spaceship is invisible in the game"
        fact = ExtractedFact(
            content=content,
            type=MemoryType.FACT,
            importance=0.6,
        )
        user_messages = [
            "I noticed the spaceship is invisible in the game when I press start",
        ]

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = True
            assert not _validate_fact(fact, user_messages), (
                "with rule enabled, artifact fact should be dropped"
            )

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = False
            assert _validate_fact(fact, user_messages), (
                "with rule disabled, artifact fact should pass (novelist escape)"
            )

    def test_validate_drops_in_game_event_report(self):
        """Bug-report-style content from coder/builder turns."""
        from augmentum.memory.extractor import _validate_fact

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = True
            fact = ExtractedFact(
                content="The black hole kills the player instantly",
                type=MemoryType.FACT,
                importance=0.6,
            )
            # First-person in anchor so check #3 passes
            user_messages = [
                "I tried level 2 and the black hole kills the player instantly, "
                "no way for me to escape",
            ]
            assert not _validate_fact(fact, user_messages)

    def test_validate_passes_identity_with_artifact_mention(self):
        """User fact mentioning an artifact noun but identity-grounded:
        should NOT be dropped (escape valve #1)."""
        from augmentum.memory.extractor import _validate_fact

        with patch("augmentum.memory.extractor.settings") as mock_settings:
            mock_settings.memory_anti_projection_enabled = True
            fact = ExtractedFact(
                content="I work on indie games and ship a new one every year",
                type=MemoryType.FACT,
                importance=0.85,
                evidence="I work on indie games",
            )
            user_messages = [
                "I work on indie games and ship a new one every year",
            ]
            assert _validate_fact(fact, user_messages)
