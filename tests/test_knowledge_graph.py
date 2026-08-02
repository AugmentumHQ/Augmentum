"""Tests for the knowledge graph system."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.memory.graph_extractor import (
    _CARD_TYPE_PROMPTS,
    _SYSTEM_PROMPT,
    GraphUpdate,
    _parse_extraction_response,
    apply_graph_updates,
    extract_graph_updates,
)

# ------------------------------------------------------------------
# GraphExtractor — parsing tests
# ------------------------------------------------------------------


class TestParseExtractionResponse:
    """Test JSON response parsing from LLM."""

    def test_valid_node(self):
        raw = json.dumps({
            "updates": [
                {"type": "node", "label": "Alice", "kind": "person", "properties": {"age": 25}},
            ]
        })
        results = _parse_extraction_response(raw)
        assert len(results) == 1
        assert results[0].update_type == "node"
        assert results[0].label == "Alice"
        assert results[0].kind == "person"
        assert results[0].properties == {"age": 25}

    def test_valid_edge(self):
        raw = json.dumps({
            "updates": [
                {"type": "edge", "source": "Alice", "target": "Bob", "relation": "trusts", "weight": 0.8},
            ]
        })
        results = _parse_extraction_response(raw)
        assert len(results) == 1
        assert results[0].update_type == "edge"
        assert results[0].source == "Alice"
        assert results[0].target == "Bob"
        assert results[0].relation == "trusts"
        assert results[0].weight == 0.8

    def test_weight_clamped(self):
        raw = json.dumps({
            "updates": [
                {"type": "edge", "source": "A", "target": "B", "relation": "r", "weight": 1.5},
            ]
        })
        results = _parse_extraction_response(raw)
        assert results[0].weight == 1.0

    def test_negative_weight_clamped(self):
        raw = json.dumps({
            "updates": [
                {"type": "edge", "source": "A", "target": "B", "relation": "r", "weight": -0.5},
            ]
        })
        results = _parse_extraction_response(raw)
        assert results[0].weight == 0.0

    def test_empty_updates(self):
        raw = json.dumps({"updates": []})
        assert _parse_extraction_response(raw) == []

    def test_invalid_json(self):
        assert _parse_extraction_response("not json") == []

    def test_markdown_fenced_json(self):
        raw = "```json\n{\"updates\": [{\"type\": \"node\", \"label\": \"Test\", \"kind\": \"thing\"}]}\n```"
        results = _parse_extraction_response(raw)
        assert len(results) == 1
        assert results[0].label == "Test"

    def test_short_label_skipped(self):
        raw = json.dumps({
            "updates": [
                {"type": "node", "label": "X", "kind": "thing"},
            ]
        })
        assert _parse_extraction_response(raw) == []

    def test_edge_missing_fields_skipped(self):
        raw = json.dumps({
            "updates": [
                {"type": "edge", "source": "A", "target": "", "relation": "r"},
            ]
        })
        assert _parse_extraction_response(raw) == []

    def test_mixed_updates(self):
        raw = json.dumps({
            "updates": [
                {"type": "node", "label": "Alice", "kind": "person"},
                {"type": "edge", "source": "Alice", "target": "Bob", "relation": "knows"},
                {"type": "node", "label": "Bob", "kind": "person"},
            ]
        })
        results = _parse_extraction_response(raw)
        assert len(results) == 3
        assert results[0].update_type == "node"
        assert results[1].update_type == "edge"
        assert results[2].update_type == "node"

    def test_non_dict_items_skipped(self):
        raw = json.dumps({"updates": ["not a dict", 42, None]})
        assert _parse_extraction_response(raw) == []

    def test_updates_not_list(self):
        raw = json.dumps({"updates": "not a list"})
        assert _parse_extraction_response(raw) == []


# ------------------------------------------------------------------
# GraphExtractor — LLM extraction
# ------------------------------------------------------------------


class TestExtractGraphUpdates:
    """Test the LLM-based extraction function."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        backend = AsyncMock()
        backend.chat.return_value = MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "updates": [
                        {"type": "node", "label": "Alice", "kind": "person"},
                    ]
                })
            ),
        )
        # Mock the return to be a dict like InternalChatResponse
        backend.chat.return_value = {
            "message": {"content": json.dumps({
                "updates": [{"type": "node", "label": "Alice", "kind": "person"}]
            })}
        }

        results = await extract_graph_updates(
            user_message="Alice entered the room",
            assistant_response="Alice looked around nervously",
            backend=backend,
        )
        assert len(results) == 1
        assert results[0].label == "Alice"

    @pytest.mark.asyncio
    async def test_extraction_with_context(self):
        backend = AsyncMock()
        backend.chat.return_value = {
            "message": {"content": json.dumps({"updates": []})}
        }

        results = await extract_graph_updates(
            user_message="Hello",
            assistant_response="Hi there",
            backend=backend,
            recent_context=["Earlier message 1", "Earlier message 2"],
        )
        assert results == []
        # Verify backend was called
        backend.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_handles_backend_error(self):
        backend = AsyncMock()
        backend.chat.side_effect = RuntimeError("Backend down")

        results = await extract_graph_updates(
            user_message="Test",
            assistant_response="Response",
            backend=backend,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_extraction_handles_empty_response(self):
        backend = AsyncMock()
        backend.chat.return_value = {"message": {"content": ""}}

        results = await extract_graph_updates(
            user_message="Test",
            assistant_response="Response",
            backend=backend,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_brace_escaping(self):
        """Ensure user messages with braces don't break .format()."""
        backend = AsyncMock()
        backend.chat.return_value = {"message": {"content": json.dumps({"updates": []})}}

        results = await extract_graph_updates(
            user_message="Use {variable} here",
            assistant_response="Result: {output}",
            backend=backend,
        )
        assert results == []
        # Verify it didn't crash


# ------------------------------------------------------------------
# GraphExtractor — card-type-aware prompt selection
# ------------------------------------------------------------------


class TestCardTypeAwareExtraction:
    """Test card-type-aware prompt selection for narrative KG extraction."""

    def test_card_type_prompts_exist(self):
        assert "character" in _CARD_TYPE_PROMPTS
        assert "narrator" in _CARD_TYPE_PROMPTS
        assert "ensemble" in _CARD_TYPE_PROMPTS

    def test_character_prompt_focuses_on_relationships(self):
        prompt = _CARD_TYPE_PROMPTS["character"]
        assert "Relationship edges" in prompt
        assert "Emotional state edges" in prompt
        assert "Shared experiences" in prompt
        assert "interpersonal" in prompt.lower()

    def test_narrator_prompt_focuses_on_world(self):
        prompt = _CARD_TYPE_PROMPTS["narrator"]
        assert "Quest/objective" in prompt
        assert "Faction/allegiance" in prompt
        assert "Location edges" in prompt
        assert "world-building" in prompt.lower()

    def test_ensemble_prompt_focuses_on_group(self):
        prompt = _CARD_TYPE_PROMPTS["ensemble"]
        assert "Inter-character edges" in prompt
        assert "Group role edges" in prompt
        assert "Alliance/conflict" in prompt
        assert "multi-character" in prompt.lower()

    def test_all_prompts_share_base_rules(self):
        """Every prompt includes the base JSON format and weight rules."""
        for card_type, prompt in _CARD_TYPE_PROMPTS.items():
            assert '"updates"' in prompt, f"{card_type} missing updates format"
            assert "weight" in prompt.lower(), f"{card_type} missing weight rule"
            assert "Return ONLY valid JSON" in prompt, f"{card_type} missing JSON rule"

    @pytest.mark.asyncio
    async def test_character_card_type_uses_character_prompt(self):
        backend = AsyncMock()
        backend.chat.return_value = {
            "message": {"content": json.dumps({"updates": []})}
        }

        await extract_graph_updates(
            user_message="She kissed him goodbye",
            assistant_response="He watched her leave with a heavy heart",
            backend=backend,
            card_type="character",
        )

        call_args = backend.chat.call_args[0][0]
        system_msg = call_args.messages[0].content
        assert "interpersonal" in system_msg.lower()
        assert "Relationship edges" in system_msg

    @pytest.mark.asyncio
    async def test_narrator_card_type_uses_narrator_prompt(self):
        backend = AsyncMock()
        backend.chat.return_value = {
            "message": {"content": json.dumps({"updates": []})}
        }

        await extract_graph_updates(
            user_message="The party enters the dungeon",
            assistant_response="The ancient door creaks open",
            backend=backend,
            card_type="narrator",
        )

        call_args = backend.chat.call_args[0][0]
        system_msg = call_args.messages[0].content
        assert "world-building" in system_msg.lower()
        assert "Quest/objective" in system_msg

    @pytest.mark.asyncio
    async def test_ensemble_card_type_uses_ensemble_prompt(self):
        backend = AsyncMock()
        backend.chat.return_value = {
            "message": {"content": json.dumps({"updates": []})}
        }

        await extract_graph_updates(
            user_message="The group debates their next move",
            assistant_response="Aria suggests the mountain pass",
            backend=backend,
            card_type="ensemble",
        )

        call_args = backend.chat.call_args[0][0]
        system_msg = call_args.messages[0].content
        assert "multi-character" in system_msg.lower()
        assert "Inter-character edges" in system_msg

    @pytest.mark.asyncio
    async def test_none_card_type_uses_universal_prompt(self):
        backend = AsyncMock()
        backend.chat.return_value = {
            "message": {"content": json.dumps({"updates": []})}
        }

        await extract_graph_updates(
            user_message="What is Python?",
            assistant_response="Python is a programming language",
            backend=backend,
            card_type=None,
        )

        call_args = backend.chat.call_args[0][0]
        system_msg = call_args.messages[0].content
        assert system_msg == _SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_unknown_card_type_falls_back_to_universal(self):
        backend = AsyncMock()
        backend.chat.return_value = {
            "message": {"content": json.dumps({"updates": []})}
        }

        await extract_graph_updates(
            user_message="Test",
            assistant_response="Response",
            backend=backend,
            card_type="unknown_type",
        )

        call_args = backend.chat.call_args[0][0]
        system_msg = call_args.messages[0].content
        assert system_msg == _SYSTEM_PROMPT


# ------------------------------------------------------------------
# GraphExtractor — apply_graph_updates
# ------------------------------------------------------------------


class TestApplyGraphUpdates:
    """Test applying extracted updates to a knowledge graph."""

    @pytest.mark.asyncio
    async def test_apply_node_creates(self):
        graph = AsyncMock()
        graph.find_node.return_value = None
        node = MagicMock()
        node.id = "n1"
        graph.upsert_node.return_value = node

        updates = [
            GraphUpdate(update_type="node", label="Alice", kind="person"),
        ]

        stats = await apply_graph_updates(updates, graph, chat_id="c1")
        assert stats["nodes_created"] == 1
        graph.upsert_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_node_merges(self):
        graph = AsyncMock()
        existing = MagicMock()
        existing.id = "n1"
        graph.find_node.return_value = existing

        updates = [
            GraphUpdate(update_type="node", label="Alice", kind="person"),
        ]

        stats = await apply_graph_updates(updates, graph, chat_id="c1")
        assert stats["nodes_merged"] == 1
        graph.upsert_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_edge_with_existing_nodes(self):
        graph = AsyncMock()
        # First pass: nodes
        graph.find_node.return_value = None
        node_a = MagicMock(id="n1")
        node_b = MagicMock(id="n2")
        graph.upsert_node.side_effect = [node_a, node_b]

        edge = MagicMock()
        edge.created_at = "2025-01-01"
        edge.updated_at = "2025-01-01"  # Same = new
        graph.upsert_edge.return_value = edge

        updates = [
            GraphUpdate(update_type="node", label="Alice", kind="person"),
            GraphUpdate(update_type="node", label="Bob", kind="person"),
            GraphUpdate(update_type="edge", source="Alice", target="Bob", relation="knows", weight=0.8),
        ]

        stats = await apply_graph_updates(updates, graph, chat_id="c1")
        assert stats["nodes_created"] == 2
        assert stats["edges_created"] == 1

    @pytest.mark.asyncio
    async def test_apply_edge_auto_creates_missing_nodes(self):
        graph = AsyncMock()
        graph.find_node.return_value = None
        auto_node = MagicMock(id="auto1")
        graph.upsert_node.return_value = auto_node

        edge = MagicMock()
        edge.created_at = "2025-01-01"
        edge.updated_at = "2025-01-01"
        graph.upsert_edge.return_value = edge

        updates = [
            GraphUpdate(update_type="edge", source="X", target="Y", relation="depends_on"),
        ]

        stats = await apply_graph_updates(updates, graph, chat_id="c1")
        assert stats["nodes_created"] == 2
        assert stats["edges_created"] == 1

    @pytest.mark.asyncio
    async def test_apply_edge_reinforced(self):
        graph = AsyncMock()
        graph.find_node.return_value = None
        node = MagicMock(id="n1")
        graph.upsert_node.return_value = node

        edge = MagicMock()
        edge.created_at = "2025-01-01T00:00:00"
        edge.updated_at = "2025-01-01T00:00:01"  # Different = reinforced
        graph.upsert_edge.return_value = edge

        updates = [
            GraphUpdate(update_type="edge", source="A", target="B", relation="r"),
        ]

        stats = await apply_graph_updates(updates, graph, chat_id="c1")
        assert stats["edges_reinforced"] == 1

    @pytest.mark.asyncio
    async def test_apply_empty_updates(self):
        graph = AsyncMock()
        stats = await apply_graph_updates([], graph)
        assert stats == {"nodes_created": 0, "nodes_merged": 0, "edges_created": 0, "edges_reinforced": 0}


# ------------------------------------------------------------------
# Integration — schedule_graph_extraction
# ------------------------------------------------------------------


class TestScheduleGraphExtraction:
    """Test that graph extraction is properly scheduled."""

    def test_schedule_called_when_kg_enabled(self):
        """Verify _schedule_graph_extraction creates a task."""
        from augmentum.memory.integration import _schedule_graph_extraction

        with patch("augmentum.memory.integration.asyncio") as mock_asyncio:
            mock_task = MagicMock()
            mock_asyncio.create_task.return_value = mock_task

            _schedule_graph_extraction(
                graph=MagicMock(),
                user_message="Hello",
                assistant_content="Hi",
                session_id="ses_test",
                backend=MagicMock(),
            )

            mock_asyncio.create_task.assert_called_once()
            mock_task.add_done_callback.assert_called_once()


# ------------------------------------------------------------------
# Context builder — graph summary injection
# ------------------------------------------------------------------


class TestGraphContextInjection:
    """Test that graph summaries are injected into narrative context."""

    def test_graph_summary_injected(self):
        from augmentum.modes.narrative.context_builder import ContextBuilder

        builder = ContextBuilder(token_budget=10000)
        result = builder.build(graph_summary="Persons: Alice, Bob\nAlice --trusts(strong)--> Bob")

        assert "known_relationships" in result.blocks_used
        assert "Alice" in result.injected_text

    def test_graph_summary_empty_not_injected(self):
        from augmentum.modes.narrative.context_builder import ContextBuilder

        builder = ContextBuilder(token_budget=10000)
        result = builder.build(graph_summary="")

        assert "known_relationships" not in result.blocks_used


# ------------------------------------------------------------------
# Decay scheduling
# ------------------------------------------------------------------


class TestDecayScheduling:
    """Test periodic edge decay."""

    @pytest.mark.asyncio
    async def test_decay_triggered_at_interval(self):
        from augmentum.memory.integration import _decay_message_counts, _maybe_decay_edges

        graph = AsyncMock()
        graph.decay_edges = AsyncMock(return_value={"decayed": 5, "pruned": 1})

        # Reset counter
        _decay_message_counts.clear()

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.kg_decay_interval = 3
            mock_settings.kg_decay_factor = 0.95
            mock_settings.kg_prune_threshold = 0.1

            # Messages 1 and 2: no decay
            await _maybe_decay_edges(graph, "chat1")
            await _maybe_decay_edges(graph, "chat1")
            graph.decay_edges.assert_not_called()

            # Message 3: decay triggered
            await _maybe_decay_edges(graph, "chat1")
            graph.decay_edges.assert_called_once_with(
                chat_id="chat1", factor=0.95, prune_threshold=0.1,
            )

        _decay_message_counts.clear()

    @pytest.mark.asyncio
    async def test_decay_handles_errors(self):
        from augmentum.memory.integration import _decay_message_counts, _maybe_decay_edges

        graph = AsyncMock()
        graph.decay_edges = AsyncMock(side_effect=RuntimeError("DB error"))

        _decay_message_counts.clear()

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.kg_decay_interval = 1
            mock_settings.kg_decay_factor = 0.95
            mock_settings.kg_prune_threshold = 0.1

            # Should not raise
            await _maybe_decay_edges(graph, "chat1")

        _decay_message_counts.clear()


# ------------------------------------------------------------------
# Auto-promotion
# ------------------------------------------------------------------


class TestAutoPromotion:
    """Test auto-promotion of chat-scoped nodes to global."""

    @pytest.mark.asyncio
    async def test_promotes_high_mention_nodes(self):
        from augmentum.memory.integration import _maybe_auto_promote

        node_local = MagicMock()
        node_local.chat_id = "chat1"
        node_local.mentions = 5
        node_local.id = "n1"
        node_local.label = "Alice"

        node_global = MagicMock()
        node_global.chat_id = None  # Already global
        node_global.mentions = 10
        node_global.id = "n2"

        graph = AsyncMock()
        graph.get_nodes_by_chat = AsyncMock(return_value=[node_local, node_global])
        graph.promote_to_global = AsyncMock(return_value=True)

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.kg_auto_promote_threshold = 3

            await _maybe_auto_promote(graph, "chat1", "default")

            # Only the local node should be promoted
            graph.promote_to_global.assert_called_once_with("n1")

    @pytest.mark.asyncio
    async def test_skips_low_mention_nodes(self):
        from augmentum.memory.integration import _maybe_auto_promote

        node = MagicMock()
        node.chat_id = "chat1"
        node.mentions = 1
        node.id = "n1"

        graph = AsyncMock()
        graph.get_nodes_by_chat = AsyncMock(return_value=[node])
        graph.promote_to_global = AsyncMock()

        with patch("augmentum.memory.integration.settings") as mock_settings:
            mock_settings.kg_auto_promote_threshold = 3

            await _maybe_auto_promote(graph, "chat1", "default")

            graph.promote_to_global.assert_not_called()


# ------------------------------------------------------------------
# RRF multi-source merge
# ------------------------------------------------------------------


class TestRRFMultiMerge:
    """Test multi-source Reciprocal Rank Fusion."""

    def test_merge_three_sources(self):
        from augmentum.memory.models import Memory, MemoryTier
        from augmentum.memory.store import MemoryStore

        def _mem(mid: str) -> Memory:
            return Memory(
                id=mid, user_id="u", content=f"content_{mid}",
                memory_type="fact", importance=0.5, confidence=0.8,
                created_at="", updated_at="", tier=MemoryTier.ACTIVE,
            )

        list1 = [(_mem("a"), 0.9), (_mem("b"), 0.8)]
        list2 = [(_mem("b"), 0.7), (_mem("c"), 0.6)]
        list3 = [(_mem("a"), 0.5), (_mem("c"), 0.4)]

        merged = MemoryStore._rrf_merge_multi([list1, list2, list3], k=60)

        # "a" appears in list1 and list3, "b" in list1 and list2, "c" in list2 and list3
        ids = [m.id for m, _ in merged]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids

        # "a" and "b" both appear in 2 lists at rank 1 each → should have similar scores
        # All three should have similar scores since they each appear in 2 of 3 lists
        scores = {m.id: s for m, s in merged}
        assert scores["a"] > 0
        assert scores["b"] > 0
        assert scores["c"] > 0

    def test_merge_empty_lists(self):
        from augmentum.memory.store import MemoryStore

        merged = MemoryStore._rrf_merge_multi([[], [], []], k=60)
        assert merged == []

    def test_backward_compat_rrf_merge(self):
        """_rrf_merge still works (delegates to _rrf_merge_multi)."""
        from augmentum.memory.models import Memory, MemoryTier
        from augmentum.memory.store import MemoryStore

        def _mem(mid: str) -> Memory:
            return Memory(
                id=mid, user_id="u", content="x",
                memory_type="fact", importance=0.5, confidence=0.8,
                created_at="", updated_at="", tier=MemoryTier.ACTIVE,
            )

        vec = [(_mem("a"), 0.9)]
        fts = [(_mem("a"), 0.7), (_mem("b"), 0.6)]

        merged = MemoryStore._rrf_merge(vec, fts, k=60)
        ids = [m.id for m, _ in merged]
        assert "a" in ids
        assert "b" in ids


# ------------------------------------------------------------------
# Smart retrieval (archived message search)
# ------------------------------------------------------------------


class TestEmbeddedArchive:
    """Test that engine no longer holds archived messages in memory."""

    def test_engine_has_no_archived_messages_list(self):
        from augmentum.modes.narrative.engine import NarrativeEngine

        engine = NarrativeEngine(session_id="test")
        assert not hasattr(engine, "_archived_messages")

    def test_prepare_overflow_batch_no_longer_populates_archive(self):
        from augmentum.modes.narrative.engine import NarrativeEngine

        engine = NarrativeEngine(session_id="test", max_history_messages=2, summary_batch_size=2)
        engine._message_history = ["msg1", "msg2", "msg3", "msg4"]
        batch = engine.prepare_overflow_batch()
        assert batch == ["msg1", "msg2"]
        assert len(engine._message_history) == 2
        assert not hasattr(engine, "_archived_messages")

    def test_process_request_accepts_retrieved_archive(self):
        from augmentum.modes.narrative.engine import NarrativeEngine

        engine = NarrativeEngine(session_id="test")
        retrieved = [
            {"user_content": "hello", "assistant_content": "hi", "summary": "Greeting exchange"},
        ]
        # Should not raise
        from augmentum.models.base import InternalChatRequest, Message
        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="test")],
        )
        result = engine.process_request(request, retrieved_archive=retrieved)
        assert result is not None
