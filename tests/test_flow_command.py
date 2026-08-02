"""Tests for /flow command — classifier detection, fuzzy matching, handler routing."""

from __future__ import annotations

import asyncio
import unittest

from augmentum.classifier.router import ClassificationResult, Mode, RequestClassifier
from augmentum.models.base import InternalChatRequest, Message


class TestFlowCommandClassifier(unittest.TestCase):
    """Test /flow and /f command detection in the classifier."""

    def setUp(self):
        self.classifier = RequestClassifier()

    def _req(self, content: str) -> InternalChatRequest:
        return InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content=content)],
        )

    # --- Detection ---

    def test_flow_command_detected(self):
        req = self._req("/flow Deep Research")
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "Deep Research")

    def test_flow_shorthand_detected(self):
        req = self._req("/f Quick Answer")
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "Quick Answer")

    def test_flow_case_insensitive(self):
        req = self._req("/FLOW Deep Research")
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "Deep Research")

    def test_flow_with_separator_query(self):
        req = self._req("/flow Deep Research -- what is quantum computing?")
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "Deep Research")
        self.assertEqual(req.messages[-1].content, "what is quantum computing?")

    def test_flow_with_dash_separator(self):
        req = self._req("/flow Code Review — review this function")
        result = self.classifier.classify(req)
        self.assertEqual(req.explicit_flow_name, "Code Review")
        self.assertEqual(req.messages[-1].content, "review this function")

    def test_flow_with_newline_separator(self):
        req = self._req("/flow Deep Research\nwhat is quantum computing?")
        result = self.classifier.classify(req)
        self.assertEqual(req.explicit_flow_name, "Deep Research")
        self.assertEqual(req.messages[-1].content, "what is quantum computing?")

    def test_bare_flow_command(self):
        req = self._req("/flow")
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "__list__")

    def test_bare_f_command(self):
        req = self._req("/f")
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "__list__")

    def test_flow_without_separator_passes_full_rest(self):
        """Without separator, entire rest is the flow name (handler splits later)."""
        req = self._req("/flow Deep Research world war 2")
        self.classifier.classify(req)
        self.assertEqual(req.explicit_flow_name, "Deep Research world war 2")
        self.assertEqual(req.messages[-1].content, "")

    def test_flow_without_query_empties_content(self):
        req = self._req("/flow Deep Research")
        self.classifier.classify(req)
        self.assertEqual(req.messages[-1].content, "")

    # --- Non-flow messages pass through ---

    def test_normal_message_not_affected(self):
        req = self._req("What is quantum computing?")
        result = self.classifier.classify(req)
        self.assertNotEqual(result.reason, "/flow command: Deep Research")
        self.assertEqual(req.explicit_flow_name, "")

    def test_flow_in_middle_not_detected(self):
        req = self._req("Tell me about /flow command")
        result = self.classifier.classify(req)
        self.assertEqual(req.explicit_flow_name, "")

    def test_empty_messages(self):
        req = InternalChatRequest(model="test", messages=[])
        result = self.classifier.classify(req)
        self.assertEqual(req.explicit_flow_name, "")

    # --- Priority: /flow takes precedence over other classification ---

    def test_flow_overrides_narrative(self):
        """Even with narrative system prompt, /flow forces passthrough."""
        req = InternalChatRequest(
            model="test-model",
            messages=[
                Message(role="system", content="You are Luna, a cheerful catgirl. {{char}} loves milk."),
                Message(role="user", content="/flow Quick Answer"),
            ],
        )
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)

    def test_flow_overrides_model_prefix(self):
        """Model prefix should not override /flow command since /flow checks first."""
        req = InternalChatRequest(
            model="n/llama3.1:8b",
            messages=[Message(role="user", content="/flow Deep Research")],
        )
        result = self.classifier.classify(req)
        self.assertEqual(result.mode, Mode.PASSTHROUGH)
        self.assertEqual(req.explicit_flow_name, "Deep Research")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reasoning_flows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    icon TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    is_default BOOLEAN DEFAULT 0,
    is_builtin BOOLEAN DEFAULT 0,
    auto_select BOOLEAN DEFAULT 1,
    trigger_domains TEXT DEFAULT '[]',
    trigger_keywords TEXT DEFAULT '[]',
    pinned_models TEXT DEFAULT '[]',
    auto_search BOOLEAN DEFAULT 1,
    max_tool_calls_per_step INTEGER DEFAULT 3,
    autonomy_level INTEGER DEFAULT 2,
    escalation_flow TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS reasoning_flow_steps (
    id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL REFERENCES reasoning_flows(id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL,
    name TEXT NOT NULL,
    system_prompt TEXT DEFAULT '',
    user_template TEXT DEFAULT '',
    role TEXT DEFAULT 'analyze',
    tool_categories TEXT DEFAULT '[]',
    tool_names TEXT DEFAULT '[]',
    complexity_gate TEXT DEFAULT '[]',
    stream_to_user BOOLEAN DEFAULT 0,
    output_cap INTEGER DEFAULT 800,
    enabled BOOLEAN DEFAULT 1,
    model_override TEXT DEFAULT ''
);
"""


async def _make_store():
    import aiosqlite
    from augmentum.reasoning.store import FlowStore

    db = await aiosqlite.connect(":memory:")
    db.row_factory = None
    await db.executescript(_SCHEMA_SQL)
    return FlowStore(db), db


class TestFlowStoreFuzzyMatch(unittest.TestCase):
    """Test FlowStore.find_by_name fuzzy matching."""

    def test_exact_match(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Deep Research", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            found = await store.find_by_name("Deep Research")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "Deep Research")
            await db.close()
        asyncio.run(go())

    def test_case_insensitive_match(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Quick Answer", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            found = await store.find_by_name("quick answer")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "Quick Answer")
            await db.close()
        asyncio.run(go())

    def test_substring_match(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Data & Comparison", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            found = await store.find_by_name("comparison")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "Data & Comparison")
            await db.close()
        asyncio.run(go())

    def test_word_overlap_match(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Deep Research", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            found = await store.find_by_name("deep")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "Deep Research")
            await db.close()
        asyncio.run(go())

    def test_no_match_returns_none(self):
        async def go():
            store, db = await _make_store()
            found = await store.find_by_name("nonexistent flow")
            self.assertIsNone(found)
            await db.close()
        asyncio.run(go())

    def test_empty_query_returns_none(self):
        async def go():
            store, db = await _make_store()
            found = await store.find_by_name("")
            self.assertIsNone(found)
            await db.close()
        asyncio.run(go())


class TestGreedyMatch(unittest.TestCase):
    """Test FlowStore.greedy_match for prefix-based flow name + query splitting."""

    def test_greedy_splits_name_and_query(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Deep Research", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            flow, remainder = await store.greedy_match("Deep Research world war 2")
            self.assertIsNotNone(flow)
            self.assertEqual(flow.name, "Deep Research")
            self.assertEqual(remainder, "world war 2")
            await db.close()
        asyncio.run(go())

    def test_greedy_exact_no_remainder(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Quick Answer", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            flow, remainder = await store.greedy_match("Quick Answer")
            self.assertIsNotNone(flow)
            self.assertEqual(flow.name, "Quick Answer")
            self.assertEqual(remainder, "")
            await db.close()
        asyncio.run(go())

    def test_greedy_single_word_match(self):
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Research", description="test",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            flow, remainder = await store.greedy_match("Research quantum physics")
            self.assertIsNotNone(flow)
            self.assertEqual(flow.name, "Research")
            self.assertEqual(remainder, "quantum physics")
            await db.close()
        asyncio.run(go())

    def test_greedy_prefers_longest_match(self):
        """If 'Research' and 'Deep Research' both exist, 'Deep Research x' matches the longer one."""
        async def go():
            store, db = await _make_store()
            from augmentum.reasoning.models import FlowStep, ReasoningFlow
            await store.create_flow(ReasoningFlow(
                name="Research", description="short",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            await store.create_flow(ReasoningFlow(
                name="Deep Research", description="long",
                steps=[FlowStep(name="s1", role="respond")],
            ))
            flow, remainder = await store.greedy_match("Deep Research world war 2")
            self.assertIsNotNone(flow)
            self.assertEqual(flow.name, "Deep Research")
            self.assertEqual(remainder, "world war 2")
            await db.close()
        asyncio.run(go())

    def test_greedy_no_match(self):
        async def go():
            store, db = await _make_store()
            flow, remainder = await store.greedy_match("nonexistent flow something")
            self.assertIsNone(flow)
            self.assertEqual(remainder, "nonexistent flow something")
            await db.close()
        asyncio.run(go())

    def test_greedy_empty_text(self):
        async def go():
            store, db = await _make_store()
            flow, remainder = await store.greedy_match("")
            self.assertIsNone(flow)
            self.assertEqual(remainder, "")
            await db.close()
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
