"""Tests for custom flow CRUD, trigger matching, and template resolution."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import aiosqlite

from augmentum.tools.custom_flows import (
    _DEFAULT_FLOWS,
    CustomFlowStore,
    flow_to_plan,
    match_trigger,
)


async def _create_db() -> aiosqlite.Connection:
    """Create an in-memory SQLite database with the custom_flows table."""
    db = await aiosqlite.connect(":memory:")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS custom_flows (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            trigger_pattern TEXT DEFAULT '',
            steps_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            enabled INTEGER DEFAULT 1,
            user_id TEXT
        )
    """)
    await db.commit()
    return db


UID = "user_test"


def _run(coro):
    return asyncio.run(coro)


class TestTriggerMatching(unittest.TestCase):
    """Test match_trigger function."""

    def test_matches_pattern(self):
        flows = [
            {"id": "1", "name": "Video", "trigger_pattern": r"analyze\s+(this\s+)?video", "enabled": True},
        ]
        result = match_trigger("analyze this video about cats", flows)
        assert result is not None
        assert result["id"] == "1"

    def test_no_match(self):
        flows = [
            {"id": "1", "name": "Video", "trigger_pattern": r"analyze\s+video", "enabled": True},
        ]
        result = match_trigger("search for cats", flows)
        assert result is None

    def test_disabled_flow_skipped(self):
        flows = [
            {"id": "1", "name": "Video", "trigger_pattern": r".*", "enabled": False},
        ]
        result = match_trigger("anything", flows)
        assert result is None

    def test_empty_pattern_skipped(self):
        flows = [
            {"id": "1", "name": "Video", "trigger_pattern": "", "enabled": True},
        ]
        result = match_trigger("anything", flows)
        assert result is None

    def test_invalid_regex_skipped(self):
        flows = [
            {"id": "1", "name": "Bad", "trigger_pattern": "[invalid", "enabled": True},
            {"id": "2", "name": "Good", "trigger_pattern": r"cats", "enabled": True},
        ]
        result = match_trigger("I like cats", flows)
        assert result is not None
        assert result["id"] == "2"

    def test_first_match_wins(self):
        flows = [
            {"id": "1", "name": "First", "trigger_pattern": r"cats", "enabled": True},
            {"id": "2", "name": "Second", "trigger_pattern": r"cats", "enabled": True},
        ]
        result = match_trigger("cats are great", flows)
        assert result["id"] == "1"


class TestFlowToPlan(unittest.TestCase):
    """Test flow_to_plan conversion."""

    def test_basic_conversion(self):
        flow = {
            "id": "abc",
            "name": "Test Flow",
            "steps_json": json.dumps([
                {"id": 1, "tool": "web_search", "input": {"query": "cats"}, "needs": [], "reason": "Search"},
                {"id": 2, "tool": "calculator", "needs": [1], "reason": "Calc"},
            ]),
        }
        plan = flow_to_plan(flow)
        assert plan.source == "custom:abc"
        assert len(plan.steps) == 2
        assert plan.steps[0].tool == "web_search"
        assert plan.steps[0].input == {"query": "cats"}
        assert plan.steps[1].needs == [1]

    def test_string_steps_json(self):
        flow = {
            "id": "x",
            "name": "Flow",
            "steps_json": '[{"id": 1, "tool": "calc"}]',
        }
        plan = flow_to_plan(flow)
        assert len(plan.steps) == 1

    def test_list_steps_json(self):
        """Already-parsed list works too."""
        flow = {
            "id": "x",
            "name": "Flow",
            "steps_json": [{"id": 1, "tool": "calc"}],
        }
        plan = flow_to_plan(flow)
        assert len(plan.steps) == 1


class TestCustomFlowStore(unittest.TestCase):
    """Test CustomFlowStore CRUD operations."""

    def test_create_and_get(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            flow = await store.create_flow(
                name="Test Flow",
                steps=[{"id": 1, "tool": "web_search", "reason": "Search"}],
                description="A test flow",
                trigger_pattern=r"test\s+flow",
                user_id=UID,
            )
            assert flow["name"] == "Test Flow"
            assert flow["description"] == "A test flow"

            fetched = await store.get_flow(flow["id"], user_id=UID)
            assert fetched is not None
            assert fetched["name"] == "Test Flow"
            await db.close()

        _run(_test())

    def test_list_flows(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Flow A", [{"id": 1, "tool": "a"}], user_id=UID)
            await store.create_flow("Flow B", [{"id": 1, "tool": "b"}], user_id=UID)
            flows = await store.list_flows(user_id=UID)
            assert len(flows) == 2
            await db.close()

        _run(_test())

    def test_list_enabled_only(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            f1 = await store.create_flow("Active", [{"id": 1, "tool": "a"}], user_id=UID)
            f2 = await store.create_flow("Inactive", [{"id": 1, "tool": "b"}], user_id=UID)
            await store.update_flow(f2["id"], user_id=UID, enabled=False)
            flows = await store.list_flows(enabled_only=True, user_id=UID)
            assert len(flows) == 1
            assert flows[0]["name"] == "Active"
            await db.close()

        _run(_test())

    def test_update_flow(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            flow = await store.create_flow("Original", [{"id": 1, "tool": "a"}], user_id=UID)
            updated = await store.update_flow(flow["id"], user_id=UID, name="Updated")
            assert updated["name"] == "Updated"
            await db.close()

        _run(_test())

    def test_delete_flow(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            flow = await store.create_flow("Delete Me", [{"id": 1, "tool": "a"}], user_id=UID)
            deleted = await store.delete_flow(flow["id"], user_id=UID)
            assert deleted is True
            assert await store.get_flow(flow["id"], user_id=UID) is None
            await db.close()

        _run(_test())

    def test_delete_nonexistent(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            deleted = await store.delete_flow("nonexistent", user_id=UID)
            assert deleted is False
            await db.close()

        _run(_test())

    def test_match_query(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow(
                "Video Analyzer",
                [{"id": 1, "tool": "web_search"}],
                trigger_pattern=r"analyze\s+(this\s+)?video",
                user_id=UID,
            )
            match = await store.match_query("analyze this video about cooking", user_id=UID)
            assert match is not None
            assert match["name"] == "Video Analyzer"

            no_match = await store.match_query("what is the weather", user_id=UID)
            assert no_match is None
            await db.close()

        _run(_test())

    def test_fuzzy_find_exact(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Video Analyzer", [{"id": 1, "tool": "a"}], user_id=UID)
            found = await store.fuzzy_find("Video Analyzer", user_id=UID)
            assert found is not None
            assert found["name"] == "Video Analyzer"
            await db.close()

        _run(_test())

    def test_fuzzy_find_prefix(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Video Analyzer", [{"id": 1, "tool": "a"}], user_id=UID)
            found = await store.fuzzy_find("video", user_id=UID)
            assert found is not None
            assert found["name"] == "Video Analyzer"
            await db.close()

        _run(_test())

    def test_fuzzy_find_substring(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Video Analyzer", [{"id": 1, "tool": "a"}], user_id=UID)
            found = await store.fuzzy_find("analyzer", user_id=UID)
            assert found is not None
            await db.close()

        _run(_test())

    def test_fuzzy_find_no_match(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Video Analyzer", [{"id": 1, "tool": "a"}], user_id=UID)
            found = await store.fuzzy_find("weather", user_id=UID)
            assert found is None
            await db.close()

        _run(_test())

    def test_invalid_trigger_pattern_rejected(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            with self.assertRaises(ValueError):
                await store.create_flow(
                    "Bad Pattern",
                    [{"id": 1, "tool": "a"}],
                    trigger_pattern="[invalid",
                    user_id=UID,
                )
            await db.close()

        _run(_test())

    def test_export_all(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Flow A", [{"id": 1, "tool": "a"}], user_id=UID)
            await store.create_flow("Flow B", [{"id": 1, "tool": "b"}], user_id=UID)
            exported = await store.export_all(user_id=UID)
            assert len(exported) == 2
            # steps_json should be parsed
            assert isinstance(exported[0]["steps_json"], list)
            await db.close()

        _run(_test())

    def test_import_flows(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            data = [
                {"name": "Imported A", "steps": [{"id": 1, "tool": "a"}]},
                {"name": "Imported B", "steps_json": '[{"id": 1, "tool": "b"}]'},
            ]
            count = await store.import_flows(data, user_id=UID)
            assert count == 2
            flows = await store.list_flows(user_id=UID)
            assert len(flows) == 2
            await db.close()

        _run(_test())

    def test_update_trigger_validates_regex(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            flow = await store.create_flow("Flow", [{"id": 1, "tool": "a"}], user_id=UID)
            with self.assertRaises(ValueError):
                await store.update_flow(flow["id"], user_id=UID, trigger_pattern="[bad")
            await db.close()

        _run(_test())


class TestGenerateFlowViaLLM(unittest.TestCase):
    """Test AI flow generation from natural language."""

    def test_valid_llm_response_parsed(self):
        from augmentum.tools.custom_flows import generate_flow_via_llm

        async def _test():
            llm_output = json.dumps({
                "name": "Test Flow",
                "description": "A generated flow",
                "trigger_pattern": "",
                "steps": [
                    {"id": 1, "tool": "web_search", "input": {"query": "{{query}}"}, "needs": [], "reason": "Search"},
                    {"id": 2, "tool": "web_fetch", "needs": [1], "reason": "Fetch top result"},
                ],
            })
            backend = MagicMock()
            resp_msg = MagicMock()
            resp_msg.content = llm_output
            backend.chat = AsyncMock(return_value=MagicMock(message=resp_msg))

            registry = MagicMock()
            tool = MagicMock()
            tool.name = "web_search"
            tool.description = "Search"
            tool.input_schema = {"properties": {"query": {"type": "string"}}, "required": ["query"]}
            registry.list_tools.return_value = [tool]

            flow = await generate_flow_via_llm("search then fetch", backend, registry)
            self.assertEqual(flow["name"], "Test Flow")
            self.assertEqual(len(flow["steps"]), 2)

        _run(_test())

    def test_markdown_fenced_response(self):
        from augmentum.tools.custom_flows import generate_flow_via_llm

        async def _test():
            llm_output = '```json\n{"name": "Fenced", "steps": [{"id": 1, "tool": "calculator"}]}\n```'
            backend = MagicMock()
            resp_msg = MagicMock()
            resp_msg.content = llm_output
            backend.chat = AsyncMock(return_value=MagicMock(message=resp_msg))
            registry = MagicMock()
            registry.list_tools.return_value = []

            flow = await generate_flow_via_llm("test", backend, registry)
            self.assertEqual(flow["name"], "Fenced")

        _run(_test())

    def test_invalid_json_raises(self):
        from augmentum.tools.custom_flows import generate_flow_via_llm

        async def _test():
            backend = MagicMock()
            resp_msg = MagicMock()
            resp_msg.content = "Not valid JSON at all"
            backend.chat = AsyncMock(return_value=MagicMock(message=resp_msg))
            registry = MagicMock()
            registry.list_tools.return_value = []

            with self.assertRaises(ValueError) as ctx:
                await generate_flow_via_llm("test", backend, registry)
            self.assertIn("parse", str(ctx.exception).lower())

        _run(_test())

    def test_missing_steps_raises(self):
        from augmentum.tools.custom_flows import generate_flow_via_llm

        async def _test():
            backend = MagicMock()
            resp_msg = MagicMock()
            resp_msg.content = json.dumps({"name": "Empty", "steps": []})
            backend.chat = AsyncMock(return_value=MagicMock(message=resp_msg))
            registry = MagicMock()
            registry.list_tools.return_value = []

            with self.assertRaises(ValueError) as ctx:
                await generate_flow_via_llm("test", backend, registry)
            self.assertIn("no steps", str(ctx.exception).lower())

        _run(_test())


class TestSeedDefaults(unittest.TestCase):
    """Test default flow seeding."""

    def test_seed_populates_empty_store(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            count = await store.seed_defaults(user_id=UID)
            self.assertGreater(count, 0)
            flows = await store.list_flows(user_id=UID)
            self.assertEqual(len(flows), count)
            names = {f["name"] for f in flows}
            self.assertIn("Deep Research", names)
            self.assertIn("Video Summary", names)
            await db.close()

        _run(_test())

    def test_seed_adds_missing_defaults_to_nonempty_store(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            await store.create_flow("Existing", [{"id": 1, "tool": "a"}], user_id=UID)
            count = await store.seed_defaults(user_id=UID)
            # Should add all 8 missing defaults (not update "Existing")
            self.assertEqual(count, len(_DEFAULT_FLOWS))
            flows = await store.list_flows(user_id=UID)
            # 1 custom + 8 defaults
            self.assertEqual(len(flows), 1 + len(_DEFAULT_FLOWS))
            await db.close()

        _run(_test())

    def test_seed_updates_changed_defaults(self):
        async def _test():
            db = await _create_db()
            store = CustomFlowStore(db)
            # First seed
            await store.seed_defaults(user_id=UID)
            flows = await store.list_flows(user_id=UID)
            vs = [f for f in flows if f["name"] == "Video Summary"][0]
            # Modify the stored steps to simulate an outdated default
            await store.update_flow(vs["id"], user_id=UID, steps=[{"id": 1, "tool": "x"}])
            # Re-seed should detect the change and update
            count = await store.seed_defaults(user_id=UID)
            self.assertGreaterEqual(count, 1)
            updated = await store.get_flow(vs["id"], user_id=UID)
            steps = json.loads(updated["steps_json"])
            self.assertEqual(len(steps), 2)  # restored to default
            await db.close()

        _run(_test())


if __name__ == "__main__":
    unittest.main()
