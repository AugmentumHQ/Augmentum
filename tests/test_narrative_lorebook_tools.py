"""Tests for the lorebook LLM-tool schemas and dispatcher."""

from __future__ import annotations

from augmentum.modes.narrative.lore_engine import LoreEngine
from augmentum.modes.narrative.lorebook_schemas import (
    LOREBOOK_TOOL_NAMES,
    LOREBOOK_TOOL_SCHEMAS,
    dispatch_lorebook_tool,
)
from augmentum.state.narrative_state import LorebookEntry

SESSION_ID = "test-session-1"


def _engine_with_entries(*entries: LorebookEntry) -> LoreEngine:
    engine = LoreEngine()
    for e in entries:
        engine.add_entry(e)
    return engine


def _dispatch(engine, tool_name, args):
    return dispatch_lorebook_tool(
        engine, SESSION_ID,
        tool_name=tool_name,
        raw_arguments=args,
    )


def _sample_entry(**overrides):
    defaults = {
        "id": "lore_001",
        "session_id": SESSION_ID,
        "keywords": ["dragons", "fire"],
        "content": "Dragons breathe fire and hoard gold.",
        "comment": "Dragon Lore",
    }
    defaults.update(overrides)
    return LorebookEntry(**defaults)


class TestSchemas:
    def test_schema_names_match_frozenset(self):
        names = {s["function"]["name"] for s in LOREBOOK_TOOL_SCHEMAS}
        assert names == LOREBOOK_TOOL_NAMES

    def test_all_schemas_have_required_fields(self):
        for schema in LOREBOOK_TOOL_SCHEMAS:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn


class TestSearch:
    def test_search_by_keyword(self):
        engine = _engine_with_entries(_sample_entry())
        text, mutations = _dispatch(engine, "lorebook_search", {"query": "dragon"})
        assert "Dragon Lore" in text
        assert "fire" in text
        assert mutations is None

    def test_search_by_content(self):
        engine = _engine_with_entries(_sample_entry())
        text, _ = _dispatch(engine, "lorebook_search", {"query": "hoard gold"})
        assert "Dragon Lore" in text

    def test_search_no_match(self):
        engine = _engine_with_entries(_sample_entry())
        text, _ = _dispatch(engine, "lorebook_search", {"query": "unicorns"})
        assert "No lorebook entries match" in text

    def test_search_empty_engine(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_search", {"query": "anything"})
        assert "0 entries total" in text

    def test_search_requires_query(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_search", {})
        assert "requires" in text

    def test_search_json_string_args(self):
        engine = _engine_with_entries(_sample_entry())
        text, _ = _dispatch(engine, "lorebook_search", '{"query": "dragon"}')
        assert "Dragon Lore" in text

    def test_search_respects_limit(self):
        entries = [
            _sample_entry(id=f"e{i}", keywords=[f"word{i}"], comment=f"Entry {i}")
            for i in range(10)
        ]
        engine = _engine_with_entries(*entries)
        text, _ = _dispatch(engine, "lorebook_search", {"query": "word", "limit": 3})
        assert "Found 3" in text


class TestCreate:
    def test_create_basic(self):
        engine = LoreEngine()
        text, mutations = _dispatch(engine, "lorebook_create", {
            "name": "Elven City",
            "keywords": ["elves", "silvana"],
            "content": "Silvana is a city of ancient elves.",
        })
        assert "Created" in text
        assert "Elven City" in text
        assert mutations is not None
        assert len(mutations) == 1
        assert mutations[0]["action"] == "create"
        assert mutations[0]["entry"]["name"] == "Elven City"
        assert len(engine.entries) == 1
        entry = list(engine.entries.values())[0]
        assert entry.keywords == ["elves", "silvana"]
        assert entry.source == "llm_authored"

    def test_create_missing_name(self):
        engine = LoreEngine()
        text, mutations = _dispatch(engine, "lorebook_create", {
            "keywords": ["x"], "content": "y",
        })
        assert "requires" in text
        assert mutations is None

    def test_create_missing_keywords(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_create", {
            "name": "Test", "content": "y",
        })
        assert "requires" in text

    def test_create_with_priority_and_constant(self):
        engine = LoreEngine()
        text, mutations = _dispatch(engine, "lorebook_create", {
            "name": "Rule", "keywords": ["law"],
            "content": "The law says...", "priority": 500, "constant": True,
        })
        entry = list(engine.entries.values())[0]
        assert entry.priority == 500
        assert entry.constant is True


class TestUpdate:
    def test_update_content(self):
        engine = _engine_with_entries(_sample_entry())
        text, mutations = _dispatch(engine, "lorebook_update", {
            "entry_id": "lore_001", "content": "Dragons are extinct now.",
        })
        assert "Updated" in text
        assert "content" in text
        assert mutations is not None
        assert mutations[0]["action"] == "update"
        assert engine.entries["lore_001"].content == "Dragons are extinct now."

    def test_update_keywords(self):
        engine = _engine_with_entries(_sample_entry())
        text, _ = _dispatch(engine, "lorebook_update", {
            "entry_id": "lore_001", "keywords": ["wyrm", "drake"],
        })
        assert engine.entries["lore_001"].keywords == ["wyrm", "drake"]

    def test_update_disable(self):
        engine = _engine_with_entries(_sample_entry())
        text, _ = _dispatch(engine, "lorebook_update", {
            "entry_id": "lore_001", "enabled": False,
        })
        assert engine.entries["lore_001"].enabled is False

    def test_update_nonexistent(self):
        engine = LoreEngine()
        text, mutations = _dispatch(engine, "lorebook_update", {
            "entry_id": "nope", "content": "x",
        })
        assert "No lorebook entry" in text
        assert mutations is None

    def test_update_no_fields(self):
        engine = _engine_with_entries(_sample_entry())
        text, _ = _dispatch(engine, "lorebook_update", {"entry_id": "lore_001"})
        assert "no fields" in text


class TestDelete:
    def test_delete(self):
        engine = _engine_with_entries(_sample_entry())
        text, mutations = _dispatch(engine, "lorebook_delete", {"entry_id": "lore_001"})
        assert "Deleted" in text
        assert mutations is not None
        assert mutations[0]["action"] == "delete"
        assert len(engine.entries) == 0

    def test_delete_nonexistent(self):
        engine = LoreEngine()
        text, mutations = _dispatch(engine, "lorebook_delete", {"entry_id": "nope"})
        assert "No lorebook entry" in text
        assert mutations is None

    def test_delete_missing_id(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_delete", {})
        assert "requires" in text


class TestDispatcherEdgeCases:
    def test_unknown_tool_name(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_foo", {})
        assert "Unknown lorebook tool" in text

    def test_invalid_json_args(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_search", "not json")
        assert "not valid JSON" in text

    def test_non_object_json_args(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_search", '"just a string"')
        assert "must be a JSON object" in text

    def test_none_args(self):
        engine = LoreEngine()
        text, _ = _dispatch(engine, "lorebook_search", None)
        assert "requires" in text
