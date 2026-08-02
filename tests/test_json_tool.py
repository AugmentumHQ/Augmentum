"""Tests for JsonTool — JSON parsing, validation, query, diff, merge."""

from __future__ import annotations

import json

import pytest

from augmentum.tools.json_tool import JsonTool, _jsonpath_query


class TestJsonPathQuery:
    """Direct tests on the JSONPath query function."""

    def test_root_query(self):
        data = {"a": 1}
        assert _jsonpath_query(data, "$") == {"a": 1}

    def test_nested_key(self):
        data = {"a": {"b": {"c": 42}}}
        assert _jsonpath_query(data, "$.a.b.c") == 42

    def test_array_index(self):
        data = {"items": [10, 20, 30]}
        assert _jsonpath_query(data, "$.items[0]") == 10

    def test_array_wildcard(self):
        data = {"items": [{"name": "a"}, {"name": "b"}]}
        result = _jsonpath_query(data, "$.items[*].name")
        assert result == ["a", "b"]

    def test_missing_path_returns_none(self):
        data = {"a": 1}
        assert _jsonpath_query(data, "$.b.c") is None

    def test_array_out_of_bounds(self):
        data = {"items": [1, 2]}
        assert _jsonpath_query(data, "$.items[5]") is None


class TestJsonToolValidate:
    """Validate action."""

    async def test_validate_valid_json(self):
        tool = JsonTool()
        result = await tool.execute(action="validate", json_text='{"key": "value"}')
        assert result.success is True
        assert result.metadata["valid"] is True
        assert "dict" in result.output.lower() or "Valid" in result.output

    async def test_validate_invalid_json(self):
        tool = JsonTool()
        result = await tool.execute(action="validate", json_text='{bad json}')
        assert result.success is True  # validate succeeds but reports invalid
        assert result.metadata["valid"] is False

    async def test_validate_array(self):
        tool = JsonTool()
        result = await tool.execute(action="validate", json_text='[1, 2, 3]')
        assert result.success is True
        assert result.metadata["valid"] is True


class TestJsonToolFormat:
    """Format and minify actions."""

    async def test_format_pretty_prints(self):
        tool = JsonTool()
        result = await tool.execute(action="format", json_text='{"a":1,"b":2}')
        assert result.success is True
        assert "\n" in result.output

    async def test_minify_removes_whitespace(self):
        tool = JsonTool()
        result = await tool.execute(action="minify", json_text='{ "a": 1, "b": 2 }')
        assert result.success is True
        assert " " not in result.output.replace('"', "")


class TestJsonToolQuery:
    """Query action with JSONPath."""

    async def test_query_nested(self):
        tool = JsonTool()
        data = json.dumps({"user": {"name": "Alice", "age": 30}})
        result = await tool.execute(action="query", json_text=data, path="$.user.name")
        assert result.success is True
        assert "Alice" in result.output

    async def test_query_not_found(self):
        tool = JsonTool()
        result = await tool.execute(action="query", json_text='{"a": 1}', path="$.b")
        assert result.success is True
        assert result.metadata["found"] is False


class TestJsonToolDiff:
    """Diff action comparing two JSON objects."""

    async def test_diff_identical(self):
        tool = JsonTool()
        j = '{"a": 1}'
        result = await tool.execute(action="diff", json_text=j, json_text2=j)
        assert result.success is True
        assert result.metadata["identical"] is True

    async def test_diff_changed_value(self):
        tool = JsonTool()
        result = await tool.execute(
            action="diff",
            json_text='{"a": 1}',
            json_text2='{"a": 2}',
        )
        assert result.success is True
        assert result.metadata["identical"] is False
        assert result.metadata["diff_count"] >= 1

    async def test_diff_added_key(self):
        tool = JsonTool()
        result = await tool.execute(
            action="diff",
            json_text='{"a": 1}',
            json_text2='{"a": 1, "b": 2}',
        )
        assert result.success is True
        assert "added" in result.output

    async def test_diff_missing_second_json(self):
        tool = JsonTool()
        result = await tool.execute(action="diff", json_text='{"a": 1}')
        assert result.success is False


class TestJsonToolMerge:
    """Merge action combining two JSON objects."""

    async def test_merge_objects(self):
        tool = JsonTool()
        result = await tool.execute(
            action="merge",
            json_text='{"a": 1}',
            json_text2='{"b": 2}',
        )
        assert result.success is True
        merged = json.loads(result.output)
        assert merged == {"a": 1, "b": 2}

    async def test_merge_arrays(self):
        tool = JsonTool()
        result = await tool.execute(
            action="merge",
            json_text='[1, 2]',
            json_text2='[3, 4]',
        )
        assert result.success is True
        merged = json.loads(result.output)
        assert merged == [1, 2, 3, 4]

    async def test_merge_deep(self):
        tool = JsonTool()
        result = await tool.execute(
            action="merge",
            json_text='{"a": {"x": 1}}',
            json_text2='{"a": {"y": 2}}',
        )
        assert result.success is True
        merged = json.loads(result.output)
        assert merged["a"] == {"x": 1, "y": 2}


class TestJsonToolEdgeCases:
    """Edge cases and error handling."""

    async def test_empty_json_text_returns_error(self):
        tool = JsonTool()
        result = await tool.execute(action="validate", json_text="")
        assert result.success is False

    async def test_keys_action_on_object(self):
        tool = JsonTool()
        result = await tool.execute(action="keys", json_text='{"a": 1, "b": 2}')
        assert result.success is True
        assert result.metadata["count"] == 2

    async def test_keys_action_on_array(self):
        tool = JsonTool()
        result = await tool.execute(action="keys", json_text='[1, 2, 3]')
        assert result.success is True
        assert "3 elements" in result.output

    async def test_unknown_action(self):
        tool = JsonTool()
        result = await tool.execute(action="explode", json_text='{}')
        assert result.success is False
