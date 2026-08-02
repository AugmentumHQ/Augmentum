"""Tests for FlowTool — wrapping custom flows as callable tools."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock

from augmentum.tools.base import ToolCategory, ToolResult
from augmentum.tools.flow_tool import FlowTool, flow_name_to_tool_name


class TestFlowNameToToolName(unittest.TestCase):
    """Test flow_name_to_tool_name sanitisation."""

    def test_simple_spaces(self):
        assert flow_name_to_tool_name("Deep Research") == "flow_deep_research"

    def test_special_characters(self):
        assert flow_name_to_tool_name("My Flow!@#v2") == "flow_my_flow_v2"

    def test_already_lowercase(self):
        assert flow_name_to_tool_name("quick answer") == "flow_quick_answer"

    def test_mixed_case(self):
        assert flow_name_to_tool_name("Code Review") == "flow_code_review"

    def test_consecutive_special_chars(self):
        assert flow_name_to_tool_name("a---b") == "flow_a_b"

    def test_leading_trailing_special(self):
        assert flow_name_to_tool_name("--hello--") == "flow_hello"

    def test_numbers_preserved(self):
        assert flow_name_to_tool_name("Report v3.1") == "flow_report_v3_1"

    def test_single_word(self):
        assert flow_name_to_tool_name("Summarize") == "flow_summarize"


class TestFlowToolProperties(unittest.TestCase):
    """Test FlowTool property accessors."""

    def _make_flow(self, **overrides):
        flow = {"name": "Test Flow", "id": "flow-001", "description": "A test flow"}
        flow.update(overrides)
        return flow

    def test_name_derived_from_flow(self):
        tool = FlowTool(self._make_flow(), AsyncMock())
        assert tool.name == "flow_test_flow"

    def test_description_from_flow(self):
        tool = FlowTool(self._make_flow(description="Custom desc"), AsyncMock())
        assert tool.description == "Custom desc"

    def test_description_fallback_when_empty(self):
        tool = FlowTool(self._make_flow(description=""), AsyncMock())
        assert "Test Flow" in tool.description
        assert "Run the" in tool.description

    def test_description_fallback_when_missing(self):
        flow = {"name": "Research", "id": "r1"}
        tool = FlowTool(flow, AsyncMock())
        assert "Research" in tool.description

    def test_category_is_execute(self):
        tool = FlowTool(self._make_flow(), AsyncMock())
        assert tool.category == ToolCategory.EXECUTE

    def test_input_schema_has_query(self):
        tool = FlowTool(self._make_flow(), AsyncMock())
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_flow_id(self):
        tool = FlowTool(self._make_flow(id="abc-123"), AsyncMock())
        assert tool.flow_id == "abc-123"

    def test_flow_id_empty_when_missing(self):
        flow = {"name": "X"}
        tool = FlowTool(flow, AsyncMock())
        assert tool.flow_id == ""

    def test_timeout_is_short(self):
        tool = FlowTool(self._make_flow(), AsyncMock())
        assert tool.timeout == 10.0

    def test_not_cacheable(self):
        tool = FlowTool(self._make_flow(), AsyncMock())
        assert tool.cacheable is False


class TestFlowToolExecute(unittest.TestCase):
    """Test FlowTool.execute() behaviour."""

    def _make_flow(self, **overrides):
        flow = {"name": "Research", "id": "flow-42"}
        flow.update(overrides)
        return flow

    def test_successful_launch(self):
        """Successful launch returns ToolResult with task metadata."""
        launcher = AsyncMock(return_value="task_abc123")
        flow = self._make_flow()
        tool = FlowTool(flow, launcher, session_id="sess-1")

        result: ToolResult = asyncio.run(tool.execute(query="find cats"))

        assert result.success is True
        assert "task_abc123" in result.output
        assert "Research" in result.output
        assert result.metadata["task_id"] == "task_abc123"
        assert result.metadata["flow_name"] == "Research"
        assert result.metadata["flow_id"] == "flow-42"
        assert result.metadata["background"] is True
        launcher.assert_awaited_once_with(flow, "find cats", "sess-1", user_id="", request_context=None)

    def test_launcher_exception_returns_error(self):
        """When the launcher raises, execute returns a failed ToolResult."""
        launcher = AsyncMock(side_effect=RuntimeError("backend unavailable"))
        flow = self._make_flow()
        tool = FlowTool(flow, launcher, session_id="sess-1")

        result: ToolResult = asyncio.run(tool.execute(query="test"))

        assert result.success is False
        assert result.error == "backend unavailable"
        assert "Failed to start flow" in result.output
        assert "Research" in result.output

    def test_missing_query_defaults_to_empty_string(self):
        """Calling execute without query passes empty string."""
        launcher = AsyncMock(return_value="t1")
        flow = self._make_flow()
        tool = FlowTool(flow, launcher, session_id="s1")

        asyncio.run(tool.execute())

        launcher.assert_awaited_once_with(flow, "", "s1", user_id="", request_context=None)

    def test_session_id_passed_through(self):
        """Session ID from constructor is passed to the launcher."""
        launcher = AsyncMock(return_value="t1")
        tool = FlowTool(self._make_flow(), launcher, session_id="custom-session")

        asyncio.run(tool.execute(query="q"))

        assert launcher.call_args[0][2] == "custom-session"


if __name__ == "__main__":
    unittest.main()
