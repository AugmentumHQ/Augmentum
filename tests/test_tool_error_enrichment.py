"""Error enrichment: what a tool tells the model after a failed call.

A tool failure is only recoverable if the model can tell WHY it failed and what
to send instead. Two layers, both asserted here:

1. Per-tool ``error_hints`` — hand-written substring→guidance, first match wins.
2. A generic schema reminder in ``Tool.enrich_error`` for malformed calls, which
   covers the ~40 tools that define no hints at all.

The layer-2 behaviour was documented in ``enrich_error``'s docstring long before
it was implemented, so these tests pin the promise to the code.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from augmentum.tools.base import Tool, ToolCategory, ToolResult


class _FakeTool(Tool):
    """Minimal tool with a known schema and no error_hints."""

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.UTILITY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="ok")


class _HintedTool(_FakeTool):
    @property
    def error_hints(self) -> dict[str, str]:
        return {"invalid arguments": "Use `query`, a plain search string."}


class TestSchemaReminderFallback:
    def test_names_the_parameter_the_model_invented(self):
        """The single most useful fact after a bad call: the wrong name."""
        err = _FakeTool().enrich_error(
            "invalid arguments: execute() got an unexpected keyword "
            "argument 'query_string'",
            {"query_string": "cats"},
        )
        assert "Not accepted: query_string" in err
        assert "required: query" in err
        assert "optional: limit" in err
        # The raw error is preserved, not replaced.
        assert "unexpected keyword argument" in err

    def test_reports_missing_required_params(self):
        err = _FakeTool().enrich_error(
            "invalid arguments: missing 1 required positional argument", {},
        )
        assert "Missing: query" in err

    def test_runtime_injected_params_are_not_reported_as_unknown(self):
        """``_context``/``_user_id`` come from the runtime, never the model —
        naming them would send it chasing a parameter it never sent."""
        err = _FakeTool().enrich_error(
            "invalid arguments: bad call",
            {"query": "x", "_context": {}, "_user_id": "u1"},
        )
        assert "_context" not in err
        assert "_user_id" not in err

    def test_non_shape_errors_are_left_alone(self):
        """A network failure is not fixable by restating the schema, so adding
        parameter lists to it would be pure noise."""
        raw = "Connection timed out after 30s"
        assert _FakeTool().enrich_error(raw, {"query": "x"}) == raw

    def test_handwritten_hint_wins_over_generic_reminder(self):
        err = _HintedTool().enrich_error("invalid arguments: nope", {"bad": 1})
        assert "Use `query`, a plain search string." in err
        assert "Not accepted" not in err  # short-circuited before layer 2

    def test_malformed_schema_degrades_to_raw_error(self):
        """Enrichment must never turn a tool failure into an enrichment
        failure."""
        class _Broken(_FakeTool):
            @property
            def input_schema(self) -> dict:
                raise RuntimeError("schema exploded")

        raw = "invalid arguments: boom"
        assert _Broken().enrich_error(raw, {"x": 1}) == raw

    def test_no_properties_yields_no_hint(self):
        class _NoParams(_FakeTool):
            @property
            def input_schema(self) -> dict:
                return {"type": "object", "properties": {}}

        raw = "invalid arguments: boom"
        assert _NoParams().enrich_error(raw, {}) == raw


class TestInvokeRoutesThroughEnrichment:
    @pytest.mark.asyncio
    async def test_bad_kwargs_return_enriched_typed_failure(self):
        """``Tool.invoke`` converts a signature mismatch into a typed result
        whose error carries the schema reminder — the model sees guidance, not a
        traceback."""
        class _Strict(_FakeTool):
            async def execute(self, query: str, limit: int = 10) -> ToolResult:
                return ToolResult(success=True, output="ok")

        res = await _Strict().invoke({"quiery": "typo"})
        assert res.success is False
        assert res.failure_kind == "invalid_input"
        # NOTE: ``coerce_params`` strips keys the schema doesn't know, so by the
        # time execute() raises, the invented name is gone and the error reads
        # "missing 1 required positional argument". The reminder's value on this
        # path is therefore the required-parameter list, not the "Not accepted"
        # line (which still fires for handlers that call execute() directly).
        assert "required: query" in (res.error or "")
        assert "Missing: query" in (res.error or "")


class TestArtifactCreatorHints:
    """The two creators exposed inline in passthrough Auto. Their hint KEYS must
    stay substrings of errors the tools really emit, or enrichment silently
    stops matching."""

    def test_chart_hint_keys_match_real_error_strings(self):
        from augmentum.tools.artifact_chart import ChartTool
        hints = ChartTool(MagicMock()).error_hints
        assert "Labels and datasets are required" in hints
        assert "Unsupported chart type" in hints
        # Non-retryable: matplotlib is a Docker-only dep.
        assert "No module named 'matplotlib'" in hints
        assert "NOT call this tool again" in hints["No module named 'matplotlib'"]

    def test_chart_missing_data_hint_explains_the_shape(self):
        from augmentum.tools.artifact_chart import ChartTool
        hint = ChartTool(MagicMock()).error_hints[
            "Labels and datasets are required"]
        assert "labels" in hint and "datasets" in hint
        assert "values" in hint  # the documented series key

    def test_spreadsheet_hint_keys_match_real_error_strings(self):
        from augmentum.tools.artifact_spreadsheet import SpreadsheetTool
        hints = SpreadsheetTool(MagicMock()).error_hints
        assert "No sheets provided" in hints
        assert "No data after cleanup" in hints
        assert "NOT call this tool again" in hints["No module named 'openpyxl'"]

    @pytest.mark.parametrize("chart_type", ["donut", "histogram", "bubble"])
    def test_unsupported_chart_type_hint_lists_the_valid_set(self, chart_type):
        """A model guessing a plausible-but-wrong type needs the real list."""
        from augmentum.tools.artifact_chart import ChartTool
        tool = ChartTool(MagicMock())
        err = tool.enrich_error(f"Unsupported chart type: {chart_type}", {})
        for valid in ("bar", "line", "pie", "scatter", "horizontal_bar"):
            assert valid in err
