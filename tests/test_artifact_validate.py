"""Tests for artifact data-quality gates + the honest low-data warning path.

Covers:
- ``chart_quality`` / ``sheet_quality`` degeneracy detection
- ``ChartTool`` / ``SpreadsheetTool`` surfacing a warning (not a silent
  blank) on thin data, while still producing the artifact
- ``normalize_sheets`` row↔header width reconciliation
- ``_render_chart`` empty-state placeholder
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_artifact_store():
    store = AsyncMock()
    store.save = AsyncMock(return_value={
        "id": "abc123",
        "filename": "Test.xlsx",
        "display_name": "Test.xlsx",
        "format": "xlsx",
        "size_bytes": 1234,
        "path": "standalone/Test.xlsx",
        "download_url": "/api/artifacts/abc123/download",
    })
    store.get_file_path = MagicMock(return_value=Path("/tmp/fake.xlsx"))
    return store


# ---------------------------------------------------------------------------
# chart_quality
# ---------------------------------------------------------------------------

class TestChartQuality:
    def _q(self, labels, datasets):
        from augmentum.tools.artifact_validate import chart_quality
        return chart_quality(labels, datasets)

    def test_healthy_dataset_is_ok(self):
        q = self._q(["Q1", "Q2", "Q3", "Q4"], [{"name": "R", "values": [1, 2, 3, 4]}])
        assert q.usable and not q.degenerate and q.ok

    def test_empty_datasets_unusable(self):
        q = self._q(["A", "B"], [])
        assert not q.usable and q.degenerate
        assert "no data" in q.reason

    def test_empty_values_unusable(self):
        q = self._q(["A", "B"], [{"name": "R", "values": []}])
        assert not q.usable and q.degenerate

    def test_all_zero_is_degenerate(self):
        q = self._q(["A", "B", "C", "D"], [{"name": "R", "values": [0, 0, 0, 0]}])
        assert q.usable and q.degenerate
        assert "zero" in q.reason

    def test_single_point_is_thin_not_degenerate(self):
        # One point is valid-but-sparse: pipeline enriches, tools don't warn.
        q = self._q(["Only"], [{"name": "R", "values": [5]}])
        assert q.usable and q.thin and not q.degenerate
        assert q.needs_repair

    def test_ragged_series_is_degenerate(self):
        q = self._q(["A", "B", "C", "D"], [{"name": "R", "values": [1, 2]}])
        assert q.usable and q.degenerate
        assert "shorter" in q.reason

    def test_few_points_is_thin_not_degenerate(self):
        # 3 labels, all filled, non-zero — below MIN_CHART_POINTS (4) but valid
        q = self._q(["A", "B", "C"], [{"name": "R", "values": [1, 2, 3]}])
        assert q.usable and q.thin and not q.degenerate
        assert q.needs_repair

    def test_bool_values_do_not_count_as_data(self):
        q = self._q(["A", "B"], [{"name": "R", "values": [True, False]}])
        assert not q.usable


# ---------------------------------------------------------------------------
# sheet_quality
# ---------------------------------------------------------------------------

class TestSheetQuality:
    def _q(self, sheets):
        from augmentum.tools.artifact_validate import sheet_quality
        return sheet_quality(sheets)

    def test_healthy_sheet_is_ok(self):
        q = self._q([{
            "name": "S", "headers": ["A", "B"],
            "rows": [["x", 1], ["y", 2], ["z", 3], ["w", 4]],
        }])
        assert q.ok

    def test_no_sheets(self):
        q = self._q([])
        assert not q.usable and q.degenerate

    def test_headers_no_rows_is_degenerate(self):
        q = self._q([{"name": "S", "headers": ["A", "B"], "rows": []}])
        assert q.usable and q.degenerate
        assert "no data rows" in q.reason

    def test_no_columns_unusable(self):
        q = self._q([{"name": "S", "headers": [], "rows": [["x"]]}])
        assert not q.usable

    def test_few_rows_is_thin_not_degenerate(self):
        # A short table is often legitimate — thin (pipeline enriches), not
        # degenerate (no user-facing warning).
        q = self._q([{"name": "S", "headers": ["A"], "rows": [["1"], ["2"]]}])
        assert q.usable and q.thin and not q.degenerate
        assert q.needs_repair

    def test_blank_cells_dont_count_as_filled(self):
        q = self._q([{"name": "S", "headers": ["A", "B"],
                      "rows": [["", ""], ["", ""]]}])
        assert q.degenerate
        assert "no data rows" in q.reason

    def test_densest_sheet_wins_over_sources(self):
        # A thin Sources sheet must not mask a healthy data sheet.
        q = self._q([
            {"name": "Data", "headers": ["A", "B"],
             "rows": [["x", 1], ["y", 2], ["z", 3], ["w", 4]]},
            {"name": "Sources", "headers": ["URL"], "rows": [["http://x"]]},
        ])
        assert q.ok


# ---------------------------------------------------------------------------
# Tool warning path (render anyway, but flag it)
# ---------------------------------------------------------------------------

class TestSpreadsheetWarning:
    def _tool(self, store):
        from augmentum.tools.artifact_spreadsheet import SpreadsheetTool
        return SpreadsheetTool(store)

    @pytest.mark.asyncio
    async def test_empty_rows_still_succeeds_with_warning(self, mock_artifact_store):
        tool = self._tool(mock_artifact_store)
        result = await tool.execute(
            title="Empty",
            sheets=[{"name": "S", "headers": ["A", "B"], "rows": []}],
        )
        assert result.success  # render anyway (user's "render + warn" choice)
        assert result.warnings, "expected a low-data warning"
        assert "limited data" in result.warnings[0]
        assert (result.card or {}).get("preview", {}).get("low_data") is True

    @pytest.mark.asyncio
    async def test_healthy_sheet_has_no_warning(self, mock_artifact_store):
        tool = self._tool(mock_artifact_store)
        result = await tool.execute(
            title="Full",
            sheets=[{"name": "S", "headers": ["A", "B"],
                     "rows": [["x", 1], ["y", 2], ["z", 3], ["w", 4]]}],
        )
        assert result.success
        assert not result.warnings
        assert (result.card or {}).get("preview", {}).get("low_data") is False


class TestChartWarning:
    """Chart rendering needs matplotlib (a Docker-only dependency); skip when
    absent so local runs stay green."""

    def _tool(self, store):
        pytest.importorskip("matplotlib")
        from augmentum.tools.artifact_chart import ChartTool
        return ChartTool(store)

    @pytest.mark.asyncio
    async def test_all_zero_chart_warns_but_succeeds(self, mock_artifact_store):
        tool = self._tool(mock_artifact_store)
        result = await tool.execute(
            title="Zeros", chart_type="bar",
            labels=["A", "B", "C", "D"],
            datasets=[{"name": "R", "values": [0, 0, 0, 0]}],
        )
        assert result.success
        assert result.warnings
        assert (result.card or {}).get("preview", {}).get("low_data") is True

    @pytest.mark.asyncio
    async def test_healthy_chart_no_warning(self, mock_artifact_store):
        tool = self._tool(mock_artifact_store)
        result = await tool.execute(
            title="Good", chart_type="bar",
            labels=["A", "B", "C", "D"],
            datasets=[{"name": "R", "values": [4, 8, 15, 16]}],
        )
        assert result.success
        assert not result.warnings


# ---------------------------------------------------------------------------
# normalize_sheets reconciliation + render empty-state
# ---------------------------------------------------------------------------

class TestNormalizeReconciliation:
    def test_short_rows_padded_to_header_width(self):
        from augmentum.tools.artifact_normalize import normalize_sheets
        out = normalize_sheets([{
            "name": "S", "headers": ["A", "B", "C"],
            "rows": [["x"], ["y", 2]],
        }])
        rows = out[0]["rows"]
        assert all(len(r) == 3 for r in rows), rows
        assert rows[0] == ["x", "", ""]
        assert rows[1] == ["y", 2, ""]

    def test_long_rows_preserved(self):
        from augmentum.tools.artifact_normalize import normalize_sheets
        out = normalize_sheets([{
            "name": "S", "headers": ["A", "B"],
            "rows": [["x", 1, "extra"]],
        }])
        assert out[0]["rows"][0] == ["x", 1, "extra"]


class TestChartEmptyStateRender:
    def test_no_values_renders_placeholder_png(self):
        pytest.importorskip("matplotlib")
        from augmentum.tools.artifact_chart import _render_chart
        data = _render_chart(
            title="Empty", chart_type="bar",
            x_label="", y_label="",
            labels=["A", "B"],
            datasets=[{"name": "R", "values": []}],
        )
        assert isinstance(data, bytes)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
