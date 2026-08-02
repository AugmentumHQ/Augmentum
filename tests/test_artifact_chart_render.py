"""Tests for the professional chart renderer (artifact_chart.py).

Two tiers:
- Pure-Python helpers (formatters, palette, format detection) — run everywhere.
- matplotlib render smoke tests — gated on matplotlib (a Docker-only dep).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.tools import artifact_chart as c

PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Value-format detection
# ---------------------------------------------------------------------------

class TestDetectValueFormat:
    def test_explicit_wins(self):
        assert c._detect_value_format("percent", "", "Revenue", [1, 2]) == "percent"

    def test_currency_from_label(self):
        assert c._detect_value_format("auto", "Quarter", "Revenue ($M)", [12, 15]) == "currency"

    def test_percent_from_label(self):
        assert c._detect_value_format("auto", "", "Growth %", [0.1, 0.2]) == "percent"

    def test_abbreviated_from_magnitude(self):
        assert c._detect_value_format("auto", "", "Users", [14000, 220000]) == "abbreviated"

    def test_plain_number_default(self):
        assert c._detect_value_format("auto", "", "Count", [3, 5, 8]) == "number"


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

class TestValueFormatter:
    def test_currency_small_and_large(self):
        f = c._make_value_formatter("currency", [142000, 98])
        assert f(142000) == "$142K"
        assert f(98) == "$98"

    def test_currency_negative(self):
        f = c._make_value_formatter("currency", [-50000])
        assert f(-50000) == "-$50K"

    def test_percent_fraction_scaled(self):
        f = c._make_value_formatter("percent", [0.45, 0.37])
        assert f(0.45) == "45%"

    def test_percent_whole_not_scaled(self):
        f = c._make_value_formatter("percent", [45, 37])
        assert f(45) == "45%"
        assert f(37.5) == "37.5%"

    def test_abbreviated_units(self):
        assert c._abbrev(1_500_000) == "1.5M"
        assert c._abbrev(2300) == "2.3K"
        assert c._abbrev(2_300_000_000) == "2.3B"
        assert c._abbrev(42) == "42"

    def test_number_thousands_separator(self):
        f = c._make_value_formatter("number", [1234])
        assert f(1234) == "1,234"
        assert f(1234.5) == "1,234.5"


# ---------------------------------------------------------------------------
# Palette (accent-anchored HSL rotation)
# ---------------------------------------------------------------------------

class TestPalette:
    def test_accent_is_first_color(self):
        pal = c._palette("#2563EB", 4)
        assert pal[0] == "#2563EB"
        assert len(pal) == 4

    def test_distinct_colors(self):
        pal = c._palette("#059669", 5)
        assert len(set(pal)) >= 4  # rotations produce distinct hues

    def test_bad_accent_falls_back(self):
        pal = c._palette("not-a-color", 3)
        assert len(pal) == 3
        assert all(p.startswith("#") for p in pal)

    def test_hsl_roundtrip_sane(self):
        # rgb -> hsl -> hex should land near the input for a saturated color
        h, s, lum = c._rgb_to_hsl(37, 99, 235)
        out = c._hsl_to_hex(h, s, lum)
        assert out.startswith("#") and len(out) == 7


# ---------------------------------------------------------------------------
# Render smoke tests (matplotlib-gated)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _mpl():
    return pytest.importorskip("matplotlib")


class TestRenderSmoke:
    @pytest.mark.parametrize("chart_type", [
        "bar", "line", "pie", "scatter", "area",
        "stacked_bar", "stacked_area", "horizontal_bar",
    ])
    def test_each_type_renders_png(self, _mpl, chart_type):
        data = c._render_chart(
            title="T", chart_type=chart_type, x_label="X", y_label="Y",
            labels=["A", "B", "C", "D"],
            datasets=[{"name": "S1", "values": [10, 25, 18, 30]},
                      {"name": "S2", "values": [5, 12, 9, 20]}],
        )
        assert data[:8] == PNG_SIG

    def test_empty_renders_placeholder(self, _mpl):
        data = c._render_chart(
            title="Empty", chart_type="bar", labels=["A", "B"],
            datasets=[{"name": "x", "values": []}],
        )
        assert data[:8] == PNG_SIG

    def test_themed_currency_sorted_bar(self, _mpl):
        data = c._render_chart(
            title="Revenue", chart_type="bar", y_label="Revenue ($M)",
            labels=["Q1", "Q2", "Q3", "Q4"],
            datasets=[{"name": "2025", "values": [12.4, 15.1, 18.9, 22.3]}],
            theme_name="emerald", sort="desc", subtitle="FY2025", caption="Source: x",
        )
        assert data[:8] == PNG_SIG

    def test_pie_with_many_slices_groups_other(self, _mpl):
        # 9 slices must not crash — renderer groups the tail into "Other".
        data = c._render_chart(
            title="Share", chart_type="pie",
            labels=[f"S{i}" for i in range(9)],
            datasets=[{"name": "v", "values": [40, 20, 12, 9, 7, 5, 3, 2, 2]}],
            theme_name="slate", donut=True,
        )
        assert data[:8] == PNG_SIG

    def test_line_trend_line_no_crash(self, _mpl):
        data = c._render_chart(
            title="Growth", chart_type="line", y_label="Users",
            labels=["Jan", "Feb", "Mar", "Apr", "May"],
            datasets=[{"name": "MAU", "values": [14000, 22000, 31000, 38000, 51000]}],
            theme_name="modern", trend_line=True,
        )
        assert data[:8] == PNG_SIG

    def test_all_themes_render(self, _mpl):
        for theme in ("slate", "corporate", "modern", "emerald", "rose"):
            data = c._render_chart(
                title="T", chart_type="bar", labels=["A", "B"],
                datasets=[{"name": "s", "values": [3, 7]}], theme_name=theme,
            )
            assert data[:8] == PNG_SIG, theme


# ---------------------------------------------------------------------------
# Tool path: new fields persist + render succeeds (matplotlib-gated)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.save = AsyncMock(return_value={
        "id": "ch1", "filename": "c.png", "display_name": "c.png",
        "format": "png", "size_bytes": 10, "path": "p/c.png",
        "download_url": "/api/artifacts/ch1/download",
    })
    store.get_file_path = MagicMock(return_value=Path("/tmp/c.png"))
    return store


class TestChartToolFields:
    @pytest.mark.asyncio
    async def test_execute_persists_presentation_fields(self, _mpl, mock_store):
        import json

        from augmentum.tools.artifact_chart import ChartTool
        tool = ChartTool(mock_store)
        result = await tool.execute(
            title="Rev", chart_type="bar", y_label="Revenue ($M)",
            labels=["Q1", "Q2", "Q3", "Q4"],
            datasets=[{"name": "s", "values": [10, 20, 30, 40]}],
            theme="emerald", value_format="currency", sort="desc",
            subtitle="FY25", caption="src",
        )
        assert result.success
        saved = json.loads(mock_store.save.call_args.kwargs["source_json"])
        assert saved["value_format"] == "currency"
        assert saved["sort"] == "desc"
        assert saved["theme"] == "emerald"
        assert saved["subtitle"] == "FY25"
        # Preview card carries the format hints for the frontend.
        assert result.card["preview"]["value_format"] == "currency"

    @pytest.mark.asyncio
    async def test_invalid_value_format_falls_back_to_auto(self, _mpl, mock_store):
        import json

        from augmentum.tools.artifact_chart import ChartTool
        tool = ChartTool(mock_store)
        result = await tool.execute(
            title="X", chart_type="bar", labels=["A", "B"],
            datasets=[{"name": "s", "values": [1, 2]}],
            value_format="bogus", sort="weird",
        )
        assert result.success
        saved = json.loads(mock_store.save.call_args.kwargs["source_json"])
        assert saved["value_format"] == "auto"
        assert saved["sort"] == "none"
