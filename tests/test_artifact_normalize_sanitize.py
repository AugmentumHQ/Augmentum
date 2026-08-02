"""Tests for artifact normalization, sanitization, and template system."""

from __future__ import annotations

import pytest

from augmentum.tools.artifact_normalize import (
    normalize_bool,
    normalize_chart_datasets,
    normalize_chart_labels,
    normalize_int,
    normalize_list,
    normalize_number,
    normalize_sections,
    normalize_sheets,
    normalize_slides,
    normalize_str,
)
from augmentum.tools.artifact_sanitize import (
    sanitize_heading,
    sanitize_sections,
    sanitize_sheets,
    sanitize_slides,
    sanitize_text,
)
from augmentum.tools.artifact_templates import ArtifactTemplate

# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeStr:
    def test_none_returns_default(self):
        assert normalize_str(None, "fallback") == "fallback"

    def test_list_joined(self):
        assert normalize_str(["a", "b"]) == "a\nb"

    def test_int_converted(self):
        assert normalize_str(42) == "42"

    def test_string_passthrough(self):
        assert normalize_str("hello") == "hello"


class TestNormalizeInt:
    def test_none_returns_default(self):
        assert normalize_int(None, 5) == 5

    def test_string_number(self):
        assert normalize_int("42") == 42

    def test_float_truncated(self):
        assert normalize_int(3.7) == 3

    def test_garbage_returns_default(self):
        assert normalize_int("abc", 0) == 0


class TestNormalizeBool:
    def test_true_string(self):
        assert normalize_bool("true") is True

    def test_false_string(self):
        assert normalize_bool("false") is False

    def test_none_returns_default(self):
        assert normalize_bool(None, True) is True

    def test_int_one_is_true(self):
        assert normalize_bool(1) is True


class TestNormalizeList:
    def test_none_returns_default(self):
        assert normalize_list(None) == []

    def test_json_string_parsed(self):
        assert normalize_list('[1, 2, 3]') == [1, 2, 3]

    def test_single_string_wrapped(self):
        assert normalize_list("hello") == ["hello"]

    def test_list_passthrough(self):
        assert normalize_list([1, 2]) == [1, 2]

    def test_empty_string_returns_default(self):
        assert normalize_list("") == []


class TestNormalizeNumber:
    def test_string_with_comma(self):
        assert normalize_number("1,234.56") == 1234.56

    def test_string_with_percent(self):
        assert normalize_number("45.6%") == 45.6

    def test_none_returns_default(self):
        assert normalize_number(None, 0.0) == 0.0


class TestNormalizeSections:
    def test_normalizes_section_fields(self):
        sections = [{"heading": "Title", "body": ["line1", "line2"], "level": "2"}]
        result = normalize_sections(sections)
        assert result[0]["heading"] == "Title"
        assert result[0]["body"] == "line1\nline2"
        assert result[0]["level"] == 2

    def test_clamps_level_high(self):
        result = normalize_sections([{"heading": "H", "body": "B", "level": 10}])
        assert result[0]["level"] == 4

    def test_clamps_level_low(self):
        result = normalize_sections([{"heading": "H", "body": "B", "level": 0}])
        assert result[0]["level"] == 1

    def test_string_item_becomes_body(self):
        result = normalize_sections(["Just text"])
        assert result[0]["body"] == "Just text"
        assert result[0]["heading"] == ""

    def test_non_dict_items_skipped(self):
        result = normalize_sections([42, None, True])
        # String "42" would be wrapped, but int 42 should be skipped
        # Actually normalize_sections calls normalize_list first, then checks isinstance
        assert len(result) == 0


class TestNormalizeSlides:
    def test_invalid_layout_defaults_to_content(self):
        result = normalize_slides([{"title": "T", "layout": "invalid"}])
        assert result[0]["layout"] == "content"

    def test_valid_layout_preserved(self):
        result = normalize_slides([{"title": "T", "layout": "two_column"}])
        assert result[0]["layout"] == "two_column"

    def test_blank_layout(self):
        result = normalize_slides([{"title": "T", "layout": "blank"}])
        assert result[0]["layout"] == "blank"


class TestNormalizeSheets:
    def test_headers_to_strings(self):
        result = normalize_sheets([{"name": "S1", "headers": [1, 2], "rows": []}])
        assert result[0]["headers"] == ["1", "2"]

    def test_dict_rows_converted(self):
        result = normalize_sheets([{
            "name": "S1",
            "headers": ["a", "b"],
            "rows": [{"a": 1, "b": 2}],
        }])
        assert result[0]["rows"] == [[1, 2]]

    def test_name_truncated_to_31(self):
        long_name = "A" * 50
        result = normalize_sheets([{"name": long_name, "headers": [], "rows": []}])
        assert len(result[0]["name"]) <= 31


class TestNormalizeChart:
    def test_labels_to_strings(self):
        result = normalize_chart_labels([1, "two", None])
        assert result == ["1", "two", ""]

    def test_datasets_values_to_numbers(self):
        result = normalize_chart_datasets([{"name": "S1", "values": ["1", "2.5"]}])
        assert result[0]["values"] == [1.0, 2.5]

    def test_datasets_missing_name(self):
        result = normalize_chart_datasets([{"values": [1, 2]}])
        assert result[0]["name"] == "Series 1"

    def test_datasets_accept_chartjs_shape(self):
        """``{label, data}`` is what LLMs emit — Chart.js is the shape they were
        trained on. Reading only ``values`` silently produced an EMPTY series,
        so the chart rendered blank with no error and the model thought it had
        succeeded. This is the regression guard for that."""
        result = normalize_chart_datasets([{"label": "Revenue", "data": [1, "2.5"]}])
        assert result == [{"name": "Revenue", "values": [1.0, 2.5]}]

    def test_datasets_accept_bare_number_list(self):
        """A flat list is ONE series, not N single-point series."""
        assert normalize_chart_datasets([1, 2, 3]) == [
            {"name": "Series 1", "values": [1.0, 2.0, 3.0]},
        ]

    def test_datasets_accept_list_of_lists(self):
        result = normalize_chart_datasets([[1, 2], [3, 4]])
        assert [d["values"] for d in result] == [[1.0, 2.0], [3.0, 4.0]]
        assert [d["name"] for d in result] == ["Series 1", "Series 2"]

    def test_datasets_accept_name_to_values_mapping(self):
        assert normalize_chart_datasets({"Q1": [1, 2]}) == [
            {"name": "Q1", "values": [1.0, 2.0]},
        ]

    def test_datasets_accept_json_string(self):
        result = normalize_chart_datasets('[{"label":"A","data":[5]}]')
        assert result == [{"name": "A", "values": [5.0]}]

    def test_empty_series_dropped_not_kept_blank(self):
        """A series with nothing plottable must DISAPPEAR so the caller's
        "labels and datasets are required" check fires and the model gets a
        real error, instead of the tool rendering an empty canvas."""
        assert normalize_chart_datasets([{"label": "X", "data": []}]) == []
        assert normalize_chart_datasets([{"name": "X", "values": None}]) == []
        assert normalize_chart_datasets(None) == []
        assert normalize_chart_datasets([{"not_a_series": 1}]) == []


# ---------------------------------------------------------------------------
# Sanitize tests
# ---------------------------------------------------------------------------


class TestSanitizeText:
    def test_strips_preamble(self):
        text = "Here is the document. The economy grew by 3%."
        result = sanitize_text(text)
        assert "Here is" not in result
        assert "3%" in result

    def test_strips_certainly_preamble(self):
        text = "Certainly! Here is your analysis. GDP grew 4%."
        result = sanitize_text(text)
        assert "Certainly" not in result

    def test_strips_placeholder_brackets(self):
        text = "Revenue was [Insert data here] million."
        result = sanitize_text(text)
        assert "[Insert" not in result

    def test_strips_filler_phrases(self):
        text = "It is important to note that the sky is blue."
        result = sanitize_text(text)
        assert "important to note" not in result
        assert "sky is blue" in result

    def test_strips_ai_disclaimer(self):
        text = "Note: This is a generated sample document. Revenue was $5M."
        result = sanitize_text(text)
        assert "generated sample" not in result

    def test_empty_input_unchanged(self):
        assert sanitize_text("") == ""

    def test_clean_text_preserved(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert sanitize_text(text) == text


class TestSanitizeHeading:
    def test_strips_number_prefix(self):
        assert sanitize_heading("1. Introduction") == "Introduction"

    def test_strips_section_prefix(self):
        assert sanitize_heading("Section 3: Methods") == "Methods"

    def test_strips_slide_prefix(self):
        assert sanitize_heading("Slide 2: Overview") == "Overview"

    def test_empty_heading(self):
        assert sanitize_heading("") == ""

    def test_clean_heading_preserved(self):
        assert sanitize_heading("Executive Summary") == "Executive Summary"


class TestSanitizeSections:
    def test_removes_empty_sections(self):
        sections = [
            {"heading": "Good", "body": "Content here"},
            {"heading": "", "body": ""},
        ]
        result = sanitize_sections(sections)
        assert len(result) == 1

    def test_adds_untitled_to_bodyonly(self):
        sections = [{"heading": "", "body": "Some content"}]
        result = sanitize_sections(sections)
        assert result[0]["heading"] == "Untitled"


class TestSanitizeSlides:
    def test_removes_empty_slides(self):
        slides = [
            {"title": "Good", "body": "Content"},
            {"title": "", "body": ""},
        ]
        result = sanitize_slides(slides)
        assert len(result) == 1

    def test_preserves_notes(self):
        slides = [{"title": "Slide", "body": "Content", "notes": "Speaker note"}]
        result = sanitize_slides(slides)
        assert result[0]["notes"] == "Speaker note"


class TestSanitizeSheets:
    def test_cleans_tbd_cells(self):
        sheets = [{"name": "S1", "headers": ["A"], "rows": [["TBD"]]}]
        result = sanitize_sheets(sheets)
        assert result[0]["rows"][0][0] == ""

    def test_numeric_cells_preserved(self):
        sheets = [{"name": "S1", "headers": ["A"], "rows": [[42]]}]
        result = sanitize_sheets(sheets)
        assert result[0]["rows"][0][0] == 42

    def test_clean_strings_preserved(self):
        sheets = [{"name": "S1", "headers": ["A"], "rows": [["Revenue"]]}]
        result = sanitize_sheets(sheets)
        assert result[0]["rows"][0][0] == "Revenue"


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestArtifactTemplate:
    def test_template_construction(self):
        t = ArtifactTemplate(
            name="test",
            description="A test template",
            format="pdf",
            category="business",
        )
        assert t.name == "test"
        assert t.format == "pdf"
        assert t.category == "business"

    def test_template_is_frozen(self):
        t = ArtifactTemplate(name="t", description="d", format="pdf", category="c")
        with pytest.raises(AttributeError):
            t.name = "changed"

    def test_template_default_fields(self):
        t = ArtifactTemplate(name="t", description="d", format="pdf", category="c")
        assert t.layout == {}
        assert t.context_prompt == ""
