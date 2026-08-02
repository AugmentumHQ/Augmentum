"""Tests for utility tools — calculator, datetime, unit converter, text analysis, JSON, hash."""

from __future__ import annotations

import math

import pytest

from augmentum.tools.calculator import CalculatorTool, safe_calculate
from augmentum.tools.datetime_tool import DateTimeTool
from augmentum.tools.hash_tool import HashTool, compute_hash
from augmentum.tools.json_tool import JsonTool
from augmentum.tools.text_analysis import TextAnalysisTool, analyze_text
from augmentum.tools.unit_converter import UnitConverterTool, convert

# ============================================================
# Calculator
# ============================================================


class TestSafeCalculate:
    def test_basic_arithmetic(self):
        assert safe_calculate("2 + 3") == 5
        assert safe_calculate("10 - 4") == 6
        assert safe_calculate("3 * 7") == 21
        assert safe_calculate("20 / 4") == 5.0

    def test_operator_precedence(self):
        assert safe_calculate("2 + 3 * 4") == 14
        assert safe_calculate("(2 + 3) * 4") == 20

    def test_nested_expressions(self):
        assert safe_calculate("(1 + 2) * (3 + 4)") == 21

    def test_unary_operators(self):
        assert safe_calculate("-5") == -5
        assert safe_calculate("-3 + 7") == 4

    def test_power(self):
        assert safe_calculate("2 ** 10") == 1024

    def test_math_functions(self):
        assert safe_calculate("sqrt(144)") == 12.0
        assert safe_calculate("abs(-42)") == 42
        assert abs(safe_calculate("sin(0)")) < 1e-10
        assert abs(safe_calculate("cos(0)") - 1.0) < 1e-10

    def test_constants(self):
        assert abs(safe_calculate("pi") - math.pi) < 1e-10
        assert abs(safe_calculate("e") - math.e) < 1e-10

    def test_complex_expression(self):
        result = safe_calculate("sqrt(3**2 + 4**2)")
        assert abs(result - 5.0) < 1e-10

    def test_forbidden_keywords(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_calculate("import os")

    def test_dunder_blocked(self):
        with pytest.raises(ValueError, match="forbidden"):
            safe_calculate("__import__('os')")

    def test_invalid_syntax(self):
        with pytest.raises(SyntaxError):
            safe_calculate("2 +")

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_calculate("1 / 0")

    def test_unknown_variable(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            safe_calculate("x + 1")

    def test_unknown_function(self):
        with pytest.raises(ValueError, match="Unknown function"):
            safe_calculate("foo(42)")


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 3 * 4")
        assert result.success
        assert result.output == "14"

    @pytest.mark.asyncio
    async def test_execute_empty(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_error(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="import os")
        assert not result.success
        assert "error" in result.error.lower() or "forbidden" in result.error.lower()

    def test_tool_metadata(self):
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert tool.input_schema["required"] == ["expression"]


# ============================================================
# DateTime
# ============================================================


class TestDateTimeTool:
    @pytest.mark.asyncio
    async def test_now(self):
        tool = DateTimeTool()
        result = await tool.execute(action="now")
        assert result.success
        assert "T" in result.output  # ISO format

    @pytest.mark.asyncio
    async def test_now_with_timezone(self):
        tool = DateTimeTool()
        result = await tool.execute(action="now", timezone="America/New_York")
        # May fail if tzdata package is not installed (Windows without it)
        if not result.success:
            assert "timezone" in result.error.lower()
        else:
            assert "T" in result.output

    @pytest.mark.asyncio
    async def test_parse_iso(self):
        tool = DateTimeTool()
        result = await tool.execute(action="parse", date="2024-03-15")
        assert result.success
        assert "2024-03-15" in result.output

    @pytest.mark.asyncio
    async def test_parse_us_format(self):
        tool = DateTimeTool()
        result = await tool.execute(action="parse", date="03/15/2024")
        assert result.success

    @pytest.mark.asyncio
    async def test_diff(self):
        tool = DateTimeTool()
        result = await tool.execute(action="diff", date="2024-01-01", date2="2024-01-31")
        assert result.success
        assert result.metadata["total_days"] == 30

    @pytest.mark.asyncio
    async def test_add_days(self):
        tool = DateTimeTool()
        result = await tool.execute(action="add", date="2024-01-01", days=10)
        assert result.success
        assert "2024-01-11" in result.output

    @pytest.mark.asyncio
    async def test_calendar(self):
        tool = DateTimeTool()
        result = await tool.execute(action="calendar", year=2024, month=2)
        assert result.success
        assert "February" in result.output
        assert result.metadata["days_in_month"] == 29  # 2024 is a leap year

    @pytest.mark.asyncio
    async def test_day_of_week(self):
        tool = DateTimeTool()
        result = await tool.execute(action="day_of_week", date="2024-01-01")
        assert result.success
        assert result.output == "Monday"

    @pytest.mark.asyncio
    async def test_parse_invalid(self):
        tool = DateTimeTool()
        result = await tool.execute(action="parse", date="not-a-date")
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = DateTimeTool()
        result = await tool.execute(action="foobar")
        assert not result.success

    def test_tool_metadata(self):
        tool = DateTimeTool()
        assert tool.name == "datetime"


# ============================================================
# Unit Converter
# ============================================================


class TestUnitConverter:
    def test_km_to_miles(self):
        result = convert(100, "km", "mi")
        assert abs(result - 62.137) < 0.01

    def test_feet_to_meters(self):
        result = convert(1, "ft", "m")
        assert abs(result - 0.3048) < 0.001

    def test_lb_to_kg(self):
        result = convert(1, "lb", "kg")
        assert abs(result - 0.4536) < 0.001

    def test_fahrenheit_to_celsius(self):
        result = convert(212, "F", "C")
        assert abs(result - 100.0) < 0.1

    def test_celsius_to_kelvin(self):
        result = convert(0, "C", "K")
        assert abs(result - 273.15) < 0.01

    def test_gallons_to_liters(self):
        result = convert(1, "gal", "l")
        assert abs(result - 3.785) < 0.01

    def test_mph_to_kph(self):
        result = convert(60, "mph", "kph")
        assert abs(result - 96.56) < 0.1

    def test_gb_to_mb(self):
        result = convert(1, "GB", "MB")
        assert abs(result - 1000.0) < 0.1

    def test_unknown_unit(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            convert(1, "foobar", "m")

    def test_incompatible_categories(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            convert(1, "kg", "m")

    def test_freezing_point(self):
        result = convert(32, "F", "C")
        assert abs(result - 0.0) < 0.1


class TestUnitConverterTool:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = UnitConverterTool()
        result = await tool.execute(value=100, from_unit="km", to_unit="mi")
        assert result.success
        assert "62.13" in result.output

    @pytest.mark.asyncio
    async def test_execute_error(self):
        tool = UnitConverterTool()
        result = await tool.execute(value=1, from_unit="kg", to_unit="mi")
        assert not result.success

    @pytest.mark.asyncio
    async def test_missing_value(self):
        tool = UnitConverterTool()
        result = await tool.execute(from_unit="km", to_unit="mi")
        assert not result.success

    def test_tool_metadata(self):
        tool = UnitConverterTool()
        assert tool.name == "unit_converter"


# ============================================================
# Text Analysis
# ============================================================


class TestTextAnalysis:
    def test_basic_analysis(self):
        text = "Hello world. This is a test. Three sentences here."
        stats = analyze_text(text)
        assert stats["words"] == 9
        assert stats["sentences"] == 3
        assert stats["characters"] > 0

    def test_readability_present(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        stats = analyze_text(text)
        assert "readability" in stats
        assert "flesch_reading_ease" in stats["readability"]
        assert "flesch_kincaid_grade" in stats["readability"]

    def test_top_words(self):
        text = "hello hello hello world world"
        stats = analyze_text(text)
        assert stats["top_words"]["hello"] == 3
        assert stats["top_words"]["world"] == 2

    def test_empty_text(self):
        stats = analyze_text("")
        assert "error" in stats

    def test_reading_time(self):
        # 238 words => ~1 minute reading time
        text = "word " * 238
        stats = analyze_text(text)
        assert abs(stats["reading_time_minutes"] - 1.0) < 0.1

    def test_lexical_diversity(self):
        text = "a b c d e f g h i j"
        stats = analyze_text(text)
        assert stats["lexical_diversity"] == 1.0  # all unique


class TestTextAnalysisTool:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = TextAnalysisTool()
        result = await tool.execute(text="Hello world. This is a test.")
        assert result.success
        assert "words" in result.output

    @pytest.mark.asyncio
    async def test_execute_empty(self):
        tool = TextAnalysisTool()
        result = await tool.execute(text="")
        assert not result.success

    def test_tool_metadata(self):
        tool = TextAnalysisTool()
        assert tool.name == "text_analysis"


# ============================================================
# JSON Tool
# ============================================================


class TestJsonTool:
    @pytest.mark.asyncio
    async def test_validate_valid(self):
        tool = JsonTool()
        result = await tool.execute(action="validate", json_text='{"key": "value"}')
        assert result.success
        assert result.metadata["valid"]

    @pytest.mark.asyncio
    async def test_validate_invalid(self):
        tool = JsonTool()
        result = await tool.execute(action="validate", json_text="{bad json")
        assert result.success  # The action itself succeeded
        assert not result.metadata["valid"]

    @pytest.mark.asyncio
    async def test_format(self):
        tool = JsonTool()
        result = await tool.execute(action="format", json_text='{"a":1,"b":2}')
        assert result.success
        assert "\n" in result.output  # pretty printed

    @pytest.mark.asyncio
    async def test_minify(self):
        tool = JsonTool()
        result = await tool.execute(action="minify", json_text='{\n  "a": 1,\n  "b": 2\n}')
        assert result.success
        assert " " not in result.output  # minified

    @pytest.mark.asyncio
    async def test_query_nested(self):
        tool = JsonTool()
        data = '{"a": {"b": {"c": 42}}}'
        result = await tool.execute(action="query", json_text=data, path="$.a.b.c")
        assert result.success
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_query_array_index(self):
        tool = JsonTool()
        data = '{"items": [10, 20, 30]}'
        result = await tool.execute(action="query", json_text=data, path="$.items[1]")
        assert result.success
        assert "20" in result.output

    @pytest.mark.asyncio
    async def test_keys(self):
        tool = JsonTool()
        result = await tool.execute(action="keys", json_text='{"a": 1, "b": 2, "c": 3}')
        assert result.success
        assert result.metadata["count"] == 3

    @pytest.mark.asyncio
    async def test_diff_identical(self):
        tool = JsonTool()
        j = '{"a": 1}'
        result = await tool.execute(action="diff", json_text=j, json_text2=j)
        assert result.success
        assert result.metadata["identical"]

    @pytest.mark.asyncio
    async def test_diff_changed(self):
        tool = JsonTool()
        result = await tool.execute(
            action="diff",
            json_text='{"a": 1, "b": 2}',
            json_text2='{"a": 1, "b": 3}',
        )
        assert result.success
        assert not result.metadata["identical"]
        assert "$.b" in result.output

    @pytest.mark.asyncio
    async def test_diff_added_removed(self):
        tool = JsonTool()
        result = await tool.execute(
            action="diff",
            json_text='{"a": 1}',
            json_text2='{"b": 2}',
        )
        assert result.success
        assert "removed" in result.output
        assert "added" in result.output

    @pytest.mark.asyncio
    async def test_merge_objects(self):
        tool = JsonTool()
        result = await tool.execute(
            action="merge",
            json_text='{"a": 1}',
            json_text2='{"b": 2}',
        )
        assert result.success
        import json
        merged = json.loads(result.output)
        assert merged == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_merge_arrays(self):
        tool = JsonTool()
        result = await tool.execute(
            action="merge",
            json_text='[1, 2]',
            json_text2='[3, 4]',
        )
        assert result.success
        import json
        merged = json.loads(result.output)
        assert merged == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_type_info(self):
        tool = JsonTool()
        result = await tool.execute(action="type", json_text='{"a": 1, "b": [1, 2]}')
        assert result.success

    def test_tool_metadata(self):
        tool = JsonTool()
        assert tool.name == "json_tool"


# ============================================================
# Hash Tool
# ============================================================


class TestHashTool:
    def test_sha256_deterministic(self):
        h1 = compute_hash("hello", "sha256")
        h2 = compute_hash("hello", "sha256")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = compute_hash("hello", "sha256")
        h2 = compute_hash("world", "sha256")
        assert h1 != h2

    def test_md5(self):
        h = compute_hash("hello", "md5")
        assert len(h) == 32

    def test_sha1(self):
        h = compute_hash("hello", "sha1")
        assert len(h) == 40

    def test_sha256_length(self):
        h = compute_hash("hello", "sha256")
        assert len(h) == 64

    def test_blake2b(self):
        h = compute_hash("hello", "blake2b")
        assert len(h) > 0

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported"):
            compute_hash("hello", "rot13")

    @pytest.mark.asyncio
    async def test_hash_action(self):
        tool = HashTool()
        result = await tool.execute(action="hash", text="hello", algorithm="sha256")
        assert result.success
        assert len(result.output) == 64

    @pytest.mark.asyncio
    async def test_hmac_action(self):
        tool = HashTool()
        result = await tool.execute(action="hmac", text="hello", key="secret", algorithm="sha256")
        assert result.success

    @pytest.mark.asyncio
    async def test_hmac_no_key(self):
        tool = HashTool()
        result = await tool.execute(action="hmac", text="hello")
        assert not result.success

    @pytest.mark.asyncio
    async def test_compare_match(self):
        tool = HashTool()
        expected = compute_hash("hello", "sha256")
        result = await tool.execute(action="compare", text="hello", expected=expected)
        assert result.success
        assert result.metadata["match"]

    @pytest.mark.asyncio
    async def test_compare_no_match(self):
        tool = HashTool()
        result = await tool.execute(action="compare", text="hello", expected="0000")
        assert result.success
        assert not result.metadata["match"]

    @pytest.mark.asyncio
    async def test_empty_text(self):
        tool = HashTool()
        result = await tool.execute(action="hash", text="")
        assert not result.success

    def test_tool_metadata(self):
        tool = HashTool()
        assert tool.name == "hash"
