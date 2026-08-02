"""Tests for UnitConverterTool — unit conversion across categories."""

from __future__ import annotations

import pytest

from augmentum.tools.unit_converter import UnitConverterTool, _convert_temperature, convert


class TestTemperatureConversion:
    """Non-linear temperature conversions."""

    def test_celsius_to_fahrenheit(self):
        result = convert(0, "C", "F")
        assert abs(result - 32.0) < 0.01

    def test_fahrenheit_to_celsius(self):
        result = convert(212, "F", "C")
        assert abs(result - 100.0) < 0.01

    def test_celsius_to_kelvin(self):
        result = convert(0, "C", "K")
        assert abs(result - 273.15) < 0.01

    def test_kelvin_to_celsius(self):
        result = convert(273.15, "K", "C")
        assert abs(result - 0.0) < 0.01

    def test_fahrenheit_to_kelvin(self):
        result = convert(32, "F", "K")
        assert abs(result - 273.15) < 0.01

    def test_body_temp_f_to_c(self):
        result = convert(98.6, "fahrenheit", "celsius")
        assert abs(result - 37.0) < 0.1


class TestLengthConversion:
    """Length unit conversions."""

    def test_km_to_miles(self):
        result = convert(1, "km", "mi")
        assert abs(result - 0.621371) < 0.001

    def test_meters_to_feet(self):
        result = convert(1, "m", "ft")
        assert abs(result - 3.28084) < 0.001

    def test_inches_to_cm(self):
        result = convert(1, "in", "cm")
        assert abs(result - 2.54) < 0.001

    def test_miles_to_km(self):
        result = convert(1, "mi", "km")
        assert abs(result - 1.60934) < 0.001


class TestWeightConversion:
    """Mass/weight unit conversions."""

    def test_kg_to_pounds(self):
        result = convert(1, "kg", "lb")
        assert abs(result - 2.20462) < 0.01

    def test_pounds_to_kg(self):
        result = convert(1, "lb", "kg")
        assert abs(result - 0.453592) < 0.001

    def test_grams_to_ounces(self):
        result = convert(28.3495, "g", "oz")
        assert abs(result - 1.0) < 0.01

    def test_tonnes_to_kg(self):
        result = convert(1, "t", "kg")
        assert abs(result - 1000.0) < 0.01


class TestInvalidUnits:
    """Error handling for invalid conversions."""

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            convert(1, "zorps", "blargs")

    def test_cross_category_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            convert(1, "kg", "km")

    def test_unknown_temperature_raises(self):
        with pytest.raises(ValueError, match="Unknown temperature"):
            _convert_temperature(100, "Q", "C")


class TestUnitConverterTool:
    """ToolResult contract for UnitConverterTool."""

    async def test_execute_basic_conversion(self):
        tool = UnitConverterTool()
        result = await tool.execute(value=100, from_unit="km", to_unit="mi")
        assert result.success is True
        assert result.metadata["result"] is not None

    async def test_execute_missing_value(self):
        tool = UnitConverterTool()
        result = await tool.execute(from_unit="km", to_unit="mi")
        assert result.success is False

    async def test_execute_missing_units(self):
        tool = UnitConverterTool()
        result = await tool.execute(value=100)
        assert result.success is False

    async def test_execute_output_format(self):
        tool = UnitConverterTool()
        result = await tool.execute(value=100, from_unit="C", to_unit="F")
        assert result.success is True
        assert "100" in result.output
        assert "C" in result.output
        assert "F" in result.output

    async def test_tool_properties(self):
        tool = UnitConverterTool()
        assert tool.name == "unit_converter"
        assert tool.category.value == "verify"
